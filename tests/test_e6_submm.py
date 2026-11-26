"""CPU checks for the released E6 source hooks and artifact lineage."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from superreach.networks import build_mlp

ROOT = Path(__file__).resolve().parents[1]
E6 = importlib.import_module('gpu_train.runtime.openyam-50hz-policyik-eef-v1.e6_submm.recipe')


class FixtureFK(nn.Module):
    def forward(self, q, jac=False):
        p = q[:, :3]
        r = q.new_tensor([1., 0., 0., 0.]).expand(len(q), 4)
        if not jac:
            return p, r
        j = torch.eye(3, 6, dtype=q.dtype, device=q.device).expand(len(q), 3, 6)
        return p, r, j, None


def test_e6_inference_masks_and_joint_dr_loss():
    torch.manual_seed(7)
    actor = E6.Actor(FixtureFK(), None, build_mlp, 'dual_encoder', 'dr_reg_mlp').eval()
    torch.nn.init.normal_(actor.decoder[-1].weight, std=.005)  # Non-vacuous feature invariance.
    hist = torch.randn(3, 16, 19)
    hist[..., 18] = 1
    hist[:, :8, 18] = 0
    obs = dict(history=hist, goal_pose=torch.tensor([[.03, 0, 0, 1, 0, 0, 0]]).repeat(3, 1),
               goal_q=torch.randn(3, 6), mode=torch.arange(3), privileged_ctx=torch.zeros(3, 71))
    changed = {k: v.clone() for k, v in obs.items()}
    changed['history'][:, :8, :18] = float('nan')
    changed['privileged_ctx'].fill_(30.)
    changed['goal_q'][:2] = float('nan')
    changed['goal_pose'][1, 3:] = float('nan')
    action = actor(obs)
    assert action.shape == (3, 6) and torch.isfinite(action).all()
    torch.testing.assert_close(actor(changed), action, rtol=0, atol=0)
    loss = actor.sysid_loss(obs, None)
    assert loss > 0
    (action.square().mean() + .01 * loss).backward()
    for layer in (actor.history_encoder, actor.decoder, actor.aux_head):
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in layer.parameters())


def test_smooth_submm_reward_and_fork_clock():
    d = torch.tensor([[0., 0, 0], [.0005, 0, 0], [.002, 0, 0]], dtype=torch.float64, requires_grad=True)
    fine = E6.position_fine(d)
    torch.testing.assert_close(fine, (-d.square().sum(-1) / .002**2).exp())
    fine.sum().backward()
    assert d.grad[0, 0] == 0 and (d.grad[1:, 0] < 0).all()
    q = d.detach().new_zeros((3, 6))
    goal = torch.cat((d.detach(), q.new_tensor([1., 0., 0., 0.]).expand(3, 4)), -1)
    ref = q.repeat(1, 2)
    _, terms, *_ = E6.reward(FixtureFK(), q, q, goal, q, q, goal, torch.zeros(3, dtype=torch.bool), ref, q, torch.arange(3))
    torch.testing.assert_close(terms['position_fine'], fine.detach() * fine.new_tensor([1.5, 2., 0.]))
    assert E6.actor_lr(0) == 5e-5 and E6.actor_lr(4999) == 5e-6
    assert E6.actor_lr(0) > E6.actor_lr(2500) > E6.actor_lr(4999)


def test_wm_adds_33_frame_batch_after_short_update():
    class Batch(dict):
        @property
        def batch_size(self):
            return self['state'].shape[:2]

        def __getitem__(self, k):
            return super().__getitem__(k) if isinstance(k, str) else Batch({key: value[k] for key, value in self.items()})

    events = []
    state = torch.zeros(8, 33, 23)
    state[..., 6] = torch.tensor([.5, .4, .3, .2, .1, .01, .15, .05])[:, None]
    state[..., 19] = torch.arange(33)
    flags = torch.zeros(8, 33, dtype=torch.bool)
    batch = Batch(state=state, terminated=flags.clone(), truncated=flags.clone())

    def sample(length, size):
        events.append(('sample', length, size))
        return batch

    def update(batches):
        events.append(('long', tuple(batches[0].batch_size)))
        torch.testing.assert_close(batches[0]['state'][:, 0, 6], torch.tensor([.5, .4, .3, .01]))
        return {'loss': torch.tensor(0.)}

    def short():
        events.append('short')
        return {'short_seen': 1}

    learner = SimpleNamespace(wm_update=short, buffer=SimpleNamespace(sample=sample), long_batch=4,
                              wm=SimpleNamespace(update=update), count={'wm_optimizer_batches': 10})
    stats = E6.wm_update(learner)
    assert events == ['short', ('sample', 33, 8), ('long', (4, 33))]
    assert stats['wm33_frames'] == 33 and stats['wm33_steps'] == 32 and stats['wm33_low_speed'] == 1
    assert learner.count['wm_optimizer_batches'] == 11
    batch['truncated'][0, 12] = True
    with pytest.raises(AssertionError):
        E6.wm_update(learner)


def test_native_evaluation_matches_pinned_checkpoint():
    bundle = json.loads((ROOT / 'web/robots/openyam/BUNDLE.json').read_text())
    checkpoint = bundle['checkpoints']['e6-submm@20k']
    report = json.loads((ROOT / 'docs/evaluation-data/e6-submm-20k-native.json').read_text())
    assert list(bundle['checkpoints']) == [bundle['default_checkpoint']] == ['e6-submm@20k']
    assert report['checkpoint_sha256'] == checkpoint['checkpoint_sha256']
    assert report['actor_updates'] == checkpoint['actor_updates'] == E6.RECIPE['updates'] == 20000
    ids = {(r['suite'], r['mode'], r['case']) for r in report['cases']}
    assert len(ids) == len(report['cases']) == 129
