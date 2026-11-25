"""Portable CPU reference trainer for the tri-mode controller."""
import argparse
from pathlib import Path

import torch
from torch import nn

from .actor import HOME
from .networks import build_mlp
from .plant import Plant, load_model, new_actor, targets
from .recipe import DT, GAMMA, SEED, TAU, actor_lr, effective, mode_at, observed, reward, schedule


def train(scene, output, updates, horizon, seed, mass):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    model = load_model(scene, mass)
    plant = Plant(model, seed + 2, noisy=True)
    actor = new_actor(model)
    dynamics = build_mlp(19, (128, 128), 12)
    critic = build_mlp(34, (128, 128), 1)
    target = build_mlp(34, (128, 128), 1)
    target.load_state_dict(critic.state_dict())
    target.requires_grad_(False)
    wm_opt = torch.optim.AdamW(dynamics.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=2e-4)
    enc_opt = torch.optim.AdamW(actor.history_encoder.parameters(), lr=2e-4)
    actor_opt = torch.optim.AdamW([p for name, p in actor.named_parameters() if not name.startswith('history_encoder.')], lr=2e-4)
    limits = torch.from_numpy(model.jnt_range[:6].copy()).float()
    home = torch.tensor(HOME)[None]
    replay = []
    clock = 0
    ctx = torch.tensor([[mass, 0., 0., 0., mass*.0002666667, mass*.0002666667, mass*.0002666667, 0., 0., 0.]])

    def ref(tick):
        phase = min(tick, 250)*DT + offset
        moving = 50 <= tick < 250
        q = home if tick < 50 else (home + amplitude*torch.sin(phase)).clamp(limits[:,0], limits[:,1])
        dq = torch.where((q > limits[:,0]) & (q < limits[:,1]), amplitude*torch.cos(phase), 0.) if moving else torch.zeros_like(q)
        return torch.cat((q, dq), -1), torch.tensor([moving])

    def critic_input(x, g, joint, mode, u):
        masks = actor.masks[mode]
        return torch.cat((x, g[:, :3], torch.where(masks[:,1:2].bool(), g[:,3:], 0.),
                          torch.where(masks[:,2:3].bool(), joint, 0.), masks, u), -1)

    def collect():
        nonlocal clock, code, amplitude, offset
        if clock % 300 == 0:
            clock = 0
            plant.dr[:, :6] = .0015*torch.randn((1, 6), generator=rng)
            plant.delay = int(torch.randint(0, 6, (), generator=rng))
            plant.reset()
            code = schedule(1, 'cpu', rng)
            amplitude = .3*(2*torch.rand((1,6), generator=rng)-1)
            offset = torch.rand((1,6), generator=rng)*6.28
        h, x, u0 = plant.history, plant.state(), plant.issued
        reference, _ = ref(clock)
        g, joint = targets(actor.fk, reference[:, :6], reference[:, :6])
        mode = mode_at(code, clock)
        actor.obs_normalizer.update(h[:, -1, :18])
        u = actor.rollout(h, g, joint, mode)
        u = u + .03*torch.randn(u.shape, generator=rng)
        xn = plant.step(u)
        if not torch.isfinite(xn).all():
            raise RuntimeError('Nonfinite plant state')
        replay.append((x, effective(u0, u, plant.delay), xn, h, u0, plant.dr.clone(), plant.delay, clock, code, amplitude, offset))
        if len(replay) > 4096:
            replay.pop(0)
        clock += 1

    def step(opt, loss):
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1., error_if_nonfinite=True)
        opt.step()

    code = amplitude = offset = None
    for update in range(1, updates+1):
        with torch.no_grad():
            for _ in range(32 if update == 1 else 4):
                collect()
        sample_ids = torch.randint(len(replay), (32,), generator=rng).tolist()
        x, u, xn, h = (torch.cat([replay[i][k] for i in sample_ids]) for k in range(4))
        wm_loss = (dynamics(torch.cat((x, u, x.new_full((len(x),1), mass)), -1))-(xn-x)).square().mean()
        step(wm_opt, wm_loss)
        enc_loss, _, _ = actor.payload_loss(h, ctx)
        step(enc_opt, .01*enc_loss)

        # Preserve the live collector's reference generator while imagining an older episode.
        live = code, amplitude, offset
        x, _, _, h, previous, dr, delay, tick, code, amplitude, offset = replay[int(torch.randint(len(replay), (), generator=rng))]
        actor.history_encoder.requires_grad_(False)
        dynamics.requires_grad_(False)
        rows, returns = [], []
        steps = min(horizon, 300-tick)
        for t in range(steps):
            r0, _ = ref(tick+t)
            r1, tracked = ref(tick+t+1)
            g0, q0 = targets(actor.fk, r0[:, :6], r0[:, :6])
            g1, _ = targets(actor.fk, r1[:, :6], r1[:, :6])
            mode = mode_at(code, tick+t)
            rows.append(critic_input(x, g0, q0, mode, previous))
            issued = actor.rollout(h, g0, q0, mode)
            applied = effective(previous, issued, delay)
            nxt = x + dynamics(torch.cat((x, applied, x.new_full((1,1), mass)), -1))
            score, *_ = reward(actor.fk, nxt[:, :6], nxt[:, 6:], g1, issued, previous, g0,
                               tracked, r1, x[:, 6:], mode_at(code, tick+t+1))
            returns.append(score)
            row = observed(torch.cat((nxt, issued, nxt.new_ones(1,1)), -1), dr, rng)
            h = torch.cat((h[:,1:], row[:,None]), 1)
            x, previous = nxt, issued
        final_tick = tick + steps
        final_ref, _ = ref(final_tick)
        final_goal, final_joint = targets(actor.fk, final_ref[:, :6], final_ref[:, :6])
        bootstrap = target(critic_input(x, final_goal, final_joint, mode_at(code, final_tick), previous)).squeeze(-1)
        if final_tick >= 300:
            bootstrap = bootstrap * 0
        objective = sum(GAMMA**i * r for i,r in enumerate(returns)) + .25*GAMMA**len(returns)*bootstrap
        for group in actor_opt.param_groups:
            group['lr'] = actor_lr(update)
        step(actor_opt, -objective.mean())
        actor.history_encoder.requires_grad_(True)
        dynamics.requires_grad_(True)
        with torch.no_grad():
            labels, value = [], bootstrap.detach()
            for r in reversed(returns):
                value = r.detach() + GAMMA*value
                labels.append(value)
        step(critic_opt, nn.functional.huber_loss(critic(torch.cat(rows).detach()).flatten(), torch.cat(labels[::-1])))
        with torch.no_grad():
            for dst, src in zip(target.parameters(), critic.parameters(), strict=True):
                dst.lerp_(src, TAU)
        code, amplitude, offset = live
        if update == 1 or update == updates or update % 50 == 0:
            print(f'update={update} world_model_mse={wm_loss.item():.6g} actor_return={objective.item():.6g}', flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save({'actor': actor.state_dict(), 'actor_updates': updates}, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--mass', type=float, default=1.)
    args = parser.parse_args()
    if args.updates < 1 or not 1 <= args.horizon <= 32:
        parser.error('Expected positive updates and a horizon of 1–32 ticks')
    if args.output.exists():
        parser.error('Output already exists')
    torch.set_num_threads(1)
    train(args.scene, args.output, args.updates, args.horizon, args.seed, args.mass)


if __name__ == '__main__':
    main()
