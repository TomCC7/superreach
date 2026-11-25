"""Local MuJoCo plant and causal sensor history."""
from pathlib import Path
import re

import mujoco as mj
import numpy as np
import torch

from .actor import HOME
from .recipe import DT, SUBSTEPS, observed


def load_model(scene: Path, mass: float = 0.):
    if not np.isfinite(mass) or mass < 0:
        raise ValueError('Payload must be finite and nonnegative')
    text = scene.read_text()
    text = re.sub(r'\bmeshdir="[^"]*"', 'meshdir="meshes"', text)
    text = re.sub(r'<geom\b[^>]*name="(?:table|target_zone)[^"]*"[^>]*/>', '', text)
    text = re.sub(r'<body\b[^>]*name="(?:object|pick_object)"[^>]*>[\s\S]*?</body>', '', text)
    if mass:
        payload = f'<body name="payload"><inertial pos="0 0 0" mass="{mass}" diaginertia="{(str(mass * .0002666667) + " ") * 3}"/></body>'
        text, count = re.subn(r'(<site name="arm_tcp_site")', lambda m: payload+m[1], text)
        if count != 1:
            raise ValueError('Expected one TCP site in the scene')
    meshes = {f'meshes/{p.name}': p.read_bytes() for p in (scene.parent/'meshes').iterdir()} if (scene.parent/'meshes').is_dir() else {}
    model = mj.MjModel.from_xml_string(text, assets=meshes)
    if (model.nq, model.nv, model.nu) not in {(6, 6, 6), (8, 8, 7)} or not np.isclose(model.opt.timestep, DT/SUBSTEPS):
        raise ValueError('Expected a 50 Hz OpenYAM scene with ten substeps per tick')
    if model.actuator_ctrllimited[:6].any():
        raise ValueError('Arm actuators must accept the direct position command')
    return model


class Plant:
    def __init__(self, model, seed, noisy=False, delay=0):
        if not 0 <= delay <= SUBSTEPS//2:
            raise ValueError('Delay must be 0 to 5 substeps')
        self.model = model
        self.data = mj.MjData(model)
        self.noisy = noisy
        self.delay = delay
        self.rng = torch.Generator().manual_seed(seed)
        self.dr = torch.zeros(1, 7)
        self.dr[:, :6] = .0015*torch.randn((1, 6), generator=self.rng)
        self.reset()

    def state(self):
        return torch.tensor(np.r_[self.data.qpos[:6], self.data.qvel[:6]], dtype=torch.float32)[None]

    def row(self):
        row = torch.cat((self.state(), self.issued, torch.ones(1, 1)), -1)
        return observed(row, self.dr, self.rng) if self.noisy else row

    def reset(self, q=HOME):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = q
        self.data.qpos[6:] = .04
        self.data.ctrl[6:] = .04
        self.data.ctrl[:6] = q
        mj.mj_forward(self.model, self.data)
        self.issued = torch.tensor([list(q)], dtype=torch.float32)
        self.history = self.row()[:, None, :].repeat(1, 16, 1)
        self.history[:, :-1, 18] = 0

    def step(self, u):
        if u.shape != (1, 6) or not torch.isfinite(u).all():
            raise ValueError('Expected a finite [1, 6] command')
        for substep in range(SUBSTEPS):
            self.data.ctrl[:6] = (self.issued if substep < self.delay else u).detach().cpu().numpy()[0]
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        self.issued = u.detach().cpu().clone()
        self.history = torch.cat((self.history[:, 1:], self.row()[:, None]), 1)
        return self.state()


def targets(fk, q, joint):
    pos, quat = fk(q)
    return torch.cat((pos, quat), -1), joint


def new_actor(model):
    from .actor import Actor
    from .geometry import TCP
    return Actor(TCP(model).float())


def load_actor(model, checkpoint):
    actor = new_actor(model).eval()
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = data.get('actor', data)
    for key, value in actor.fk.state_dict().items():
        torch.testing.assert_close(weights['fk.' + key], value, atol=0, rtol=0)
    actor.load_state_dict(weights, strict=True)
    return actor
