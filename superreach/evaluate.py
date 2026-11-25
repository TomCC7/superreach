"""Evaluate a tri-mode checkpoint on paired native HOME, TRACK and STEP cases."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .actor import HOME
from .evaluation import joint_step_metrics
from .plant import Plant, load_actor, load_model, targets
from .recipe import DT


@torch.inference_mode()
def evaluate(scene, checkpoint, ticks):
    model = load_model(scene)
    actor = load_actor(model, checkpoint)
    goal_mode = torch.zeros(1, dtype=torch.long)
    home = torch.tensor(HOME)[None]
    limit = torch.tensor(model.jnt_range[:6], dtype=torch.float32)
    offset = torch.tensor([[.25, -.18, .12, -.20, .10, .18]])
    result = {}
    for label, delay in [('clean', 0), ('noisy-10ms', 5)]:
        result[label] = {}
        for k, name in enumerate(['EEF_6D', 'XYZ_ONLY', 'JOINT_6D']):
            goal_mode[0] = k
            rows = result[label][name] = []
            for mass in (0., 0.5, 1., 2.):
                model = load_model(scene, mass)
                site = model.site('arm_tcp_site').id
                for kind in ('HOME', 'TRACK', 'STEP'):
                    plant = Plant(model, 937701, bool(delay), delay)
                    xyz_err, rot_err, q_err, qs, xyzs, cmds, velocities = [], [], [], [], [], [], []
                    target = (home + offset).clamp(limit[:,0], limit[:,1])
                    for tick in range(ticks):
                        q_ref = home
                        if kind == 'STEP' and tick >= 75:
                            q_ref = target
                        if kind == 'TRACK':
                            q_ref = (home + offset*np.sin(2*np.pi*tick/150)).clamp(limit[:,0], limit[:,1])
                        pose, joint = targets(actor.fk, q_ref, q_ref)
                        command = actor.rollout(plant.history, pose, joint, goal_mode)
                        state = plant.step(command)
                        if not torch.isfinite(state).all():
                            break
                        q = state[:, :6]
                        _, quat = actor.fk(q)
                        xyz = plant.data.site_xpos[site].copy()
                        xyzs.append(xyz)
                        qs.append(q.numpy()[0].copy())
                        cmds.append(command.numpy()[0].copy())
                        velocities.append(state.numpy()[0,6:].copy())
                        xyz_err.append(float(np.linalg.norm(xyz-pose.numpy()[0,:3]))*1000)
                        rot_err.append(float(torch.rad2deg(2*torch.acos((quat*pose[:,3:]).sum().abs().clamp(max=1)))))
                        q_err.append(float(torch.rad2deg((q-q_ref).norm())))
                    row: dict[str, object] = dict(kind=kind, mass_kg=mass, observed_ticks=len(qs), physical_failure=len(qs)<ticks)
                    if qs:
                        row.update(tcp_mean_mm=float(np.mean(xyz_err)), tcp_p95_mm=float(np.percentile(xyz_err,95)),
                                   orientation_p95_deg=float(np.percentile(rot_err,95)), joint_mean_deg=float(np.mean(q_err)))
                    if len(qs)>1:
                        row.update(command_slew_rms_rad_s=float(np.sqrt(np.mean((np.diff(cmds,axis=0)/DT)**2))),
                                   joint_accel_rms_rad_s2=float(np.sqrt(np.mean((np.diff(velocities,axis=0)/DT)**2))))
                    if k==2 and kind=='STEP':
                        row.update(joint_step_metrics(dict(kind=kind,target=target.numpy()[0],q=home.numpy()[0]),qs,xyzs,len(qs),HOME))
                    rows.append(row)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--ticks', type=int, default=300)
    args = ap.parse_args()
    if args.ticks < 1:
        ap.error('Ticks must be positive')
    torch.set_num_threads(1)
    result = evaluate(args.scene, args.checkpoint, args.ticks)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(f'Wrote {sum(len(rows) for modes in result.values() for rows in modes.values())} cases')


if __name__ == '__main__':
    main()
