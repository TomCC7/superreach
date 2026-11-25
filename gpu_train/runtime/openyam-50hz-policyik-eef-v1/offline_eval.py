"""Extended, reset-free off-HOME hold diagnostics for the 7D EEF actor."""
import copy
import numpy as np
import torch
import mujoco as mj
from objective import DT, Recipe
from policy import Actor


def actor_from_checkpoint(checkpoint, fk, normalizer, mlp):
    """Rebuild an offline/inference actor from trusted checkpoint metadata."""
    actor = Actor(fk, Recipe(**checkpoint['recipe']), normalizer, mlp)
    actor.load_state_dict(checkpoint['actor'], strict=True)
    return actor.eval()

HOME = np.array([0., 1.047, 1.047, 0., 0., 0.])
# Exact two poses and switch tick from browser_probe_trace.json (actor14000).
BROWSER_INITIAL = np.array([0.3722517363367851, 4.616852332123585e-06, 0.3295063253566963,
                            0.9999999999995002, 4.998289253351356e-07, 7.828424338030606e-12, 8.65926611233702e-07])
BROWSER_TARGET = np.array([0.4622504303995114, -0.1199947728685973, 0.24950634094884086,
                           0.9843038601978671, 0.006229721329362225, 0.08766559228204898, -0.15304197235260075])
PRE_TICKS, HOLD_TICKS = 107, 1500


def summarize(tcp, dq, u, goal):
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    def window(a, b):
        if b > len(tcp) or b-a < 2:
            return None
        x, v, command = tcp[a:b, :3], dq[a:b], u[a:b]
        return dict(tcp_peak_to_peak_mm=1000*float(np.linalg.norm(np.ptp(x, axis=0))),
                    tcp_centered_rms_mm=1000*rms(np.linalg.norm(x-x.mean(0), axis=-1)),
                    dq_rms_rad_s=rms(v), ddq_rms_rad_s2=rms(np.diff(v, axis=0)/DT),
                    command_slew_rms_rad_s=rms(np.diff(command, axis=0)/DT),
                    command_peak_to_peak_rad=float(np.ptp(command, axis=0).max()))

    tcp, dq, u = tcp[PRE_TICKS:], dq[PRE_TICKS:], u[PRE_TICKS:]
    if not len(tcp):
        return dict(observed_hold_ticks=0, final_error_mm=None, final_error_deg=None,
                    sustained_below_10mm_s=None, tail_2s=None, browser_probe_2s=None, hold_windows_2s={})
    err = np.linalg.norm(tcp[:, :3]-goal[:3], axis=-1)
    complete = len(tcp)==HOLD_TICKS
    stays = np.flatnonzero(np.maximum.accumulate(err[::-1])[::-1] < .010) if complete else []
    r = np.abs(tcp[-1, 3:] @ goal[3:])
    return dict(observed_hold_ticks=len(tcp), final_error_mm=1000*float(err[-1]),
                final_error_deg=float(np.degrees(2*np.arccos(np.clip(r, 0, 1)))),
                sustained_below_10mm_s=(int(stays[0])+1)*DT if len(stays) else None,
                tail_2s=window(HOLD_TICKS-100,HOLD_TICKS),
                browser_probe_2s=window(300-PRE_TICKS,400-PRE_TICKS),
                hold_windows_2s={str(s):window(int(s/DT),int(s/DT)+100) for s in (0,2,4,6,8,10,20,28)})


@torch.inference_mode()
def evaluate_holds(actor, bank, E, matrix_quat):
    del bank  # Browser fixture, independent of training/reference-bank sampling.
    policy = copy.deepcopy(actor).cpu().eval()
    models, datas, rows, histories = [], [], [], []
    peak = np.asarray(E.o.PEAK)
    variants = E.continuous.load_catalog()
    for mass in (0.,1.,2.):
        for orientation in ('fixed', 'rotated'):
            goal = BROWSER_TARGET.copy()
            if orientation=='fixed':
                goal[3:] = BROWSER_INITIAL[3:]
            m = mj.MjModel.from_xml_path(E.o.os.environ['SUPERREACH_SCENE'])
            E.old.configure_model(m); m.opt.timestep = .002
            m.actuator_ctrllimited[:] = False
            m.actuator_forcelimited[:] = True
            m.actuator_forcerange[:] = np.column_stack((-peak,peak))
            idx = int(np.argmin(np.abs(variants['masses']-mass)))
            assert abs(variants['masses'][idx]-mass)<1e-6
            for name in ('body_mass','body_ipos','body_inertia','body_iquat'):
                getattr(m,name)[:] = variants[name][idx]
            d = mj.MjData(m); mj.mj_setConst(m,d)
            d.qpos[:6] = HOME; d.qpos[6:] = .055; mj.mj_forward(m,d)
            history = np.tile(np.r_[HOME,np.zeros(6),HOME,1.,0.],(16,1)); history[:-1,18] = 0
            models.append(m); datas.append(d); histories.append(history)
            rows.append(dict(mass_kg=mass,orientation=orientation,goal_pose=goal.tolist(),
                             exact_browser_repro_goal=mass==2 and orientation=='rotated',
                             physical_failure=False,failure_tick=None,tcp=[],dq=[],u=[],
                             saturated_joint_substeps=0,scored_joint_substeps=0))
    variants.close()
    history = np.asarray(histories)
    targets = np.array([r['goal_pose'] for r in rows])
    for tick in range(PRE_TICKS+HOLD_TICKS):
        goals = np.tile(BROWSER_INITIAL,(len(rows),1)) if tick<PRE_TICKS else targets
        cmd = policy({'history':torch.tensor(history,dtype=torch.float32),
                      'goal_pose':torch.tensor(goals,dtype=torch.float32)}).numpy()
        for i,(m,d,r) in enumerate(zip(models,datas,rows)):
            if r['physical_failure']:
                continue
            if not np.isfinite(cmd[i]).all():
                r.update(physical_failure=True,failure_tick=tick)
                continue
            d.ctrl[:] = cmd[i]
            for _ in range(10):
                mj.mj_step(m,d)
                r['saturated_joint_substeps'] += int((np.abs(d.actuator_force)>=peak-1e-4).sum())
                r['scored_joint_substeps'] += len(peak)
            mj.mj_forward(m,d)
            sid = m.site('arm_tcp_site').id
            q, v = d.qpos[:6], d.qvel[:6]
            if not np.isfinite(np.r_[q,v]).all() or np.abs(q).max()>12 or np.abs(v).max()>100:
                r.update(physical_failure=True,failure_tick=tick)
                continue
            r['tcp'].append(np.r_[d.site_xpos[sid],matrix_quat(d.site_xmat[sid])])
            r['dq'].append(v.copy()); r['u'].append(cmd[i].copy())
            history[i,:-1] = history[i,1:]
            history[i,-1] = np.r_[q,v,cmd[i],1.,(tick+1)*DT]
    for r in rows:
        traces = [np.asarray(r.pop(k), dtype=np.float64) for k in ('tcp','dq','u')]
        r.update(summarize(*traces,np.asarray(r['goal_pose'])))
    return dict(protocol_version=1, pre_hold_ticks=PRE_TICKS, hold_ticks=HOLD_TICKS,
                control_dt=DT, physics_dt=.002, substeps=10, sustained_threshold_mm=10,
                window_seconds=2, browser_probe_window_absolute_ticks=[300,400],
                tcp_peak_to_peak='norm of the XYZ peak-to-peak spans',
                command_peak_to_peak='maximum per-joint peak-to-peak span',
                initial_q=HOME.tolist(), initial_goal_pose=BROWSER_INITIAL.tolist(), cases=rows)
