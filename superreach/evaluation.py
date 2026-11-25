"""Metrics for native MuJoCo rollouts."""
import numpy as np


def joint_step_metrics(case, measured_q, tcp, ticks, home):
    """Absolute joint angles; score after the STEP at tick 75 only."""
    if case['kind'] != 'STEP':
        return {}
    target, start = case['target'][:6], case['q']
    q = np.asarray(measured_q)[75:]
    xyz = np.asarray(tcp)[75:]
    if len(q)==0 or not np.isfinite(q).all() or not np.isfinite(xyz).all():
        return dict(joint_overshoot_max_by_joint_rad=None, joint_off_home_tail_2s=None)
    direction = np.sign(target-start)
    overshoot = np.maximum((q-target)*direction,0.).max(0)
    tail = None
    if len(q)>=100 and not np.allclose(target,home,atol=1e-6,rtol=0):
        q_span = np.ptp(q[-100:],axis=0)
        tail = dict(ticks=100, start_tick=ticks-100, end_tick_exclusive=ticks,
                    tcp_peak_to_peak_mm=1000*float(np.linalg.norm(np.ptp(xyz[-100:],axis=0))),
                    joint_peak_to_peak_by_joint_rad=q_span.tolist(),
                    joint_peak_to_peak_max_rad=float(q_span.max()))
    return dict(joint_overshoot_max_by_joint_rad=[float(v) if s else None for v,s in zip(overshoot,direction)],
                joint_off_home_tail_2s=tail)
