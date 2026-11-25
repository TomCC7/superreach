"""Shared real/imagined Cartesian scoring, SI units and world-frame twists."""
from dataclasses import dataclass
import math
import torch
from geometry import inv, log_rotation, mul

DT = .02
POSITION_TOL = .001
ROTATION_TOL = math.radians(.2)
GLOBAL_DQ_WEIGHT = .8
GLOBAL_DDQ_WEIGHT = .03
MAX_RATE = 6.5  # position 3.5 + orientation 2 + tolerance <= 1


@dataclass(frozen=True)
class Recipe:
    command: str = 'accumulate'
    eta: float = .04
    fk_feedback: bool = True
    tolerance: bool = True
    static_calibration: bool = True
    anti_jitter: bool = False
    stable_finetune: bool = False
    minimal_slew: bool = False
    fk_mode: str = 'multiscale'
    equilibrium_perturb: bool = False
    uprev_jacobian_l2: float = 0.0
    global_damping: bool = False
    wm32_critic_v2: bool = False
    no_terminal_critic: bool = False
    clean_obs: bool = False


def pose_rates(ep, er, use_tolerance):
    return dict(
        position=2.5*(1-((ep.square()+.005**2).sqrt()-.005)/.10)+.5*torch.exp(-(ep/.005).square())+.5*torch.exp(-(ep/.05).square()),
        rotation=1.5*(1-((er.square()+.01**2).sqrt()-.01)/.35)+.25*torch.exp(-(er/.02).square())+.25*torch.exp(-(er/.15).square()),
        tolerance=tolerance(ep,er) if use_tolerance else torch.zeros_like(ep),
    )


def clean_reward(fk, q, dq, goal, u, previous, previous_dq, use_tolerance):
    xyz, quat = fk(q)
    ep = (goal[:, :3]-xyz).norm(dim=-1)
    er = log_rotation(mul(inv(quat), goal[:, 3:7])).norm(dim=-1)
    terms = pose_rates(ep, er, use_tolerance)
    terms.update(joint_speed=-.05*dq.square().mean(-1),
                 joint_accel=-.001*((dq-previous_dq)/DT).square().mean(-1),
                 command_slew=-.10*((u-previous)/DT).square().mean(-1))
    return DT*sum(terms.values()), terms, ep, er


def plateau_weight(error, inner, outer):
    """C1: one through inner, zero from outer onward."""
    return near_weight(error - inner, outer - inner)


def near_weight(error, outer):
    """C1 falloff, exactly zero outside the near-target band."""
    x = (error / outer).clamp(0, 1)
    return (1-x).square()*(1+2*x)


def damping_rate(ep, er, dq, previous_dq, u, previous, hold_weight):
    gate = hold_weight*near_weight(ep, .025)*near_weight(er, math.radians(5))
    return -gate*(.5*dq.square().mean(-1)
                  + .003*((dq-previous_dq)/DT).square().mean(-1)
                  + .20*((u-previous)/DT).square().mean(-1))


def tolerance(p,r):
    def precision(error, fine):
        return .7/(1+(error/(3*fine)).square()) + .3/(1+(error/fine).square())
    return .5*(precision(p, POSITION_TOL) + precision(r, ROTATION_TOL))


def direction(delta):
    return delta / delta.norm(dim=-1,keepdim=True).clamp_min(1e-8)


def goal_twist(goal,previous,tracking):
    v = (goal[:,:3]-previous[:,:3])/DT
    w = log_rotation(mul(goal[:,3:7],inv(previous[:,3:7])))/DT
    return v*tracking[:,None],w*tracking[:,None]


def task_reward(fk,q,dq,goal,u,previous,v_ref,w_ref,q_nom,use_tolerance=True,*,previous_dq=None,hold_weight=None,stable=False,slew_coeff=.40,global_damping=False):
    xyz,quat,jv,jw = fk(q,jac=True)
    dp = goal[:,:3]-xyz
    ep = dp.norm(dim=-1)
    er = log_rotation(mul(inv(quat),goal[:,3:7])).norm(dim=-1)
    dv = (jv@dq.unsqueeze(-1)).squeeze(-1)-v_ref
    dw = (jw@dq.unsqueeze(-1)).squeeze(-1)-w_ref
    speed = .35*torch.tanh(3*ep/.35)
    v_star_rel = direction(dp)*speed[:,None]
    gate = torch.exp(-(ep/.03).square()-(er/.15).square())
    terms = pose_rates(ep, er, use_tolerance)
    terms.update(
        approach=-.8*((dv-v_star_rel)/.35).square().sum(-1),
        away=-8*torch.relu((-dp*dv).sum(-1)),
        braking=-1.5*gate*(dv.square().sum(-1)+.05*dw.square().sum(-1)),
        posture=-.005*(q-q_nom).norm(dim=-1),
        command_slew=-slew_coeff*((u-previous)/DT).square().mean(-1),
    )
    if global_damping:
        terms['global_joint_damping'] = -(GLOBAL_DQ_WEIGHT*dq.square().mean(-1)
            + GLOBAL_DDQ_WEIGHT*((dq-previous_dq)/DT).square().mean(-1))
    elif stable:
        hold_gate = hold_weight*plateau_weight(ep,.012,.040)*plateau_weight(er,math.radians(2),math.radians(8))
        terms['wide_hold_damping'] = -hold_gate*(.5*dq.square().mean(-1)
            + .006*((dq-previous_dq)/DT).square().mean(-1)
            + .20*((u-previous)/DT).square().mean(-1))
        close = (dp*dv).sum(-1)/(ep.square()+.002**2).sqrt()
        cap = .35*torch.tanh(2*ep/.35)
        terms['approach_overspeed'] = -2.*(torch.relu(close-cap)/.35).square()
    elif previous_dq is not None:
        terms["near_hold_damping"] = damping_rate(ep, er, dq, previous_dq, u, previous, hold_weight)
    return DT*sum(terms.values()),terms,ep,er
