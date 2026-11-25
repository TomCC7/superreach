"""Tri-mode objective, sensor noise and command-delay rules."""
import math

import torch

from .actor import HOME
from .geometry import inv, log_rotation, mul

DT = .02
SEED = 20261118


SUBSTEPS = 10


TAU = .005


GAMMA = .99


RECIPE = dict(name="tri-mode-dr10ms-joint075-15k-v2", pose_encoder=False, pose_critic=False, horizon=32,
              gamma=GAMMA, ema_tau=TAU, terminal_weight=.25, huber_delta=1.,
              actor_command='direct_u', goal_dim=16, action_dim=6, delay_substeps=[0,5], delay_step_ms=2, joint_speed_coeff=.05,
              directional_deceleration=dict(approach_weight=.8, speed_cap=.35, speed_slope=3.,
                                             away_weight=8., direction_eps=1e-8, track_reference=True), sigma_q=.0025, sigma_bias=.0015, sigma_dq=.065,
              modes=["EEF_6D","XYZ_ONLY","JOINT_6D"], mode_probabilities=[.4,.3,.3],
              switch_probability=.15,switch_ticks=[80,220],joint_approach=[.8,.75,3.,8.],
              wrist_comfort_coeff=.05, actor_lr=[2e-4, 5e-5], decay_start=10000, updates=15000, eval_every=500)


def actor_lr(n):
    x = min(1., max(0., (n-10000)/5000))
    return 5e-5 + .5*(2e-4-5e-5)*(1+math.cos(math.pi*x))


def directional_terms(dp, ep, velocity, goal_xyz, prev_xyz, tracking):
    c = RECIPE['directional_deceleration']
    v_ref = torch.where(tracking[:, None], (goal_xyz-prev_xyz)/DT, 0.)
    dv = velocity-v_ref
    cap = c['speed_cap']
    speed = cap*torch.tanh(c['speed_slope']*ep/cap)
    v_star_rel = (dp/ep[:, None].clamp_min(c['direction_eps']))*speed[:, None]
    approach = -c['approach_weight']*((dv-v_star_rel)/cap).square().sum(-1)
    away = -c['away_weight']*torch.relu((-dp*dv).sum(-1))
    return approach, away


def reward(fk, q, dq, goal, u, previous, prev_goal, tracking, ref, old_dq, mode):
    p, r, jv, _ = fk(q, jac=True)
    dp = goal[:, :3]-p
    ep = dp.norm(dim=-1)
    approach, away = directional_terms(dp, ep, (jv@dq.unsqueeze(-1)).squeeze(-1),
                                      goal[:, :3], prev_goal[:, :3], tracking)
    eef,xyz,joint = mode==0,mode==1,mode==2
    er = log_rotation(mul(inv(r),torch.where(eef[:,None],goal[:,3:7],r))).norm(dim=-1)
    delta = torch.where(joint[:,None],ref[:,:6]-q,0.)
    eq = delta.norm(dim=-1)
    dqr = dq-torch.where((tracking & joint)[:,None],ref[:,6:12],0.)
    weight, cap, slope, away_weight = RECIPE['joint_approach']
    dq_star = delta/eq[:,None].clamp_min(1e-8)*(cap*torch.tanh(slope*eq/cap))[:,None]
    joint_approach = -weight*((dqr-dq_star)/cap).square().sum(-1)
    joint_away = -away_weight*torch.relu((-delta*dqr).sum(-1))
    pos = 1-((ep.square()+.005**2).sqrt()-.005)/.10
    terms = dict(position=torch.where(joint,0.,torch.where(xyz,5.5*pos,3.5*pos)),
                 rotation=torch.where(eef,2.*(1-((er.square()+.01**2).sqrt()-.01)/.35),0.),
                 joint_position=torch.where(joint,5.5*(1-((eq.square()+.01**2).sqrt()-.01)/.35),0.),
                 joint_speed=-.05*dq.square().mean(-1),
                 joint_accel=-.001*((dq-old_dq)/DT).square().mean(-1),
                 command_slew=-.10*((u-previous)/DT).square().mean(-1),
                 approach=torch.where(joint,joint_approach,approach),
                 away=torch.where(joint,joint_away,away),
                 wrist_comfort=torch.where(xyz,-.05*(q[:,3:]-q.new_tensor(HOME[3:])).square().mean(-1),0.))
    return DT*sum(terms.values()), terms, ep, er


def schedule(n,device,rng):
    first=torch.bucketize(torch.rand(n,device=device,generator=rng),torch.tensor([.4,.7],device=device))
    second=(first+torch.randint(1,3,(n,),device=device,generator=rng))%3
    switch=torch.rand(n,device=device,generator=rng)<.15
    tick=torch.randint(80,221,(n,),device=device,generator=rng)
    return first+3*second+9*torch.where(switch,tick,301)


def mode_at(code,tick):
    code=code.long()
    return torch.where(tick<code//9,code%3,(code//3)%3)


def switch_goal(fk, bank_pose, ref, code):
    p,r=fk(ref[:,:6]); exact=torch.cat((p,r),-1)
    return torch.where((code.long()//9<301)[:,None],exact,bank_pose)


def observed(row, dr, rng=None):
    noise = torch.randn(row.shape[:-1]+(12,), device=row.device, dtype=row.dtype, generator=rng)
    return torch.cat((row[..., :6]+dr[..., :6]+.0025*noise[..., :6],
                      row[..., 6:12]+.065*noise[..., 6:], row[...,12:]), -1)


def effective(previous, issued, delay):
    return torch.lerp(issued, previous, delay/SUBSTEPS)
