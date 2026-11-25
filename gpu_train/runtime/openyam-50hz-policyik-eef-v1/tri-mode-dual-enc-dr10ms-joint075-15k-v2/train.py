"""Fresh OpenYAM learner; frozen 50 Hz plant, references and WM optimizer."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent), str(ROOT)]
import experiment as parent
from geometry import inv, log_rotation, mul
from actor import Actor, MASKS, HOME
from objective import DT, Recipe

CFG = Recipe(command='direct', clean_obs=True, tolerance=False, static_calibration=False)
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
                 wrist_comfort=torch.where(xyz,-.05*q[:,3:].square().mean(-1),0.))
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


def observations(td,fk,h,goal,ref,mode,state,u,ctx):
    m=torch.stack((torch.ones_like(mode),mode!=1,mode==2),-1).to(goal.dtype)
    # Zero hidden targets themselves, not merely error channels.
    jp,jr=fk(ref[:,:6]); goal=torch.where((mode==2)[:,None],torch.cat((jp,jr),-1),goal)
    g=torch.cat((goal[:,:3],torch.where(m[:,1:2].bool(),goal[:,3:7],0.)), -1)
    q=torch.where(m[:,2:3].bool(),ref[:,:6],0.)
    return td(dict(history=h,goal_pose=g,goal_q=q,mode=mode,privileged_ctx=ctx,
                   critic=torch.cat((state,g,q,m,u,ctx),-1)),[len(goal)],device=goal.device)


def observed(row, dr, rng=None):
    noise = torch.randn(row.shape[:-1]+(12,), device=row.device, dtype=row.dtype, generator=rng)
    return torch.cat((row[..., :6]+dr[..., :6]+.0025*noise[..., :6],
                      row[..., 6:12]+.065*noise[..., 6:], row[...,12:]), -1)


def effective(previous, issued, delay):
    return torch.lerp(issued, previous, delay/SUBSTEPS)


def setup(arch):
    b = parent.setup(CFG)
    E, patch = b.T.E, b.A.patch
    L, O = E.learning, E.o.OpenYAMEnv
    b.eef_source_digest = hashlib.sha256((b.eef_source_digest+arch+parent.file_sha256(HERE/'train.py')+parent.file_sha256(HERE/'actor.py')).encode()).hexdigest()
    identity = lambda: b.eef_source_digest
    E.identity = E.core.identity = E.core.base.identity = E.base.identity = identity
    old_reset, old_sensor, old_state = O.reset, O.sensor, O.fullstate
    td=L.TensorDict
    def reset(self, ids, *a, **kw):
        if not hasattr(self, 'dr'):
            self.dr = torch.zeros((self.num_envs,7), device=self.device)
            self.dr_rng = torch.Generator(device=self.device).manual_seed(SEED+1)
        if not hasattr(self,'mode_schedule'):
            self.mode_schedule=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
            self.mode_rng=torch.Generator(device=self.device).manual_seed(SEED+2)
        if len(ids):
            self.mode_schedule[ids]=schedule(len(ids),self.device,self.mode_rng) if (not a or a[0] is None) and kw.get("ref_ids") is None else 2709
            self.dr[ids,:6] = .0015*torch.randn((len(ids),6), device=self.device, generator=self.dr_rng)
            self.dr[ids,6] = torch.randint(0,6,(len(ids),), device=self.device,generator=self.dr_rng).to(self.dr.dtype)
        old_reset(self, ids, *a, **kw)
        if len(ids):
            if hasattr(self,"q_nom"):
                ref=self.bank.lookup(self.ref_id[ids],self.start_tick[ids])
                self.goal[ids]=switch_goal(self.fk,self.goal[ids],ref,self.mode_schedule[ids])
            row = observed(old_sensor(self)[ids], self.dr[ids], self.dr_rng)
            self.history[ids,:,:18] = row[:,None,:18]
    O.reset = reset
    O.fullstate = lambda self: torch.cat((old_state(self),self.mode_schedule[:,None]),-1)
    def real_obs(self):
        ref=self.bank.lookup(self.ref_id,self.start_tick+self.tick)
        return observations(td,self.fk,self.history.clone(),self.goal.clone(),ref,mode_at(self.mode_schedule,self.tick),self.state(),self.sent,self.context.read(self))
    O.obs = real_obs
    rep_step=L.Representation.apply_prediction
    def advance(self,aux,rep,pred,dt):
        nxt,state=rep_step(self,aux,rep,pred,dt)
        ref=self.bank.lookup(nxt[:,8],nxt[:,9]+nxt[:,7])
        return torch.cat((switch_goal(self.fk,nxt[:,:7],ref,aux[:,10]),nxt[:,7:],aux[...,10:11]),-1),state
    L.Representation.apply_prediction=advance
    # The constructor invokes reset, so keep its sensor method clean; inject noise
    # exactly once when the finalized post-step row is appended.
    step = O.step
    pose_lookup='pose(self.bank,self.ref_id,self.start_tick+self.tick)'
    installed_pose='switch_goal(self.fk,'+pose_lookup+',self.bank.lookup(self.ref_id,self.start_tick+self.tick),self.mode_schedule)'
    O.step = patch(step, [
        (pose_lookup+');r,terms',installed_pose+');r,terms'),
        (pose_lookup+');self.history',installed_pose+');self.history'),
        ('self.q_nom,old_dq)', 'self.bank.lookup(self.ref_id,self.start_tick+self.tick),old_dq,mode_at(self.mode_schedule,self.tick))'),
        ('r=r-departure.float()', 'r=r-DT*departure.float()'),
        ('self.sim.step()',  'self.d.ctrl[:]=torch.where(self.dr[:,6:7]>substep,previous,self.sent);self.sim.step()'),
        ('self.sensor()[:,None]', 'observed(self.sensor(),self.dr,self.dr_rng)[:,None]'),
        ('last=dict(ctrl_eff=ctrl_eff,', 'last=dict(applied_effective=effective(previous,self.sent,self.dr[:,6:7]),ctrl_eff=ctrl_eff,'),
    ], 'unified-real', score=reward, observed=observed, effective=effective,mode_at=mode_at,switch_goal=switch_goal)
    def make_actor(env):
        limits = torch.as_tensor(env.model.jnt_range[:6].copy(), device=env.device, dtype=torch.float32)
        return Actor(env.fk, limits, L.build_mlp,arch).to(env.device)
    L.Learner.__init__ = patch(L.Learner.__init__, [
        ('actor_obs=SENSOR','actor_obs=SENSOR,dr=7'),
        ('state=22','state=23'),
        ('12+7+ACTION_DIM+CTX','12+7+6+3+ACTION_DIM+CTX'),
    ], 'unified-learner', make_actor=make_actor)
    L.Learner.collect = patch(L.Learner.collect, [
        ('ctx=e.context.read(e);sensor=e.sensor()', 'ctx=e.context.read(e);sensor=obs["history"][:,-1];dr=e.dr.clone()'),
        ('action=raw,prev_action=previous', 'action=last["applied_effective"],prev_action=previous,dr=dr'),
        ('last["sent"]','last["applied_effective"]'),
    ], 'unified-collect')
    # Match the exact noisy replay packet on the initial imagined frame; new
    # imagined packets use the same noise law and replay episode bias.
    W = L.MetalWorldModelEnv
    def on_reset(self, batch):
        self.dr = batch['dr']
        self._past_sensors = self._warmstart_past_sensors(batch['sensor_window'],batch['pad_mask']).clone()
        self._past_sensors[...,18] = (~batch['pad_mask'][:,:-1]).float()
        self._latest_sensor = batch['sensor_window'][:,-1]
    W._on_reset = on_reset
    synth = W._synth_sensor
    W._synth_sensor = lambda self: observed(synth(self), self.dr)
    def is_track(bank, ids, ticks):
        return (ids < bank.step_offset) & (~E.hold.decode(ticks)[0])
    def wm_obs(self):
        a=self.aux_state
        ref=self.representation.bank.lookup(a[:,8],a[:,9]+a[:,7])
        return observations(td,self.representation.fk,self._sensor_history(),a[:,:7],ref,mode_at(a[:,10],a[:,7]),self.rep_state,self.last_actions,self.sim_ctx)
    W.get_observations=wm_obs
    def wm_step(self, rep, aux, u, previous, ctx):
        next_aux, nxt = self.dynamics(rep,effective(previous,u,self.dr[:,6:7]),aux,ctx)
        r, *_ = reward(self.representation.fk,nxt[:,:6],nxt[:,6:],next_aux[:,:7],
                       u,previous,aux[:,:7],is_track(self.representation.bank,aux[:,8],aux[:,9]),self.representation.bank.lookup(next_aux[:,8],next_aux[:,9]+next_aux[:,7]),rep[:,6:],mode_at(next_aux[:,10],next_aux[:,7]))
        fail = b.task.physical_failure(nxt[:,:6],nxt[:,6:])
        return next_aux,nxt,nxt,((next_aux[:,7]>=300)|fail).float(),r-DT*fail.float()
    W._wm_step = wm_step
    collect=L.Learner.collect
    def measured_collect(self):
        m=mode_at(self.e.mode_schedule,self.e.tick)
        stats=collect(self)
        counts=torch.stack([(m==i).sum() for i in range(3)]).tolist()
        stats['collection_mode_counts']=dict(zip(RECIPE['modes'],counts,strict=True))
        return stats
    L.Learner.collect=measured_collect
    b.eef_score = reward
    return b


def make(b, worlds, gpu, wm_batch, actor_batch, smoke=False):
    E = b.T.E
    e = E.core.install(E.o.OpenYAMEnv(worlds)) if gpu else b.cpu_environment(worlds)
    e.payload_enabled = True
    e.reset(e.ids)
    l = E.Learner(e)
    b.T.configure(l,'separate',wm_batch,smoke)
    for opt in l.policy.actor_optimizers:
        # Remove predecessor constant-LR assertion only from actor optimizers.
        opt._optimizer_step_pre_hooks.clear()
    l.policy.cfg.batch_size = l.wm_env.num_envs = actor_batch
    l.target = copy.deepcopy(l.critic).eval().requires_grad_(False)
    policy = l.policy
    def bootstrap(self, obs, survival, term):
        return .25*self.cfg.gamma**self.cfg.horizon*survival*l.target(obs['critic']).squeeze(-1).clamp(max=self._v_max)
    def fit(self, rollout, final):
        from supermimic.learners.policy_learner import gae_returns
        steps = torch.stack(rollout)
        x = steps['critic']
        with torch.no_grad():
            y = gae_returns(l.target(x),l.target(final['critic']), steps['reward'].unsqueeze(-1),
                            steps['done'].unsqueeze(-1), self.cfg.gamma,self.cfg.lam).clamp(max=self._v_max).flatten(0,1)
        x = x.flatten(0,1)
        loss = F.huber_loss(self.critic(x), y, delta=1.)
        norm = b.T._optimize_step(self.critic_optimizers, loss, list(self.critic.parameters()),1.)
        with torch.no_grad():
            for target, source in zip(l.target.parameters(),l.critic.parameters(),strict=True):
                target.lerp_(source,TAU)
        return dict(value_function=loss.detach(), critic_grad_norm=norm)
    policy._bootstrap = MethodType(bootstrap,policy)
    policy._fit_critic = MethodType(fit,policy)
    b.A.checkplant(e,b.T)
    return l


def policy_step(b,l):
    starts = torch.stack([l.wm_env.sample_initial_states()])
    obs = l.wm_env.reset(starts[0])
    loss, _, _ = l.actor.payload_loss(obs['history'],obs['privileged_ctx'])
    norm = b.T._optimize_step(l.encoder_opts,.01*loss,list(l.actor.history_encoder.parameters()),1.)
    l.encoder_updates += 1
    l.actor.history_encoder.requires_grad_(False)
    for mod in (l.dynamics,l.rep):
        mod.eval().requires_grad_(False)
    lr = actor_lr(l.count['actor_updates']+1)
    for opt in l.policy.actor_optimizers:
        for group in opt.param_groups:
            group['lr'] = lr
    try:
        logs = l.policy._captured_update(starts)
    finally:
        l.actor.history_encoder.requires_grad_(True)
    l.count['actor_updates'] += 1
    l.count['critic_minibatches'] += 1
    return dict(actor_lr=lr,payload_loss=float(loss.detach()),payload_grad_norm=float(norm)) | {k:float(v.detach()) for k,v in logs.items()}


def save(b,l,out,n):
    if shutil.disk_usage(out).free < 12*2**30:
        raise RuntimeError('Insufficient free space for checkpoint (12 GiB required)')
    z = b.T.E.core.snapshot(l)
    z.update(recipe=RECIPE, target=l.target.state_dict(),
             encoder_optim=[o.state_dict() for o in l.encoder_opts], encoder_updates=l.encoder_updates,
             dr=l.e.dr, dr_rng=l.e.dr_rng.get_state(), mode_schedule=l.e.mode_schedule, mode_rng=l.e.mode_rng.get_state())
    p = out/f'{n}.pt'
    tmp = p.with_suffix('.tmp'); torch.save(z,tmp); os.replace(tmp,p)
