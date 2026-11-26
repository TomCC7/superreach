"""E6 training hooks for the portable 50 Hz GPU runtime.

The historical checkpoint restore and host admission are not part of these hooks.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from . import dr as hardware
from .actor import Actor, HOME
from superreach.geometry import inv, log_rotation, mul

HERE = Path(__file__).resolve().parent
DT = .02
SEED = 20261118
SUBSTEPS = 10
TAU = .005
RECIPE = json.loads((HERE / 'recipe.json').read_text())

EXTRA = 5000
WM_FRAMES = 33

def actor_lr(completed):
    if completed <= 0: return 5e-5
    if completed >= EXTRA-1: return 5e-6
    return 5e-6 + .5*(5e-5-5e-6)*(1+math.cos(math.pi*completed/(EXTRA-1)))

def position_fine(dp):
    return torch.exp(-dp.square().sum(-1)/.002**2)

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
                 position_fine=torch.where(joint,0.,torch.where(xyz,2.,1.5)*position_fine(dp)),
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


def setup(parent, *, arch='dual_encoder', sysid_mode='dr_reg_mlp'):
    """Install E6 hooks once in a fresh process using the portable EEF runtime."""
    # Resolve the EEF runtime's own Recipe rather than the portable CPU toy recipe.
    cfg = parent.Recipe(command='direct', clean_obs=True, tolerance=False, static_calibration=False)
    b = parent.setup(cfg)
    E, patch = b.T.E, b.A.patch
    E.continuous.sample_indices = hardware.sample_payload
    L, O = E.learning, E.o.OpenYAMEnv
    h = hashlib.sha256((b.eef_source_digest + arch + sysid_mode).encode())
    for name in ('recipe.py', 'actor.py', 'dr.py', 'recipe.json'):
        h.update((HERE / name).read_bytes())
    b.eef_source_digest = h.hexdigest()
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
        return observations(td,self.fk,self.history.clone(),self.goal.clone(),ref,mode_at(self.mode_schedule,self.tick),self.state(),self.sent,hardware.theta(self))
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
        return Actor(env.fk, limits, L.build_mlp,arch,sysid_mode).to(env.device)
    L.Learner.__init__ = patch(L.Learner.__init__, [
        ('actor_obs=SENSOR','actor_obs=SENSOR,dr=7'),
        ('state=22','state=23'),
        ('12+7+ACTION_DIM+CTX','12+7+6+3+ACTION_DIM+CTX'),
    ], 'sysid-learner', make_actor=make_actor,CTX=hardware.THETA)
    L.Learner.collect = patch(L.Learner.collect, [
        ('ctx=e.context.read(e);sensor=e.sensor()', 'ctx=obs["privileged_ctx"].clone();sensor=obs["history"][:,-1];dr=e.dr.clone()'),
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
    hardware.install(e,SEED+3)
    e.payload_enabled = True
    e.reset(e.ids)
    l = E.Learner(e)
    l.wm_env.max_step_reward = DT*7.5
    b.T.configure(l,'e2e',wm_batch,smoke)
    assert l.wm.cfg.horizon == 8 and l.policy._captured_update == l.policy._update_from
    l.long_batch = 4 if smoke else 64
    l.fork_start = 0
    l.actor_step_lrs = {}
    def witness(opt, *_):
        l.actor_step_lrs[opt] = [float(g['lr']) for g in opt.param_groups]
    # Both SYSID auxiliary objectives backpropagate jointly with BPTT.
    cls=type('SysIDPolicy',(type(l.policy),),{})
    l.policy.__class__=cls
    cls._train_batch=b.A.patch(cls._train_batch,[
        ('  payload_loss, payload_dims, prediction = self.actor.payload_loss(obs["history"], obs["privileged_ctx"])\n',''),
        ('      "payload_mse": payload_loss.detach(),\n      "payload_weighted": AUX_WEIGHT * payload_loss.detach(),\n      **{f"payload_mse_{i}": payload_dims[i] for i in range(10)},\n',''),
        ('loss = (-ret + aux_loss).mean()',
         'sysid_aux = self.actor.sysid_loss(start_obs, self.wm_env.replay_buffer)\n  loss = (-ret + aux_loss).mean() + .01 * sysid_aux'),
        ('obs = self.wm_env.reset(start)','obs = self.wm_env.reset(start); start_obs=obs'),
        ('"return": ret.mean().detach(),','"return": ret.mean().detach(), "sysid_aux":sysid_aux.detach(),'),
    ],'sysid-train-batch')

    for opt in l.policy.actor_optimizers:
        # Remove predecessor constant-LR assertion only from actor optimizers.
        opt._optimizer_step_pre_hooks.clear()
        opt.register_step_pre_hook(witness)
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


def wm_update(l):
    stats = l.wm_update()  # Existing short-window objective and cadence.
    n = l.long_batch
    td = l.buffer.sample(WM_FRAMES, 2*n)
    speed = td['state'][...,6:12].norm(dim=-1).amax(-1)
    # Keep 3/4 unconditioned; favor low peak speed over the entire long window
    # in the other 1/4, ranking independent candidates rather than gating loss.
    ordinary = n - n//4
    slow = speed[ordinary:].topk(n//4,largest=False).indices + ordinary
    idx = torch.cat((torch.arange(ordinary,device=speed.device),slow))
    td = td[idx]
    assert td.batch_size == torch.Size([n,WM_FRAMES])
    assert not (td['terminated'][:,:-1] | td['truncated'][:,:-1]).any()
    assert (td['state'][...,19].diff(dim=-1)==1).all()
    # Inherited eager update starts only at frame 0 and predicts frames 1..32
    # with stored applied-effective actions, without a reset or teacher forcing.
    logs = l.wm.update([td])
    l.count['wm_optimizer_batches'] += 1
    return stats | dict(wm33_frames=WM_FRAMES, wm33_steps=WM_FRAMES-1,
                        wm33_batch=n, wm33_slow_ranked=n//4,
                        wm33_low_speed=int((speed[idx]<=.05).sum())) | {
        'WM33/'+k:float(v.detach()) for k,v in logs.items()}


def policy_step(b,l):
    starts = torch.stack([l.wm_env.sample_initial_states()])
    for mod in (l.dynamics,l.rep):
        mod.eval().requires_grad_(False)
    lr = actor_lr(l.count['actor_updates']-l.fork_start)
    l.actor_step_lrs.clear()
    for opt in l.policy.actor_optimizers:
        for group in opt.param_groups:
            if torch.is_tensor(group['lr']): group['lr'].fill_(lr)
            else: group['lr'] = lr
    logs = l.policy._captured_update(starts)
    l.count['actor_updates'] += 1
    l.count['critic_minibatches'] += 1
    observed_lrs = [l.actor_step_lrs[o] for o in l.policy.actor_optimizers]
    assert all(math.isclose(x,lr,rel_tol=1e-7) for rates in observed_lrs for x in rates)
    return dict(actor_lr=observed_lrs[0][0],actor_optimizer_group_lrs=observed_lrs) | {k:float(v.detach()) for k,v in logs.items()}
