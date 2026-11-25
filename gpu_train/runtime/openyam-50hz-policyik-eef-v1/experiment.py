"""Fresh EEF-only 50 Hz experiment atop the frozen six-position plant."""
from pathlib import Path
import argparse, fcntl, hashlib, importlib.util, inspect, json, os, signal, sys, time
import math
import numpy as np
import torch
from geometry import TCP
from objective import Recipe, DT, MAX_RATE, goal_twist, near_weight, plateau_weight, task_reward, clean_reward
from policy import Actor
import calibration

P=Path(__file__).resolve().parent
BASE=P.parent/'openyam-50hz-supermimic-6d-v1'
BASE_LR=2e-4

def file_sha256(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def source_digest():
    h=hashlib.sha256()
    for path in sorted([*P.glob('*.py'),*(BASE.glob('*.py'))]):
        h.update(path.name.encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def setup(cfg):
    spec=importlib.util.spec_from_file_location('_eef_frozen_6d',BASE/'experiment.py')
    b=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b); b.setup()
    E=b.T.E; l=E.learning; o=E.o; patch=b.A.patch
    digest=source_digest()
    def identity(): return digest
    E.identity=E.core.identity=E.core.base.identity=E.base.identity=identity
    b.eef_source_digest=digest
    if cfg.stable_finetune or cfg.minimal_slew or cfg.global_damping or cfg.clean_obs:
        b.T.configure=patch(b.T.configure,[('opt.register_step_pre_hook(lambda *_:assert_lr(l))','pass')],
                            'eef-stable-configure')
        # Inherited WM-update wrapper normally resets every optimizer LR.
        E.set_lr=lambda l: BASE_LR
    TensorDict=l.TensorDict
    def pose(bank,ids,ticks):
        ids=ids.long(); _,ticks=E.hold.decode(ticks)
        return bank.table[bank.offsets[ids]+ticks.clamp_min(0).minimum(bank.lengths[ids]-1),12:19]
    def is_track(bank,ids,ticks):
        return (ids<bank.step_offset)&(~E.hold.decode(ticks)[0])
    def observations(history,goal,state,u,ctx):
        return TensorDict(dict(history=history,goal_pose=goal,privileged_ctx=ctx,
                               critic=torch.cat((state,goal,u,ctx),-1)),[goal.shape[0]],device=goal.device)
    def real_obs(self):
        return observations(self.history.clone(),self.goal.clone(),self.state(),self.sent,self.context.read(self))
    def wm_obs(self):
        return observations(self._sensor_history(),self.aux_state[:,:7],self.rep_state,self.last_actions,self.sim_ctx)
    o.OpenYAMEnv.obs=real_obs; l.MetalWorldModelEnv.get_observations=wm_obs
    def apply_prediction(self,aux,rep,pred,dt):
        tick=aux[...,7:8]+1
        goal=pose(self.bank,aux[...,8],aux[...,9]+tick[...,0])
        return torch.cat((goal,tick,aux[...,8:10]),-1),rep+pred*dt
    l.Representation.apply_prediction=apply_prediction
    old_init=o.OpenYAMEnv.__init__
    def env_init(self,*args,**kwargs):
        old_init(self,*args,**kwargs)
        self.fk=TCP(self.model).to(device=self.device,dtype=torch.float32)
        self.q_nom=torch.as_tensor([0.,1.047,1.047,0.,0.,0.],device=self.device)
        self.bank.goal_pose=pose
    o.OpenYAMEnv.__init__=env_init
    def score(fk,q,dq,goal,u,previous,prev_goal,tracking,q_nom,previous_dq):
        if cfg.clean_obs:
            return clean_reward(fk,q,dq,goal,u,previous,previous_dq,cfg.tolerance)
        if not (cfg.anti_jitter or cfg.stable_finetune):
            v,w=goal_twist(goal,prev_goal,tracking)
            return task_reward(fk,q,dq,goal,u,previous,v,w,q_nom,cfg.tolerance,
                               slew_coeff=.60 if cfg.minimal_slew else .40,
                               previous_dq=previous_dq if cfg.global_damping else None,global_damping=cfg.global_damping)
        # Unmasked goal motion suppresses damping on STEP switch ticks too.
        v,w=goal_twist(goal,prev_goal,torch.ones_like(tracking))
        hold=(plateau_weight(v.norm(dim=-1),.005,.02)*plateau_weight(w.norm(dim=-1),.025,.10)
              if cfg.stable_finetune else near_weight(v.norm(dim=-1),.02)*near_weight(w.norm(dim=-1),.10))
        return task_reward(fk,q,dq,goal,u,previous,v*tracking[:,None],w*tracking[:,None],q_nom,
                           cfg.tolerance,previous_dq=previous_dq,hold_weight=hold,stable=cfg.stable_finetune)
    b.eef_score=score
    lookup='self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,12:19]'
    call='reward(self.fk,old,self.d.qpos._tensor[:,:6],self.goal,self.raw,previous,self.d.qvel._tensor[:,:6],self.bank.lookup(self.ref_id,self.start_tick+self.tick),old_dq,self.bank.lookup(self.ref_id,self.start_tick+(self.tick-1).clamp(min=0)))'
    # The frozen physical step updates the cached goal twice; replace both reads.
    original=o.OpenYAMEnv.step
    o.OpenYAMEnv.step=patch(original,[
        ('old=self.d.qpos._tensor[:,:6].clone();','old_goal=self.goal.clone();'),
        (f'self.goal.copy_({lookup});r,terms,p,a={call}',
         'self.goal.copy_(pose(self.bank,self.ref_id,self.start_tick+self.tick));r,terms,p,a=score(self.fk,self.d.qpos._tensor[:,:6],self.d.qvel._tensor[:,:6],self.goal,self.raw,previous,old_goal,is_track(self.bank,self.ref_id,self.start_tick),self.q_nom,old_dq)'),
        (lookup,'pose(self.bank,self.ref_id,self.start_tick+self.tick)'),
        *([('r=r+torque_cost','torque_cost=torch.zeros_like(torque_cost)')] if cfg.clean_obs else []),
    ],'eef-physical-step',pose=pose,score=score,is_track=is_track)
    def wm_step(self,rep,aux,u,previous,ctx):
        next_aux,next_rep=self.dynamics(rep,u,aux,ctx)
        r,_,_,_=score(self.representation.fk,next_rep[:,:6],next_rep[:,6:],next_aux[:,:7],u,previous,aux[:,:7],is_track(self.representation.bank,aux[:,8],aux[:,9]),self.q_nom,rep[:,6:])
        torque=0. if cfg.clean_obs else b.task.effort_cost(rep[:,:6],rep[:,6:],u)[0]
        departure=b.task.physical_failure(next_rep[:,:6],next_rep[:,6:])
        return next_aux,next_rep,next_rep,((next_aux[:,7]>=300)|departure).float(),r-departure.float()+torque
    l.MetalWorldModelEnv._wm_step=wm_step
    old_wm=l.MetalWorldModelEnv.__init__
    def wm_init(self,dynamics,representation,env,*args,**kwargs):
        old_wm(self,dynamics,representation,env,*args,**kwargs)
        self.q_nom=env.q_nom
        self.max_step_reward=DT*(MAX_RATE if cfg.tolerance else 5.5)
    l.MetalWorldModelEnv.__init__=wm_init
    old_factory=E.make_actor
    def actor_factory(env):
        # Reuse only the frozen validity-aware proprioception normalizer.
        if cfg.clean_obs:
            return Actor(env.fk,cfg,None,l.build_mlp).to(env.device)
        old=old_factory(env.device)
        return Actor(env.fk,cfg,old.obs_normalizer,l.build_mlp).to(env.device)
    l.Learner.__init__=patch(l.Learner.__init__,[
        ('make_actor(e.device)','make_actor(e)'),
        ('12+7+12+ACTION_DIM+CTX','12+7+ACTION_DIM+CTX'),
    ],'eef-learner',make_actor=actor_factory)
    l.Learner.collect=patch(l.Learner.collect,[
        ('self.actor(obs)+torch.randn(e.num_envs,ACTION_DIM,device=e.device)*SIGMA_Q',
         'self.actor.command(obs,torch.randn(e.num_envs,ACTION_DIM,device=e.device)*exploration)'),
        *([('self.actor.update_normalization(obs)','self.actor.obs_normalizer.eval()')] if cfg.stable_finetune else []),
    ],'eef-collect',exploration=.12 if cfg.command=='accumulate' else .003)
    if cfg.uprev_jacobian_l2:
        cls=l.Learner.__init__.__globals__['ActorCriticPolicyLearner']
        cls._train_batch=patch(cls._train_batch,[
            ('obs = self.wm_env.reset(start)',
             'obs = self.wm_env.reset(start)\n  jac_mse = self.actor.previous_command_jacobian(obs).square().sum((1,2)).mean()'),
            ('loss = (-ret + aux_loss).mean()',
             'loss = (-ret + aux_loss).mean()\n  jac_cost = self.actor.cfg.uprev_jacobian_l2 * jac_mse\n'
             '  jac_diagnostics = jacobian_diagnostics(loss, jac_cost, self.actor_params) if getattr(self, "verify_jacobian", False) else {}\n'
             '  loss = loss + jac_cost'),
            ('"grad_norm": grad_norm,',
             '"grad_norm": grad_norm,\n    "uprev_jacobian_frob_rms": jac_mse.detach().sqrt(),\n'
             '    "uprev_jacobian_l2": jac_cost.detach(),\n    **jac_diagnostics,'),
        ],'eef-uprev-reg',jacobian_diagnostics=jacobian_diagnostics)
    return b


def jacobian_diagnostics(main, regularizer, params):
    def norm(loss):
        grads=torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
        return torch.stack([g.square().sum() for g in grads if g is not None]).sum().sqrt()
    main_norm, reg_norm = norm(main), norm(regularizer)
    assert main_norm > 0 and reg_norm > 0
    return dict(uprev_main_loss=main.detach(), uprev_loss_ratio=regularizer.detach()/main.detach().abs(),
                uprev_main_grad_norm=main_norm, uprev_reg_grad_norm=reg_norm,
                uprev_grad_ratio=reg_norm/main_norm)
