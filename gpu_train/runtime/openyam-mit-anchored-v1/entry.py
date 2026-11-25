"""Shared smooth-hold student recipe with fresh and verified warm-start arms."""
from pathlib import Path
import sys,importlib.util,json,hashlib,copy,os
P=Path(__file__).resolve().parent;R=P.parent;OUT=R/'output'/P.name

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
old=load('_velocity_adapt_frozen_task',R/'openyam-policyonly-resmooth-v1/entry.py')
o=old.openyam;learning=old.learning;core=old.core;base=old.state
# Parent import reorders sys.path; resolve our own overlay by explicit identity.
adapt=load('_velocity_adaptation',P/'adaptation.py')
continuous=load('_charb_continuous',P/'continuous.py')
rewards=load('_charb_rewards',P/'rewards.py')
rewards.install(o,learning,old)
hold=load('_smoothhold_reference',P/'hold.py')
original_lookup=hold.install(o,rewards.recompile)
mit=load('_mit_policy12_control',P/'policy12.py')
mit_control=mit.install(o,learning,old,rewards.recompile)
adapt.install_context(o,learning)
Actor=adapt.make_actor_class(learning.MultiEncoderAE,learning.build_mlp)
MODE='student'
INIT='fresh'
def configure_mode(mode):
    global MODE,OUT
    assert mode in ('teacher','student');MODE=mode
    OUT=R/'output'/f'openyam-mit-anchored-{INIT}-v1'
    base.OUT=core.OUT=core.base.OUT=old.contract.OUT=OUT
    return OUT
core.install=continuous.install;core.load_catalog=continuous.load_catalog
def make_actor(device):
    a=Actor(o.SENSOR,12,[learning.TaskSpec(name='motion',goal_dim=19,extract_training_goal=lambda *x:None)],{'motion':1.},32,(256,256),0.,o.H,mode=MODE).to(device)
    learning.nn.init.zeros_(a.decoder[-1].weight);learning.nn.init.zeros_(a.decoder[-1].bias)
    return a
learning.make_actor=make_actor
PayloadPolicy=adapt.install_policy(learning)
learning.Learner.__init__.__globals__.update(make_actor=make_actor,ActorCriticPolicyLearner=PayloadPolicy)
Learner=learning.Learner
def configure_init(init):
    global INIT
    assert init=='fresh';INIT=init
    return configure_mode(MODE)

def cosine_lr(init,actor_updates):
    import math
    maximum,minimum,horizon,origin=(1e-4,2e-5,60000,0)
    elapsed=min(max(actor_updates-origin,0),horizon)
    return minimum+(maximum-minimum)*.5*(1+math.cos(math.pi*elapsed/horizon))

def set_lr(l):
    lr=cosine_lr(INIT,l.count['actor_updates'])
    for opts in [l.wm.optimizers,l.policy.actor_optimizers,l.policy.critic_optimizers]:
        for opt in opts:
            for group in opt.param_groups:group['lr']=lr
    return lr

# Wrap the actual learner calls, not only main(), so all callers share the schedule.
def scheduled(fn):
    def update(self,*args,**kwargs):
        lr=set_lr(self);result=fn(self,*args,**kwargs);result['cosine_lr']=lr;return result
    return update
Learner.wm_update=scheduled(Learner.wm_update)
Learner.actor_update=scheduled(Learner.actor_update)
configure_mode(MODE)

def hold_evaluate(e,actor):
    import torch,numpy as np
    assert e.num_envs>=32
    ids=torch.tensor(np.resize(e.bank.dev_ids,e.num_envs),device=e.device)
    starts=(e.bank.lengths[ids]-181)//2
    results=[]
    with torch.no_grad():
        actor.eval()
        for mi,mass in [(0,0.),(4,2.)]:
            e.eval_mass_index=mi;e.reset(e.ids,ids,hold.encode(starts))
            position=[];slew=[];accel=[];qdes_offsets=[];dqdes_errors=[]
            for _ in range(180):
                previous=e.sent[:32].clone();dq=e.d.qvel._tensor[:32,:6].clone()
                _,_,_,last=e.step(actor(e.obs()),auto_reset=False)
                ref=e.bank.lookup(last['ref_id'][:32],last['start_tick'][:32]+last['tick'][:32])
                qdes_offsets.append((last['sent'][:32]-ref[:,:6]).abs())
                dqdes_errors.append(e.sent_dq[:32]-ref[:,6:12])
                position.append(last['pos'][:32]);slew.append((last['sent'][:32]-previous)/o.DT);accel.append((last['dq'][:32]-dq)/o.DT)
            results.append(dict(mass_kg=mass,cases=32,ticks=180,qdes_offset_deg_p95=float(torch.rad2deg(torch.quantile(torch.stack(qdes_offsets).flatten(),.95))),dqdes_error_rms_rad_s=float(torch.stack(dqdes_errors).square().mean().sqrt()),position_mean_mm=float(torch.stack(position).mean()*1000),command_slew_rms_rad_s=float(torch.stack(slew).square().mean().sqrt()),joint_acceleration_rms_rad_s2=float(torch.stack(accel).square().mean().sqrt())))
    return results
