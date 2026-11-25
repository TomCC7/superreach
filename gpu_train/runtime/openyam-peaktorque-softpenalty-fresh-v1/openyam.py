"""OpenYAM dynamic reference task; official SuperMimic learning bodies reused."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'metal-history-absolute-v1'))
from metal import FK,errors,reward,EffectivePayloadSpec,quatmat
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from supermimic.sim_context import SimContext
import json,hashlib
DT=1/30;H=16;SENSOR=20;GOAL=7;CTX=10
ASSET=Path(os.environ['SUPERREACH_BANK']).parent
DATA_VERSION=Path(os.environ['SUPERREACH_BANK']).name
RATED=[9.,9.,9.,3.,3.,3.];PEAK=[27.,27.,27.,7.,7.,7.]
KP=[50.,60.,50.,25.,15.,10.];KD=[3.,4.,3.,2.,1.,.8]
def effort_cost(q,dq,qdes):
 kp=q.new_tensor(KP);kd=q.new_tensor(KD);rated=q.new_tensor(RATED);peak=q.new_tensor(PEAK)
 requested=kp*(qdes-q)-kd*dq
 excess=torch.relu(requested.abs()-rated)/(peak-rated)
 return -.01*excess.square().mean(-1)*DT,requested
class ReferenceBank(nn.Module):
 def __init__(self,device):
  super().__init__();p=ASSET/DATA_VERSION;self.manifest=json.loads((p/'MANIFEST.json').read_text());assert self.manifest['ready'] and self.manifest['accepted_train']>=128
  if DATA_VERSION=='bank-full-v2':assert self.manifest['accepted_train']>=16000
  arrays=[]
  for row in self.manifest['rows']:
   f=p/row['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'];z=np.load(f);t=z['time_s'];ticks=np.arange(int(np.floor(t[-1]*30))+1);ix=np.searchsorted(t,ticks/30-1e-8).clip(0,len(t)-1)
   arrays.append(np.concatenate((z['q'][ix],z['dq'][ix],z['tcp_pos'][ix],z['tcp_quat'][ix]),axis=1))
  lengths=np.array([len(a) for a in arrays]);assert np.all(lengths>=181)
  offsets=np.concatenate(([0],np.cumsum(lengths)[:-1]));table=np.concatenate(arrays,axis=0).astype(np.float32)
  assert [r['id'] for r in self.manifest['rows']]==list(range(len(arrays)))
  # Immutable inputs are source-hash pinned, not copied into each learnable checkpoint.
  self.register_buffer('table',torch.tensor(table,device=device),persistent=False);self.register_buffer('lengths',torch.tensor(lengths,device=device),persistent=False);self.register_buffer('offsets',torch.tensor(offsets,device=device),persistent=False);self.train_ids=np.array(self.manifest['train_ids']);self.dev_ids=np.array(self.manifest['dev_ids'])
 def lookup(self,ids,ticks):
  ids=ids.long();ticks=ticks.long().clamp(min=0);ticks=torch.minimum(ticks,self.lengths[ids]-1);return self.table[self.offsets[ids]+ticks]
class OpenYAMEnv:
 def __init__(self,n=4096,device='cuda:0'):
  import mujoco
  from mjlab.sim import Simulation,SimulationCfg,MujocoCfg
  self.device=device;self.num_envs=n;self.bank=ReferenceBank(device);self.model=mujoco.MjModel.from_xml_path(os.environ['SUPERREACH_SCENE']);assert (self.model.nq,self.model.nv,self.model.nu)==(8,8,6)
  self.site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'arm_tcp_site');self.payload_body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'hand')
  self.model.actuator_ctrllimited[:]=False
  self.model.actuator_forcelimited[:]=True
  self.model.actuator_forcerange[:]=np.array([[-x,x] for x in PEAK],dtype=float)
  self.sim=Simulation(n,SimulationCfg(nconmax=64,njmax=128,mujoco=MujocoCfg(timestep=1/480,iterations=32,ls_iterations=10)),self.model,device);self.d=self.sim.data;self.fk=FK(self.model,self.site).to(device)
  limits=torch.tensor(self.model.jnt_range[:6],dtype=torch.float32,device=device);self.center=torch.zeros(6,device=device);self.scale=torch.full((6,),.1,device=device)
  self.sent=torch.zeros(n,6,device=device);self.raw=torch.zeros_like(self.sent);self.mapped=torch.zeros_like(self.sent);self.clipped=torch.zeros_like(self.sent)
  self.tick=torch.zeros(n,dtype=torch.long,device=device);self.ref_id=torch.zeros_like(self.tick);self.start_tick=torch.zeros_like(self.tick);self.goal=torch.zeros(n,7,device=device)
  self.history=torch.zeros(n,H,SENSOR,device=device);self.history_time=torch.zeros(n,H,device=device);self.valid=torch.zeros(n,H,dtype=torch.bool,device=device);self.ids=torch.arange(n,device=device);self.total=0;self.goal_rng=np.random.default_rng(81115)
  self.shuffle_order=self.goal_rng.permutation(self.bank.train_ids);self.shuffle_cursor=0;self.shuffle_cycles=0;self.reference_draw_counts=np.zeros(len(self.bank.lengths),dtype=np.int64);self.reference_real_steps=torch.zeros(len(self.bank.lengths),device=device,dtype=torch.long)
  spec=EffectivePayloadSpec();self.context=SimContext([spec],spec.bind(self)).to(device);self.reset(self.ids)
 def decode(self,action):return self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,:6]+self.scale*action
 def sensor(self):return torch.cat((self.d.qpos._tensor[:,:6],self.d.qvel._tensor[:,:6],self.sent,torch.ones(self.num_envs,1,device=self.device),self.tick[:,None]*DT),-1)
 def state(self):return torch.cat((self.d.qpos._tensor[:,:6],self.d.qvel._tensor[:,:6]),-1)
 def fullstate(self):return torch.cat((self.state(),self.goal,self.tick[:,None].float(),self.ref_id[:,None].float(),self.start_tick[:,None].float()),-1)
 def obs(self):return TensorDict(dict(history=self.history.clone(),goal_motion=torch.cat((self.goal,self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,:12]),-1),critic=torch.cat((self.state(),self.goal,self.context.read(self)),-1)),[self.num_envs],device=self.device)
 def reset(self,ids,ref_ids=None,start_ticks=None):
  if not len(ids):return
  if ref_ids is None:
   pieces=[];needed=len(ids)
   while needed:
    available=len(self.shuffle_order)-self.shuffle_cursor;take=min(available,needed);pieces.append(self.shuffle_order[self.shuffle_cursor:self.shuffle_cursor+take]);self.shuffle_cursor+=take;needed-=take
    if self.shuffle_cursor==len(self.shuffle_order):self.shuffle_order=self.goal_rng.permutation(self.bank.train_ids);self.shuffle_cursor=0;self.shuffle_cycles+=1
   selected=np.concatenate(pieces);np.add.at(self.reference_draw_counts,selected,1);ref_ids=torch.as_tensor(selected,device=self.device);lengths=self.bank.lengths[ref_ids].cpu().numpy();start_ticks=torch.as_tensor([self.goal_rng.integers(0,int(k)-180) for k in lengths],device=self.device)
  assert torch.all(start_ticks>=0) and torch.all(start_ticks+180<self.bank.lengths[ref_ids])
  self.sim.reset(ids);self.ref_id[ids]=ref_ids;self.start_tick[ids]=start_ticks;self.tick[ids]=0;initial=self.bank.lookup(ref_ids,start_ticks);q=initial[:,:6];dq=initial[:,6:12]
  self.d.qpos[ids,:6]=q;self.d.qpos[ids,6:]=.055;self.d.qvel[ids,:6]=dq;self.d.qvel[ids,6:]=0;self.sent[ids]=q;self.d.ctrl[ids]=q
  self.raw[ids]=0;self.mapped[ids]=q;self.clipped[ids]=q;self.goal[ids]=initial[:,12:19]
  self.sim.forward();sensor=self.sensor()[ids];self.history[ids]=sensor[:,None];self.history[ids,:,18]=0;self.history[ids,-1,18]=1;self.valid[ids]=False;self.valid[ids,-1]=True;self.history_time[ids]=0
 @torch.no_grad()
 def step(self,action,auto_reset=True):
  assert action.shape==(self.num_envs,6) and torch.isfinite(action).all();old=self.d.qpos._tensor[:,:6].clone();previous=self.sent.clone();self.raw.copy_(action);self.mapped.copy_(self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,:6]+self.scale*action);self.clipped.copy_(self.decode(action));self.sent.copy_(self.clipped);self.d.ctrl[:]=self.sent
  torque_cost,tau_requested=effort_cost(self.d.qpos._tensor[:,:6],self.d.qvel._tensor[:,:6],self.sent)
  applied=[]
  for substep in range(16):
   self.sim.step()
   applied.append(self.d.actuator_force._tensor.clone())
   if substep==0:first_physics_q=self.d.qpos._tensor.clone()
  self.sim.forward();self.tick+=1;self.total+=self.num_envs;self.reference_real_steps+=torch.bincount(self.ref_id,minlength=len(self.bank.lengths))
  # Match upstream score poststep against cached reference, then advance clip for next observation.
  self.goal.copy_(self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,12:19]);r,terms,p,a=reward(self.fk,old,self.d.qpos._tensor[:,:6],self.goal,self.sent,previous);departure=(p>.05)|(a>0.3490658503988659);r=r-departure.float();terminal=(self.tick>=180)|departure
  r=r+torque_cost
  actual_torque=torch.stack(applied);rated=actual_torque.new_tensor(RATED);peak=actual_torque.new_tensor(PEAK)
  assert torch.isfinite(actual_torque).all()
  last=dict(torque_cost=torque_cost.clone(),tau_requested=tau_requested.clone(),actual_torque_peak=actual_torque.abs().amax(0),actual_rated_exceedance=(actual_torque.abs()>rated).float().mean(0),actual_peak_saturation=(actual_torque.abs()>=peak-1e-4).float().mean(0),q=self.d.qpos._tensor[:,:6].clone(),dq=self.d.qvel._tensor[:,:6].clone(),raw=self.raw.clone(),sent=self.sent.clone(),goal=self.goal.clone(),pos=p,angle=a,ref_id=self.ref_id.clone(),start_tick=self.start_tick.clone(),tick=self.tick.clone(),finger_q=self.d.qpos._tensor[:,6:].clone(),first_physics_q=first_physics_q,departure=departure.clone())
  self.goal.copy_(self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,12:19]);self.history=torch.cat((self.history[:,1:],self.sensor()[:,None]),1);self.valid=torch.cat((self.valid[:,1:],torch.ones(self.num_envs,1,dtype=torch.bool,device=self.device)),1);self.history_time=torch.cat((self.history_time[:,1:],self.tick[:,None]*DT),1)
  if auto_reset:self.reset(self.ids[terminal])
  assert torch.isfinite(self.state()).all() and torch.isfinite(r).all();return self.obs(),r,terminal,last
