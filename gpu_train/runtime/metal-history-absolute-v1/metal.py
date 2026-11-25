"""Fixed-base MetalArm adapter; no G11 policy/task numerical code imported."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from supermimic.sim_context import SimContext,SimContextSpec
ROOT=Path(__file__).resolve().parent
ASSET=ROOT.parent/'policyik-metal-right-g11-v1'
DT=1/30
H=16
SENSOR=20
GOAL=7
CTX=10

def quatmat(q):
 w,x,y,z=q.unbind(-1)
 return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)

class FK(nn.Module):
 """MuJoCo local body/joint/site frames -> differentiable world TCP transform."""
 def __init__(self,m,site):
  super().__init__();self.site=int(site);self.chain=[]
  b=int(m.site_bodyid[site])
  while b: self.chain.insert(0,b);b=int(m.body_parentid[b])
  for name,v in [('pos',m.body_pos),('rot',m.body_quat),('axis',m.jnt_axis),('jpos',m.jnt_pos),('sp',m.site_pos[site]),('sr',m.site_quat[site])]:
   self.register_buffer(name,torch.tensor(v,dtype=torch.float32))
  self.js={b:[j for j in range(m.njnt) if m.jnt_bodyid[j]==b] for b in self.chain}
  self.qadr=[int(x) for x in m.jnt_qposadr]
 def forward(self,q):
  shape=q.shape[:-1];r=torch.eye(3,device=q.device,dtype=q.dtype).expand(*shape,3,3);p=torch.zeros(*shape,3,device=q.device,dtype=q.dtype)
  for b in self.chain:
   p=p+(r@self.pos[b].expand(*shape,3).unsqueeze(-1)).squeeze(-1);r=r@quatmat(self.rot[b])
   for j in self.js[b]:
    a=q[...,self.qadr[j]]*.5;qq=torch.cat((torch.cos(a)[...,None],torch.sin(a)[...,None]*self.axis[j]),-1);jr=quatmat(qq)
    p=p+(r@(self.jpos[j]-(jr@self.jpos[j].expand(*shape,3).unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1);r=r@jr
  return p+(r@self.sp.expand(*shape,3).unsqueeze(-1)).squeeze(-1),r@quatmat(self.sr)

def errors(fk,q,goal):
 p,r=fk(q);gr=quatmat(goal[...,3:]);dist=(p-goal[...,:3]).norm(dim=-1)
 # atan2 SO(3) is stable at identity and differentiable away from pi.
 rel=gr.transpose(-1,-2)@r
 v=torch.stack((rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]),-1)*.5
 c=((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
 angle=torch.atan2(v.norm(dim=-1),c)
 return dist,angle

def reward(fk,qold,qnew,goal,sent,previous):
 p,a=errors(fk,qnew,goal);po,ao=errors(fk,qold,goal)
 velocity=(sent-previous)/DT
 # Joint command velocity scaled by 1 rad/s. No raw incremental-action penalty.
 terms=dict(position=4*torch.exp(-p/.10),orientation=2*torch.exp(-a/.30),progress=10*(po-p)+.5*(ao-a),hold=.5*((p<.005)&(a<math.radians(.5))).float(),command_slew=-.002*velocity.square().mean(-1))
 return sum(terms.values())*DT,terms,p,a

class EffectivePayloadSpec(SimContextSpec):
 """Actual merged gripper mass, body COM, and symmetric full inertia (10).

 Baseline contains original tool mass too: not a YAML draw or actor input.
 Stage1 changes no physics parameters. Nonzero variants require later validation.
 """
 name='effective_rigid_payload';dim=10
 def bind(self,e):
  self.bid=e.payload_body
  return [(0,4),(-.3,.3),(-.3,.3),(-.3,.3)]+[(-.1,.1)]*6
 def read(self,e):
  m=e.sim.model;b=self.bid
  def field(name):
   x=getattr(m,name);x=x._tensor if hasattr(x,'_tensor') else x
   # mjlab constants may be shared or per-world; never substitute sampled values.
   expected={'body_mass':1,'body_ipos':2,'body_inertia':2,'body_iquat':2}[name]
   return (x[b].expand(e.num_envs,*x[b].shape) if x.ndim==expected else x[:,b])
  mass=field('body_mass');com=field('body_ipos');principal=field('body_inertia');rot=quatmat(field('body_iquat'))
  it=rot@torch.diag_embed(principal)@rot.transpose(-1,-2)
  iv=torch.stack((it[:,0,0],it[:,1,1],it[:,2,2],it[:,0,1],it[:,0,2],it[:,1,2]),-1)
  return torch.cat((mass[:,None],com,iv),-1).clone()

class MetalEnv:
 def __init__(self,n=4096,device='cuda:0'):
  import mujoco
  from mjlab.sim import Simulation,SimulationCfg,MujocoCfg
  self.device=device;self.num_envs=n
  self.model=mujoco.MjModel.from_xml_path(str(ASSET/'assets/right_runtime.xml'))
  assert (self.model.nq,self.model.nv,self.model.nu)==(6,6,6)
  self.site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'right_tip')
  self.payload_body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'right_gripper_base')
  self.sim=Simulation(n,SimulationCfg(nconmax=64,njmax=128,mujoco=MujocoCfg(timestep=1/60,iterations=32,ls_iterations=10)),self.model,device)
  self.d=self.sim.data;self.fk=FK(self.model,self.site).to(device)
  limits=torch.tensor(self.model.jnt_range,dtype=torch.float32,device=device)
  self.center=limits.mean(-1);self.scale=(limits[:,1]-limits[:,0])*.5
  self.sent=torch.zeros(n,6,device=device);self.raw=torch.zeros_like(self.sent);self.mapped=torch.zeros_like(self.sent);self.clipped=torch.zeros_like(self.sent)
  self.tick=torch.zeros(n,dtype=torch.long,device=device);self.goal=torch.zeros(n,7,device=device)
  self.history=torch.zeros(n,H,SENSOR,device=device);self.history_time=torch.zeros(n,H,device=device);self.valid=torch.zeros(n,H,dtype=torch.bool,device=device)
  self.ids=torch.arange(n,device=device);self.total=0;self.goal_rng=np.random.default_rng(713);self.stage=0;self.near_fraction=.10
  self.pool={}
  for name in ['train_start_pool','train_goal_pool']:
   z=np.load(ASSET/f'data/{name}.npz');self.pool[name]={k:torch.tensor(z[k],dtype=torch.float32,device=device) for k in ['q','ee_pos','ee_quat']}
  spec=EffectivePayloadSpec();self.context=SimContext([spec],spec.bind(self)).to(device)
  self.reset(self.ids)
 def decode(self,action):return self.center+self.scale*action.clamp(-1,1)
 def sensor(self):return torch.cat((self.d.qpos._tensor,self.d.qvel._tensor,self.sent,torch.ones(self.num_envs,1,device=self.device),self.tick[:,None]*DT),-1)
 def state(self):return torch.cat((self.d.qpos._tensor,self.d.qvel._tensor),-1)
 def obs(self):
  state=self.state();ctx=self.context.read(self)
  return TensorDict(dict(history=self.history.clone(),goal_motion=self.goal.clone(),critic=torch.cat((state,self.goal,ctx),-1)),[self.num_envs],device=self.device)
 def reset(self,ids):
  if not len(ids):return
  self.sim.reset(ids)
  p=self.pool['train_start_pool'];rows=self.goal_rng.integers(len(p['q']),size=len(ids));q=p['q'][torch.as_tensor(rows,device=self.device)]
  self.d.qpos[ids]=q;self.d.qvel[ids]=0;self.sent[ids]=q;self.d.ctrl[ids]=q
  self.raw[ids]=(q-self.center)/self.scale;self.mapped[ids]=q;self.clipped[ids]=q;self.tick[ids]=0
  # Near goals are geometry-selected from the ORIGINAL valid TRAIN goal pool.
  # Stage0: nearest of 64 IID candidates in normalized joint distance; no IK/ref
  # joint target goes to actor. Stage1 widening requires saved acquisition gate.
  g=self.pool['train_goal_pool'];rows=self.goal_rng.integers(len(g['q']),size=(len(ids),64))
  candidates=g['q'][torch.as_tensor(rows,device=self.device)]
  nearest=((candidates-q[:,None])/self.scale).square().sum(-1).argmin(-1)
  qtarget=candidates[torch.arange(len(ids),device=self.device),nearest]
  if self.stage>0:qtarget=candidates[:,0]
  pos,rot=self.fk(qtarget)
  import mujoco
  quats=[]
  for mat in rot.detach().cpu().numpy():
   out=np.zeros(4);mujoco.mju_mat2Quat(out,mat.reshape(-1).astype(np.float64));quats.append(out)
  self.goal[ids]=torch.cat((pos,torch.tensor(np.array(quats),device=self.device,dtype=torch.float32)),-1)
  self.sim.forward();sensor=self.sensor()[ids]
  self.history[ids]=sensor[:,None];self.history[ids,:,18]=0;self.history[ids,-1,18]=1
  self.valid[ids]=False;self.valid[ids,-1]=True;self.history_time[ids]=0
 @torch.no_grad()
 def step(self,action):
  assert action.shape==(self.num_envs,6) and torch.isfinite(action).all()
  old=self.d.qpos.clone();previous=self.sent.clone();self.raw.copy_(action);self.mapped.copy_(self.center+self.scale*action)
  self.clipped.copy_(self.decode(action));self.sent.copy_(self.clipped);self.d.ctrl[:]=self.sent
  for _ in range(2):self.sim.step()
  self.sim.forward();self.tick+=1;self.total+=self.num_envs
  r,terms,p,a=reward(self.fk,old,self.d.qpos,self.goal,self.sent,previous)
  terminal=self.tick>=360
  last=dict(q=self.d.qpos.clone(),dq=self.d.qvel.clone(),raw=self.raw.clone(),mapped=self.mapped.clone(),clipped=self.clipped.clone(),sent=self.sent.clone(),goal=self.goal.clone(),pos=p,angle=a)
  self.history=torch.cat((self.history[:,1:],self.sensor()[:,None]),1);self.valid=torch.cat((self.valid[:,1:],torch.ones(self.num_envs,1,dtype=torch.bool,device=self.device)),1)
  self.history_time=torch.cat((self.history_time[:,1:],self.tick[:,None]*DT),1)
  self.reset(self.ids[terminal])
  assert torch.isfinite(self.state()).all() and torch.isfinite(r).all()
  return self.obs(),r,terminal,last
