"""Metal reach adapter to byte-identical official SuperMimic learners/history.

Execution is eager (no CUDA graphs/torch.compile). Optimizers and loss/rollout
bodies are official. No humanoid reference/RSI/reset or tracking reward imported.
"""
from openyam import *
from supermimic.models import Dynamics,MultiEncoderAE,build_mlp
from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.wm_learner import WorldModelLearner,WorldModelLearnerCfg
from supermimic.learners.policy_learner import ActorCriticPolicyLearner,ActorCriticPolicyLearnerCfg
from types import SimpleNamespace as TaskSpec # structural TaskSpec; avoids unused humanoid task imports
from supermimic.replay_buffer import ReplayBuffer,FieldSpec

class Representation(nn.Module):
 LOSS_WEIGHT_CONFIG={'q':'nominal_error_joint_pos','dq':'nominal_error_joint_vel'}
 rep_state_dim=12;prediction_dim=12;dt=DT
 def __init__(self,e):
  super().__init__();self.fk=e.fk;self.bank=e.bank
  self.register_buffer('center',e.center.clone());self.register_buffer('scale',e.scale.clone())
 def represent(self,full):return full[...,:12]
 def extract_aux_state(self,full):return full[...,12:]
 def joint_state(self,rep):return rep
 def apply_prediction(self,aux,rep,pred,dt):
  tick=aux[...,7:8]+1
  goal=self.bank.lookup(aux[...,8],aux[...,9]+tick[...,0])[:,12:19]
  return torch.cat((goal,tick,aux[...,8:10]),-1),rep+pred*dt
 def loss_blocks(self,rep,aux):return {'q':rep[...,:6],'dq':rep[...,6:12]}
 def add_proprio_noise(self,s):return s # Stage1: no observation randomization.
 def compute_actor_sensor(self,rep,ctx,aux_state,joint_state=None,last_action=None,noisy=False):
  sent=self.bank.lookup(aux_state[...,8],aux_state[...,9]+(aux_state[...,7]-1).clamp(min=0))[...,:6]+self.scale*last_action
  return torch.cat((rep,sent,torch.ones_like(aux_state[...,7:8]),aux_state[...,7:8]*DT),-1)

class CommandDynamics(Dynamics):
 def __init__(self,representation,*args,**kwargs):
  super().__init__(representation,*args,**kwargs);self.reference_bank=representation.bank
 def proc(self,action):return action
 def forward(self,rep,action,aux,ctx):
  sent=self.reference_bank.lookup(aux[...,8],aux[...,9]+aux[...,7])[...,:6]+.1*action
  return super().forward(rep,sent,aux,ctx)

class MetalWorldModelEnv(WorldModelEnv):
 """Reuse upstream step/ring/sample_causal; replace humanoid task methods only."""
 def __init__(self,dynamics,representation,env,buffer,batch_size=4096):
  self.device=env.device;self.dynamics=dynamics;self.representation=representation
  self.num_envs=batch_size;self.replay_buffer=buffer;self.history_length=H
  self.max_step_reward=(4+2+.5+25)*DT # conservative finite progress ceiling
  self._num_actions=6
 def get_observations(self):
  return TensorDict(dict(history=self._sensor_history(),goal_motion=torch.cat((self.aux_state[:,:7],self.representation.bank.lookup(self.aux_state[:,8],self.aux_state[:,9]+self.aux_state[:,7])[:,:12]),-1),critic=torch.cat((self.rep_state,self.aux_state[:,:7],self.sim_ctx),-1)),[self.num_envs],device=self.device)
 def _on_reset(self,batch):
  # Upstream causal window and episode mask, plus explicit actor-visible padding bit.
  self._past_sensors=self._warmstart_past_sensors(batch['sensor_window'],batch['pad_mask'])
  self._past_sensors=self._past_sensors.clone();self._past_sensors[...,18]=(~batch['pad_mask'][:,:-1]).float()
  self._latest_sensor=self._synth_sensor()
 def _compute_commands(self):
  # Reference identity/time are privileged aux, never predicted by learned dynamics.
  self.motion_times=self.aux_state[:,9]+self.aux_state[:,7]
 def _wm_step(self,rep,aux,action,last_action,ctx):
  next_aux,next_rep=self.dynamics(rep,action,aux,ctx)
  sent=self.representation.bank.lookup(aux[:,8],aux[:,9]+aux[:,7])[:,:6]+.1*action
  previous=self.representation.bank.lookup(aux[:,8],aux[:,9]+(aux[:,7]-1).clamp(min=0))[:,:6]+.1*last_action
  r,terms,p,a=reward(self.representation.fk,rep[:,:6],next_rep[:,:6],next_aux[:,:7],sent,previous)
  torque_cost,_=effort_cost(rep[:,:6],rep[:,6:12],sent)
  if getattr(self,'gradient_requested',False) and action.requires_grad:
   self.gradient_requested=False;params=[p for p in self.gradient_actor.parameters() if p.requires_grad]
   def gradients(value):return torch.autograd.grad(value.mean(),params,retain_graph=True,allow_unused=True)
   gs=gradients(torque_cost);gp=gradients(terms['position']*DT);gb=gradients(r)
   def norm(g):return torch.sqrt(sum((v.square().sum() for v in g if v is not None),r.new_tensor(0.)))
   ns,np_,nb=norm(gs),norm(gp),norm(gb)
   dot=sum((u.mul(v).sum() for u,v in zip(gs,gp) if u is not None and v is not None),r.new_tensor(0.))
   self.gradient_receipt=dict(kind='same actual imagined batch first tick; actor parameter gradients, NOT full BPTT objective',soft_lambda=.01,batch_size=action.shape[0],soft_mean=float(torque_cost.detach().mean()),position_mean=float((terms['position']*DT).detach().mean()),base_reward_mean=float(r.detach().mean()),soft_actor_gradient_l2=float(ns.detach()),position_actor_gradient_l2=float(np_.detach()),base_reward_actor_gradient_l2=float(nb.detach()),soft_to_position_grad_ratio=float((ns/(np_+1e-30)).detach()),soft_position_grad_cosine=float((dot/(ns*np_+1e-30)).detach()))
   assert all(np.isfinite(v) for v in self.gradient_receipt.values() if isinstance(v,float))
  r=r+torque_cost
  departure=(p>.05)|(a>0.3490658503988659)
  return next_aux,next_rep,next_rep,((next_aux[:,7]>=180)|departure).float(),r-departure.float()

def make_actor(device):
 actor=MultiEncoderAE(SENSOR,6,[TaskSpec(name='motion',goal_dim=19,extract_training_goal=lambda *x:None)],{'motion':1.},32,(256,256),0.,H).to(device)
 nn.init.zeros_(actor.decoder[-1].weight);nn.init.zeros_(actor.decoder[-1].bias)
 return actor

class Learner:
 def __init__(self,e):
  self.e=e;self.device=e.device;self.rep=Representation(e).to(e.device)
  self.actor=make_actor(e.device)
  self.critic=build_mlp(12+7+CTX,(256,256),1).to(e.device)
  self.dynamics=CommandDynamics(self.rep,6,CTX,e.scale,e.center,(512,512,512)).to(e.device)
  schema={k:FieldSpec(v) for k,v in dict(state=22,action=6,prev_action=6,sim_ctx=CTX,actor_obs=SENSOR).items()}
  schema.update(terminated=FieldSpec(dtype=torch.bool),truncated=FieldSpec(dtype=torch.bool))
  self.buffer=ReplayBuffer(schema,256,e.num_envs,e.device)
  self.wm=WorldModelLearner(self.dynamics,self.rep,WorldModelLearnerCfg(),e.device)
  self.wm_env=MetalWorldModelEnv(self.dynamics,self.rep,e,self.buffer,e.num_envs)
  self.policy=ActorCriticPolicyLearner(self.actor,self.critic,self.wm_env,ActorCriticPolicyLearnerCfg(batch_size=e.num_envs),e.device)
  # Exact numerical learner bodies; eager scheduling avoids compile/capture overhead.
  self.wm._compiled_rollout=self.wm._rollout;self.wm._compiled_compute_losses=self.wm._compute_losses;self.wm._compiled_represent=self.rep.represent
  self.wm._captured_update=self.wm._update_from;self.policy._captured_update=self.policy._update_from;self.policy._rollout_actor=self.actor
  self.count=dict(real_steps=0,real_transitions=0,wm_calls=0,wm_optimizer_batches=0,actor_updates=0,critic_minibatches=0,diagnostic=False,learning_seconds=0.,collection_seconds=0.,wm_seconds=0.,actor_seconds=0.)
 @torch.no_grad()
 def collect(self):
  e=self.e;obs=e.obs();self.actor.eval()
  raw=self.actor(obs)+torch.randn(e.num_envs,6,device=e.device)*(.005/e.scale)
  state=e.fullstate()
  previous=e.raw.clone()
  ctx=e.context.read(e);sensor=e.sensor()
  _,rew,done,last=e.step(raw)
  self.buffer.insert(TensorDict(dict(state=state,action=raw,prev_action=previous,sim_ctx=ctx,actor_obs=sensor,terminated=last["departure"],truncated=(done & ~last["departure"])),[e.num_envs]))
  self.wm.update_normalization(state,last["sent"])
  self.actor.train();self.actor.update_normalization(obs)
  # Freeze running statistics during imagined forwards; update only on real data.
  for mod in [self.actor.obs_normalizer,*self.actor.goal_normalizers.values()]:mod.eval()
  self.count['real_steps']+=1;self.count['real_transitions']+=e.num_envs
  return dict(reward=float(rew.mean()),pos_mean=float(last['pos'].mean()),angle_mean=float(last['angle'].mean()),raw_abs_gt_one_fraction=float((raw.abs()>1).float().mean()),actual_command_clip_fraction=0.,torque_cost=float(last['torque_cost'].mean()),actual_torque_peak_by_joint=last['actual_torque_peak'].amax(0).cpu().tolist(),actual_rated_exceedance_by_joint=last['actual_rated_exceedance'].mean(0).cpu().tolist(),actual_peak_saturation_by_joint=last['actual_peak_saturation'].mean(0).cpu().tolist())
 def wm_update(self):
  data=(self.buffer.sample(self.wm.cfg.horizon,self.wm.cfg.batch_size) for _ in range(self.wm.cfg.num_batches))
  logs=self.wm.update(data);self.count['wm_calls']+=1;self.count['wm_optimizer_batches']+=self.wm.cfg.num_batches
  return {k:float(v.detach()) for k,v in logs.items()}
 def actor_update(self):
  # update() calls actor.train; normalizers must remain frozen during imagination.
  self.actor.train()
  for mod in [self.actor.obs_normalizer,*self.actor.goal_normalizers.values()]:mod.eval()
  for mod in (self.dynamics,self.rep):mod.eval();mod.requires_grad_(False)
  self.wm_env.gradient_actor=self.actor;self.wm_env.gradient_requested=(self.count['actor_updates']%100==0 or self.count['actor_updates']==95)
  starts=torch.stack([self.wm_env.sample_initial_states() for _ in range(self.policy.cfg.num_batches)])
  logs=self.policy._captured_update(starts)
  self.count['actor_updates']+=1;self.count['critic_minibatches']+=20
  result={k:float(v.detach()) for k,v in logs.items()}
  if hasattr(self.wm_env,'gradient_receipt'):result['same_batch_actor_gradient']=dict(self.wm_env.gradient_receipt,actor_update=self.count['actor_updates'])
  return result
 def finite(self):
  for m in [self.actor,self.critic,self.dynamics]:
   assert all(torch.isfinite(t).all() for t in m.state_dict().values())
