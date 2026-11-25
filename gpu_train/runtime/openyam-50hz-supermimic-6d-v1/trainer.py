"""Fresh matched direct-10D screening. No historical checkpoint loading."""
from pathlib import Path
import os,sys,json,hashlib,time,inspect,textwrap,linecache,random,copy,argparse,fcntl
R=Path(__file__).resolve().parent.parent
P=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'openyam-mit-anchored-v1'))
import entry as E
import torch,numpy as np
from supermimic.learners.learner_utils import build_optimizers,_optimize_step

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,z):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(z,indent=2,allow_nan=False));os.replace(tmp,p)
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
def optimizer_groups(l):
 return {name:[g for o in opts for g in o.param_groups] for name,opts in [('wm',l.wm.optimizers),('actor',l.policy.actor_optimizers),('critic',l.policy.critic_optimizers),('encoder',getattr(l,'encoder_opts',[]))]}
def set_constant_lr(l):
 for groups in optimizer_groups(l).values():
  for g in groups:g['lr']=2e-4
 return 2e-4
def assert_lr(l):
 assert all(g['lr']==2e-4 for groups in optimizer_groups(l).values() for g in groups)
def prepare(method):
 E.configure_init('fresh');E.configure_mode('student');E.set_lr=set_constant_lr
 # Use the inherited goal encoder unchanged. This experiment adds NO goal encoder.
 # Existing physical 10D prediction is the context itself, not a new latent/head.
 base=E.PayloadPolicy
 s=textwrap.dedent(inspect.getsource(base._train_batch))
 # Keep the same real replay-start supervision in both arms. Separate only its optimizer path.
 if method=='separate':
  assert s.count('loss = (-ret + aux_loss).mean() + AUX_WEIGHT * payload_loss')==1
  s=s.replace('loss = (-ret + aux_loss).mean() + AUX_WEIGHT * payload_loss','loss = (-ret + aux_loss).mean()')
 ns=dict(base._train_batch.__globals__);fn=str(P/f'{method}_actual_train_batch.py')
 linecache.cache[fn]=(len(s),None,s.splitlines(True),fn)
 exec(compile(s,fn,'exec'),ns)
 cls=type('Fresh'+method.title(),(base,),{'_train_batch':ns['_train_batch']})
 E.learning.Learner.__init__.__globals__['ActorCriticPolicyLearner']=cls
 return cls

def configure(l,method,batch=512,smoke=False):
 import alignment as A
 l.wm.cfg.batch_size=batch;l.wm.cfg.num_batches=1 if smoke else 10;l.wm.cfg.horizon=A.WM_FRAMES
 l.policy.cfg.batch_size=batch;l.policy.cfg.num_batches=1;l.policy.cfg.horizon=A.ACTOR_HORIZON
 l.policy.cfg.gamma=A.GAMMA;l.policy.cfg.lam=A.LAMBDA
 # The inherited constructor caches this before the runtime cfg override.
 l.policy._v_max=l.wm_env.max_step_reward/(1-l.policy.cfg.gamma)
 l.wm_env.num_envs=batch
 l.encoder_opts=[]
 if method=='separate':
  encids={id(p) for p in l.actor.history_encoder.parameters()}
  for opt in l.policy.actor_optimizers:
   for group in opt.param_groups:group['params']=[p for p in group['params'] if id(p) not in encids]
  l.policy.actor_params=[p for p in l.policy.actor_params if id(p) not in encids]
  l.encoder_opts=build_optimizers([l.actor.history_encoder],2e-4,betas=(.9,.95),device=l.e.device)
 l.encoder_updates=0
 set_constant_lr(l)
 # Assert actual groups at every step, including internal critic minibatches.
 for opts in [l.wm.optimizers,l.policy.actor_optimizers,l.policy.critic_optimizers,l.encoder_opts]:
  for opt in opts:opt.register_step_pre_hook(lambda *_:assert_lr(l))
