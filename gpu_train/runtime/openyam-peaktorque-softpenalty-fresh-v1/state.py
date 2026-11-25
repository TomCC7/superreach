"""Full continuation, source identity, predictive disk floors, independent equality."""
from learning import *
import copy, random, json, hashlib, shutil, time
OUT=ROOT.parent/'output/openyam-onreference-fresh-v1'

def cpu(x):
 if hasattr(x,'_tensor'):x=x._tensor
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,np.ndarray):return x.copy()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
 return copy.deepcopy(x)
def move(x,dev):
 if torch.is_tensor(x):return x.to(dev).clone()
 if isinstance(x,dict):return {k:move(v,dev) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(move(v,dev) for v in x)
 return copy.deepcopy(x)
def equal(a,b,path=''):
 if torch.is_tensor(a):assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),path
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
 else:assert a==b,(path,a,b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity():return 'portable-tri-mode'
def reserve(n):
 OUT.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(OUT).free-n>=20*2**30,'disk floor'
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(x,indent=2,allow_nan=False);reserve(len(text.encode())+65536)
 tmp=Path(str(p)+'.tmp');tmp.write_text(text);tmp.replace(p)
def seed():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
EFIELDS=['sent','raw','mapped','clipped','tick','goal','history','history_time','valid','total','ref_id','start_tick','shuffle_order','shuffle_cursor','shuffle_cycles','reference_draw_counts','reference_real_steps']
SFIELDS=['qpos','qvel','qacc_warmstart','time','ctrl','qfrc_applied']
def snapshot(l):
 e=l.e
 return cpu(dict(ppo=None if not hasattr(l,'ppo') else dict(logstd=l.ppo.logstd,opt=l.ppo.opt.state_dict(),lr=l.ppo.lr,updates=l.ppo.updates),identity=identity(),count=l.count,actor=l.actor.state_dict(),critic=l.critic.state_dict(),dynamics=l.dynamics.state_dict(),representation=l.rep.state_dict(),context=e.context.state_dict(),optimizers={k:[o.state_dict() for o in getattr(l,k)] for k in []},wm_optim=[o.state_dict() for o in l.wm.optimizers],actor_optim=[o.state_dict() for o in l.policy.actor_optimizers],critic_optim=[o.state_dict() for o in l.policy.critic_optimizers],replay=l.buffer._storage.to_dict(),replay_step=l.buffer._step,replay_filled=l.buffer.num_transitions,env={k:getattr(e,k) for k in EFIELDS},sim={k:getattr(e.d,k) for k in SFIELDS},effective_context=e.context.read(e),rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate(),goal=e.goal_rng.bit_generator.state)))
