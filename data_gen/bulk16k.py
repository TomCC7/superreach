"""Durable resumable >=16000 distinct TRAIN + separate32DEV native RoboPlan bank."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ.setdefault('MUJOCO_GL','egl')
from generate_bank import *
import fcntl,shutil
from native_validation import validate
B=R/'output/openyam-history-window-v1/bank-full-v2'
def atomic(p,z):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(z,indent=2));t.replace(p)
def key(q0,q1):
 a=np.asarray(q0,dtype='<f8').tobytes();b=np.asarray(q1,dtype='<f8').tobytes();return hashlib.sha256(min(a+b,b+a)).hexdigest()
def main():
 import argparse
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--init-only', action='store_true', help='Seed the ledgers for pool_generator.py')
 args = parser.parse_args()
 B.mkdir(parents=True,exist_ok=True);lock=(B/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (B/'MANIFEST.json').exists(),'immutable full bank already READY'
 m=mj.MjModel.from_xml_path(str(M/'runtime.xml'));d=mj.MjData(m);meta=json.loads((M/'MODEL_VALIDATION.json').read_text());scene=Scene('openyam16000',loadMjcfModel(M/'planner.xml'));scene.setCollisions([tuple(p) for p in meta['disabled_body_pairs']],False);scene.importJointLimitsFromConfig(loadJointLimitsConfig(M/'limits.yaml'));scene.setRngSeed(91114)
 planner=RRT(scene,RRTOptions(max_nodes=4000,max_connection_distance=.4,collision_check_step_size=.008,max_planning_time=5.,rrt_connect=True));shortcut=PathShortcutter(scene,PathShortcuttingOptions(max_step_size=.008,max_iters=100,seed=91114));timer=PathParameterizerTOPPRA(scene)
 ledger=B/'accepted.jsonl';events=B/'candidates.jsonl';rows=[];counts=dict(endpoint_draws=0,endpoint_collision=0,planner_failure=0,timer_failure=0,too_short=0,trajectory_collision=0,duplicate=0,numerical_invalid=0)
 if ledger.exists():rows=[json.loads(line) for line in ledger.read_text().splitlines()]
 else:
  seedbank=json.loads((O/'MANIFEST.json').read_text());assert seedbank['ready']
  with ledger.open('a') as f:
   for row in seedbank['rows']:
    old=O/row['file'];assert sha(old)==row['sha256'];new=dict(row,file='../bank/'+row['file'],endpoint_identity=key(row['q_start'],row['q_goal']),geometric_path_sha256=hashlib.sha256(np.asarray(np.load(old)['geometric_path'],dtype='<f8').tobytes()).hexdigest(),origin='preserved_initial160');rows.append(new);f.write(json.dumps(new)+'\n')
   f.flush();os.fsync(f.fileno())
 if args.init_only:
  events.touch(exist_ok=True)
  return
 if events.exists() and events.stat().st_size:
  for line in events.read_text().splitlines():
   ev=json.loads(line)
   for k,v in ev['delta'].items():counts[k]+=v
  attempt=ev['attempt']+1
 else:attempt=0
 endpoints={r['endpoint_identity'] for r in rows};geom={r['geometric_path_sha256'] for r in rows};assert len(endpoints)==len(rows) and len(geom)==len(rows)
 start=time.monotonic();initial=len(rows);lo,hi=m.jnt_range[:6].T;site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'arm_tcp_site');train_count=sum(r['split']=='train' for r in rows);new_id=max(r['id'] for r in rows)+1
 def status(ready=False):
  elapsed=time.monotonic()-start;added=len(rows)-initial;rate=added/elapsed if added else 0
  out=dict(ready=ready,target_TRAIN=16000,accepted=len(rows),accepted_train=sum(r['split']=='train' for r in rows),accepted_dev=sum(r['split']=='dev' for r in rows),counts=counts,unique_physical_duration_s=sum(r['duration_s'] for r in rows),resumed_initial=initial,new_since_resume=added,session_elapsed_s=elapsed,measured_new_trajectories_per_second=rate,estimated_remaining_s=(16000-sum(r['split']=='train' for r in rows))/rate if rate else None,throughput_scope='this actual process session, includes rejections and persistence; ETA is extrapolation not promise',updated=time.time(),pid=os.getpid(),model_sha256=sha(M/'runtime.xml'),model_commit=meta['commit'],seed=91114,window_duration_s=6.,pipeline='RoboPlan0.8 RRT -> native PathShortcutter -> native TOPPRA Adaptive',no_hidden_repeat_hold_or_padding=True,endpoint_and_geometric_uniqueness=True,initial_bank_manifest_sha256=sha(O/'MANIFEST.json'),generator_source_sha256=sha(__file__))
  atomic(B/'STATUS.json',out)
  if ready:
   out.update(rows=rows,train_ids=[r['id'] for r in rows if r['split']=='train'],dev_ids=[r['id'] for r in rows if r['split']=='dev'],sampling='shuffled TRAIN ID cycles without replacement, uniform valid30Hz random6s start; WM causal replay unchanged');atomic(B/'MANIFEST.json',out)
  return out
 status()
 while train_count<16000:
  assert shutil.disk_usage(B).free>22*2**30,'disk floor';rng=np.random.default_rng(91114+attempt);delta={k:0 for k in counts};reason=None;planner.setRngSeed(91114+attempt)
  def sample():
   for _ in range(1000):
    delta['endpoint_draws']+=1;q=rng.uniform(lo+.015,hi-.015)
    if scene.hasCollisions(q) or contacts(m,d,q):delta['endpoint_collision']+=1;continue
    return q
   raise RuntimeError('valid endpoint sampler exhausted')
  q0,q1=sample(),sample();identity=key(q0,q1);x,y=JointConfiguration(),JointConfiguration();x.positions=q0;y.positions=q1
  if identity in endpoints:reason='duplicate'
  if reason is None:
   try:path=shortcut.shortcut(planner.plan(x,y))
   except RuntimeError:reason='planner_failure'
  if reason is None:
   path_sha=hashlib.sha256(np.asarray(path.positions,dtype='<f8').tobytes()).hexdigest()
   if path_sha in geom:reason='duplicate'
  vs=float(rng.uniform(.12,.45));ac=float(rng.uniform(.12,.5))
  if reason is None:
   try:tr=timer.generate(path,TOPPRAOptions(dt=1/60,mode=SplineFittingMode.Adaptive,velocity_scale=vs,acceleration_scale=ac))
   except RuntimeError:reason='timer_failure'
  if reason is None and tr.times[-1]<6.1:reason='too_short'
  if reason is None:
   t,q,dq,ddq=map(np.array,[tr.times,tr.positions,tr.velocities,tr.accelerations]);invalid=validate(t,q,dq,ddq,lo,hi)
   if invalid:
    reject=B/'rejected';reject.mkdir(exist_ok=True);np.savez_compressed(reject/f'candidate_{attempt:07d}.npz',time_s=t,q=q,dq=dq,ddq=ddq,q_start=q0,q_goal=q1,lo=lo,hi=hi,geometric_path=np.array(path.positions));atomic(reject/f'candidate_{attempt:07d}.json',dict(attempt=attempt,violations=invalid,velocity_scale=vs,acceleration_scale=ac));reason='numerical_invalid';print('NUMERICAL_REJECT',attempt,invalid,flush=True)
  if reason is None:
   ps=[];qs=[]
   for qt in q:
    if scene.hasCollisions(qt) or contacts(m,d,qt):reason='trajectory_collision';break
    ps.append(d.site_xpos[site].copy());qq=np.zeros(4);mj.mju_mat2Quat(qq,d.site_xmat[site]);qs.append(qq)
  if reason is None:
   f=B/f'train_{new_id:05d}.npz';assert not f.exists(),('uncommitted output preserved, inspect before resume',str(f));tmp=f.with_suffix('.writing.npz');np.savez_compressed(tmp,time_s=t,q=q,dq=dq,ddq=ddq,tcp_pos=ps,tcp_quat=qs,geometric_path=np.array(path.positions));tmp.replace(f);fd=float(np.max(abs(np.diff(dq,axis=0)/np.diff(t)[:,None])))
   row=dict(id=new_id,split='train',file=f.name,sha256=sha(f),duration_s=float(t[-1]),valid_6s_start_ticks=int(np.floor((t[-1]-6)*30))+1,velocity_scale=vs,acceleration_scale=ac,fd_acceleration_max=fd,acceleration_warning=fd>1.79*ac*1.01,reference_samples=len(t),q_start=q0.tolist(),q_goal=q1.tolist(),endpoint_identity=identity,geometric_path_sha256=path_sha,candidate_attempt=attempt)
   with ledger.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   rows.append(row);endpoints.add(identity);geom.add(path_sha);train_count+=1;new_id+=1
  else:delta[reason]+=1
  for k,v in delta.items():counts[k]+=v
  with events.open('a') as f:f.write(json.dumps(dict(attempt=attempt,reason=reason,delta=delta,time=time.time()))+'\n');f.flush()
  attempt+=1
  if len(rows)%25==0 or reason is not None:progress=status();print('PROGRESS',progress['accepted_train'],progress['accepted_dev'],progress['measured_new_trajectories_per_second'],flush=True)
 result=status(True);print('FULL_BANK_READY',result['accepted_train'],result['accepted_dev'],result['unique_physical_duration_s'],flush=True)
if __name__=='__main__':main()
