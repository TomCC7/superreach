"""Four independent CPU planning processes; single deterministic ordered writer."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ.setdefault('MUJOCO_GL','egl')
from bulk16k import B,key,atomic
from generate_bank import *
from native_validation import validate
import concurrent.futures as cf
import multiprocessing as mp
import fcntl,shutil,signal
W={}
def worker_init():
 signal.signal(signal.SIGINT,signal.SIG_IGN)
 m=mj.MjModel.from_xml_path(str(M/'runtime.xml'));d=mj.MjData(m);meta=json.loads((M/'MODEL_VALIDATION.json').read_text());s=Scene('openyam_pool_'+str(os.getpid()),loadMjcfModel(M/'planner.xml'));s.setCollisions([tuple(x) for x in meta['disabled_body_pairs']],False);s.importJointLimitsFromConfig(loadJointLimitsConfig(M/'limits.yaml'))
 W.update(m=m,d=d,s=s,site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'arm_tcp_site'))
def candidate(attempt):
 m,d,s=W['m'],W['d'],W['s'];lo,hi=m.jnt_range[:6].T;seed=91114+attempt;rng=np.random.default_rng(seed);s.setRngSeed(seed);planner=RRT(s,RRTOptions(max_nodes=4000,max_connection_distance=.4,collision_check_step_size=.008,max_planning_time=5.,rrt_connect=True));planner.setRngSeed(seed);shortcut=PathShortcutter(s,PathShortcuttingOptions(max_step_size=.008,max_iters=100,seed=seed));timer=PathParameterizerTOPPRA(s)
 delta=dict(endpoint_draws=0,endpoint_collision=0,planner_failure=0,timer_failure=0,too_short=0,trajectory_collision=0,duplicate=0,numerical_invalid=0)
 def result(reason,**extra):
  if reason:delta[reason]+=1
  return dict(attempt=attempt,reason=reason,delta=delta,worker_pid=os.getpid(),**extra)
 def sample():
  for _ in range(1000):
   delta['endpoint_draws']+=1;q=rng.uniform(lo+.015,hi-.015)
   if s.hasCollisions(q) or contacts(m,d,q):delta['endpoint_collision']+=1;continue
   return q
  raise RuntimeError('endpoint sampler exhausted')
 q0,q1=sample(),sample();x,y=JointConfiguration(),JointConfiguration();x.positions=q0;y.positions=q1
 try:path=shortcut.shortcut(planner.plan(x,y))
 except RuntimeError:return result('planner_failure')
 vs=float(rng.uniform(.12,.45));ac=float(rng.uniform(.12,.5))
 try:tr=timer.generate(path,TOPPRAOptions(dt=1/60,mode=SplineFittingMode.Adaptive,velocity_scale=vs,acceleration_scale=ac))
 except RuntimeError:return result('timer_failure')
 if tr.times[-1]<6.1:return result('too_short')
 t,q,dq,ddq=map(np.array,[tr.times,tr.positions,tr.velocities,tr.accelerations]);invalid=validate(t,q,dq,ddq,lo,hi)
 if invalid:
  rej=B/'rejected';rej.mkdir(exist_ok=True);prefix=f'pool_candidate_{attempt:07d}';np.savez_compressed(rej/(prefix+'.npz'),time_s=t,q=q,dq=dq,ddq=ddq,lo=lo,hi=hi,q_start=q0,q_goal=q1);atomic(rej/(prefix+'.json'),dict(attempt=attempt,violations=invalid,worker_pid=os.getpid()));return result('numerical_invalid')
 ps=[];qs=[]
 for qt in q:
  if s.hasCollisions(qt) or contacts(m,d,qt):return result('trajectory_collision')
  ps.append(d.site_xpos[W['site']].copy());qq=np.zeros(4);mj.mju_mat2Quat(qq,d.site_xmat[W['site']]);qs.append(qq)
 shard=B/'shards'/f'candidate_{attempt:07d}_pid{os.getpid()}.npz';np.savez_compressed(shard,time_s=t,q=q,dq=dq,ddq=ddq,tcp_pos=ps,tcp_quat=qs,geometric_path=np.array(path.positions))
 fd=float(np.max(abs(np.diff(dq,axis=0)/np.diff(t)[:,None])))
 row=dict(duration_s=float(t[-1]),valid_6s_start_ticks=int(np.floor((t[-1]-6)*30))+1,velocity_scale=vs,acceleration_scale=ac,fd_acceleration_max=fd,acceleration_warning=fd>1.79*ac*1.01,reference_samples=len(t),q_start=q0.tolist(),q_goal=q1.tolist(),endpoint_identity=key(q0,q1),geometric_path_sha256=hashlib.sha256(np.asarray(path.positions,dtype='<f8').tobytes()).hexdigest(),candidate_attempt=attempt,worker_pid=os.getpid())
 return result(None,row=row,shard=str(shard))
def main():
 lock=(B/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (B/'MANIFEST.json').exists();(B/'shards').mkdir(exist_ok=True)
 ledger=B/'accepted.jsonl';events=B/'candidates.jsonl';rows=[json.loads(l) for l in ledger.read_text().splitlines()];evs=[json.loads(l) for l in events.read_text().splitlines()];next_attempt=max([e['attempt'] for e in evs]+[r.get('candidate_attempt',-1) for r in rows])+1;counts=dict(endpoint_draws=0,endpoint_collision=0,planner_failure=0,timer_failure=0,too_short=0,trajectory_collision=0,duplicate=0,numerical_invalid=0)
 for e in evs:
  for k,v in e['delta'].items():counts[k]+=v
 endpoints={r['endpoint_identity'] for r in rows};geom={r['geometric_path_sha256'] for r in rows};assert len(endpoints)==len(rows) and len(geom)==len(rows);initial=len(rows);initial_train=sum(r['split']=='train' for r in rows);train_count=initial_train;new_id=max(r['id'] for r in rows)+1;start=time.monotonic();worker_pids=set();duration=sum(r['duration_s'] for r in rows);stopping=[False]
 signal.signal(signal.SIGINT,lambda *x:stopping.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *x:stopping.__setitem__(0,True))
 def status(ready=False):
  elapsed=time.monotonic()-start;added=len(rows)-initial;rate=added/elapsed if elapsed else 0
  out=dict(ready=ready,target_TRAIN=16000,accepted=len(rows),accepted_train=train_count,accepted_dev=sum(r['split']=='dev' for r in rows),counts=counts,counts_scope='expansion candidates; original160 ledger in linked initial manifest',unique_physical_duration_s=duration,resumed_initial=initial,new_since_resume=added,session_elapsed_s=elapsed,measured_new_trajectories_per_second=rate,estimated_remaining_s=(16000-train_count)/rate if rate else None,throughput_scope='this4worker process session, rejections/disk included; extrapolation only',workers=4,worker_pids=sorted(worker_pids),updated=time.time(),pid=os.getpid(),model_sha256=sha(M/'runtime.xml'),model_commit='f9853344bbc9fa00aade6ff3ffb7db6532d185e1',seed='91114+unique candidate attempt; independent per-worker scene',window_duration_s=6.,pipeline='native RoboPlan RRT -> PathShortcutter -> TOPPRA Adaptive',no_hidden_repeat_hold_or_padding=True,endpoint_and_geometric_uniqueness=True,initial_bank_manifest_sha256=sha(O/'MANIFEST.json'),generator_source_sha256=sha(__file__),merge='ordered candidate attempt, parent-only deterministic accepted IDs and atomic manifest')
  atomic(B/'STATUS.json',out)
  if ready:out.update(rows=rows,train_ids=[r['id'] for r in rows if r['split']=='train'],dev_ids=[r['id'] for r in rows if r['split']=='dev'],sampling='shuffledTRAIN ID cycles without replacement, uniform valid30Hz random6s starts, official causal replay unchanged');atomic(B/'MANIFEST.json',out)
  return out
 status();ctx=mp.get_context('spawn')
 with cf.ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=worker_init) as pool:
  pending={i:pool.submit(candidate,i) for i in range(next_attempt,next_attempt+4)};submit_id=next_attempt+4;merge_id=next_attempt
  while train_count<16000 and not stopping[0]:
   assert shutil.disk_usage(B).free>22*2**30,'disk reserve';res=pending.pop(merge_id).result(timeout=120);worker_pids.add(res['worker_pid']);reason=res['reason'];row=res.get('row')
   if row and (row['endpoint_identity'] in endpoints or row['geometric_path_sha256'] in geom):reason='duplicate';res['reason']=reason;res['delta']['duplicate']+=1
   if reason is None:
    f=B/f'train_{new_id:05d}.npz';assert not f.exists(),('uncommitted immutable file, preserve and inspect',str(f));Path(res['shard']).replace(f);row.update(id=new_id,split='train',file=f.name,sha256=sha(f))
    with ledger.open('a') as out:out.write(json.dumps(row)+'\n');out.flush();os.fsync(out.fileno())
    rows.append(row);duration+=row['duration_s'];endpoints.add(row['endpoint_identity']);geom.add(row['geometric_path_sha256']);train_count+=1;new_id+=1
   for k,v in res['delta'].items():counts[k]+=v
   event={k:v for k,v in res.items() if k not in ['row','shard']};event['time']=time.time()
   with events.open('a') as out:out.write(json.dumps(event)+'\n');out.flush()
   merge_id+=1
   if train_count<16000 and not stopping[0]:pending[submit_id]=pool.submit(candidate,submit_id);submit_id+=1
   if len(rows)%25==0 or reason:out=status();print('POOL_PROGRESS',train_count,out['workers'],out['worker_pids'],out['measured_new_trajectories_per_second'],flush=True)
  # At most3 completed/inflight shards remain unadmitted and preserved; they never inflate counts.
 status(train_count>=16000);print('POOL_EXIT',train_count,'ready',train_count>=16000,flush=True)
if __name__=='__main__':main()
