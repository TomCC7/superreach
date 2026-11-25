"""Actual policy before any optimizer: t0, 480Hz first response, native six seconds."""
from state import *
import fcntl

def evaluate(e,actor,path):
 path=Path(path);path.mkdir(parents=True,exist_ok=False)
 ids=torch.tensor(np.resize(e.bank.dev_ids,e.num_envs),device=e.device);starts=(e.bank.lengths[ids]-181)//2
 assert e.num_envs>=32
 e.reset(e.ids,ids,starts);actor.eval()
 initial=e.bank.lookup(ids,starts)
 assert torch.equal(e.d.qpos._tensor[:,:6],initial[:,:6]);assert torch.equal(e.d.qvel._tensor[:,:6],initial[:,6:12]);assert torch.equal(e.sent,initial[:,:6]);assert torch.equal(e.d.ctrl._tensor,initial[:,:6])
 assert (e.valid.sum(-1)==1).all() and (e.history[:,:-1,18]==0).all()
 reset={k:cpu(getattr(e.d,k)[:32]) for k in SFIELDS}
 reset.update(history=cpu(e.history[:32]),valid=cpu(e.valid[:32]),sent=cpu(e.sent[:32]),ref_id=cpu(ids[:32]),start_tick=cpu(starts[:32]))
 np.savez_compressed(path/'t0_reset.npz',**{k:np.asarray(v) for k,v in reset.items()})
 rows=[];q=[cpu(e.d.qpos[:32])];goals=[cpu(e.goal[:32])];positions=[];angles=[]
 failed=torch.zeros(32,dtype=torch.bool,device=e.device);first_failure=torch.full((32,),181,device=e.device)
 with torch.no_grad():
  for tick in range(180):
   action=actor(e.obs());_,r,done,last=e.step(action,auto_reset=False)
   first_failure=torch.where(last['departure'][:32] & ~failed,tick+1,first_failure);failed|=last['departure'][:32]
   rows.append({k:cpu(v[:32]) for k,v in last.items()});q.append(cpu(e.d.qpos[:32]));goals.append(cpu(last['goal'][:32]));positions.append(cpu(last['pos'][:32]));angles.append(cpu(last['angle'][:32]))
 arrays={k:np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}
 arrays.update(full_q=np.stack(q),frame_goal=np.stack(goals),time_s=np.arange(181)/30,first_failure_tick=cpu(first_failure).numpy())
 np.savez_compressed(path/'actual_policy.npz',**arrays)
 p=np.stack(positions);a=np.stack(angles);delta=arrays['sent'][0]-initial[:32,:6].cpu().numpy()
 result=dict(completed=True,kind='actual closed-loop fresh policy; stage and optimizer counts in companion receipt',cases=32,poststep_frames=5760,t0_frames=32,native_seconds=6.,reference_scored_at='poststep t+1; actor goal current t',no_autoreset=True,failed_case_count=int(failed.sum()),failed_case_ids=ids[:32][failed].cpu().tolist(),first_failure_tick=cpu(first_failure).tolist(),first_command_max_delta_deg=float(np.rad2deg(abs(delta)).max()),first_physics_max_delta_deg=float(np.rad2deg(abs(arrays['first_physics_q'][0,:,:6]-initial[:32,:6].cpu().numpy())).max()),first_control_position_max_mm=float(p[0].max()*1000),position_mean_mm=float(p.mean()*1000),position_p95_mm=float(np.quantile(p,.95)*1000),angle_mean_deg=float(np.rad2deg(a.mean())),angle_p95_deg=float(np.rad2deg(np.quantile(a,.95))),within_domain_frame_fraction=float(((p<=.05)&(a<=.3490658503988659)).mean()),strict_frame_count=int(((p<=.005)&(a<=np.deg2rad(.5))).sum()),strict_case_count=int(((p<=.005)&(a<=np.deg2rad(.5))).all(0).sum()),first_second_position_mean_mm=float(p[:30].mean()*1000),late_position_mean_mm=float(p[150:].mean()*1000),case_mean_mm=(p.mean(0)*1000).tolist(),not_claimed=['learned tracking','safety success','hardware fidelity','payload robustness'])
 atomic(path/'RESULT.json',result);print(json.dumps(result),flush=True);return result

if __name__=='__main__':
 lock=(ROOT.parent/'metal-history-gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);seed();e=OpenYAMEnv(32);actor=make_actor(e.device)
 evaluate(e,actor,OUT/'residual_initial_contract')
