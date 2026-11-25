"""Random valid joints -> native RoboPlan RRT/shortcut/TOPPRA, OpenYAM only."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,subprocess,time
from pathlib import Path
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
from roboplan.core import Scene,loadMjcfModel,loadJointLimitsConfig,JointConfiguration,PathShortcutter,PathShortcuttingOptions
from roboplan.rrt import RRT,RRTOptions
from roboplan.toppra import PathParameterizerTOPPRA,TOPPRAOptions,SplineFittingMode
from prepare_model import contacts,OPENING
R=Path(os.environ.get('SUPERREACH_GEN_DIR', 'data/generated')).resolve();M=R/'output/openyam-history-window-v1/model';O=R/'output/openyam-history-window-v1/bank'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(m,z,out,label):
 m.vis.global_.offwidth=960;m.vis.global_.offheight=720;d=mj.MjData(m);ren=mj.Renderer(m,720,960);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.lookat[:]=[0,0,.15];cam.distance=1.6;cam.azimuth=135;cam.elevation=-20
 font=ImageFont.truetype('/usr/share/fonts/noto/NotoSans-Regular.ttf',16)
 q=z['q'];t=z['time_s'];frames=int(np.floor(t[-1]*30))+1
 proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-','-an','-c:v','libvpx-vp9','-threads','4','-crf','33','-b:v','0',str(out)],stdin=subprocess.PIPE)
 for i in range(frames):
  ix=min(int(np.searchsorted(t,i/30)),len(t)-1);d.qpos[:6]=q[ix];d.qpos[6:]=OPENING;mj.mj_forward(m,d);ren.update_scene(d,cam);im=Image.fromarray(ren.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,960,67),fill='black');dr.text((12,5),'OpenYAM | PLANNED REFERENCE - NO POLICY - NATIVE 1x',font=font,fill='yellow');dr.text((12,26),f'{label} | t={i/30:.3f}s | RRT -> PathShortcutter -> TOPPRA',font=font,fill='white');dr.text((12,47),'Source inertials ESTIMATED; kinematic playback, not hardware / PD proof',font=font,fill='white');proc.stdin.write(np.asarray(im).tobytes())
  if i in {0,frames//2,frames-1}:im.save(out.with_name(out.stem+f'_{i:05d}.jpg'))
 proc.stdin.close();assert proc.wait(timeout=120)==0;ren.close();subprocess.run(['ffmpeg','-v','error','-i',str(out),'-f','null','-'],check=True,timeout=120)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-show_entries','format=duration','-of','json',str(out)]));assert int(probe['streams'][0]['nb_read_frames'])==frames
 return dict(file=out.name,frames=frames,fps=30,native1x=True,sha256=sha(out),full_decode=True,probe=probe)
def main():
 O.mkdir(parents=True,exist_ok=False);meta=json.loads((M/'MODEL_VALIDATION.json').read_text());scene=Scene('openyam_bank',loadMjcfModel(M/'planner.xml'));scene.setCollisions([tuple(p) for p in meta['disabled_body_pairs']],False);scene.importJointLimitsFromConfig(loadJointLimitsConfig(M/'limits.yaml'));scene.setRngSeed(81114)
 m=mj.MjModel.from_xml_path(str(M/'runtime.xml'));d=mj.MjData(m);site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'arm_tcp_site');rng=np.random.default_rng(81114);lo,hi=m.jnt_range[:6].T
 planner=RRT(scene,RRTOptions(max_nodes=4000,max_connection_distance=.4,collision_check_step_size=.008,max_planning_time=5.,rrt_connect=True));planner.setRngSeed(81114);shortcut=PathShortcutter(scene,PathShortcuttingOptions(max_step_size=.008,max_iters=100,seed=81114));timer=PathParameterizerTOPPRA(scene);rows=[];media=[];counts=dict(endpoint_draws=0,endpoint_collision=0,planner_failure=0,timer_failure=0,too_short=0,trajectory_collision=0)
 def sample():
  for _ in range(1000):
   counts['endpoint_draws']+=1;q=rng.uniform(lo+.015,hi-.015)
   if scene.hasCollisions(q) or contacts(m,d,q):counts['endpoint_collision']+=1;continue
   return q
  raise RuntimeError('no valid q')
 for attempt in range(1000):
  q0,q1=sample(),sample();x,y=JointConfiguration(),JointConfiguration();x.positions=q0;y.positions=q1
  try:path=shortcut.shortcut(planner.plan(x,y))
  except RuntimeError as e:counts['planner_failure']+=1;print('PLAN_REJECT',str(e),flush=True);continue
  vs=float(rng.uniform(.12,.45));ac=float(rng.uniform(.12,.5))
  try:tr=timer.generate(path,TOPPRAOptions(dt=1/60,mode=SplineFittingMode.Adaptive,velocity_scale=vs,acceleration_scale=ac))
  except RuntimeError as e:counts['timer_failure']+=1;print('TIME_REJECT',str(e),flush=True);continue
  t,q,dq,ddq=map(np.array,[tr.times,tr.positions,tr.velocities,tr.accelerations])
  if t[-1]<6.1:counts['too_short']+=1;continue
  if not all(np.isfinite(a).all() for a in [t,q,dq,ddq]) or np.any(np.diff(t)<=0) or np.any(q<lo-1e-6) or np.any(q>hi+1e-6):raise RuntimeError('invalid native numerical output')
  ps=[];qs=[];bad=False
  for qt in q:
   if scene.hasCollisions(qt) or contacts(m,d,qt):bad=True;break
   ps.append(d.site_xpos[site].copy());qq=np.zeros(4);mj.mju_mat2Quat(qq,d.site_xmat[site]);qs.append(qq)
  if bad:counts['trajectory_collision']+=1;continue
  i=len(rows);split='train' if i<128 else 'dev';f=O/f'{split}_{i:04d}.npz';a=dict(time_s=t,q=q,dq=dq,ddq=ddq,tcp_pos=np.array(ps),tcp_quat=np.array(qs),geometric_path=np.array(path.positions));np.savez_compressed(f,**a)
  fd=float(np.max(abs(np.diff(dq,axis=0)/np.diff(t)[:,None])))
  rows.append(dict(id=i,split=split,file=f.name,sha256=sha(f),duration_s=float(t[-1]),valid_6s_start_ticks=int(np.floor((t[-1]-6)*30))+1,velocity_scale=vs,acceleration_scale=ac,fd_acceleration_max=fd,acceleration_warning=fd>1.79*ac*1.01,reference_samples=len(t),q_start=q0.tolist(),q_goal=q1.tolist()))
  status=dict(ready=False,model_sha256=sha(M/'runtime.xml'),model_commit=meta['commit'],seed=81114,accepted=len(rows),counts=counts,unique_physical_duration_s=sum(r['duration_s'] for r in rows),rows=rows,train_ids=list(range(128)),dev_ids=list(range(128,160)),window_duration_s=6.,sampling='uniform TRAIN trajectory then uniform valid native30Hz window start; no holds/repeats/hidden retiming',pipeline='RoboPlan0.8 RRT -> PathShortcutter -> PathParameterizerTOPPRA Adaptive',script_sha256=sha(__file__))
  (O/'PROGRESS.json').write_text(json.dumps(status,indent=2));print('BANK_ACCEPT',len(rows),status['unique_physical_duration_s'],flush=True)
  if i<3:
   media.append(render(m,a,O/f'reference_{i:04d}_native1x.webm',f'path {i}, duration={t[-1]:.3f}s'));(O/'MEDIA.json').write_text(json.dumps(media,indent=2));print('FIRST_MEDIA_READY',media[-1],flush=True)
  if len(rows)==160:break
 assert len(rows)==160,(len(rows),counts)
 status['ready']=True;(O/'MANIFEST.json').write_text(json.dumps(status,indent=2));print('BANK_READY',status['unique_physical_duration_s'],counts,flush=True)
if __name__=='__main__':main()
