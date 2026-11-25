"""Pinned OpenYAM adapter: preserve articulated hand, hold fingers by equalities."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco as mj
from roboplan.core import Scene,loadMjcfModel
R=Path(os.environ.get('SUPERREACH_GEN_DIR', 'data/generated')).resolve()
S=R/'openyam-description/openyam_single'
O=R/'output/openyam-history-window-v1/model'
OPENING=.055
NAMES=[f'arm_joint_{i}' for i in range(1,7)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contacts(m,d,q):
 d.qpos[:6]=q
 if m.nq==8:d.qpos[6:]=OPENING
 d.qvel[:]=0;mj.mj_forward(m,d)
 return [(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom1])),mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom2])),float(c.dist)) for c in d.contact if c.dist<0]
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'MODEL_VALIDATION.json').exists(),'completed model is immutable'
 tree=ET.parse(S/'openyam_single.mjcf.xml');root=tree.getroot();root.find('compiler').set('meshdir',str(S/'meshes'))
 ET.SubElement(root,'option',timestep=str(1/60),gravity='0 0 -9.81',integrator='implicitfast')
 tcp=root.find('.//body[@name="arm_tcp"]');ET.SubElement(tcp,'site',name='arm_tcp_site',size='.006',rgba='1 0 0 1')
 eq=root.find('equality');ET.SubElement(eq,'joint',name='fixed_registered_gripper_opening',joint1='hand_finger_joint1',polycoef=f'{OPENING} 0 0 0 0')
 for equality in eq.findall('joint'):equality.set('solimp','.9999 .9999 .001 .5 2')
 act=ET.SubElement(root,'actuator')
 # Engineering simulation-only gains and conservative effort limits, not hardware calibration.
 kp=[50,60,50,25,15,10];kd=[3,4,3,2,1,.8];eff=[12,12,10,6,4,3]
 for i,n in enumerate(NAMES):
  j=root.find(f'.//joint[@name="{n}"]');j.set('damping','.05');j.set('armature','.005')
  ET.SubElement(act,'position',name=n+'_sim_pd',joint=n,kp=str(kp[i]),kv=str(kd[i]),forcerange=f'{-eff[i]} {eff[i]}',ctrlrange=j.get('range'))
 ct=ET.SubElement(root,'contact');ET.SubElement(ct,'exclude',body1='base',body2='arm_link_1')
 runtime=O/'runtime.xml';tree.write(runtime)
 m=mj.MjModel.from_xml_path(str(runtime));d=mj.MjData(m)
 orig=mj.MjModel.from_xml_path(str(S/'openyam_single.mjcf.xml'))
 for name in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_axis','jnt_range']:
  assert np.array_equal(getattr(m,name),getattr(orig,name)),name
 assert m.nq==8 and m.nu==6 and m.neq==2
 assert [mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i) for i in range(6)]==NAMES
 # Planner-only reduction holds the same two slider positions by fixed transforms; never deletes bodies/inertia.
 pt=ET.parse(runtime);pr=pt.getroot();pr.remove(pr.find('equality'));pr.remove(pr.find('actuator'))
 for b in pr.iter('body'):
  for j in list(b.findall('joint')):
   if j.get('type')=='slide':
    axis=np.fromstring(j.get('axis'),sep=' ');axis/=np.linalg.norm(axis)
    p=np.fromstring(b.get('pos','0 0 0'),sep=' ')+axis*OPENING;b.set('pos',' '.join(map(str,p)));b.remove(j)
 wb=pr.find('worldbody');base=wb.find('body');wb.remove(base);anchor=ET.SubElement(wb,'body',name='fixed_world_anchor');anchor.append(base)
 plannerfile=O/'planner.xml';pt.write(plannerfile);pm=mj.MjModel.from_xml_path(str(plannerfile));pd=mj.MjData(pm)
 scene=Scene('openyam',loadMjcfModel(plannerfile));assert scene.getJointNames()==NAMES
 disabled=[]
 for a in range(1,pm.nbody):
  for b in range(a+1,pm.nbody):
   wa,ww=int(pm.body_weldid[a]),int(pm.body_weldid[b]);x,y=sorted((a,b))
   if wa==ww or wa==pm.body_weldid[pm.body_parentid[ww]] or ww==pm.body_weldid[pm.body_parentid[wa]] or ((x<<16)+y in pm.exclude_signature):
    na,nb=mj.mj_id2name(pm,mj.mjtObj.mjOBJ_BODY,a),mj.mj_id2name(pm,mj.mjtObj.mjOBJ_BODY,b)
    if na and nb:disabled.append((na,nb))
 scene.setCollisions(disabled,False)
 u=mj.MjModel.from_xml_path(str(S/'openyam_single.urdf'));ud=mj.MjData(u);ub=mj.mj_name2id(u,mj.mjtObj.mjOBJ_BODY,'arm_tcp');mb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'arm_tcp')
 rng=np.random.default_rng(81113);lo,hi=m.jnt_range[:6].T;maxp=maxr=maxaxis=0.;hist=Counter();safe=0;negative=None
 for i in range(128):
  q=rng.uniform(lo+.015,hi-.015);cs=contacts(m,d,q)
  ud.qpos[:6]=q;ud.qpos[6:]=OPENING;mj.mj_forward(u,ud)
  maxp=max(maxp,float(np.max(abs(ud.xpos[ub]-d.xpos[mb]))));maxr=max(maxr,float(np.max(abs(ud.xmat[ub]-d.xmat[mb]))))
  T=scene.forwardKinematics(q,'arm_tcp');maxp=max(maxp,float(np.max(abs(T[:3,3]-d.xpos[mb]))));maxr=max(maxr,float(np.max(abs(T[:3,:3]-d.xmat[mb].reshape(3,3)))))
  maxaxis=max(maxaxis,float(np.max(abs(u.jnt_axis[:6]-m.jnt_axis[:6]))))
  for a,b,dist in cs:hist[(a,b)]+=1
  collide=scene.hasCollisions(q)
  if not cs and not collide:safe+=1
  if cs and collide and negative is None:negative=dict(q=q.tolist(),contacts=cs,roboplan_collision=True)
 assert maxp<1e-6 and maxr<1e-6 and maxaxis<1e-10,(maxp,maxr,maxaxis)
 assert safe>0 and negative is not None,(safe,hist)
 # Equality dynamics at registered opening: no artificial mass removal.
 d.qpos[:6]=(lo+hi)/2;d.qpos[6:]=OPENING;d.ctrl[:]=d.qpos[:6];d.qvel[:]=0
 for _ in range(60):mj.mj_step(m,d)
 finger_error=float(np.max(abs(d.qpos[6:]-OPENING)))
 assert finger_error<1e-4,finger_error
 limits='joint_limits:\n'+''.join(f'  {n}:\n    max_velocity: [1.19]\n    max_acceleration: [1.79]\n' for n in NAMES);(O/'limits.yaml').write_text(limits)
 out=dict(commit='f9853344bbc9fa00aade6ff3ffb7db6532d185e1',tcp='arm_tcp',joint_order=NAMES,opening_each_slider_m=OPENING,mimic='q_finger2=q_finger1; q_finger1=.055m equality; actor never controls gripper',finger_dynamics_max_error_m=finger_error,fk_URDF_MJCF_RoboPlan_max_m=maxp,fk_rotation_max=maxr,axis_max=maxaxis,random_test_count=128,accepted_safe=safe,collision_pair_counts=[dict(bodies=k,count=v) for k,v in hist.items()],disabled_body_pairs=disabled,negative=negative,actuator=dict(kp=kp,kd=kd,effort_Nm=eff,provenance='ENGINEERING SIMULATION ASSUMPTIONS; URDF effort30 and velocity3 are source limits only, not hardware-validated; no vendor gains available'),inertials='source ESTIMATED, unchanged',gravity=[0,0,-9.81],runtime_sha256=sha(runtime),planner_sha256=sha(plannerfile),source_xml_sha256=sha(S/'openyam_single.mjcf.xml'),source_urdf_sha256=sha(S/'openyam_single.urdf'))
 (O/'MODEL_VALIDATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
