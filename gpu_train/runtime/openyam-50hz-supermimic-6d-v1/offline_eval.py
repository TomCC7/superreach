"""Offline native MuJoCo validation; no training state/replay mutation."""
import copy,json
from pathlib import Path
import numpy as np,torch,mujoco as mj
HOME=np.array([0.,1.047,1.047,0.,0.,0.])

def evaluate(actor,bank,E,path,ticks=300):
    policy=copy.deepcopy(actor).cpu().eval()
    qbank=bank.table.detach().cpu().numpy();off=bank.offsets.cpu().numpy()
    dev=np.asarray(bank.dev_ids)
    # Held-out reachable far pairs, not training goal templates.
    dev_indices=np.concatenate([off[rid]+np.arange(0,int(bank.lengths[rid])-301,25) for rid in dev])
    candidates=qbank[dev_indices,:]
    cases=[]
    for mass in [0.,.5,1.,2.]:cases.append(dict(kind='HOME',mass=mass,q=HOME.copy()))
    for mass in [0.,1.,2.]:
        cases.append(dict(kind='TRACK',mass=mass,ref=candidates[0],rid=int(dev[0])))
        for lo,hi in [(.3,.45),(.45,.6)]:
            origin=candidates[0]
            dist=np.linalg.norm(candidates[:,12:15]-origin[12:15],axis=1)
            ids=np.flatnonzero((dist>=lo)&(dist<=hi));assert len(ids),(lo,hi)
            cases.append(dict(kind='STEP',mass=mass,q=origin[:6],target=candidates[ids[0]],distance=float(dist[ids[0]])))
    variants=E.continuous.load_catalog();models=[];datas=[];refs=[];hist=[];records=[]
    for case in cases:
        m=mj.MjModel.from_xml_path(E.o.os.environ['SUPERREACH_SCENE'])
        E.old.configure_model(m);m.opt.timestep=.002
        m.actuator_ctrllimited[:]=False
        m.actuator_forcelimited[:]=True
        m.actuator_forcerange[:]=np.array([[-x,x] for x in E.o.PEAK])
        mass_index=int(np.argmin(np.abs(variants['masses']-case['mass'])))
        for field in ['body_mass','body_ipos','body_inertia','body_iquat']:getattr(m,field)[:]=variants[field][mass_index]
        d=mj.MjData(m);mj.mj_setConst(m,d)
        initial=case['ref'][:6] if case['kind']=='TRACK' else case['q']
        d.qpos[:6]=initial;d.qpos[6:]=.055;mj.mj_forward(m,d)
        sid=m.site('arm_tcp_site').id
        def pose():return np.r_[d.site_xpos[sid],matrix_quat(d.site_xmat[sid])]
        if case['kind']=='HOME':ref=np.r_[initial,np.zeros(6),pose()]
        elif case['kind']=='STEP':
            ref=case['target'].copy();ref[6:12]=0;case['target']=ref
        else:ref=qbank[off[case['rid']]].copy()
        if case['kind']=='STEP':case['hold_ref']=np.r_[initial,np.zeros(6),pose()]
        frame=np.r_[d.qpos[:6],d.qvel[:6],initial,1.,0.]
        h=np.tile(frame,(16,1));h[:-1,18]=0
        models.append(m);datas.append(d);refs.append(ref);hist.append(h)
        records.append(dict(position=[],velocity=[],saturation=[],overshoot=[],failure=False,initial_xyz=d.site_xpos[sid].copy(),target=ref[12:15].copy()))
    history=np.array(hist)
    with torch.no_grad():
        for tick in range(ticks):
            goals=[]
            for i,c in enumerate(cases):
                if c['kind']=='TRACK':refs[i]=qbank[off[c['rid']]+tick]
                elif c['kind']=='STEP':refs[i]=c['hold_ref'] if tick<75 else c['target']
                goals.append(np.r_[refs[i][12:19],refs[i][:12]])
            obs={'history':torch.tensor(history,dtype=torch.float32),'goal_motion':torch.tensor(np.array(goals),dtype=torch.float32)}
            command=policy(obs).cpu().numpy()
            assert command.shape==(len(cases),6) and np.isfinite(command).all()
            for i,(m,d,c,r) in enumerate(zip(models,datas,cases,records)):
                d.ctrl[:]=command[i]
                saturation=[]
                for _ in range(10):
                    mj.mj_step(m,d);saturation.append(np.abs(d.actuator_force)>=np.array(E.o.PEAK)-1e-4)
                mj.mj_forward(m,d);sid=m.site('arm_tcp_site').id
                if c['kind']=='TRACK':refs[i]=qbank[off[c['rid']]+tick+1]
                p=d.site_xpos[sid];r['position'].append(float(np.linalg.norm(p-refs[i][12:15])));r['velocity'].append(d.qvel[:6].copy());r['saturation'].append(np.mean(saturation))
                direction=r['target']-r['initial_xyz'];direction=direction/max(np.linalg.norm(direction),1e-12)
                r['overshoot'].append(float(max(0,np.dot(p-r['target'],direction))))
                r['failure']|=bool(not np.isfinite(d.qpos).all() or np.max(np.abs(d.qvel[:6]))>100)
                frame=np.r_[d.qpos[:6],d.qvel[:6],d.ctrl[:],1.,(tick+1)*.02]
                history[i]=np.concatenate((history[i,1:],frame[None]),0)
    results=[]
    for c,r in zip(cases,records):
        results.append(dict(kind=c['kind'],mass_kg=c['mass'],jump_m=c.get('distance'),ticks=ticks,position_error_mean_m=float(np.mean(r['position'])),position_error_final_m=r['position'][-1],position_error_tail_rms_m=float(np.sqrt(np.mean(np.square(r['position'][-min(50,ticks):])))),overshoot_m=max(r['overshoot']) if c['kind']=='STEP' else None,tail_velocity_rms_rad_s=float(np.sqrt(np.mean(np.square(r['velocity'][-min(50,ticks):])))),peak_torque_saturation=float(np.mean(r['saturation'])),physical_failure=r['failure']))
    payload=dict(completed=True,offline_native_MuJoCo=True,physics_dt=.002,control_dt=.02,substeps=10,no_training_mutation=True,HOME=HOME.tolist(),cases=results)
    E.base.atomic(Path(path),payload);return payload

def matrix_quat(matrix):
    out=np.empty(4);mj.mju_mat2Quat(out,np.asarray(matrix).reshape(9));return out
