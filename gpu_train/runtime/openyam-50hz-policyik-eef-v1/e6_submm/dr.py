"""Episode-wise inertial and actuator DR, shared by native and Warp paths."""
import numpy as np
import torch
import mujoco as mj

N = 60
THETA = 71
PAYLOAD = ('body_mass','body_ipos','body_inertia','body_iquat','body_subtreemass','body_invweight0','dof_invweight0','actuator_acc0')
HARDWARE = ('actuator_gainprm','actuator_biasprm','dof_damping','dof_frictionloss','dof_armature')
SLICES = (slice(0,6),slice(6,24),slice(24,30),slice(30,36),slice(36,42),slice(42,48),slice(48,54),slice(54,60))
LOW = np.repeat([.85,-.01,.85,.8,.75,.01,0,.5],[6,18,6,6,6,6,6,6])
HIGH = np.repeat([1.15,.01,1.15,1.2,1.25,.2,.30,2.],[6,18,6,6,6,6,6,6])
PAY_CENTER = np.array([2.]+[0.]*9)
PAY_SCALE = np.array([2.]+[.3]*3+[.1]*6)


def ids(m):
    b = [m.body(f'arm_link_{i}').id for i in range(1,7)]
    j = [m.joint(f'arm_joint_{i}').id for i in range(1,7)]
    d = m.jnt_dofadr[j]
    assert np.array_equal(m.actuator_trnid[:6,0],j)
    return b,d


def spec(m):
    b,d = ids(m)
    base = {k: np.array(getattr(m,k),copy=True) for k in PAYLOAD + HARDWARE}
    assert np.all(base['body_mass'][b]>0) and np.all(base['body_inertia'][b]>0)
    assert np.all(base['dof_armature'][d]>0)
    nominal=np.r_[np.ones(6),np.zeros(18),np.ones(18),base['dof_damping'][d],base['dof_frictionloss'][d],np.ones(6)]
    assert nominal.size == N
    return base,b,d,nominal


def normalized(raw):
    return (raw-(LOW+HIGH)*.5)/((HIGH-LOW)*.5)


def payload_context(m):
    b=m.body('hand').id
    rot=np.empty(9); mj.mju_quat2Mat(rot,m.body_iquat[b]);rot=rot.reshape(3,3)
    I=(rot*m.body_inertia[b])@rot.T
    return (np.r_[m.body_mass[b],m.body_ipos[b],I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]-PAY_CENTER)/PAY_SCALE


def apply_native(m,d,base,raw):
    """Call after writing canonical payload inertial fields. Preserve dynamic state."""
    p,b,j,_=base
    mass,com,inertia,kp,kd,damping,friction,armature=(raw[s] for s in SLICES)
    m.body_mass[b]=p['body_mass'][b]*mass
    m.body_ipos[b]=p['body_ipos'][b]+com.reshape(6,3)
    m.body_inertia[b]=p['body_inertia'][b]*inertia[:,None]
    m.actuator_gainprm[:6,0]=p['actuator_gainprm'][:6,0]*kp
    m.actuator_biasprm[:6,1]=-m.actuator_gainprm[:6,0]
    m.actuator_biasprm[:6,2]=p['actuator_biasprm'][:6,2]*kd
    m.dof_damping[j]=damping; m.dof_frictionloss[j]=friction
    m.dof_armature[j]=p['dof_armature'][j]*armature
    # mj_setConst uses qpos0 and can overwrite workspace state; never use live data.
    mj.mj_setConst(m,mj.MjData(m))


def sample_payload(n,device,generator,count):
    rows=torch.randint(9,count,(n,),device=device,generator=generator)
    return torch.where(torch.rand(n,device=device,generator=generator)<.1,torch.zeros_like(rows),rows)


def install(e,seed):
    """Wrap the existing payload catalog at its pre-reset physical write boundary."""
    base,b,d,nominal=spec(e.model)
    cast=lambda v: torch.as_tensor(v,dtype=torch.float32,device=e.device)
    b=torch.as_tensor(b,device=e.device);d=torch.as_tensor(d,device=e.device)
    base={k:cast(v) for k,v in base.items()}
    lo,hi,nom=map(cast,(LOW,HIGH,nominal))
    e.hw_raw=nom.expand(e.num_envs,-1).clone()
    e.hw_rng=torch.Generator(device=e.device).manual_seed(seed)
    e.hw_nominal=torch.ones(e.num_envs,device=e.device,dtype=torch.bool)
    sim=e.sim
    cpu=torch.device(e.device).type=='cpu'
    if cpu:
        for k in HARDWARE:
            setattr(sim.model,k,base[k].expand(e.num_envs,*base[k].shape).clone())
        dirty=set()
        push=sim._push
        def native_push(i):
            push(i)
            m=sim.models[i]
            for k in HARDWARE:getattr(m,k)[:]=getattr(sim.model,k)[i].numpy()
            if i in dirty:
                mj.mj_setConst(m,mj.MjData(m))
                # The inherited facade pushes model bridge constants on every
                # step, so publish recomputed values back to the same row.
                for k in PAYLOAD[4:]:
                    getattr(sim.model,k)[i].copy_(cast(getattr(m,k)))
                e.meaninertia[i]=m.stat.meaninertia
                dirty.remove(i)
        sim._push=native_push
    else:
        sim.expand_model_fields(HARDWARE)
        import warp as wp
        import mujoco_warp as mw
    prior=e.apply_payload
    def apply(rows):
        if not len(rows):return
        gpu_stream = wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(e.device))) if not cpu else None
        if gpu_stream is not None:gpu_stream.__enter__()
        try:
            prior(rows)
            uniform=torch.rand((len(rows),N+1),device=e.device,generator=e.hw_rng)
            e.hw_nominal[rows]=uniform[:,0]<.1
            x=torch.where(e.hw_nominal[rows,None],nom,lo+(hi-lo)*uniform[:,1:])
            e.hw_raw[rows]=x
            mass,com,inertia,kp,kd,damping,friction,armature=(x[:,s] for s in SLICES)
            m=sim.model; bi=rows[:,None]; di=d[None,:]
            m.body_mass[bi,b]=base['body_mass'][b]*mass
            m.body_ipos[bi,b]=base['body_ipos'][b]+com.reshape(-1,6,3)
            m.body_inertia[bi,b]=base['body_inertia'][b]*inertia[...,None]
            m.actuator_gainprm[rows,:6,0]=base['actuator_gainprm'][:6,0]*kp
            m.actuator_biasprm[rows,:6,1]=-m.actuator_gainprm[rows,:6,0]
            m.actuator_biasprm[rows,:6,2]=base['actuator_biasprm'][:6,2]*kd
            m.dof_damping[bi,di]=damping; m.dof_frictionloss[bi,di]=friction
            m.dof_armature[bi,di]=base['dof_armature'][d]*armature
            if cpu:dirty.update(rows.tolist())
            else:mw.set_const(sim.wp_model,sim.wp_data)
        finally:
            if gpu_stream is not None:gpu_stream.__exit__(None,None,None)
    e.apply_payload=apply
    return e


def theta(e):
    x=e.hw_raw
    lo=x.new_tensor(LOW);hi=x.new_tensor(HIGH)
    return torch.cat((e.context.read(e),(x-(hi+lo)/2)/((hi-lo)/2),e.dr[:,6:7]/2.5-1),-1)
