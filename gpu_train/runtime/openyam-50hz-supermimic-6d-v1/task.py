"""Shared immutable reachable goal schedules: real/WM use identical lookup.

Schedule identity and tick live in existing privileged aux (ref_id/start/tick).
Actor ABI remains 16*20+19=339 -> 6; last history command is q_des.
No future goals enter actor observations. Templates are immutable for replay.
"""
import torch,numpy as np
DT=.02
KP=[50.,60.,50.,25.,15.,10.]; KD=[3.,4.,3.,2.,1.,.8]

def physical_failure(q,dq):
    return (~torch.isfinite(torch.cat((q,dq),-1)).all(-1)) | (q.abs().amax(-1)>12) | (dq.abs().amax(-1)>100)

def build_schedules(bank,count=2048):
    rng=np.random.default_rng(20261117)
    table=bank.table.detach().cpu().numpy();offsets=bank.offsets.cpu().numpy();lengths=bank.lengths.cpu().numpy()
    # Only training poses; held-out dev clips remain untouched.
    candidates=[]
    for rid in bank.train_ids:
        candidates.extend((offsets[rid]+np.arange(0,lengths[rid]-301,50)).tolist())
    pool=table[np.array(candidates,dtype=np.int64)]
    from scipy.spatial import cKDTree
    tree=cKDTree(pool[:,12:15])
    rows=[];initial=[];bins=[];switches=[];distances=[]
    for i in range(count):
        A=pool[rng.integers(len(pool))].copy(); initial.append(A[:12].copy())
        row=np.repeat(A[None],601,axis=0); row[:,6:12]=0
        goal=A;tick=0;events=[]
        for epoch in range(5):
            category=i%3
            lo,hi=[(.01,.10),(.10,.30),(.30,1.5)][category]
            subset=tree.query_ball_point(goal[12:15],hi) if category<2 else rng.integers(len(pool),size=2048)
            sample=pool[np.asarray(subset,dtype=np.int64)]
            dist=np.linalg.norm(sample[:,12:15]-goal[12:15],axis=1)
            eligible=np.flatnonzero((dist>=lo)&(dist<hi))
            if not len(eligible):raise RuntimeError(('unavailable reachable distance bin',lo,hi,float(dist.max())))
            B=sample[rng.choice(eligible)].copy(); B[6:12]=0
            if epoch==0:distances.append(float(np.linalg.norm(B[12:15]-A[12:15])))
            end=min(601,tick+int(rng.integers(75,151)))
            row[tick:end]=B; events.append(tick); tick=end;goal=B
            if tick>=601:break
        row[tick:]=goal
        rows.append(row);bins.append(category);switches.append(events)
    n=len(lengths);extra=torch.tensor(np.concatenate(rows),device=bank.table.device)
    bank.table=torch.cat((bank.table,extra),0)
    bank.offsets=torch.cat((bank.offsets,torch.arange(count,device=extra.device)*601+len(table)))
    bank.lengths=torch.cat((bank.lengths,torch.full((count,),601,device=extra.device,dtype=torch.long)))
    bank.step_initial=torch.tensor(np.array(initial),device=extra.device)
    bank.step_ids=torch.arange(n,n+count,device=extra.device)
    bank.step_offset=n
    bank.schedule_receipt=dict(templates=count,mode_probabilities=dict(TRACK=.4,HOLD=.2,STEP=.4),distance_bins_m=[[.01,.1],[.1,.3],[.3,1.5]],max_initial_distance_m=max(distances),over_0_6m=sum(d>=.6 for d in distances),switch_intervals_ticks=[75,150],switches=switches[:3])
    assert max(distances)>=.6,bank.schedule_receipt

def components(p,a,q,dq_new,ref_new,sent,previous,dq_old,ref_old):
    qref=ref_new[:,:6];dqref=ref_new[:,6:12]
    dq_star=dqref+1.5*torch.tanh(2.0*(qref-q)/1.5)
    # Legitimate TRACK derivatives only; switches between static targets have
    # dqref=0. Crossing dynamic->static or an implausible jump is masked too.
    changed=(ref_new[:,:6]-ref_old[:,:6]-DT*ref_old[:,6:12]).abs().amax(-1)>.08
    qdd_ref=torch.where(changed[:,None],torch.zeros_like(dqref),(dqref-ref_old[:,6:12])/DT)
    e=(q-qref).norm(dim=-1)
    # Multi-scale precision bonuses retain a far-field gradient; max sum=7.5.
    return dict(position=2.5*(1-(torch.sqrt(p*p+.005**2)-.005)/.10)+.5*torch.exp(-(p/.005)**2)+.5*torch.exp(-(p/.05)**2),
        rotation=1.5*(1-(torch.sqrt(a*a+.01**2)-.01)/.35)+.25*torch.exp(-(a/.02)**2)+.25*torch.exp(-(a/.15)**2),
        joint_position=1.0*(1-(torch.sqrt(e*e+.01**2)-.01)/.30)+.25*torch.exp(-(e/.03)**2)+.25*torch.exp(-(e/.20)**2),
        joint_velocity=-.8*(dq_new-dq_star).square().mean(-1),
        joint_acceleration=-.015*((dq_new-dq_old)/DT-qdd_ref).square().mean(-1),
        hold=.5*((p<.005)&(a<.008726646)).float(),
        command_slew=-.40*((sent-previous)/DT).square().mean(-1),
        overshoot=-8*torch.relu((q-qref)*(dq_new-dqref)).mean(-1),
        braking=-1.5*(torch.exp(-((q-qref)/.10).square())*(dq_new-dqref).square()).mean(-1),
        progress=torch.zeros_like(p))

# Cached catalog-derived pool: never assume variant index is proportional to mass.
_PAYLOAD_POOLS={}
def sample_indices(n,device,generator,count):
    key=(str(torch.device(device)),count)
    if key not in _PAYLOAD_POOLS:_PAYLOAD_POOLS[key]=_PAYLOAD_POOLS[('cpu',count)].to(device)
    pool=_PAYLOAD_POOLS[key]
    ix=torch.randint(9,count,(n,),device=device,generator=generator)
    heavy=pool[torch.randint(len(pool),(n,),device=device,generator=generator)]
    draw=torch.rand(n,device=device,generator=generator)
    ix=torch.where(draw<.45,heavy,ix)
    ix=torch.where(draw<.30,torch.full_like(ix,4),ix)
    return torch.where(draw<.15,torch.zeros_like(ix),ix)

def install_sampling(E):
    with E.continuous.load_catalog() as catalog:
        masses=np.asarray(catalog['masses'])
        assert masses[0]==0 and masses[4]==2
        heavy=np.flatnonzero((masses>=1.75)&(masses<2)&(np.arange(len(masses))>=9))
        assert len(heavy)>0
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            _PAYLOAD_POOLS[(device,len(masses))]=torch.tensor(heavy,device=device)
    E.continuous.sample_indices=sample_indices

def effort_cost(q,dq,qdes):
    requested=q.new_tensor(KP)*(qdes-q)-q.new_tensor(KD)*dq
    rated=q.new_tensor(_RATED);peak=q.new_tensor(_PEAK)
    # Physical torque, not command offset: steady gravity hold remains legitimate.
    rated_excess=torch.relu(requested.abs()-rated)/(peak-rated)
    peak_excess=torch.relu(requested.abs()-peak)/peak
    return (-.05*rated_excess.square().mean(-1)-.20*peak_excess.square().mean(-1))*DT,requested

def verification_gates(e,T):
    """Empirical sampler + actual reward/plant command-boundary regressions."""
    import inspect,ast,mujoco as mj
    from scipy.optimize import brentq
    E=T.E;device=e.device
    g=torch.Generator(device=device).manual_seed(20261117)
    ix=E.continuous.sample_indices(200000,device,g,len(e.mass_values))
    masses=e.mass_values[ix]
    fractions=dict(zero=float((ix==0).float().mean()),exact_2kg=float((ix==4).float().mean()),heavy_1_75_to_2kg=float(((masses>=1.75)&(masses<=2)).float().mean()))
    assert .14<fractions['zero']<.16 and .14<fractions['exact_2kg']<.16
    assert fractions['heavy_1_75_to_2kg']>=.25
    assert not ((ix>0)&(ix<9)&(ix!=4)).any()
    # Source and executable checks: only reference tick/rotation/physical torque
    # handling may bound values; no clamp in command decoding or reward targets.
    tree=ast.parse(inspect.getsource(components))
    assert not any(isinstance(n,ast.Name) and n.id=='q_star' for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr.startswith('clamp') for n in ast.walk(tree))
    for fn in [E.o.OpenYAMEnv.decode,E.learning.CommandDynamics.proc]:
        source=inspect.getsource(fn)
        assert 'clamp' not in source and 'clip(' not in source,source
    assert inspect.unwrap(E.o.OpenYAMEnv.step).__globals__['effort_cost'] is effort_cost
    assert E.learning.MetalWorldModelEnv._wm_step.__globals__['effort_cost'] is effort_cost
    # Native 2kg HOME with exact payload constants and training motor gains.
    model=mj.MjModel.from_xml_path(E.o.os.environ['SUPERREACH_SCENE'])
    E.old.configure_model(model);model.opt.timestep=.002
    model.actuator_ctrllimited[:]=False
    model.actuator_forcelimited[:]=True
    model.actuator_forcerange[:]=np.array([[-x,x] for x in E.o.PEAK])
    assert not e.model.actuator_ctrllimited.any()
    np.testing.assert_allclose(e.model.actuator_forcerange,model.actuator_forcerange)
    with E.continuous.load_catalog() as catalog:
        for field in ['body_mass','body_ipos','body_inertia','body_iquat']:getattr(model,field)[:]=catalog[field][4]
    data=mj.MjData(model);mj.mj_setConst(model,data)
    home=np.array([0.,1.047,1.047,0.,0.,0.]);data.qpos[:6]=home;data.qpos[6:]=.055
    mj.mj_forward(model,data);bias=data.qfrc_bias[:6].copy();offset=bias/np.array(KP)
    sid=model.site('arm_tcp_site').id;xyz=data.site_xpos[sid].copy()
    quat=np.empty(4);mj.mju_mat2Quat(quat,data.site_xmat[sid])
    def sag_distance(scale):
        data.qpos[:6]=home-scale*offset;mj.mj_forward(model,data)
        return np.linalg.norm(data.site_xpos[sid]-xyz)
    scales=np.linspace(0,8,161);bracket=next(s for s in scales if sag_distance(s)>=.2)
    scale=brentq(lambda s:sag_distance(s)-.2,0,bracket)
    sag=home-scale*offset
    tensor=lambda a:torch.as_tensor(np.asarray(a),device=device,dtype=torch.float32)[None]
    q=tensor(home);zero=torch.zeros_like(q)
    ref=tensor(np.r_[home,np.zeros(6),xyz,quat]);goal=ref[:,12:19]
    cmd=tensor(home+offset);sagcmd=tensor(home)
    def score(state,command):
        reward,terms,p,a=E.o.reward(e.fk,state,state,goal,command,command,zero,ref,zero,ref)
        cost,_=effort_cost(state,zero,command)
        assert terms['command_slew'].item()==0 and 'command_anchor' not in terms
        return float((reward+cost).item()),float(p.item()),{k:float(v.item()) for k,v in terms.items()}
    good,gp,goodterms=score(q,cmd);bad,bp,badterms=score(tensor(sag),sagcmd)
    assert good>bad and gp<1e-5 and abs(bp-.2)<1e-5,(good,bad,gp,bp)
    data.qpos[:6]=home;data.qvel[:]=0;data.ctrl[:]=home+offset
    mj.mj_forward(model,data);mj.mj_step(model,data,nstep=10)
    drift=float(np.max(np.abs(data.qpos[:6]-home)))
    assert drift<1e-4,drift
    # Exercise real command path with offsets far outside the removed bounds.
    e.reset(e.ids)
    q0=e.d.qpos._tensor[:,:6].clone();dq0=e.d.qvel._tensor[:,:6].clone()
    raw=q0+1.0
    torch.testing.assert_close(e.decode(raw),raw,rtol=0,atol=0)
    _,_,_,last=e.step(raw,auto_reset=False)
    for value in (last['raw'],last['sent'],last['ctrl_eff'],e.d.ctrl._tensor,E.learning.CommandDynamics.proc(None,raw)):
        torch.testing.assert_close(value,raw,rtol=0,atol=0)
    torch.testing.assert_close(last['tau_requested'],q0.new_tensor(KP)*(raw-q0)-dq0.new_tensor(KD)*dq0)
    assert (last['actual_torque_peak']<=q0.new_tensor(_PEAK)+1e-4).all()
    # Retain away-motion/braking and match the old zero-dq_des slew exactly.
    near=q+.02;vel=torch.ones_like(q)*.5
    away=components(gp*torch.ones(1,device=device),torch.zeros(1,device=device),near,vel,ref,cmd,cmd,vel,ref)
    toward=components(gp*torch.ones(1,device=device),torch.zeros(1,device=device),near,-vel,ref,cmd,cmd,-vel,ref)
    assert away['overshoot'].item()<toward['overshoot'].item() and away['braking'].item()<0
    jump=cmd+cmd.new_tensor([.1,-.04,.03,0.,.01,-.02])
    terms=components(torch.zeros(1,device=device),torch.zeros(1,device=device),q,zero,ref,jump,cmd,zero,ref)
    sq=((jump-cmd)/DT).square().mean(-1)
    torch.testing.assert_close(terms['command_slew'],-.25*sq-.15*sq)
    e.reset(e.ids)
    return dict(passed=True,samples=len(ix),payload_fractions=fractions,no_q_star_clamp=True,no_command_anchor=True,raw_6D_passthrough=True,real_imagined_effort_same=True,gravity_home_2kg=dict(tau_bias_Nm=bias.tolist(),qdes_offset_rad=offset.tolist(),hold_reward=good,sag_200mm_reward=bad,sag_error_m=bp,hold_error_m=gp,native_hold_20ms_drift_rad=drift,hold_terms=goodterms,sag_terms=badterms),physical_braking_verified=True,steady_command_slew_zero=True)

def install(T,A):
    E=T.E;o=E.o;l=E.learning
    global _RATED,_PEAK
    _RATED=o.RATED;_PEAK=o.PEAK
    install_sampling(E)
    o.effort_cost=l.effort_cost=effort_cost
    import inspect
    inspect.unwrap(o.OpenYAMEnv.step).__globals__['effort_cost']=effort_cost
    l.MetalWorldModelEnv._wm_step.__globals__['effort_cost']=effort_cost
    E.rewards.components=components
    # make_reward resolves components through the original rewards globals.
    o.reward.__globals__['components']=components
    base_init=o.ReferenceBank.__init__
    def bank_init(self,device):base_init(self,device);build_schedules(self)
    o.ReferenceBank.__init__=bank_init
    real_reset=o.OpenYAMEnv.reset
    def reset(self,ids,ref_ids=None,start_ticks=None):
        training=ref_ids is None
        real_reset(self,ids,ref_ids,start_ticks)
        if not training or not len(ids):return
        # Inherited 20% HOLD draw is replaced by an explicit 40/20/40 draw.
        decoded=E.hold.decode(self.start_tick[ids])[1]
        self.start_tick[ids]=decoded
        mode=torch.as_tensor(self.goal_rng.random(len(ids)),device=self.device)
        h=ids[(mode>=.4)&(mode<.6)]
        self.start_tick[h]=E.hold.encode(self.start_tick[h])
        step_ids=ids[mode>=.6]
        nonstep=ids[mode<.6]
        if len(nonstep):
            initial=self.bank.lookup(self.ref_id[nonstep],self.start_tick[nonstep])
            self.d.qpos[nonstep,:6]=initial[:,:6];self.d.qvel[nonstep,:6]=initial[:,6:12]
        if len(step_ids):
            j=torch.as_tensor(self.goal_rng.integers(len(self.bank.step_ids),size=len(step_ids)),device=self.device)
            self.ref_id[step_ids]=self.bank.step_ids[j];self.start_tick[step_ids]=0
            initial=self.bank.step_initial[j]
            self.d.qpos[step_ids,:6]=initial[:,:6];self.d.qvel[step_ids,:6]=initial[:,6:12]
        # Rebuild padding only at a physical reset. Payload wrapper remains
        # outside this method and runs once; switching is only bank.lookup.
        self.sent[ids]=self.d.qpos._tensor[ids,:6]
        self.raw[ids]=self.sent[ids]
        self.d.ctrl[ids]=self.sent[ids]
        self.goal[ids]=self.bank.lookup(self.ref_id[ids],self.start_tick[ids])[:,12:19]
        self.sim.forward();s=self.sensor()[ids];self.history[ids]=s[:,None];self.history[ids,:,18]=0;self.history[ids,-1,18]=1
    o.OpenYAMEnv.reset=reset
    o.OpenYAMEnv.step=A.patch(o.OpenYAMEnv.step,[('departure=(p>.05)|(a>0.3490658503988659)','departure=physical_failure(self.d.qpos._tensor[:,:6],self.d.qvel._tensor[:,:6])')],'physical-step')
    import inspect
    inspect.unwrap(o.OpenYAMEnv.step).__globals__['physical_failure']=physical_failure
    l.MetalWorldModelEnv._wm_step=A.patch(l.MetalWorldModelEnv._wm_step,[('departure=(p>.05)|(a>0.3490658503988659)','departure=physical_failure(next_rep[:,:6],next_rep[:,6:12])')],'physical-imagined-step')
    l.MetalWorldModelEnv._wm_step.__globals__['physical_failure']=physical_failure
    # 7.5 = 2.5 position +1 precision +2 rotation +1.5 joints +.5 hold.
    def sensor(self):
        return torch.cat((self.state(),self.sent,torch.ones(self.num_envs,1,device=self.device),self.tick[:,None]*DT),-1)
    o.OpenYAMEnv.sensor=sensor
    def wm_sensor(self,rep,ctx,aux_state,joint_state=None,last_action=None,noisy=False):
        return torch.cat((rep,last_action,torch.ones_like(aux_state[...,7:8]),aux_state[...,7:8]*DT),-1)
    l.Representation.compute_actor_sensor=wm_sensor
    real_obs=o.OpenYAMEnv.obs
    def obs(self):
        td=real_obs(self);ref=self.bank.lookup(self.ref_id,self.start_tick+self.tick)
        td['critic']=torch.cat((self.state(),self.goal,ref[:,:12],self.raw,self.context.read(self)),-1)
        return td
    o.OpenYAMEnv.obs=obs
    wm_obs=l.MetalWorldModelEnv.get_observations
    def get_observations(self):
        td=wm_obs(self);ref=self.representation.bank.lookup(self.aux_state[:,8],self.aux_state[:,9]+self.aux_state[:,7])
        td['critic']=torch.cat((self.rep_state,self.aux_state[:,:7],ref[:,:12],self.last_actions,self.sim_ctx),-1)
        return td
    l.MetalWorldModelEnv.get_observations=get_observations
