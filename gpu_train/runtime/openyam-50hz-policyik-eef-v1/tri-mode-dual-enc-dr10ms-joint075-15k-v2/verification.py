from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from train import policy_step, actor_lr, reward, directional_terms, RECIPE, DT, effective, SEED, observed, observations, mode_at, schedule, switch_goal


def verify(b,l,gpu,out):
    e = l.e
    track=int(e.bank.dev_ids[0]); step=int(e.bank.step_ids[0]); base=int(e.bank.offsets[step])
    rows=e.bank.table[base:base+301,12:19]
    switch=int(torch.nonzero((rows[1:]!=rows[:-1]).any(-1))[0])+1
    ids=torch.tensor([track,track,step,step],device=e.device)
    starts=torch.tensor([0,2048,0,0],device=e.device)
    e.reset(e.ids,ids,starts)
    assert torch.all((e.dr[:,6]>=0)&(e.dr[:,6]<=5)&(e.dr[:,6]==e.dr[:,6].round()))
    clean=e.sensor(); delivered=e.obs()['history'][:,-1]
    assert (clean[:,:12]-delivered[:,:12]).abs().max()>0
    assert torch.equal(clean[:,12:],delivered[:,12:])
    assert torch.equal(e.obs()['critic'][:,:12],e.state())
    e.dr[:,6]=0
    e.tick[2]=switch-1
    e.goal.copy_(e.bank.goal_pose(e.bank,ids,e.start_tick+e.tick))
    before=e.fullstate(); previous=e.sent.clone(); ctx=e.context.read(e)
    u=previous.clone(); u[:,1]+=.016
    t=e.d.time._tensor.clone()
    _, rr, done, last=e.step(u,auto_reset=False)
    torch.testing.assert_close(e.d.time._tensor-t,torch.full_like(t,.02),atol=1e-7,rtol=0)
    assert (e.goal[2]!=before[2,12:19]).any()
    assert (e.goal[0]!=before[0,12:19]).any()
    assert torch.equal(e.goal[1],before[1,12:19])
    post=torch.cat((last['q'],last['dq']),-1); aux=e.fullstate()[:,12:]
    def dyn(rep,cmd,old_aux,ctx):
        assert torch.equal(cmd,u)
        return aux,post
    fake=SimpleNamespace(dynamics=dyn,representation=l.rep,dr=e.dr)
    *_,wd,wr=l.wm_env._wm_step.__func__(fake,before[:,:12],before[:,12:],u,previous,ctx)
    assert torch.equal(rr,wr) and torch.equal(done,wd.bool())
    assert before.shape[1] == 23 and e.goal.shape == (4,7) and e.sent.shape == (4,6)
    assert e.obs()["history"].shape == (4,16,20)
    # A terminal transition through the real auto-reset route at max delay.
    e.tick.fill_(299); e.dr[:,6]=5
    e.goal.copy_(e.bank.goal_pose(e.bank,e.ref_id,e.start_tick+e.tick))
    initial=e.fullstate(); prev=e.sent.clone(); command=prev+.005; ctx_term=e.context.read(e)
    terminal_goal=e.bank.goal_pose(e.bank,e.ref_id,e.start_tick+300).clone()
    _,rterm,dterm,term=e.step(command,auto_reset=True)
    term_post=torch.cat((term['q'],term['dq']),-1)
    term_aux=torch.cat((terminal_goal,torch.full_like(initial[:,19:20],300),initial[:,20:]),-1)
    fake.dr=initial.new_zeros((4,7)); fake.dr[:,6]=5
    def term_dyn(rep,ctrl,old_aux,ctx):
        assert torch.equal(ctrl,effective(prev,command,fake.dr[:,6:7]))
        return term_aux,term_post
    fake.dynamics=term_dyn
    *_,wdone,wterm=l.wm_env._wm_step.__func__(fake,initial[:,:12],initial[:,12:],command,prev,ctx_term)
    assert dterm.all() and torch.count_nonzero(e.tick)==0 and torch.equal(dterm,wdone.bool())
    assert torch.equal(rterm,wterm)
    # Every native substep at all six bins. Raw out-of-envelope commands must
    # be held/copied exactly, not clipped or interpolated.
    transport_delta=0.; duration_average_delta=0.; randomized_reward_delta=0.
    for bins in ([0,1,2,3],[4,5,0,5]):
        e.reset(e.ids,ids,starts); e.dr[:,6]=e.dr.new_tensor(bins)
        prev=e.sent.clone(); initial=e.fullstate(); ctx_mix=e.context.read(e)
        cmd=e.sent.new_tensor([3.25,-2.75,2.5,-3.5,3.75,-4.]).expand_as(e.sent).clone()
        stored=[]; real=e.sim.step
        def spy():
            stored.append(e.d.ctrl._tensor.clone()); return real()
        e.sim.step=spy
        _,r_mix,done_mix,last=e.step(cmd,auto_reset=False)
        e.sim.step=real
        assert len(stored)==10 and not torch.equal(prev,cmd)
        assert torch.equal(e.sent,cmd) and torch.equal(last['sent'],cmd)
        assert torch.equal(e.obs()['history'][:,-1,12:18],cmd)
        for sub,actual in enumerate(stored):
            expected=torch.where(e.dr[:,6:7]>sub,prev,cmd)
            transport_delta=max(transport_delta,float((expected-actual).abs().max()))
        averaged=torch.stack(stored).mean(0)
        expected_average=cmd+(prev-cmd)*e.dr[:,6:7]/10
        duration_average_delta=max(duration_average_delta,float((averaged-expected_average).abs().max()))
        torch.testing.assert_close(last['applied_effective'],averaged,atol=6e-7,rtol=0)
        mixed_post=torch.cat((last['q'],last['dq']),-1); mixed_aux=e.fullstate()[:,12:]
        fake.dr=e.dr.clone()
        def mix_dyn(rep,ctrl,old_aux,ctx):
            assert torch.equal(ctrl,last['applied_effective'])
            return mixed_aux,mixed_post
        fake.dynamics=mix_dyn
        *_,d_wm,r_wm=l.wm_env._wm_step.__func__(fake,initial[:,:12],initial[:,12:],cmd,prev,ctx_mix)
        assert torch.equal(done_mix,d_wm.bool()) and torch.equal(r_mix,r_wm)
        randomized_reward_delta=max(randomized_reward_delta,float((r_mix-r_wm).abs().max()))
    assert transport_delta==0.
    assert torch.equal(l.dynamics.proc(cmd),cmd)
    assert not e.model.actuator_ctrllimited.any()
    q,dq = post[:,:6].detach().requires_grad_(), post[:,6:].detach().requires_grad_()
    u_req = u.detach().requires_grad_()
    tracking = torch.tensor([True,False,False,False],device=e.device)
    r0, terms, _, _ = reward(l.rep.fk,q,dq,aux[:,:7],u_req,previous,before[:,12:19],tracking,e.bank.lookup(ids,e.start_tick+1),before[:,6:12],torch.zeros_like(ids))
    assert torch.equal(r0,rr)  # Independent row-mode mask, before failure penalty (none here).
    assert set(terms)=={'position','rotation','joint_speed','joint_accel','command_slew','approach','away','joint_position','wrist_comfort'}
    factory_kp=[50.,60.,50.,25.,15.,10.]; factory_kd=[3.,4.,3.,2.,1.,.8]
    np.testing.assert_array_equal(e.model.actuator_gainprm[:6,0],factory_kp)
    np.testing.assert_array_equal(e.model.actuator_biasprm[:6,1],-np.array(factory_kp))
    np.testing.assert_array_equal(e.model.actuator_biasprm[:6,2],-np.array(factory_kd))
    # Same position error, equal speed in opposite directions. Test a TRACK
    # moving target, and verify STEP/HOLD ignore reference jumps.
    x=q.new_tensor([[.05,0,0],[.05,0,0],[0,0,0],[.05,0,0],[.05,0,0]])
    speed=.35*np.tanh(3*.05/.35)
    goalx=q.new_tensor([[0.,0,0],[0.,0,0],[.01,0,0],[.01,0,0],[.01,0,0]])
    oldx=torch.zeros_like(goalx)
    vel=q.new_tensor([[speed,0,0],[-speed,0,0],[.5,0,0],[.5+speed,0,0],[speed,0,0]])
    app,away=directional_terms(x,x.norm(dim=-1),vel,goalx,oldx,
                              torch.tensor([False,False,True,True,False],device=e.device))
    torch.testing.assert_close(app[[0,2,3,4]],torch.zeros_like(app[:4]),atol=1e-10,rtol=0)
    torch.testing.assert_close(away[[0,2,3,4]],torch.zeros_like(away[:4]),atol=0,rtol=0)
    torch.testing.assert_close(app[1],q.new_tensor(-.8*(2*speed/.35)**2))
    torch.testing.assert_close(away[1],q.new_tensor(-8*.05*speed))
    assert app[1]+away[1] < app[0]+away[0]
    # Test the shared installed reward with independent oracle constants,
    # including reference-relative TRACK and reference-velocity-free STEP.
    err = q.new_tensor([.05, .05, 10., 10., .05, 0.])
    expected_speed = .75*torch.tanh(3.*err/.75)
    delta = torch.zeros((6,6),dtype=q.dtype,device=q.device)
    delta[:,0] = err
    z = torch.zeros_like(delta)
    ref_vel = z.clone(); ref_vel[:,0] = .25
    tracking_joint = torch.tensor([False,False,False,False,True,False],device=q.device)
    dq_probe = z.clone()
    dq_probe[:,0] = expected_speed*q.new_tensor([1.,-1.,1.,-1.,1.,0.]) + .25*tracking_joint
    dq_probe.requires_grad_(True)
    q_probe = q[0:1].detach().expand(6,-1).contiguous()
    p_probe,r_probe = e.fk(q_probe)
    goal_probe = torch.cat((p_probe,r_probe),-1)
    _, joint_terms, *_ = reward(e.fk,q_probe,dq_probe,goal_probe,z,z,goal_probe,tracking_joint,
                                 torch.cat((q_probe+delta,ref_vel),-1),z,torch.full_like(err,2,dtype=torch.long))
    a_joint = joint_terms['approach']; w_joint = joint_terms['away']
    expected_app = -.8*(2*expected_speed/.75).square()
    expected_app[[0,2,4,5]] = 0.
    expected_away = -8.*err*expected_speed
    expected_away[[0,2,4,5]] = 0.
    torch.testing.assert_close(a_joint,expected_app,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(w_joint,expected_away,atol=1e-6,rtol=1e-6)
    away_grad, = torch.autograd.grad(w_joint.sum(),dq_probe,retain_graph=True)
    approach_grad, = torch.autograd.grad(a_joint.sum(),dq_probe)
    torch.testing.assert_close(away_grad[[1,3],0],8.*err[[1,3]])
    torch.testing.assert_close(approach_grad[[1,3],0],3.2*expected_speed[[1,3]]/.75**2)
    assert torch.isfinite(away_grad).all() and torch.isfinite(approach_grad).all()
    assert float(expected_speed[2]) == .75
    assert RECIPE['joint_speed_coeff']==.05
    assert torch.equal(terms['joint_speed'],-.05*dq.square().mean(-1))
    term_grad,=torch.autograd.grad(terms['joint_speed'].sum(),dq,retain_graph=True)
    torch.testing.assert_close(term_grad,-.10*dq/6)
    assert torch.isfinite(term_grad).all() and term_grad.abs().sum()>0
    assert all(g.isfinite().all() and g.abs().sum()>0 for g in torch.autograd.grad(r0.sum(),(q,dq,u_req)))
    # Test the actual real step and actual imagined scorer on identical native next states,
    # including all six ordered switches. Non-switch rows cover TRACK/HOLD/STEP.
    rows={}
    for src in range(3):
        for dst in range(3):
            e.reset(e.ids,ids,starts)
            e.dr[:,6]= torch.arange(4,device=e.device)+2
            e.tick.fill_(99)
            e.mode_schedule.fill_(src+3*dst+900)
            if src!=dst: e.mode_schedule[:3]=src+3*src+2709
            ref_step=e.bank.lookup(ids,starts+e.tick)
            e.goal.copy_(switch_goal(e.fk,ref_step[:,12:19],ref_step,e.mode_schedule))
            initial=e.fullstate();prev=e.sent.clone();ctx_test=e.context.read(e)
            obs_pre=e.obs(); assert torch.all(obs_pre['mode']==src)
            cmd=prev.clone();cmd[:,1]+=.015
            _,a_r,a_d,last=e.step(cmd,auto_reset=False)
            assert e.tick.tolist()==[100]*4
            assert e.obs()['mode'].tolist()==[src,src,src,dst]
            assert torch.equal(e.sent,cmd) and torch.equal(e.history[:,-1,12:18],cmd)
            ref_post=e.bank.lookup(ids,starts+e.tick)
            p,r=e.fk(ref_post[:,:6]); assert torch.equal(e.goal[3],torch.cat((p,r),-1)[3])
            nxt=e.fullstate()[:,12:]; ps=torch.cat((last['q'],last['dq']),-1)
            fake.dr=e.dr.clone()
            def dyn(rep,ctl,aux,ctx):
                assert torch.equal(ctl,last['applied_effective'])
                naux,_=l.rep.apply_prediction(aux,rep,torch.zeros_like(rep),.02)
                assert torch.equal(naux,nxt)
                return naux,ps
            fake.dynamics=dyn
            *_,w_d,w_r=l.wm_env._wm_step.__func__(fake,initial[:,:12],initial[:,12:],cmd,prev,ctx_test)
            assert torch.equal(a_d,w_d.bool()) and torch.equal(w_r,a_r)
            rows[f'{src}->{dst}']=float((a_r-w_r).abs().max())
    sg=torch.Generator(device=e.device).manual_seed(123)
    codes=schedule(100000,e.device,sg); switches=codes//9<301
    rates=torch.bincount(codes%3,minlength=3).float()/len(codes)
    torch.testing.assert_close(rates,rates.new_tensor([.4,.3,.3]),atol=.006,rtol=0)
    assert abs(float(switches.float().mean())-.15)<.004
    assert ((codes[switches]//9>=80)&(codes[switches]//9<=220)).all()
    assert (mode_at(codes[switches],codes[switches]//9)!=codes[switches]%3).all()
    e.reset(e.ids,ids,starts)
    a=l.actor; q0=e.state()[:,:6]; hist=e.obs()['history']; pose=e.goal.clone(); ref=e.bank.lookup(ids,starts)
    m=torch.tensor([1,1,1,1],device=e.device)
    ctx=e.context.read(e); td=b.T.E.learning.TensorDict
    def make_obs(g,rf,mode): return observations(td,e.fk,hist,g,rf,mode,e.state(),e.sent,ctx)
    with torch.no_grad(): a.decoder[-1].weight.normal_(0,.01)
    obs=make_obs(pose,ref,m)
    hidden=pose.clone(); hidden[:,3:]=pose.new_tensor([0.,1.,0.,0.]); poison=ref.clone();poison[:,:6]+=10
    altered=make_obs(hidden,poison,m)
    assert torch.equal(obs['critic'],altered['critic'])
    # Also bypass the observer and poison the actor's packet directly.
    direct=altered.clone();direct['goal_pose']=hidden;direct['goal_q']=poison[:,:6]
    vals=[]
    for fn,x,y in [(a,obs,direct),(l.critic,obs['critic'],altered['critic'])]:
        vals.append(float((fn(x)-fn(y)).abs().max().detach()))
    dq=e.state()[:,6:];u=e.sent
    xyz_score=lambda g,r:reward(e.fk,q0,dq,g,u,u,g,torch.zeros_like(m,dtype=torch.bool),r,dq,m)[0]
    vals.append(float((xyz_score(pose,ref)-xyz_score(hidden,poison)).abs().max()))
    assert vals==[0.,0.,0.]
    pos=pose.clone();pos[:,0]+=.05
    visible=make_obs(pos,ref,m)
    assert (a(obs)!=a(visible)).any() and (l.critic(obs['critic'])!=l.critic(visible['critic'])).any() and (xyz_score(pose,ref)!=xyz_score(pos,ref)).any()
    for mode in range(3):
        m.fill_(mode); obs=make_obs(pos,ref,m)
        obs['history']=obs['history'].detach().requires_grad_();obs['goal_pose']=obs['goal_pose'].detach().requires_grad_();obs['goal_q']=obs['goal_q'].detach().requires_grad_()
        out_cmd=a(obs)
        gs=torch.autograd.grad(out_cmd.square().sum(),(obs['history'],obs['goal_q'] if mode==2 else obs['goal_pose']))
        assert all(g.isfinite().all() and g.abs().sum()>0 for g in gs)
    assert a.decoder[0].in_features==(43 if a.arch=='masked' else 60) and l.critic[0].in_features==44
    with torch.no_grad():a.decoder[-1].weight.zero_()
    assert abs(actor_lr(10000)-2e-4)<1e-12 and actor_lr(15000)==5e-5
    records=[]; insert=l.buffer.insert
    def record(row):
        records.append(row.clone()); return insert(row)
    l.buffer.insert=record
    try:
        for _ in range(20):
            packet=e.obs()['history'][:,-1]; episode_dr=e.dr.clone(); true=e.fullstate()
            l.collect()
            row=records.pop()
            assert torch.equal(row['actor_obs'],packet) and torch.equal(row['dr'],episode_dr)
            assert torch.equal(row['state'],true)
    finally:
        l.buffer.insert=insert
    batch=l.wm_env.sample_initial_states()
    imagined=l.wm_env.reset(batch)
    assert torch.equal(imagined['history'][:,-1],batch['sensor_window'][:,-1])
    assert torch.equal(l.wm_env.dr,batch['dr'])
    assert torch.equal(imagined['critic'][:,:12],l.wm_env.rep_state)
    for _ in range(2):
        wm=l.wm_update()
        target_before=[p.clone() for p in l.target.parameters()]
        pol=policy_step(b,l);l.finite()
        for prev,target,critic in zip(target_before,l.target.parameters(),l.critic.parameters(),strict=True):
            assert torch.equal(target,prev.lerp(critic.detach(),.005))
        assert pol['Policy/grad_norm']>0 and pol['Policy/critic_grad_norm']>0
        for module in (l.actor,l.critic,l.dynamics):
            assert all(p.grad is None or p.grad.isfinite().all() for p in module.parameters())
    assert l.count['actor_updates']==l.count['critic_minibatches']==2
    assert all(np.isfinite(v) for v in [*wm.values(),*pol.values()] if isinstance(v,(int,float)))
    return dict(passed=True,source_digest=b.eef_source_digest,device=str(e.device), reward_max_abs_delta=max(float((rr-wr).abs().max()),float((rterm-wterm).abs().max()),randomized_reward_delta), termination_parity=True,
                reward_rows=['TRACK','HOLD','STEP switch','STEP'], leakage_max_abs_delta=vals, all_mode_pairs=rows, mode_probabilities=rates.tolist(), switch_fraction=float(switches.float().mean()), actor_decoder_width=a.decoder[0].in_features, critic_width=44, finite_history_and_goal_gradients=True,
                factory_kp=factory_kp,factory_kd=factory_kd,
                joint_directional_probe=dict(far_speed_rad_s=float(expected_speed[2]), signed_away_coefficient=-float(away_grad[3,0]/err[3]), approach=a_joint.detach().tolist(), away=w_joint.detach().tolist()),
                directional_probe_dt_scaled=(DT*(app+away)).tolist(),
                transitions=l.count,wm=wm,policy=pol,transport_substeps_verified=10,transport_substep_max_abs_delta=transport_delta,delay_bins_ms=[0,2,4,6,8,10],duration_average_max_abs_delta=duration_average_delta,exact_actor_packet_replay=True,goal_dim=16,action_dim=6,joint_speed_coeff=RECIPE['joint_speed_coeff'],
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
