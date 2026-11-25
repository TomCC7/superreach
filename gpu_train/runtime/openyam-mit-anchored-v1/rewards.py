"""Shared post-step Charbonnier and Markovian state acceleration at 30 Hz."""
import inspect,textwrap,linecache,math
import torch
DT=1/30
def components(p,a,q,dq_new,ref_new,sent,previous,dq_old,ref_old):
    velocity_slew=velocity_anchor=0
    if sent.shape[-1]==12:
        q_des=sent[:,:6];dq_des=sent[:,6:12]
        prev_qdes=previous[:,:6];prev_dqdes=previous[:,6:12]
        ratio=q_des.new_tensor([19.34244747/50.,21.93853498/60.,12.81202128/50.,4.27472737/25.,2.28399773/15.,0.8/10.])
        ctrl_new=q_des+ratio*dq_des;ctrl_old=prev_qdes+ratio*prev_dqdes
        velocity_slew=-.05*((dq_des-prev_dqdes)/DT).square().mean(-1)-.15*((ctrl_new-ctrl_old)/DT).square().mean(-1)
        velocity_anchor=-2.0*(dq_des-ref_new[:,6:12]).square().mean(-1)
        sent=q_des;previous=prev_qdes
    e_q=(q-ref_new[:,:6]).norm(dim=-1)
    qdd=(dq_new-dq_old)/DT
    qdd_ref=(ref_new[:,6:12]-ref_old[:,6:12])/DT
    return dict(position=2.5*(1-(torch.sqrt(p.square()+.005**2)-.005)/.10),
        rotation=2*(1-(torch.sqrt(a.square()+.01**2)-.01)/.35),
        joint_position=1.5*(1-(torch.sqrt(e_q.square()+.01**2)-.01)/.30),
        joint_velocity=-.80*(dq_new-ref_new[:,6:12]).square().mean(-1),
        joint_acceleration=-.015*(qdd-qdd_ref).square().mean(-1),
        hold=.5*((p<.005)&(a<math.radians(.5))).float(),
        command_slew=velocity_slew-.25*((sent-previous)/DT).square().mean(-1),
        command_anchor=velocity_anchor-5.0*torch.relu((sent-ref_new[:,:6]).abs()-.04).square().mean(-1)-.50*(sent-ref_new[:,:6]).square().mean(-1),
        overshoot=-2.0*torch.relu((q-ref_new[:,:6])*(dq_new-ref_new[:,6:12])).mean(-1),
        progress=torch.zeros_like(p))
def make_reward(errors):
    def reward(fk,old,q,goal,sent,previous,dq_new=None,ref_new=None,dq_old=None,ref_old=None):
        p,a=errors(fk,q,goal)
        # Preserve the historical six-argument diagnostic API only; installed
        # real/imagined paths always supply both physical states and references.
        if dq_new is None:dq_new=torch.zeros_like(q)
        if ref_new is None:ref_new=torch.cat((q.detach(),dq_new.detach()),-1)
        if dq_old is None:dq_old=dq_new
        if ref_old is None:ref_old=ref_new
        terms=components(p,a,q,dq_new,ref_new,sent,previous,dq_old,ref_old)
        return sum(terms.values())*DT,terms,p,a
    return reward
def recompile(fn,changes,namespace,label):
    src=textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    for before,after in changes:
        assert src.count(before)==1,(label,before);src=src.replace(before,after)
    filename=__file__+':'+label;linecache.cache[filename]=(len(src),None,src.splitlines(True),filename)
    exec(compile(src,filename,'exec'),namespace)
    return namespace[fn.__name__]
def install(o,l,parent):
    reward=make_reward(o.errors);o.reward=l.reward=reward
    # Bypass predecessor curvature wrappers entirely. Neither installed path
    # reads u_prev2, nor installs the second-order command-curvature hook.
    real=parent.bindings['real_original'];imagined=parent.bindings['imagined_original']
    # Parent import also wraps Learner methods for curvature-only diagnostics.
    # Remove those wrappers, not just the numerical step hooks: otherwise collect
    # expects a curvature_real_log that no longer exists and update requests a
    # removed diagnostic. Preserve the actual underlying learning operations.
    for method,captured in [('collect','old_collect'),('actor_update','old_update')]:
        wrapped=getattr(l.Learner,method)
        assert inspect.getsourcefile(wrapped)==parent._policy_curvature.__file__,method
        setattr(l.Learner,method,inspect.getclosurevars(wrapped).nonlocals[captured])
    realglobals=inspect.unwrap(real).__globals__;realglobals['reward']=reward
    o.OpenYAMEnv.step=recompile(real,[
        (' applied=[]',' old_dq=self.d.qvel._tensor[:,:6].clone()\n applied=[]'),
        ('self.goal,self.sent,previous)','self.goal,self.sent,previous,self.d.qvel._tensor[:,:6],self.bank.lookup(self.ref_id,self.start_tick+self.tick),old_dq,self.bank.lookup(self.ref_id,self.start_tick+(self.tick-1).clamp(min=0)))')],realglobals,'real')
    wmglobals=imagined.__globals__;wmglobals['reward']=reward
    l.MetalWorldModelEnv._wm_step=recompile(imagined,[
        (' r=r+torque_cost',' # Integrate effort after departure, exactly like the real transition.'),
        ('r-departure.float()','(r-departure.float())+torque_cost'),
        ('next_aux[:,:7],sent,previous)','next_aux[:,:7],sent,previous,next_rep[:,6:12],self.representation.bank.lookup(next_aux[:,8],next_aux[:,9]+next_aux[:,7]),rep[:,6:12],self.representation.bank.lookup(next_aux[:,8],next_aux[:,9]+(next_aux[:,7]-1).clamp(min=0)))')],wmglobals,'imagined')
    return reward
