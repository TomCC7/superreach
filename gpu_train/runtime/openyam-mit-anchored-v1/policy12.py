"""Pure 12D policy MIT command boundary; references never drive actuators."""
import torch,inspect

def effective(qdes,dq_des,kp,kd):
    return qdes+(qdes.new_tensor(kd)/qdes.new_tensor(kp))*dq_des

def install(o,l,parent,recompile):
    def control(qdes,dq_des):return effective(qdes,dq_des,o.KP,o.KD)
    def effort_cost(q,dq,qdes,dq_des):
        kp=q.new_tensor(o.KP);kd=q.new_tensor(o.KD)
        rated=q.new_tensor(o.RATED);peak=q.new_tensor(o.PEAK)
        requested=kp*(qdes-q)+kd*(dq_des-dq)
        excess=torch.relu(requested.abs()-rated)/(peak-rated)
        return -.01*excess.square().mean(-1)*o.DT,requested
    o.effort_cost=l.effort_cost=effort_cost
    init=o.OpenYAMEnv.__init__
    def env_init(self,*args,**kwargs):
        init(self,*args,**kwargs)
        self.raw=torch.cat((self.sent,self.sent_dq),-1).clone()
    o.OpenYAMEnv.__init__=env_init
    fn=o.OpenYAMEnv.reset
    o.OpenYAMEnv.reset=recompile(fn,[
        ('if not len(ids):return','if not hasattr(self,"sent_dq"):\n  self.sent_dq=torch.zeros_like(self.sent);self.raw=torch.zeros(self.num_envs,12,device=self.device)\n if not len(ids):return'),
        ('self.d.ctrl[ids]=self.sent[ids]','self.sent_dq[ids]=0;self.d.ctrl[ids]=mit_control(self.sent[ids],self.sent_dq[ids])'),
        ('self.raw[ids]=self.sent[ids]','self.raw[ids]=torch.cat((self.sent[ids],self.sent_dq[ids]),-1)')],dict(inspect.unwrap(fn).__globals__,mit_control=control),'policy12_reset')
    fn=o.OpenYAMEnv.step
    o.OpenYAMEnv.step=recompile(fn,[
        ('action.shape==(self.num_envs,6)','action.shape==(self.num_envs,12)'),
        ('previous=self.sent.clone()','previous=torch.cat((self.sent,self.sent_dq),-1).clone()'),
        ('self.mapped.copy_(action);self.clipped.copy_(self.decode(action));self.sent.copy_(self.clipped);self.d.ctrl[:]=self.sent','self.mapped.copy_(action[:,:6]);self.clipped.copy_(action[:,:6]);self.sent.copy_(action[:,:6]);self.sent_dq.copy_(action[:,6:12]);self.d.ctrl[:]=mit_control(self.sent,self.sent_dq);ctrl_eff=self.d.ctrl._tensor.clone()'),
        ('self.d.qvel._tensor[:,:6],self.sent)','self.d.qvel._tensor[:,:6],self.sent,self.sent_dq)'),
        ('self.goal,self.sent,previous,','self.goal,self.raw,previous,'),
        ('last=dict(torque_cost=','last=dict(ctrl_eff=ctrl_eff,sent_dq=self.sent_dq.clone(),torque_cost=')],dict(inspect.unwrap(fn).__globals__,mit_control=control,effort_cost=effort_cost),'policy12_step')
    fn=l.MetalWorldModelEnv._wm_step
    l.MetalWorldModelEnv._wm_step=recompile(fn,[
        ('previous=self._sensor_history()[:, -1, 12:18]','previous=last_action'),
        ('effort_cost(rep[:,:6],rep[:,6:12],sent)','effort_cost(rep[:,:6],rep[:,6:12],sent[:,:6],sent[:,6:12])')],dict(inspect.unwrap(fn).__globals__,effort_cost=effort_cost),'policy12_wm_step')
    def proc(self,action):
        assert action.shape[-1] in (6,12)
        return control(action[...,:6],action[...,6:12]) if action.shape[-1]==12 else action
    def forward(self,rep,action,aux,ctx):
        return super(l.CommandDynamics,self).forward(rep,action,aux,ctx)
    l.CommandDynamics.proc=proc;l.CommandDynamics.forward=forward
    fn=l.Representation.compute_actor_sensor
    l.Representation.compute_actor_sensor=recompile(fn,[('sent=last_action','sent=last_action[...,:6]')],dict(fn.__globals__),'policy12_sensor')
    fn=l.Learner.__init__
    l.Learner.__init__=recompile(fn,[('action=6,prev_action=6','action=12,prev_action=12')],dict(fn.__globals__),'policy12_learner_init')
    wm_init=l.MetalWorldModelEnv.__init__
    def init_wm(self,*args,**kwargs):
        wm_init(self,*args,**kwargs);self._num_actions=12
    l.MetalWorldModelEnv.__init__=init_wm
    fn=l.Learner.collect
    l.Learner.collect=recompile(fn,[
        ('torch.randn(e.num_envs,6,device=e.device)*(.005/e.scale)','torch.randn(e.num_envs,12,device=e.device)*torch.tensor([.005]*6+[.02]*6,device=e.device)'),
        ('last["sent"]','last["ctrl_eff"]')],dict(inspect.unwrap(fn).__globals__),'policy12_collect')
    if 'sent_dq' not in parent.state.EFIELDS:parent.state.EFIELDS.append('sent_dq')
    return control
