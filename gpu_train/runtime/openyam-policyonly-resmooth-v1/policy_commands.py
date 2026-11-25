"""Only actuator semantics change. Reference remains conditioning/reward only."""
import inspect
import textwrap


def _replace_method(cls, name, changes):
    old = inspect.unwrap(getattr(cls, name))
    src = textwrap.dedent(inspect.getsource(old))
    for before, after in changes:
        assert src.count(before) == 1, (name, before)
        src = src.replace(before, after)
    import linecache
    ns = old.__globals__
    filename = __file__ + ':' + cls.__name__ + ':' + name
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    exec(compile(src, filename, 'exec'), ns)
    setattr(cls, name, ns[name])


def install(o, l):
    _replace_method(o.OpenYAMEnv, '__init__', [
        ('self.scale=torch.full((6,),.1,device=device)', 'self.scale=torch.ones(6,device=device)')])
    _replace_method(o.OpenYAMEnv, 'decode', [
        ('self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,:6]+self.scale*action', 'action')])
    _replace_method(o.OpenYAMEnv, 'step', [
        ('self.bank.lookup(self.ref_id,self.start_tick+self.tick)[:,:6]+self.scale*action', 'action')])
    _replace_method(o.OpenYAMEnv, 'reset', [
        ('self.sent[ids]=q;self.d.ctrl[ids]=q', 'self.sent[ids]=self.d.qpos._tensor[ids,:6];self.d.ctrl[ids]=self.sent[ids]'),
        ('self.raw[ids]=0;self.mapped[ids]=q;self.clipped[ids]=q', 'self.raw[ids]=self.sent[ids];self.mapped[ids]=self.sent[ids];self.clipped[ids]=self.sent[ids]')])
    _replace_method(l.Representation, 'compute_actor_sensor', [
        ('self.bank.lookup(aux_state[...,8],aux_state[...,9]+(aux_state[...,7]-1).clamp(min=0))[...,:6]+self.scale*last_action', 'last_action')])
    _replace_method(l.CommandDynamics, 'forward', [
        ('self.reference_bank.lookup(aux[...,8],aux[...,9]+aux[...,7])[...,:6]+.1*action', 'action'),
        ('super().forward(rep,sent,aux,ctx)', 'super(CommandDynamics,self).forward(rep,sent,aux,ctx)')])
    _replace_method(l.MetalWorldModelEnv, '_wm_step', [
        ('self.representation.bank.lookup(aux[:,8],aux[:,9]+aux[:,7])[:,:6]+.1*action', 'action'),
        ('self.representation.bank.lookup(aux[:,8],aux[:,9]+(aux[:,7]-1).clamp(min=0))[:,:6]+.1*last_action', 'self._sensor_history()[:, -1, 12:18]')])
    return dict(policy_only=True, units='radians', mapping='q_cmd = policy_action',
                action_scale=1., reference_bypass=False, output_clamp=False,
                reset_command='measured initial q', replay_action='executed noisy radians')
