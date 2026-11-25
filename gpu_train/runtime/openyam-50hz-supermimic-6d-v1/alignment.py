"""50 Hz task overlay. All inherited sources are read-only and hash-verified."""
from pathlib import Path
import hashlib, inspect, json, linecache, textwrap, types
import numpy as np
import torch
from torch import nn
from scipy.interpolate import CubicHermiteSpline, CubicSpline

GAMMA=0.99
LAMBDA=0.95
ACTOR_HORIZON=32
WM_FRAMES=8
REPLAY_TICKS=1000
ACTION_DIM=6
SIGMA_Q=.003
EXPLORATION_STD=[SIGMA_Q]*ACTION_DIM

def resample(z,times):
    t=z['time_s']
    spline=CubicHermiteSpline(t,z['q'],z['dq'],axis=0,extrapolate=False)
    q=spline(times);dq=spline(times,1)
    pos=CubicSpline(t,z['tcp_pos'],axis=0,extrapolate=False)(times)
    source=z['tcp_quat'].copy()
    signs=np.where(np.sum(source[1:]*source[:-1],axis=1)<0,-1.,1.)
    source*=np.r_[1.,np.cumprod(signs)][:,None]
    quat=CubicSpline(t,source,axis=0,extrapolate=False)(times)
    norm=np.linalg.norm(quat,axis=1,keepdims=True)
    assert np.all(norm>.5)
    quat/=norm
    result=np.concatenate((q,dq,pos,quat),axis=1)
    assert np.isfinite(result).all()
    return result
P=Path(__file__).resolve().parent
DT=.02
PHYSICS_DT=.002
SUBSTEPS=10
EPISODE_STEPS=300

def compile_source(src, namespace, label, name):
    filename=str(P/f'generated-{label}.py')
    linecache.cache[filename]=(len(src),None,src.splitlines(True),filename)
    exec(compile(src,filename,'exec'),namespace)
    return namespace[name]

def patch(fn, changes, label, **bindings):
    src=textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    for before,after in changes:
        assert src.count(before)==1,(label,before,src.count(before))
        src=src.replace(before,after)
    ns=dict(inspect.unwrap(fn).__globals__,DT=DT,SUBSTEPS=SUBSTEPS,EPISODE_STEPS=EPISODE_STEPS,**bindings)
    return compile_source(src,ns,label,fn.__name__)

def reference_init(o):
    def init(self,device):
        nn.Module.__init__(self)
        p=o.ASSET/o.DATA_VERSION
        self.manifest=json.loads((p/'MANIFEST.json').read_text())
        assert self.manifest['ready'] and self.manifest['accepted_train']>=16000
        arrays=[]
        for row in self.manifest['rows']:
            f=p/row['file']
            assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],f
            with np.load(f) as z:
                t=z['time_s']
                # Cover the complete physical clip; never reinterpret old sample count as 50Hz.
                times=np.arange(int(np.floor(t[-1]/DT))+1,dtype=np.float64)*DT
                arrays.append(resample(z,times).astype(np.float32))
        lengths=np.array([len(a) for a in arrays],dtype=np.int64)
        assert np.all(lengths>=EPISODE_STEPS+1)
        assert lengths.max()<o.HOLD_OFFSET if hasattr(o,'HOLD_OFFSET') else lengths.max()<2048
        assert (int(lengths.max())-(EPISODE_STEPS+1))*4096+2048+EPISODE_STEPS<2**24
        offsets=np.concatenate(([0],np.cumsum(lengths)[:-1]))
        assert [r['id'] for r in self.manifest['rows']]==list(range(len(arrays)))
        self.register_buffer('table',torch.tensor(np.concatenate(arrays),device=device),persistent=False)
        self.register_buffer('lengths',torch.tensor(lengths,device=device),persistent=False)
        self.register_buffer('offsets',torch.tensor(offsets,device=device),persistent=False)
        self.train_ids=np.array(self.manifest['train_ids']);self.dev_ids=np.array(self.manifest['dev_ids'])
        self.sample_dt=DT
        print(json.dumps(dict(reference_hz=50,clips=len(arrays),min_ticks=int(lengths.min()),max_ticks=int(lengths.max()),episode_steps=EPISODE_STEPS)),flush=True)
    return init

def install(T):
    E=T.E;o=E.o;l=E.learning
    assert o.DATA_VERSION=='bank-full-v2'
    o.DT=l.DT=E.rewards.DT=DT
    o.SUBSTEPS=l.SUBSTEPS=SUBSTEPS;o.EPISODE_STEPS=l.EPISODE_STEPS=EPISODE_STEPS
    E.rewards.components.__globals__['DT']=DT
    # Recompiled methods carry copied globals; update each reachable closure explicitly.
    seen=set()
    def update(fn):
        if isinstance(fn,(staticmethod,classmethod)):fn=fn.__func__
        if not isinstance(fn,types.FunctionType) or id(fn) in seen:return
        seen.add(id(fn))
        if 'DT' in fn.__globals__:fn.__globals__['DT']=DT
        for cell in fn.__closure__ or ():
            try:value=cell.cell_contents
            except ValueError:continue
            if isinstance(value,types.FunctionType):update(value)
    for cls in (o.OpenYAMEnv,l.Representation,l.CommandDynamics,l.MetalWorldModelEnv,l.Learner):
        for value in vars(cls).values():update(value)
    update(o.reward);update(o.effort_cost)
    l.Representation.dt=DT
    # Recompile the inherited student with a native 6D decoder and constructor.
    make_actor_class=patch(E.adapt.make_actor_class,[('build_mlp(60,(256,256),12)','build_mlp(60,(256,256),ACTION_DIM)')],'actor-class',ACTION_DIM=ACTION_DIM)
    E.Actor=make_actor_class(l.MultiEncoderAE,l.build_mlp)
    E.make_actor=l.make_actor=patch(E.make_actor,[('Actor(o.SENSOR,12,','Actor(o.SENSOR,ACTION_DIM,')],'actor-factory',ACTION_DIM=ACTION_DIM,Actor=E.Actor)
    l.Learner.__init__=patch(l.Learner.__init__,[
        ('ReplayBuffer(schema,256,','ReplayBuffer(schema,1000,'),
        ('action=12,prev_action=12','action=ACTION_DIM,prev_action=ACTION_DIM'),
        ('build_mlp(12+7+CTX,','build_mlp(12+7+12+ACTION_DIM+CTX,'),
    ],'learner-6d',ACTION_DIM=ACTION_DIM,make_actor=E.make_actor)
    l.Learner.collect=patch(l.Learner.collect,[
        ('torch.randn(e.num_envs,12,device=e.device)*torch.tensor([.005]*6+[.02]*6,device=e.device)',
         'torch.randn(e.num_envs,ACTION_DIM,device=e.device)*SIGMA_Q'),
        ('last["ctrl_eff"]','last["sent"]'),
    ],'collect-6d',ACTION_DIM=ACTION_DIM,SIGMA_Q=SIGMA_Q)
    def proc(self,action):
        assert action.shape[-1]==ACTION_DIM
        return action
    l.CommandDynamics.proc=proc
    # The 12D overlay added a state field; it is not needed in this lineage.
    E.base.EFIELDS.remove('sent_dq')
    o.ReferenceBank.__init__=reference_init(o)
    # Rebuild the actual pre-upload constructor, preserving the policy-only scale and PD override.
    src=E.old._init_source
    assert src.count('timestep=1/480')==1
    src=src.replace('timestep=1/480','timestep=PHYSICS_DT')
    assert src.count('configure_model(self.model)')==1
    src=src.replace('configure_model(self.model)','configure_model(self.model)\n self.model.opt.timestep=PHYSICS_DT')
    ns=dict(E.old._init_globals,ReferenceBank=o.ReferenceBank,DT=DT,PHYSICS_DT=PHYSICS_DT)
    ctor=compile_source(src,ns,'constructor','__init__')
    def env_init(self,*args,**kwargs):
        ctor(self,*args,**kwargs)
        self.sim.substeps=SUBSTEPS
    o.OpenYAMEnv.__init__=env_init
    o.OpenYAMEnv.reset=patch(o.OpenYAMEnv.reset,[('int(k)-180','int(k)-EPISODE_STEPS'),('decoded_ticks+180','decoded_ticks+EPISODE_STEPS'),
        ('if not hasattr(self,"sent_dq"):\n  self.sent_dq=torch.zeros_like(self.sent);self.raw=torch.zeros(self.num_envs,12,device=self.device)\n ',''),
        ('self.sent_dq[ids]=0;self.d.ctrl[ids]=mit_control(self.sent[ids],self.sent_dq[ids])','self.d.ctrl[ids]=self.sent[ids]'),
        ('torch.cat((self.sent[ids],self.sent_dq[ids]),-1)','self.sent[ids]')],'reset')
    o.OpenYAMEnv.step=patch(o.OpenYAMEnv.step,[('range(16)','range(SUBSTEPS)'),('self.tick>=180','self.tick>=EPISODE_STEPS'),
        ('action.shape==(self.num_envs,12)','action.shape==(self.num_envs,ACTION_DIM)'),
        ('torch.cat((self.sent,self.sent_dq),-1).clone()','self.sent.clone()'),
        ('self.mapped.copy_(action[:,:6]);self.clipped.copy_(action[:,:6]);self.sent.copy_(action[:,:6]);self.sent_dq.copy_(action[:,6:12]);self.d.ctrl[:]=mit_control(self.sent,self.sent_dq)',
         'self.mapped.copy_(action);self.clipped.copy_(action);self.sent.copy_(action);self.d.ctrl[:]=self.sent'),
        ('self.sent,self.sent_dq)','self.sent)'),
        ('sent_dq=self.sent_dq.clone(),','')],'step',ACTION_DIM=ACTION_DIM)
    l.MetalWorldModelEnv._wm_step=patch(l.MetalWorldModelEnv._wm_step,[('next_aux[:,7]>=180','next_aux[:,7]>=EPISODE_STEPS'),('sent[:,:6],sent[:,6:12]','sent')],'imagined-step')
    wm_init=l.MetalWorldModelEnv.__init__
    def init_wm(self,*args,**kwargs):
        wm_init(self,*args,**kwargs)
        self._num_actions=ACTION_DIM
        self.max_step_reward=7.5*DT
    l.MetalWorldModelEnv.__init__=init_wm
    # Keep dormant evaluation entry points aligned too.
    E.hold_evaluate=patch(E.hold_evaluate,[('181','(EPISODE_STEPS+1)'),('range(180)','range(EPISODE_STEPS)'),('ticks=180','ticks=EPISODE_STEPS'),('e.sent_dq[:32]-ref[:,6:12]','-ref[:,6:12]')],'hold-evaluate')
    evaluation=textwrap.dedent(inspect.getsource(E.old.contract.evaluate))
    for before,after,count in [('181','(EPISODE_STEPS+1)',3),('range(180)','range(EPISODE_STEPS)',1),('/30','*DT',1),('poststep_frames=5760','poststep_frames=32*EPISODE_STEPS',1),('p[:30]','p[:50]',1),('p[150:]','p[250:]',1)]:
        assert evaluation.count(before)==count,(before,evaluation.count(before))
        evaluation=evaluation.replace(before,after)
    E.old.contract.evaluate=compile_source(evaluation,dict(E.old.contract.evaluate.__globals__,DT=DT,EPISODE_STEPS=EPISODE_STEPS),'evaluate','evaluate')
    # A new timing/trajectory lineage must never masquerade as the inherited identity.
    def identity():return 'portable-tri-mode'
    E.identity=E.core.identity=E.core.base.identity=E.base.identity=identity

def checkplant(e,T):
    assert T.E.o.DT==T.E.learning.DT==T.E.rewards.DT==DT
    assert T.E.rewards.components.__globals__['DT']==DT
    assert e.model.opt.timestep==PHYSICS_DT
    assert e.sim.substeps==SUBSTEPS and e.bank.sample_dt==DT
    assert T.E.learning.Representation.dt==DT
    if hasattr(e.sim,'cfg'):assert e.sim.cfg.mujoco.timestep==PHYSICS_DT
    if hasattr(e.sim,'wp_model'):
        import warp as wp
        timestep=wp.to_torch(e.sim.wp_model.opt.timestep)
        torch.testing.assert_close(timestep,torch.full_like(timestep,PHYSICS_DT),rtol=0,atol=1e-9)
    return True
