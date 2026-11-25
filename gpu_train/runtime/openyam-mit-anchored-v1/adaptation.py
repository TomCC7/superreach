"""Time-free 18+32+10 bottleneck and causal payload auxiliary objective."""
import inspect,linecache,textwrap
import torch
from torch import nn
AUX_WEIGHT=.01
# EffectivePayloadSpec.bind fixed physical bounds, not learnable statistics.
LOW=(0.,-.3,-.3,-.3,-.1,-.1,-.1,-.1,-.1,-.1)
HIGH=(4.,.3,.3,.3,.1,.1,.1,.1,.1,.1)
def normalize_raw(raw):
    lo=raw.new_tensor(LOW);hi=raw.new_tensor(HIGH)
    return (raw-(lo+hi)/2)/((hi-lo)/2)
class BinaryValidityNormalizer(nn.Module):
    def __init__(self,original):super().__init__();self.original=original
    def forward(self,value):
        h=self.original(value).clone();h[...,18]=value[...,18]
        return torch.where(value[...,18:19]>.5,h,torch.zeros_like(h))
    @torch.no_grad()
    def update(self,value):self.original.update(value)
def make_actor_class(base,build_mlp):
    class NarrowActor(base):
        def __init__(self,*args,mode='student',**kwargs):
            super().__init__(*args,**kwargs);assert mode in ('student','teacher');self.mode=mode
            self.obs_normalizer=BinaryValidityNormalizer(self.obs_normalizer)
            self.history_encoder=nn.Sequential(nn.Linear(304,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,10))
            self.decoder=build_mlp(60,(256,256),12)
            nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
        def normalized_history(self,history):return self.obs_normalizer(history)
        def payload_prediction(self,history):return self.history_encoder(self.normalized_history(history)[...,:19].flatten(1))
        def forward(self,obs_td):
            h=self.normalized_history(obs_td['history']);hflat=h[...,:19].flatten(1)
            current=h[:,-1,:18]
            goal=torch.tanh(self.encoders['motion'](self.goal_normalizers['motion'](obs_td['goal_motion'])))
            ctx=obs_td['privileged_ctx'] if self.mode=='teacher' else self.history_encoder(hflat)
            self._aux_loss=h.new_zeros(h.shape[0])
            return self.decoder(torch.cat((current,goal,ctx),-1))
        def payload_loss(self,history,normalized_context):
            hflat=self.normalized_history(history)[...,:19].flatten(1)
            if self.mode=='teacher':hflat=hflat.detach()
            prediction=self.history_encoder(hflat);target=normalized_context.detach()
            dims=(prediction-target).square().mean(0)
            return dims.mean(),dims.detach(),prediction.detach()
    return NarrowActor
def install_context(o,l):
    # IMPORTANT: installed vendor SimContext.read ALREADY returns normalized
    # context, and replay sim_ctx preserves it. Never normalize it a second time.
    real=o.OpenYAMEnv.obs;imagined=l.MetalWorldModelEnv.get_observations
    def obs(self):
        td=real(self);td['privileged_ctx']=self.context.read(self);return td
    def get_observations(self):
        td=imagined(self);td['privileged_ctx']=self.sim_ctx;return td
    o.OpenYAMEnv.obs=obs;l.MetalWorldModelEnv.get_observations=get_observations
    init=l.MetalWorldModelEnv.__init__
    def wm_init(self,*args,**kwargs):
        init(self,*args,**kwargs);self.max_step_reward=7.5/30
    l.MetalWorldModelEnv.__init__=wm_init
def install_policy(l):
    base=l.ActorCriticPolicyLearner
    source=textwrap.dedent(inspect.getsource(base._train_batch))
    changes={
        'obs = self.wm_env.reset(start)':'obs = self.wm_env.reset(start)\n  payload_loss, payload_dims, prediction = self.actor.payload_loss(obs["history"], obs["privileged_ctx"])',
        'loss = (-ret + aux_loss).mean()':'loss = (-ret + aux_loss).mean() + AUX_WEIGHT * payload_loss',
        '"return": ret.mean().detach(),':'"return": ret.mean().detach(),\n      "payload_mse": payload_loss.detach(),\n      "payload_weighted": AUX_WEIGHT * payload_loss.detach(),\n      **{f"payload_mse_{i}": payload_dims[i] for i in range(10)},'}
    for before,after in changes.items():
        assert source.count(before)==1;source=source.replace(before,after)
    filename=__file__+':causal_payload_train_batch';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(base._train_batch.__globals__,AUX_WEIGHT=AUX_WEIGHT)
    exec(compile(source,filename,'exec'),namespace)
    class PayloadPolicy(base):_train_batch=namespace['_train_batch']
    l.ActorCriticPolicyLearner=PayloadPolicy
    return PayloadPolicy
