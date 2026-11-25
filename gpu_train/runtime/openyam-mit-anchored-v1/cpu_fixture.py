"""CPU facade with independent real MuJoCo model/data per physical world."""
import copy,types
import numpy as np
import torch
import mujoco as mj
import entry as E
class Proxy:
    def __init__(self,x):self._tensor=x
    def __getitem__(self,k):return self._tensor[k]
    def __setitem__(self,k,v):self._tensor[k]=v
class CPUSim:
    def __init__(self,n,cfg,model,device):
        assert device=='cpu';self.n=n;self.m=model;model.opt.timestep=1/480
        self.models=[copy.copy(model) for _ in range(n)];self.md=[mj.MjData(m) for m in self.models]
        self.data=types.SimpleNamespace(**{k:Proxy(torch.tensor(np.stack([np.array(getattr(d,k),copy=True) for d in self.md]),dtype=torch.float32)) for k in E.base.SFIELDS+['actuator_force']})
        self.model=types.SimpleNamespace(**{k:torch.tensor(np.array(getattr(model,k)),dtype=torch.float32) for k in E.core.FIELDS})
        self.expanded=False
    def expand_model_fields(self,fields):
        assert not self.expanded
        for k in fields:setattr(self.model,k,getattr(self.model,k).expand(self.n,*getattr(self.model,k).shape).clone())
        self.expanded=True
    def create_graph(self):pass
    def reset(self,ids):
        for i in ids.tolist():
            mj.mj_resetData(self.models[i],self.md[i]);self._pull(i,E.base.SFIELDS)
    def _push(self,i):
        m=self.models[i];d=self.md[i]
        for k in E.core.FIELDS:
            value=getattr(self.model,k)
            getattr(m,k)[:]=(value[i] if self.expanded else value).numpy()
        for k in E.base.SFIELDS:
            if k=='time':d.time=float(getattr(self.data,k)._tensor[i])
            else:getattr(d,k)[:]=getattr(self.data,k)._tensor[i].numpy()
    def _pull(self,i,fields):
        for k in fields:getattr(self.data,k)._tensor[i]=torch.tensor(np.array(getattr(self.md[i],k)))
    def forward(self):
        for i,d in enumerate(self.md):
            self._push(i);mj.mj_forward(self.models[i],d);self._pull(i,E.base.SFIELDS+['actuator_force'])
    def step(self):
        for i,d in enumerate(self.md):
            self._push(i);mj.mj_step(self.models[i],d);self._pull(i,E.base.SFIELDS+['actuator_force'])
def environment(n=4):
    from unittest.mock import patch
    import mjlab.sim
    with patch.object(mjlab.sim,'Simulation',CPUSim):e=E.o.OpenYAMEnv(n,'cpu')
    return E.core.install(e)
