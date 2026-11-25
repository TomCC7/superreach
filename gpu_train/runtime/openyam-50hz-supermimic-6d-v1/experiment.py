"""50Hz upstream learner with a registered fresh 4096-world pure-position run."""
from pathlib import Path
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
import sys,json,os,time,signal,fcntl,argparse,shutil
P=Path(__file__).resolve().parent;R=P.parent
MAX_UPDATES=25_000
SAVE_EVERY=1_000
WARMUP_TICKS=16
sys.path.insert(0,str(P))
import trainer as T
# Parent imports reorder sys.path: load own files by absolute identity.
A=T.E.load('_sixd_alignment',P/'alignment.py');task=T.E.load('_sixd_task',P/'task.py')
import torch,numpy as np

def setup():
    T.prepare('separate')
    T.E.o.KP=T.E.learning.KP=task.KP.copy();T.E.o.KD=T.E.learning.KD=task.KD.copy()
    T.E.old.configure_model.__globals__['_FRESH_KP']=task.KP.copy()
    T.E.old.configure_model.__globals__['_FRESH_KD']=task.KD.copy()
    A.install(T);task.install(T,A)
    # configure() imports alignment; pin exact local module despite parent paths.
    sys.modules['alignment']=A

def cpu_environment(n):
    sys.path.insert(0,str(R/'openyam-mit-anchored-v1'));import cpu_fixture
    if not getattr(cpu_fixture.CPUSim,'step_recipe',False):
        original=cpu_fixture.CPUSim
        class Sim(original):
            step_recipe=True
            def __init__(self,n,cfg,model,device):
                assert model.opt.timestep==.002 and cfg.mujoco.timestep==.002
                super().__init__(n,cfg,model,device)
                model.opt.timestep=.002
                for m in self.models:m.opt.timestep=.002
        cpu_fixture.CPUSim=Sim
    return cpu_fixture.environment(n)
