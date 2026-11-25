"""Original smoothness coefficients RESTORED; exact pinned full-state continuation."""
from pathlib import Path
import hashlib,json,sys,inspect,textwrap,copy
_FRESH_P=Path(__file__).resolve().parent
_FRESH_R=_FRESH_P.parent
_OLD=_FRESH_R/'openyam-peaktorque-softpenalty-fresh-v1'
sys.path.insert(0,str(_OLD));sys.path.insert(0,str(_FRESH_P))
import openyam,learning,core,state,contract
_FRESH_KP=[50.,60.,50.,25.,15.,10.]
_FRESH_KD=[3.,4.,3.,2.,1.,.8]

def configure_model(model):
 import numpy as np
 assert model.nu==6 and model.nv==8
 assert np.array_equal(model.actuator_trnid[:,0],np.arange(6))
 assert np.allclose(model.actuator_gear[:,0],1)
 model.actuator_gainprm[:]=0;model.actuator_biasprm[:]=0
 model.actuator_gainprm[:,0]=_FRESH_KP
 model.actuator_biasprm[:,1]=-np.array(_FRESH_KP)
 model.actuator_biasprm[:,2]=-np.array(_FRESH_KD)
 assert np.allclose(model.dof_armature[:6],.005) and np.allclose(model.dof_damping[:6],.05)
 assert np.all(model.body_gravcomp==0)
 return model
from policy_commands import install as _install_policy_commands
_POLICY_COMMAND_CONTRACT = _install_policy_commands(openyam, learning)
# Change gains in host model BEFORE the actual native Simulation constructor/upload.
_init_source=textwrap.dedent(inspect.getsource(openyam.OpenYAMEnv.__init__))
_needle='self.sim=Simulation('
assert _init_source.count(_needle)==1
_init_source=_init_source.replace(_needle,'configure_model(self.model)\n '+_needle)
_init_globals=dict(openyam.OpenYAMEnv.__init__.__globals__,configure_model=configure_model)
exec(compile(_init_source,str(_FRESH_P/'compiled_constructor.py'),'exec'),_init_globals)
openyam.OpenYAMEnv.__init__=_init_globals['__init__']
# The SAME original effort helper reads globals from openyam, including imagined alias.
openyam.KP=_FRESH_KP;openyam.KD=_FRESH_KD;learning.KP=_FRESH_KP;learning.KD=_FRESH_KD
assert learning.effort_cost is openyam.effort_cost
# Restore the original actual base slew coefficient; preserve every other term.
_old_reward=openyam.reward
_reward_source=textwrap.dedent(inspect.getsource(_old_reward))
assert _reward_source.count('command_slew=-.002*velocity.square().mean(-1)')==1
_reward_source=_reward_source.replace('command_slew=-.002*velocity.square().mean(-1)', 'command_slew=-.004*velocity.square().mean(-1)')
_reward_globals=dict(_old_reward.__globals__)
exec(compile(_reward_source,str(_FRESH_P/'compiled_reward.py'),'exec'),_reward_globals)
reward_restored=_reward_globals['reward']
openyam.reward=learning.reward=reward_restored
assert inspect.unwrap(openyam.OpenYAMEnv.step).__globals__['reward'] is reward_restored
assert learning.MetalWorldModelEnv._wm_step.__globals__['reward'] is reward_restored
import importlib.util as _importutil
_spec=_importutil.spec_from_file_location('_policyonly_resmooth_curvature',_FRESH_P/'curvature001_hooks.py')
_policy_curvature=_importutil.module_from_spec(_spec);_spec.loader.exec_module(_policy_curvature)
sys.modules['curvature001_hooks']=_policy_curvature
bindings=_policy_curvature.install(openyam,learning)
from replay_bootstrap import install as _install_replay_bootstrap
_install_replay_bootstrap(learning)
