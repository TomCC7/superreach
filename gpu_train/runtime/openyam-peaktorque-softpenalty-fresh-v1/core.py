"""Add physical per-world tool inertias without changing the aligned learner."""
from pathlib import Path
import sys,types,copy,json,hashlib
P=Path(__file__).resolve().parent;R=P.parent
sys.path.insert(0,str(P))
import state as base
from state import *
base.OUT=R/'output/openyam-peaktorque-softpenalty-fresh-v1'
OUT=base.OUT
identity=base.identity
FIELDS=['body_mass','body_ipos','body_inertia','body_iquat','body_subtreemass','body_invweight0','dof_invweight0','actuator_acc0']

def physical(e):
 return cpu(dict(fields={k:getattr(e.sim.model,k) for k in FIELDS},meaninertia=e.meaninertia))
def extension(e):
 return cpu(dict(stage_sha=identity(),mass_indices=e.mass_indices,mass_rng=e.mass_rng.get_state(),enabled=e.payload_enabled,episode_counts=e.mass_episode_counts,transitions=e.mass_real_transitions,physical=physical(e)))
def snapshot(l):return dict(base.snapshot(l),payload=extension(l.e))
