"""Continuous reset sampling; retain five-bin checkpoint accounting."""
import os
import types
import numpy as np
import torch
# dof_M0 is also retained in the exact catalog; Warp derives its runtime matrix.
FIELDS=['body_mass','body_ipos','body_inertia','body_iquat','body_subtreemass','body_invweight0','dof_invweight0','actuator_acc0']
def load_catalog():return np.load(os.environ['SUPERREACH_PAYLOAD'])
def mass_bins(values):
    return torch.bucketize(values,values.new_tensor([.25,.75,1.25,1.75]),right=True)
def sample_indices(n,device,generator,count):
    ix=torch.randint(9,count,(n,),device=device,generator=generator)
    return torch.where(torch.rand(n,device=device,generator=generator)<.1,torch.zeros_like(ix),ix)
def install(e):
    variants=load_catalog()
    e.payload_variants={k:torch.tensor(variants[k],device=e.device,dtype=torch.float32) for k in FIELDS}
    e.payload_means=torch.tensor(variants['meaninertia'],device=e.device,dtype=torch.float32)
    e.mass_values=torch.tensor(variants['masses'],device=e.device,dtype=torch.float32)
    e.mass_indices=torch.zeros(e.num_envs,dtype=torch.long,device=e.device)
    e.mass_rng=torch.Generator(device=e.device).manual_seed(20261113)
    e.payload_enabled=False;e.eval_mass_index=None
    e.mass_episode_counts=torch.zeros(5,dtype=torch.long,device=e.device)
    e.mass_real_transitions=torch.zeros_like(e.mass_episode_counts)
    e.sim.expand_model_fields(FIELDS)
    if torch.device(e.device).type=='cpu':
        # CPU MuJoCo verification facade only; production uses Warp constants.
        e.meaninertia=torch.zeros(e.num_envs,device=e.device)
    else:
        import warp as wp
        e.sim.wp_model.stat.meaninertia=wp.array(np.repeat(e.model.stat.meaninertia,e.num_envs).astype(np.float32),device=e.device)
        e.meaninertia=wp.to_torch(e.sim.wp_model.stat.meaninertia)
    e.sim.create_graph()
    def apply(ids):
        ix=e.mass_indices[ids]
        for k in FIELDS:getattr(e.sim.model,k)[ids]=e.payload_variants[k][ix]
        e.meaninertia[ids]=e.payload_means[ix]
    e.apply_payload=apply
    oldreset=e.reset
    def reset(this,ids,ref_ids=None,start_ticks=None):
        if not len(ids):return
        if this.eval_mass_index is not None:
            assert 0<=this.eval_mass_index<9
            ix=torch.full((len(ids),),this.eval_mass_index,dtype=torch.long,device=this.device)
        elif this.payload_enabled:ix=sample_indices(len(ids),this.device,this.mass_rng,len(this.mass_values))
        else:ix=torch.zeros(len(ids),dtype=torch.long,device=this.device)
        this.mass_indices[ids]=ix;this.apply_payload(ids)
        if ref_ids is None:this.mass_episode_counts+=torch.bincount(mass_bins(this.mass_values[ix]),minlength=5)
        return oldreset(ids,ref_ids,start_ticks)
    e.reset=types.MethodType(reset,e)
    oldstep=e.step
    def step(this,action,auto_reset=True):
        before=this.mass_indices.clone()
        this.mass_real_transitions+=torch.bincount(mass_bins(this.mass_values[before]),minlength=5)
        ans=oldstep(action,auto_reset);ans[3]['external_mass_kg']=this.mass_values[before].clone()
        if not auto_reset:assert torch.equal(before,this.mass_indices)
        return ans
    e.step=types.MethodType(step,e);e.apply_payload(e.ids)
    return e
