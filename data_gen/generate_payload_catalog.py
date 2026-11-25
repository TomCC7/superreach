"""Deterministic MuJoCo-exact inertial catalog; canonical rows are immutable copies."""
import argparse
from pathlib import Path
import hashlib, json

N=2048; SEED=20261113
OFFSET=(-.00201655,-.00024609,.0897254)
FIELDS=['body_mass','body_ipos','body_inertia','body_iquat','body_subtreemass','body_invweight0','dof_M0','dof_invweight0','actuator_acc0']
def variant(scene, mass, dc):
    import numpy as np
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(scene)); b=m.body('hand').id
    if mass:
        base=m.body_mass[b]; c=m.body_ipos[b].copy(); p=OFFSET+dc
        center=(base*c+mass*p)/(base+mass)
        rot=np.empty(9);mujoco.mju_quat2Mat(rot,m.body_iquat[b]);rot=rot.reshape(3,3)
        inertia=rot@np.diag(m.body_inertia[b])@rot.T + np.eye(3)*(.4*mass*.02**2)
        for pos,weight in [(c,base),(p,mass)]:
            v=pos-center;inertia+=weight*(v@v*np.eye(3)-np.outer(v,v))
        val,vec=np.linalg.eigh(inertia)
        if np.linalg.det(vec)<0:vec[:,0]*=-1
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,vec.ravel())
        assert np.all(val>0) and 2*val.max()<=val.sum()+1e-12
        m.body_mass[b]=base+mass;m.body_ipos[b]=center;m.body_inertia[b]=val;m.body_iquat[b]=quat
    d=mujoco.MjData(m);mujoco.mj_setConst(m,d);mujoco.mj_forward(m,d)
    return {**{k:getattr(m,k).copy() for k in FIELDS},'meaninertia':np.array(m.stat.meaninertia)}
def build(scene, canonical_path):
    import numpy as np
    rng=np.random.default_rng(SEED)
    masses=np.r_[np.arange(5)*.5,[.25,.75,1.25,1.75],rng.uniform(0,2,N-9)]
    offsets=np.zeros((N,3));offsets[9:]=rng.uniform(-.02,.02,(N-9,3))
    with np.load(canonical_path, allow_pickle=False) as archive:
        canonical={k: archive[k][:5].copy() for k in FIELDS+['meaninertia']}
    data={k:[] for k in canonical}
    for i in range(N):
        row={k:canonical[k][i] for k in data} if i<5 else variant(scene,masses[i],offsets[i])
        for k in data:data[k].append(row[k])
    return {**{k:np.array(v) for k,v in data.items()},'masses':masses,'com_offsets':offsets,'seed':np.array(SEED)}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if hashlib.sha256(args.canonical.read_bytes()).hexdigest() != '3bde4d5c82c4c948cacc7dd5853b23c391c480cb0bead4c0c68222bf99c36964':
        parser.error('Canonical catalog does not match the registered five-row input')
    import numpy as np
    data = build(args.scene, args.canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        np.savez_compressed(output, **data)
    print(json.dumps(dict(entries=N, seed=SEED, output=str(args.output),
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
