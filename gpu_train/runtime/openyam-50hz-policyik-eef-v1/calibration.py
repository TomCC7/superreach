"""Native force-limited equilibrium windows, used only as WM supervision."""
import copy
import numpy as np
import torch
import mujoco as mj
from geometry import log_rotation, mul, inv

FRAMES = 33
PERTURB_FRAMES = 8
PERTURB_MIX = .05


def build_perturbations(e, E, count, held_out=False):
    """Paired native trajectories with mirrored excitation at the same loaded hold."""
    rng = np.random.default_rng(15503 if held_out else 51769)
    bank = e.bank
    clips = bank.dev_ids if held_out else bank.train_ids
    m = copy.copy(e.model)
    d = mj.MjData(m)
    kp = m.actuator_gainprm[:, 0].copy()
    sid = m.site('arm_tcp_site').id
    assert m.opt.timestep == .002 and not m.actuator_ctrllimited.any() and m.actuator_forcelimited.all()
    rows, actions, contexts, aux, meta = [], [], [], [], []
    home_q = e.q_nom.detach().cpu().double().numpy()
    unit = lambda: rng.normal(size=6) / np.sqrt(6.)
    with E.continuous.load_catalog() as cat:
        for _ in range(count * 24):
            i = len(rows)
            mi, home = (i // 2) % 5, i % 2 == 0
            rid = int(rng.choice(clips))
            tick = int(rng.integers(int(bank.lengths[rid]) - 301))
            q = home_q if home else bank.table[int(bank.offsets[rid])+tick, :6].detach().cpu().double().numpy()
            for key in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat'):
                getattr(m, key)[:] = cat[key][mi]
            mj.mj_setConst(m, d)
            mj.mj_resetData(m, d)
            d.qpos[:6], d.qpos[6:] = q, .055
            mj.mj_forward(m, d)
            lower, upper = m.jnt_range[:6].T
            if d.ncon or np.any(q < lower + .01) or np.any(q > upper - .01):
                continue
            u = q + (d.qfrc_bias[:6] - d.qfrc_passive[:6]) / kp
            d.ctrl[:] = u
            p = d.site_xpos[sid].copy()
            quat = np.empty(4)
            mj.mju_mat2Quat(quat, d.site_xmat[sid])
            mj.mj_step(m, d, nstep=10 * (PERTURB_FRAMES - 1))
            if np.max(np.abs(d.qpos[:6] - q)) > 1e-5 or np.max(np.abs(d.qvel[:6])) > 1e-4:
                continue
            # Separate the q, dq and u directions as well as exciting them jointly.
            mode = (i // 10 + i % 10) % 4
            dq = .002 * unit() if mode in (0, 3) else np.zeros(6)
            dv = .02 * unit() if mode in (1, 3) else np.zeros(6)
            hz = rng.uniform(2., 4.)
            phase = 2 * np.pi * hz * .02 * np.arange(PERTURB_FRAMES) + rng.uniform(0, 2*np.pi)
            du = .003 * np.sin(phase[:, None]) * unit() if mode in (2, 3) else np.zeros((PERTURB_FRAMES, 6))
            states, commands = [], []
            good = True
            for sign in (-1., 1.):
                mj.mj_resetData(m, d)
                d.qpos[:6], d.qpos[6:], d.qvel[:6] = q + sign*dq, .055, sign*dv
                cmd = u + sign*du
                seq = [np.r_[d.qpos[:6], d.qvel[:6]]]
                for target in cmd[:-1]:
                    d.ctrl[:] = target
                    for _ in range(10):
                        mj.mj_step(m, d)
                        good &= d.ncon == 0 and np.all(np.abs(d.actuator_force) < .99*m.actuator_forcerange[:, 1])
                    seq.append(np.r_[d.qpos[:6], d.qvel[:6]])
                states.append(seq)
                commands.append(cmd)
            if not good or not np.isfinite(states).all():
                continue
            body = e.payload_body
            R = np.empty(9)
            mj.mju_quat2Mat(R, m.body_iquat[body])
            R = R.reshape(3, 3)
            I = (R*m.body_inertia[body]) @ R.T
            raw = np.r_[m.body_mass[body], m.body_ipos[body], I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]]
            contexts.append((raw-np.r_[2., np.zeros(9)]) / np.r_[2., [.3]*3, [.1]*6])
            aux.append(np.r_[p, quat, 0, rid, tick*4096+2048])
            rows.append(states)
            actions.append(commands)
            meta.append((float(cat['masses'][mi]), home))
            if len(rows) == count:
                break
    if len(rows) != count:
        raise RuntimeError(f'accepted {len(rows)}/{count} equilibrium pairs')
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=e.device)
    return tuple(map(tensor, (rows, actions, contexts, aux))), dict(
        pairs=count, frames=PERTURB_FRAMES, payloads_kg=sorted({mass for mass, _ in meta}),
        home_pairs=sum(home for _, home in meta), frequency_hz=[2., 4.])


def perturbation_errors(wm, bank, ix):
    gt, u, ctx, aux = (x[ix] for x in bank)
    pred, _ = wm._rollout(gt[:, :, 0].flatten(0, 1), u.flatten(0, 1),
                          aux.repeat_interleave(2, 0), ctx.repeat_interleave(2, 0))
    error = pred.reshape_as(gt[:, :, 1:]) - gt[:, :, 1:]
    scaled = error / error.new_tensor([.002]*6 + [.02]*6)
    # The paired differential isolates response from common free-rollout drift.
    response = .5 * (scaled[:, 1] - scaled[:, 0])
    state = scaled.square().mean()
    response_mse = response.square().mean()
    return PERTURB_MIX * (.1*state + 10.*response_mse), state.detach(), response_mse.detach()


def build(e,E,count,held_out=False):
    rng=np.random.default_rng(93853 if held_out else 44210)
    bank=e.bank; clip_ids=bank.dev_ids if held_out else bank.train_ids
    m=copy.copy(e.model); d=mj.MjData(m); sid=m.site('arm_tcp_site').id
    kp=m.actuator_gainprm[:,0].copy()
    assert m.opt.timestep==.002 and (kp>0).all()
    rows=[]; actions=[]; contexts=[]; goals=[]; aux=[]; masses=[]; max_drift=0.
    with E.continuous.load_catalog() as cat:
        for _ in range(count*12):
            i=len(rows); mi=[0,4,4,1][i%4]
            rid=int(rng.choice(clip_ids)); tick=int(rng.integers(int(bank.lengths[rid])-301))
            q=bank.table[int(bank.offsets[rid])+tick,:6].detach().cpu().double().numpy()
            for key in ('body_mass','body_ipos','body_inertia','body_iquat'):getattr(m,key)[:]=cat[key][mi]
            mj.mj_setConst(m,d); mj.mj_resetData(m,d); d.qpos[:6]=q; d.qpos[6:]=.055
            mj.mj_forward(m,d)
            if d.ncon or np.any(q<m.jnt_range[:6,0]+.005) or np.any(q>m.jnt_range[:6,1]-.005):continue
            offset=d.qfrc_bias[:6]/kp
            if np.any(np.abs(kp*offset)>=m.actuator_forcerange[:,1]*.95):continue
            u=q+offset; d.ctrl[:]=u
            p=d.site_xpos[sid].copy(); r=np.empty(4); mj.mju_mat2Quat(r,d.site_xmat[sid]); goal=np.r_[p,r]
            states=[np.r_[q,d.qvel[:6]]]; drift=0.; rot=0.
            for k in range(32):
                mj.mj_step(m,d,nstep=10); mj.mj_forward(m,d)
                states.append(np.r_[d.qpos[:6],d.qvel[:6]])
                drift=max(drift,np.linalg.norm(d.site_xpos[sid]-p))
                rt=np.empty(4); mj.mju_mat2Quat(rt,d.site_xmat[sid]); rot=max(rot,2*np.arccos(min(abs(rt@r),1)))
            if drift>.001 or rot>np.deg2rad(.2) or np.max(np.abs(np.asarray(states)[:,6:]))>.02:continue
            b=e.payload_body; R=np.empty(9); mj.mju_quat2Mat(R,m.body_iquat[b]); R=R.reshape(3,3)
            I=(R*m.body_inertia[b])@R.T
            raw=np.r_[m.body_mass[b],m.body_ipos[b],I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]
            contexts.append((raw-np.r_[2.,np.zeros(9)])/np.r_[2.,[.3]*3,[.1]*6]); rows.append(states); actions.append(u); goals.append(goal); masses.append(float(cat['masses'][mi]))
            aux.append(np.r_[goal,0,rid,tick*4096+2048]); max_drift=max(max_drift,drift)
            if len(rows)==count:break
    assert len(rows)==count, len(rows)
    tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32,device=e.device)
    return tuple(map(tensor,(rows,actions,contexts,goals,aux))),dict(windows=count,frames=FRAMES,max_native_drift_mm=max_drift*1000,payloads_kg=masses)


def errors(wm,bank,ix):
    gt,u,ctx,goal,aux=(t[ix] for t in bank)
    pred,_=wm._rollout(gt[:,0],u[:,None].expand(-1,FRAMES,-1),aux,ctx)
    fk=wm.representation.fk
    p,r=fk(pred[...,:6]); p_gt,r_gt=fk(gt[:,1:,:6])
    pos=p-p_gt; rot=log_rotation(mul(inv(r_gt),r))
    pose_loss=(pos/.01).square().mean()+(rot/.10).square().mean()
    # Penalize terminal drift and residual velocity while preserving the measured
    # equilibrium. No static-mask gradients and no modification of dynamics at inference.
    eq=((p[:,-1]-goal[:,:3])/.01).square().mean()+(.1*(pred[...,6:]/.10).square().mean())
    eq=eq+(log_rotation(mul(inv(goal[:,3:7]),r[:,-1]))/.10).square().mean()
    return .05*pose_loss+.05*eq, pos.norm(dim=-1), rot.norm(dim=-1)
