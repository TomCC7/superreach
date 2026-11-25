"""Paired native held-out evaluations: identical cases, separate sensor transport."""
import numpy as np
import torch
from torch import nn
from geometry import inv, log_rotation, mul


def joint_step_metrics(case, measured_q, tcp, ticks, home):
    """Absolute joint angles; score after the STEP at tick 75 only."""
    if case['kind'] != 'STEP':
        return {}
    target, start = case['target'][:6], case['q']
    q = np.asarray(measured_q)[75:]
    xyz = np.asarray(tcp)[75:]
    if len(q)==0 or not np.isfinite(q).all() or not np.isfinite(xyz).all():
        return dict(joint_overshoot_max_by_joint_rad=None, joint_off_home_tail_2s=None)
    direction = np.sign(target-start)
    overshoot = np.maximum((q-target)*direction,0.).max(0)
    tail = None
    if len(q)>=100 and not np.allclose(target,home,atol=1e-6,rtol=0):
        q_span = np.ptp(q[-100:],axis=0)
        tail = dict(ticks=100, start_tick=ticks-100, end_tick_exclusive=ticks,
                    tcp_peak_to_peak_mm=1000*float(np.linalg.norm(np.ptp(xyz[-100:],axis=0))),
                    joint_peak_to_peak_by_joint_rad=q_span.tolist(),
                    joint_peak_to_peak_max_rad=float(q_span.max()))
    return dict(joint_overshoot_max_by_joint_rad=[float(v) if s else None for v,s in zip(overshoot,direction)],
                joint_off_home_tail_2s=tail)


class Measurement(nn.Module):
    def __init__(self, actor, noisy, task):
        super().__init__()
        self.actor = actor
        self.noisy = noisy
        self.task = task
        self.rng = torch.Generator().manual_seed(937701)
        self.hist = None

    def forward(self, obs):
        h, g = obs['history'],obs['goal_pose']
        if self.noisy:
            from train import observed
            if self.hist is None:
                self.dr = h.new_zeros(len(h),7)
                self.dr[:,:6] = .0015*torch.randn((len(h),6),generator=self.rng)
                row = observed(h[:,-1],self.dr,self.rng)
                self.hist = torch.cat((row[:,None,:18].expand(-1,16,-1),h[...,18:]),-1)
            else:
                self.hist = torch.cat((self.hist[:,1:],observed(h[:,-1],self.dr,self.rng)[:,None]),1)
            h = self.hist
        q=obs.get('goal_q',h.new_zeros(len(h),6))
        if self.task==2:
            p,r=self.actor.fk(q); g=torch.cat((p,r),-1)
        return self.actor.rollout(h,g,q,torch.full((len(h),),self.task,dtype=torch.long,device=h.device))


def evaluate(b,l,out,n):
    E,patch = b.T.E,b.A.patch
    p = __import__('pathlib').Path(__file__).resolve().parent
    native = E.load('_unified_native_eval',p.parents[1]/'openyam-50hz-supermimic-6d-v1/offline_eval.py')
    extended = E.load('_unified_extended_eval',p.parent/'offline_eval.py')
    result = dict(actor_updates=n,source_digest=b.eef_source_digest)
    for mode, delay in [('clean',0),('noisy-10ms',5)]:
      result[mode]={}
      for task,label in enumerate(('EEF_6D','XYZ_ONLY','JOINT_6D')):
        def capture(actor):
            return Measurement(actor,bool(delay),task)
        run = patch(native.evaluate,[
            ('policy=copy.deepcopy(actor).cpu().eval()', 'policy=capture(copy.deepcopy(actor).cpu().eval())'),
            ('dev=np.asarray(bank.dev_ids)', 'dev=np.asarray(bank.dev_ids); lengths=bank.lengths.cpu().numpy()'),
            ('int(bank.lengths[rid])','int(lengths[rid])'),

            ("'goal_motion':","'goal_pose':"),
            ("'goal_pose':torch.tensor(np.array(goals),dtype=torch.float32)","'goal_pose':torch.tensor(np.array(goals)[:,:7],dtype=torch.float32),'goal_q':torch.tensor(np.array(goals)[:,7:13],dtype=torch.float32)"),
            ('d.ctrl[:]=command[i]',  'previous=history[i,-1,12:18].copy()'),
            ('for _ in range(10):\n                    mj.mj_step', 'for sub in range(10):\n                    d.ctrl[:]=previous if sub<delay else command[i];mj.mj_step'),
            ('position=[],velocity=[]','position=[],orientation=[],joint=[],wrist=[],commands=[],velocity=[],joint_q=[],tcp_xyz=[]'),
            ("p=d.site_xpos[sid];r['position'].append", "p=d.site_xpos[sid]\n                if task==2: r['joint_q'].append(d.qpos[:6].copy());r['tcp_xyz'].append(p.copy())\n                r['commands'].append(command[i].copy());r['joint'].append(float(np.linalg.norm(d.qpos[:6]-refs[i][:6])));r['wrist'].append(d.qpos[3:6].copy());r['orientation'].append(float(np.degrees(2*np.arccos(np.clip(abs(matrix_quat(d.site_xmat[sid])@refs[i][15:19]),0,1)))));r['position'].append"),
            ('position_error_final_m=', "joint_error_mean_rad=float(np.mean(r['joint'])),joint_error_p95_rad=float(np.percentile(r['joint'],95)),joint_error_mean_deg=float(np.degrees(np.mean(r['joint']))),joint_error_p95_deg=float(np.degrees(np.percentile(r['joint'],95))),tcp_error_mean_mm=1000*float(np.mean(r['position'])),tcp_error_p95_mm=1000*float(np.percentile(r['position'],95)),wrist_offset_rms_rad=float(np.sqrt(np.mean(np.square(r['wrist'])))),position_error_p95_m=float(np.percentile(r['position'],95)),orientation_error_p95_deg=float(np.percentile(r['orientation'],95)),command_slew_rms_rad_s=float(np.sqrt(np.mean((np.diff(r['commands'],axis=0)/.02)**2))),joint_accel_rms_rad_s2=float(np.sqrt(np.mean((np.diff(r['velocity'],axis=0)/.02)**2))),position_error_final_m="),
            ('    payload=dict(completed=True', "    if task==2:\n        for c,r,result in zip(cases,records,results,strict=True):\n            result.update(joint_step_metrics(c,r['joint_q'],r['tcp_xyz'],ticks,HOME))\n    payload=dict(completed=True"),
            ('E.base.atomic(Path(path),payload);return payload','return payload'),
        ],'unified-standard-eval',capture=capture,delay=delay,task=task,joint_step_metrics=joint_step_metrics)
        holds = patch(extended.evaluate_holds,[
            ('policy = copy.deepcopy(actor).cpu().eval()', 'policy = capture(copy.deepcopy(actor).cpu().eval())'),
            ('d.ctrl[:] = cmd[i]\n            for _ in range(10):\n                mj.mj_step', 'previous = history[i,-1,12:18].copy()\n            for sub in range(10):\n                d.ctrl[:] = previous if sub<delay else cmd[i]\n                mj.mj_step'),
        ],'unified-hold-eval',capture=capture,delay=delay)
        data = run(l.actor,l.e.bank,E,None)
        if task!=2:
            data['off_home_hold'] = holds(l.actor,l.e.bank,E,native.matrix_quat)
            assert len(data['off_home_hold']['cases'])==6
        assert len(data['cases'])==13
        result[mode][label]=data
    b.T.atomic(out/f'eval-{n}.json',result)
    return result


def convergence(out):
    import json
    def get(i):
        return json.loads((out/f'eval-{i}.json').read_text())
    before,after = get(0),get(200)
    per = {}
    for mode in ('clean','noisy-10ms'):
        a,b = before[mode]['EEF_6D']['cases'],after[mode]['EEF_6D']['cases']
        for kind in ('TRACK','STEP','HOME'):
            x,y = ([r for r in rows if r['kind']==kind] for rows in (a,b))
            p0,p1 = (float(np.mean([r['position_error_mean_m'] for r in rows])) for rows in (x,y))
            per[f'{mode}/{kind}'] = dict(before_m=p0,after_m=p1,failures=sum(r['physical_failure'] for r in y),
                                       max_command_slew=max(r['command_slew_rms_rad_s'] for r in y),
                                       max_orientation_p95_deg=max(r['orientation_error_p95_deg'] for r in y))
    passed = all(v['failures']==0 and np.isfinite(v['max_command_slew']) for v in per.values()) and all(
        np.mean([per[f'{m}/{k}']['after_m'] for k in ('TRACK','STEP','HOME')]) <
        np.mean([per[f'{m}/{k}']['before_m'] for k in ('TRACK','STEP','HOME')]) for m in ('clean','noisy-10ms'))
    return dict(passed=bool(passed), metrics=per)
