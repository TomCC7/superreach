"""Issued-target second-difference overlay. No persistent numerical state.

PRESTEP ring at tick t: [-1] is u[t-1], [-2] is u[t-2].
Tick 0 reset sensor is NOT an issued action; first two actions are masked.
Native WM step calls _wm_step before shifting/synthesizing the same ring.
"""
import inspect
import torch

COEFFICIENT = .01
SCALE = .1
DT = 1 / 30


def command_curvature(sent, history, tick):
    previous = history[..., -1, 12:18]
    previous2 = history[..., -2, 12:18]
    # Require same episode causal timestamps as well as explicit padding bits.
    valid = ((tick >= 2) & (history[..., -1, 18] > .5)
             & (history[..., -2, 18] > .5)
             & torch.isclose(history[..., -1, 19], tick * DT, atol=2e-6, rtol=1e-5)
             & torch.isclose(history[..., -2, 19], (tick - 1) * DT, atol=2e-6, rtol=1e-5))
    # Premask operands, not a posthoc 0*potentially invalid old episode spike.
    delta = torch.where(valid[..., None], sent - 2 * previous + previous2,
                        torch.zeros_like(sent)) / SCALE
    cost = -COEFFICIENT * delta.square().mean(-1) * DT
    return cost, delta, valid


def install(openyam, learning):
    assert openyam.DT == DT
    old_real = openyam.OpenYAMEnv.step
    old_wm = learning.MetalWorldModelEnv._wm_step
    assert not getattr(old_real, '_curvature001', False), 'overlay installed twice'

    @torch.no_grad()
    def real_step(self, action, auto_reset=True):
        # Read before sent.copy_, tick++, ring shift and possible terminal reset.
        cost, delta, valid = command_curvature(self.decode(action), self.history, self.tick)
        obs, reward, done, last = old_real(self, action, auto_reset)
        last.update(command_curvature_cost=cost, command_curvature_normalized_d2=delta,
                    command_curvature_valid=valid)
        self.curvature_real_log = dict(command_curvature_cost=float(cost.mean()),
                                      command_curvature_valid_fraction=float(valid.float().mean()),
                                      command_second_difference_rms_rad=float((delta * SCALE).square().mean().sqrt()))
        return obs, reward + cost, done, last

    def wm_step(self, rep, aux, action, last_action, ctx):
        sent = action
        cost, delta, valid = command_curvature(sent, self._sensor_history(), aux[:, 7])
        result = old_wm(self, rep, aux, action, last_action, ctx)
        if getattr(self, 'curvature_measure_requested', False) and action.requires_grad:
            self.curvature_measure_requested = False
            # Diagnostics only: same ordinary imagined first tick, no optimizer work.
            # No cache relabeling: actor_update consumes/deletes this fresh receipt.
            try:
                previous = self._sensor_history()[:, -1, 12:18]
                _, terms, _, _ = learning.reward(self.representation.fk, rep[:, :6], result[1][:, :6], result[0][:, :7], sent, previous)
                position = terms['position'] * DT
                params = [p for p in self.gradient_actor.parameters() if p.requires_grad]
                def gradients(value):
                    return torch.autograd.grad(value.mean(), params, retain_graph=True, allow_unused=True)
                gc, gp = gradients(cost), gradients(position)
                def norm(gs):
                    return torch.sqrt(sum((g.square().sum() for g in gs if g is not None), cost.new_tensor(0.)))
                nc, np = norm(gc), norm(gp)
                self.curvature_gradient_receipt = dict(
                    kind='ordinary same-batch first imagined tick actor-parameter gradient; NOT full BPTT',
                    actor_update=self.curvature_measure_update, batch_size=action.shape[0],
                    coefficient=COEFFICIENT, curvature_mean=float(cost.detach().mean()),
                    position_mean=float(position.detach().mean()),
                    normalized_d2_rms=float(delta.detach().square().mean().sqrt()),
                    valid_fraction=float(valid.float().mean()),
                    curvature_actor_gradient_l2=float(nc.detach()),
                    position_actor_gradient_l2=float(np.detach()),
                    curvature_to_position_gradient_ratio=float((nc / (np + 1e-30)).detach()))
            except Exception as exc:
                self.curvature_gradient_receipt = dict(actor_update=self.curvature_measure_update,
                                                       diagnostic_error=repr(exc))
        return (*result[:4], result[4] + cost)

    real_step._curvature001 = True
    wm_step._curvature001 = True
    openyam.OpenYAMEnv.step = real_step
    learning.MetalWorldModelEnv._wm_step = wm_step
    old_collect = learning.Learner.collect
    def collect(self):
        logs = old_collect(self)
        logs.update(self.e.curvature_real_log)
        return logs
    learning.Learner.collect = collect
    old_update = learning.Learner.actor_update
    def actor_update(self):
        wm = self.wm_env
        if hasattr(wm, 'curvature_gradient_receipt'):
            del wm.curvature_gradient_receipt
        # First ordinary update after restore plus each 100 thereafter.
        wm.curvature_measure_requested = (not hasattr(wm, 'curvature_measure_update')
                                         or self.count['actor_updates'] % 100 == 0)
        wm.curvature_measure_update = self.count['actor_updates'] + 1
        logs = old_update(self)
        if hasattr(wm, 'curvature_gradient_receipt'):
            logs['same_batch_curvature_actor_gradient'] = wm.curvature_gradient_receipt
        return logs
    learning.Learner.actor_update = actor_update
    assert inspect.unwrap(openyam.OpenYAMEnv.step).__globals__['command_curvature'] is command_curvature
    assert learning.MetalWorldModelEnv._wm_step.__globals__['command_curvature'] is command_curvature
    return dict(real_original=old_real, imagined_original=old_wm,
                real=real_step, imagined=wm_step)
