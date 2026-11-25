"""Pure 7D EEF command ABI; measured joint history is proprioception."""
import math
import torch
from torch import nn
from geometry import canonical, inv, log_rotation, mul


class RunningMeanStd(nn.Module):
    """Population moments from real observations, never from WM rollouts."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('variance', torch.ones(width))
        self.register_buffer('count', torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def update(self, x):
        variance, mean = torch.var_mean(x, dim=0, correction=0)
        self.count.add_(x.shape[0])
        weight = x.shape[0] / self.count.to(x.dtype)
        delta = mean - self.mean
        self.variance.copy_((1-weight)*self.variance + weight*variance + weight*(1-weight)*delta.square())
        self.mean.add_(weight*delta)

    def forward(self, x):
        # A std floor bounds sensitivity even on nearly constant HOLD features.
        return ((x-self.mean)*self.variance.clamp_min(.25**2).rsqrt()).clamp(-5., 5.)


class Actor(nn.Module):
    def __init__(self, fk, cfg, normalizer, mlp):
        super().__init__()
        self.fk, self.cfg = fk, cfg
        self.obs_normalizer = RunningMeanStd(18).to(fk.q0) if cfg.clean_obs else normalizer
        self.goal_normalizers = nn.ModuleDict()  # compatibility with upstream learner
        self.history_encoder = mlp(304,(128,64),10)
        fk_dim = (28 if cfg.fk_mode == 'raw' else 34) if cfg.fk_feedback else 0
        if cfg.clean_obs:
            # 7D goal + 7D TCP + dp/dr + command offset + 10D inferred context.
            self.goal_normalizers['clean'] = RunningMeanStd(36).to(fk.q0)
            fk_dim = 19
        self.decoder = mlp(18+7+10+fk_dim,(256,256),6)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        if cfg.command == 'direct':
            with torch.no_grad():
                self.decoder[-1].bias.copy_(torch.tensor([0.,1.047,1.047,0.,0.,0.]))

    def normalized_history(self,h):
        if self.cfg.clean_obs:
            valid = h[..., 18:19]
            x = self.obs_normalizer(torch.where(valid.bool(), h[..., :18], 0.))
            return torch.cat((torch.where(valid.bool(), x, 0.), valid), -1)
        return self.obs_normalizer(h)[..., :19]

    def clean_features(self, history, goal_pose, h):
        q = history[:, -1, :6]
        goal = torch.cat((goal_pose[:, :3], canonical(goal_pose[:, 3:])), -1)
        p, r = self.fk(q)
        dp = goal[:, :3]-p
        dr = log_rotation(mul(inv(r), goal[:, 3:]))
        ctx = self.history_encoder(h.flatten(1))
        return torch.cat((goal, p, r, dp, dr, history[:, -1, 12:18]-q, ctx), -1)

    def payload_loss(self, history, target):
        z = self.history_encoder(self.normalized_history(history).flatten(1))
        dims = (z-target.detach()).square().mean(0)
        return dims.mean(),dims.detach(),z.detach()

    @torch.no_grad()
    def update_normalization(self,obs):
        history = obs['history']
        if self.cfg.clean_obs:
            self.obs_normalizer.update(history[:, -1, :18])
            self.goal_normalizers['clean'].update(self.clean_features(history, obs['goal_pose'], self.normalized_history(history)))
        else:
            self.obs_normalizer.update(history[:, -1])

    def command(self, obs, noise=None):
        return self.rollout(obs['history'],obs['goal_pose'],noise)

    def rollout(self, history, goal_pose, noise=None):
        """No bank, clock, target joints, privileged context or external IK."""
        h = self.normalized_history(history)
        if self.cfg.clean_obs:
            x = self.goal_normalizers['clean'](self.clean_features(history, goal_pose, h))
            a = self.decoder(torch.cat((h[:, -1, :18], x), -1))
            self._aux_loss = a.new_zeros(a.shape[0])
            return a if noise is None else a+noise
        q, u_prev = history[:,-1,:6], history[:,-1,12:18]
        goal = torch.cat((goal_pose[:,:3],canonical(goal_pose[:,3:7])),-1)
        assert goal_pose.shape[-1] == 7
        parts = [h[:,-1,:18],goal,self.history_encoder(h.flatten(1))]
        if self.cfg.fk_feedback or self.cfg.command == 'accumulate':
            p,r,dp,dr,kp = self.fk.residuals(q,goal)
        if self.cfg.fk_feedback:
            errors = (dp,dr) if self.cfg.fk_mode == 'raw' else (dp/.01,dr/math.radians(20),dp/.001,dr/math.radians(.25))
            parts.extend((p,r,*errors,kp,u_prev-q))
        a = self.decoder(torch.cat(parts,-1))
        self._aux_loss = a.new_zeros(a.shape[0])
        if noise is not None:
            a = a+noise
        if self.cfg.command == 'direct':
            return a  # issued absolute radians; no anchor at inference
        rho = torch.maximum(dp.norm(dim=-1),.10*dr.norm(dim=-1))
        beta = .03+.97*torch.tanh(rho/.03)
        return u_prev + self.cfg.eta*beta[:,None]*a.tanh()

    def previous_command_jacobian(self, obs):
        """Total local derivative, holding other replay observations fixed."""
        h = obs['history'].detach()
        prev = h[:, -1:, 12:18].requires_grad_()
        latest = torch.cat((h[:, -1:, :12], prev, h[:, -1:, 18:]), -1)
        u = self.rollout(torch.cat((h[:, :-1], latest), 1), obs['goal_pose'].detach())
        # Actor is batch-separable with its running normalizer in eval mode.
        return torch.stack([torch.autograd.grad(col.sum(), prev, create_graph=True)[0][:, 0]
                            for col in u.unbind(-1)], dim=1)

    def forward(self,obs):
        return self.command(obs)

    def get_aux_loss(self):
        return self._aux_loss
