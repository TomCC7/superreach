"""Causal tri-mode goal conditioning to one absolute 6D joint command."""
import torch
from torch import nn
from geometry import canonical, inv, log_rotation, mul
from policy import RunningMeanStd

MASKS = ((1,1,0),(1,0,0),(1,1,1))

HOME = (0., 1.047, 1.047, 0., 0., 0.)


class Actor(nn.Module):
    def __init__(self, fk, limits, mlp, arch):
        super().__init__()
        self.fk = fk
        self.arch = arch
        self.register_buffer("masks", torch.tensor(MASKS, dtype=torch.float32))
        self.obs_normalizer = RunningMeanStd(18)
        self.goal_normalizers = nn.ModuleDict()  # fixed physical scaling, no amplification
        self.history_encoder = mlp(16*19, (128, 64), 10)
        # Match initial history and shared decoder weights wherever shapes overlap.
        self.decoder = mlp(43, (256,256), 6)
        if arch=='dual_encoder':
            with torch.random.fork_rng(devices=[]):
                wide=mlp(60,(256,256),6)
                with torch.no_grad():
                    wide[0].weight[:,:28].copy_(self.decoder[0].weight[:,:28])
                    wide[0].bias.copy_(self.decoder[0].bias)
                    for dst,src in zip(wide[1:],self.decoder[1:],strict=True):
                        dst.load_state_dict(src.state_dict(),strict=True)
                self.decoder=wide
                self.eef_encoder = nn.Sequential(mlp(7,(64,),32), nn.LayerNorm(32))
                self.joint_encoder = nn.Sequential(mlp(6,(64,),32), nn.LayerNorm(32))
        nn.init.zeros_(self.decoder[-1].weight)
        with torch.no_grad():
            self.decoder[-1].bias.copy_(torch.tensor(HOME))

    def history_features(self, h):
        valid = h[..., 18:19]
        x = self.obs_normalizer(torch.where(valid.bool(), h[..., :18], 0.))
        return torch.cat((torch.where(valid.bool(), x, 0.), valid), -1)

    def features(self, history, goal, goal_q, mode):
        q = history[:, -1, :6]
        p,r = self.fk(q)
        m = self.masks[mode.long()]
        active_rot = m[:,1:2].bool()
        # Mask before quaternion operations so hidden invalid values cannot leak.
        dr = log_rotation(mul(inv(r),canonical(torch.where(active_rot,goal[:,3:7],r))))
        dp = goal[:,:3]-p
        dq = torch.where(m[:,2:3].bool(),goal_q-q,0.)
        return dp/.5, torch.where(active_rot,dr,0.)/torch.pi, dq/torch.pi, m

    def rollout(self, history, goal, goal_q, mode, noise=None):
        h = self.history_features(history)
        dp,dr,dq,m = self.features(history,goal,goal_q,mode)
        if self.arch=='masked':
            cmd = torch.cat((dp,dr,dq,m),-1)
        else:
            ze = self.eef_encoder(torch.cat((dp,dr,m[:,1:2]),-1))
            zj = self.joint_encoder(dq)
            cmd = torch.where(m[:,2:3].bool(),zj,ze)
        u = self.decoder(torch.cat((h[:,-1,:18],self.history_encoder(h.flatten(1)),cmd),-1))
        return u if noise is None else u+noise

    def command(self,obs,noise=None):
        return self.rollout(obs['history'],obs['goal_pose'],obs['goal_q'],obs['mode'],noise)

    def forward(self,obs):
        return self.command(obs)

    @torch.no_grad()
    def update_normalization(self,obs):
        self.obs_normalizer.update(obs['history'][:,-1,:18])

    def payload_loss(self, h, target):
        z = self.history_encoder(self.history_features(h).flatten(1))
        dims = (z-target.detach()).square().mean(0)
        return dims.mean(), dims.detach(), z.detach()
