"""Causal tri-mode goal conditioning to one absolute 6D joint command."""
import torch
from torch import nn
from superreach.geometry import canonical, inv, log_rotation, mul
from superreach.networks import RunningMeanStd
from .dr import THETA

MODES=("no_history","oracle","e2e_mlp","dyn_pred_mlp","short_window_gru","dr_reg_mlp")
LATENT=16
DELTA_DQ_SCALE=.5

MASKS = ((1,1,0),(1,0,0),(1,1,1))

HOME = (0., 1.047, 1.047, 0., 0., 0.)


class Actor(nn.Module):
    def __init__(self, fk, limits, mlp, arch, sysid_mode):
        super().__init__()
        self.fk = fk
        self.arch = arch
        self.register_buffer("masks", torch.tensor(MASKS, dtype=torch.float32))
        self.obs_normalizer = RunningMeanStd(18)
        self.goal_normalizers = nn.ModuleDict()  # fixed physical scaling, no amplification
        self.sysid_mode=sysid_mode
        assert sysid_mode in MODES
        zdim=0 if sysid_mode=='no_history' else LATENT
        if sysid_mode=='oracle': self.priv_encoder=mlp(THETA,(64,),LATENT)
        elif sysid_mode=='short_window_gru': self.history_encoder=nn.GRUCell(19,LATENT)
        elif zdim: self.history_encoder=mlp(16*19,(128,64),LATENT)

        self.decoder=mlp(18+zdim+(32 if arch=='dual_encoder' else 15),(256,256),6)
        if arch=='dual_encoder':
            self.eef_encoder = nn.Sequential(mlp(7,(64,),32), nn.LayerNorm(32))
            self.joint_encoder = nn.Sequential(mlp(6,(64,),32), nn.LayerNorm(32))
        nn.init.zeros_(self.decoder[-1].weight)
        with torch.no_grad():
            self.decoder[-1].bias.copy_(torch.tensor(HOME))

        # Do not shift the matched actor/critic/WM initialization streams.
        with torch.random.fork_rng(devices=[]):
            if sysid_mode=='dyn_pred_mlp': self.aux_head=mlp(18+LATENT,(64,),6)
            if sysid_mode=='dr_reg_mlp': self.aux_head=mlp(LATENT,(64,),THETA)

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

    def latent(self,h,theta=None):
        if self.sysid_mode=='no_history':return h[:,0,:0]
        if self.sysid_mode=='oracle':
            if theta is None:raise ValueError('oracle requires theta_true')
            return self.priv_encoder(theta.detach())
        if self.sysid_mode=='short_window_gru':
            z=h.new_zeros((len(h),LATENT))
            for frame in h[:,-8:].unbind(1):
                z=torch.where(frame[:,18:19].bool(),self.history_encoder(frame,z),z)
            return z
        return self.history_encoder(h.flatten(1))

    def rollout(self, history, goal, goal_q, mode, noise=None, theta=None):
        h = self.history_features(history)
        dp,dr,dq,m = self.features(history,goal,goal_q,mode)
        if self.arch=='masked':
            cmd = torch.cat((dp,dr,dq,m),-1)
        else:
            ze = self.eef_encoder(torch.cat((dp,dr,m[:,1:2]),-1))
            zj = self.joint_encoder(dq)
            cmd = torch.where(m[:,2:3].bool(),zj,ze)
        u = self.decoder(torch.cat((h[:,-1,:18],self.latent(h,theta),cmd),-1))
        return u if noise is None else u+noise

    def command(self,obs,noise=None):
        return self.rollout(obs['history'],obs['goal_pose'],obs['goal_q'],obs['mode'],noise,
                            obs['privileged_ctx'] if self.sysid_mode=='oracle' else None)

    def forward(self,obs):
        return self.command(obs)

    @torch.no_grad()
    def update_normalization(self,obs):
        self.obs_normalizer.update(obs['history'][:,-1,:18])

    def sysid_loss(self, obs, replay):
        if self.sysid_mode=='dr_reg_mlp':
            h=self.history_features(obs['history'].detach())
            return (self.aux_head(self.latent(h))-obs['privileged_ctx'].detach()).square().mean()
        if self.sysid_mode=='dyn_pred_mlp':
            win,pad=replay.sample_causal(17,len(obs['history']))
            raw=win['actor_obs'][:,:-1].detach().clone()
            raw[...,18]=(~pad[:,:-1]).to(raw.dtype)
            h=self.history_features(raw)
            # Successor's prev_action is the issued u_t, not u_eff or a future action.
            u=win['prev_action'][:,-1].detach()
            pred=self.aux_head(torch.cat((h[:,-1,:12],u/torch.pi,self.latent(h)),-1))
            delta=(win['state'][:,-1,6:12]-win['state'][:,-2,6:12]).detach()/DELTA_DQ_SCALE
            valid=(~pad[:,-2]).to(pred.dtype)
            return ((pred-delta).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
        return obs['history'].new_zeros(())
