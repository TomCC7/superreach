"""Fixed-base TCP FK, world spatial Jacobian and quaternion residuals (wxyz)."""
import numpy as np
import torch
from torch import nn


def mul(a, b):
    w, v = a[..., :1], a[..., 1:]
    s, u = b[..., :1], b[..., 1:]
    v,u = torch.broadcast_tensors(v,u)
    return torch.cat((w*s-(v*u).sum(-1, keepdim=True), w*u+s*v+torch.linalg.cross(v,u)), -1)


def inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def rotate(q, x):
    v, x = torch.broadcast_tensors(q[..., 1:], x)
    t = 2*torch.linalg.cross(v,x)
    return x + q[..., :1]*t + torch.linalg.cross(v,t)


def canonical(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def log_rotation(q):
    q = canonical(q)
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    # atan2 is well-defined at identity; small-angle branch has slope 2.
    s = torch.where(n > 1e-7, 2*torch.atan2(n, q[..., :1])/n.clamp_min(1e-7), 2.)
    return s*v


class TCP(nn.Module):
    def __init__(self, model):
        super().__init__()
        import mujoco as mj
        self.site = model.site('arm_tcp_site').id
        body = int(model.site_bodyid[self.site]); ids = []
        while body:
            ids.append(body); body = int(model.body_parentid[body])
        chain = []; joints = []
        for body in reversed(ids):
            n = int(model.body_jntnum[body]); j = int(model.body_jntadr[body])
            assert n <= 1
            if n:
                assert model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE
                joints.append(int(model.jnt_qposadr[j]))
                chain.append(joints[-1]); axis, pivot = model.jnt_axis[j], model.jnt_pos[j]
            else:
                chain.append(None); axis = pivot = np.zeros(3)
            for key, value in [('p',model.body_pos[body]),('r',model.body_quat[body]),('a',axis),('c',pivot)]:
                self.register_buffer(f'{key}{len(chain)-1}',torch.from_numpy(value.copy()))
        assert joints == list(range(6))
        self.chain = tuple(chain)
        for key, value in [('site_p',model.site_pos[self.site]),('site_r',model.site_quat[self.site]),('q0',model.qpos0[:6]),('points',.30*np.eye(3))]:
            self.register_buffer(key,torch.from_numpy(value.copy()))

    def forward(self, q, jac=False):
        def buf(name): return getattr(self,name).to(dtype=q.dtype)
        xyz = torch.zeros_like(q[..., :3]); quat = torch.cat((torch.ones_like(q[..., :1]),xyz),-1)
        pivots, axes = [], []
        for i,j in enumerate(self.chain):
            xyz = xyz + rotate(quat, buf(f'p{i}'))
            quat = mul(quat, buf(f'r{i}'))
            if j is not None:
                axis,pivot = buf(f'a{i}'),buf(f'c{i}')
                anchor = xyz + rotate(quat,pivot)
                if jac:
                    pivots.append(anchor); axes.append(rotate(quat,axis))
                h = .5*(q[..., j:j+1]-buf('q0')[j])
                quat = mul(quat,torch.cat((h.cos(),h.sin()*axis),-1))
                xyz = anchor - rotate(quat,pivot)
        xyz = xyz + rotate(quat,buf('site_p'))
        quat = canonical(mul(quat,buf('site_r')))
        if jac:
            axis = torch.stack(axes,-2)
            linear = torch.linalg.cross(axis,xyz.unsqueeze(-2)-torch.stack(pivots,-2))
            return xyz,quat,linear.transpose(-1,-2),axis.transpose(-1,-2)
        return xyz,quat

    def residuals(self, q, goal):
        p, r = self(q)
        gp, gr = goal[..., :3], canonical(goal[..., 3:7])
        dp = gp-p
        dr = log_rotation(mul(inv(r), gr))  # body-frame SO(3) logarithm
        keypoints = dp.unsqueeze(-2) + rotate(gr.unsqueeze(-2), self.points.to(q)) - rotate(r.unsqueeze(-2),self.points.to(q))
        return p,r,dp,dr,keypoints.flatten(-2)
