"""Dark Channel Prior dehazing (He, Sun, Tang, CVPR 2009 / TPAMI 2011), batched torch.

Classical baseline built from course tools: a min-filter (dark channel),
a guided filter (edge-preserving, He et al. 2010) for transmission refinement,
and the atmospheric scattering model inverted: J = (I - A) / max(t, t0) + A.

Parameters follow the paper: patch 15, omega 0.95, t0 0.1, A from the brightest
0.1 % of the dark channel, guided-filter radius 60 (at ~600 px; scaled with size), eps 1e-3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _box(x, r):
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), k, stride=1)


def guided_filter(I, p, r, eps):
    mI, mp = _box(I, r), _box(p, r)
    a = (_box(I * p, r) - mI * mp) / (_box(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return _box(a, r) * I + _box(b, r)


class DarkChannelPrior(nn.Module):
    def __init__(self, patch=15, omega=0.95, t0=0.1, r_ref=60, eps=1e-3):
        super().__init__()
        self.patch, self.omega, self.t0, self.r_ref, self.eps = patch, omega, t0, r_ref, eps
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)  # gives the module a device

    def dark(self, x):
        m = x.min(1, keepdim=True)[0]
        p = self.patch // 2
        return -F.max_pool2d(F.pad(-m, (p, p, p, p), mode="replicate"), self.patch, stride=1)

    @torch.no_grad()
    def forward(self, I):
        B, _, H, W = I.shape
        dc = self.dark(I)
        n = max(1, int(0.001 * H * W))
        idx = dc.flatten(1).topk(n, dim=1).indices                    # brightest 0.1 % of dark channel
        lum = I.mean(1).flatten(1).gather(1, idx)
        best = idx.gather(1, lum.argmax(1, keepdim=True))              # brightest input pixel among them
        A = I.flatten(2).gather(2, best[:, None, :].expand(B, 3, 1)).view(B, 3, 1, 1).clamp_min(0.05)
        t = 1 - self.omega * self.dark(I / A)
        r = max(4, int(round(self.r_ref * min(H, W) / 600)))
        t = guided_filter(I.mean(1, keepdim=True), t, r, self.eps)
        J = (I - A) / t.clamp_min(self.t0) + A
        return J.clamp(0, 1)
