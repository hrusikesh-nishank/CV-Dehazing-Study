"""PSNR and SSIM for [0,1] RGB images (H, W, 3) numpy arrays or (3, H, W) tensors.

`ssim` reproduces skimage.metrics.structural_similarity(a, b, channel_axis=2, data_range=1,
gaussian_weights=True, sigma=1.5, use_sample_covariance=False) -- the Wang et al. 2004
setting used by most dehazing papers -- but runs on the GPU (about 30x faster on 1-megapixel
images). Equality with skimage is unit-tested (tests/test_models.py).
"""
import numpy as np
import torch
import torch.nn.functional as F

_SIGMA, _RADIUS = 1.5, 5            # skimage: radius = int(3.5 * sigma + 0.5) -> 11x11 window


def _as_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
    return x.to(device=device, dtype=torch.float64)


def psnr(a, b):
    a = torch.as_tensor(a, dtype=torch.float64); b = torch.as_tensor(b, dtype=torch.float64)
    return float(10 * torch.log10(1.0 / ((a - b) ** 2).mean().clamp_min(1e-12)))


@torch.no_grad()
def ssim(a, b, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x, y = _as_tensor(a, device)[:, None], _as_tensor(b, device)[:, None]        # (3,1,H,W)
    ax = torch.arange(-_RADIUS, _RADIUS + 1, dtype=torch.float64, device=device)
    k = torch.exp(-0.5 * ax ** 2 / _SIGMA ** 2); k = k / k.sum()

    def filt(z):                       # 'valid' separable Gaussian = skimage's filter + border crop
        return F.conv2d(F.conv2d(z, k.view(1, 1, -1, 1)), k.view(1, 1, 1, -1))

    ux, uy = filt(x), filt(y)
    vx, vy, vxy = filt(x * x) - ux * ux, filt(y * y) - uy * uy, filt(x * y) - ux * uy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(s.mean())


@torch.no_grad()
def dark_channel_mean(x, patch=15):
    """Mean of the dark channel (He et al.) per image; x (B,3,H,W) in [0,1]. A cheap proxy for
    haze density: about 0.1 for clear outdoor photos, 0.25-0.7 for real hazy photos."""
    m = x.min(1, keepdim=True)[0]
    p = patch // 2
    dc = -F.max_pool2d(F.pad(-m, (p, p, p, p), mode="replicate"), patch, stride=1)
    return dc.mean((1, 2, 3))
