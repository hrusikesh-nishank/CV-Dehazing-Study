"""Camera / ISP model: sRGB <-> linear radiance <-> camera RAW.

PyTorch port of the "Unprocessing Images for Learned Raw Denoising" pipeline
(Brooks et al., CVPR 2019), batched and GPU-friendly, extended with
  * exact invertibility when the same parameters are used both ways
    (needed for the R1 unit test: linear compositing with an identity camera),
  * randomised forward ISP (white-balance error under haze, tone-curve
    variation) for the "randomised camera" arm.

All tensors are (B, C, H, W) float in [0, 1] unless stated otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Colour matrices (from Brooks et al. / dcraw): XYZ -> camera for 4 cameras
# --------------------------------------------------------------------------
_XYZ2CAMS = torch.tensor([
    [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
    [[0.4913, -0.0541, -0.0202], [-0.613, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
    [[0.838, -0.263, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
    [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.097, 0.1581, 0.5181]],
])
_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])


@dataclass
class CameraParams:
    rgb2cam: torch.Tensor   # (B,3,3)
    cam2rgb: torch.Tensor   # (B,3,3)
    rgb_gain: torch.Tensor  # (B,)
    red_gain: torch.Tensor  # (B,)
    blue_gain: torch.Tensor  # (B,)
    shot: torch.Tensor      # (B,)  shot-noise variance scale
    read: torch.Tensor      # (B,)  read-noise variance

    def to(self, device):
        return CameraParams(*[getattr(self, f).to(device) for f in self.__dataclass_fields__])


def random_camera(batch: int, generator: torch.Generator | None = None,
                  shot_range=(2e-5, 8e-4), device="cpu") -> CameraParams:
    """Sample camera parameters (Brooks et al. distributions).

    shot_range: daylight range by default (visually checked: ISO-100..800-like grain). Brooks used (1e-4, 1.2e-2), which
    covers dim indoor scenes; daylight hazy outdoor scenes are on the low end.
    """
    g = generator
    w = torch.rand(batch, 4, generator=g)
    w = w / w.sum(1, keepdim=True)
    xyz2cam = torch.einsum("bk,kij->bij", w, _XYZ2CAMS)
    rgb2cam = xyz2cam @ _RGB2XYZ
    rgb2cam = rgb2cam / rgb2cam.sum(-1, keepdim=True)
    cam2rgb = torch.linalg.inv(rgb2cam)
    # Brooks: 1 / N(0.8, 0.1). Clamped so the gain is always >= 1: the RAW signal (image / gains)
    # then never exceeds 1 and sensor clipping cannot darken highlights in the round trip.
    rgb_gain = 1.0 / (0.8 + 0.1 * torch.randn(batch, generator=g)).clamp(0.5, 1.0)
    red_gain = 1.9 + 0.5 * torch.rand(batch, generator=g)
    blue_gain = 1.5 + 0.4 * torch.rand(batch, generator=g)
    lo, hi = math.log(shot_range[0]), math.log(shot_range[1])
    log_shot = lo + (hi - lo) * torch.rand(batch, generator=g)
    log_read = 2.18 * log_shot + 1.20 + 0.26 * torch.randn(batch, generator=g)
    p = CameraParams(rgb2cam, cam2rgb, rgb_gain, red_gain, blue_gain,
                     log_shot.exp(), log_read.exp())
    return p.to(device)


# --------------------------------------------------------------------------
# Tone curve and gamma
# --------------------------------------------------------------------------
def smoothstep(x):
    x = x.clamp(0, 1)
    return 3 * x ** 2 - 2 * x ** 3


def inverse_smoothstep(y):
    y = y.clamp(0, 1)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * y) / 3.0)


def gamma_expand(x, g=2.2):
    return x.clamp_min(1e-8) ** g


def gamma_compress(x, g=2.2):
    return x.clamp_min(1e-8) ** (1.0 / g)


def srgb_to_linear(x):
    """Standard piecewise sRGB EOTF (used for metrics / analysis only)."""
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def to_linear(img):
    """sRGB-ish display image -> linear radiance (Brooks: inverse tone, inverse gamma)."""
    return gamma_expand(inverse_smoothstep(img))


def to_display(lin, tone_mix=None):
    """linear radiance -> display. tone_mix (B,) in [0,1] blends smoothstep with identity."""
    g = gamma_compress(lin.clamp(0, 1))
    s = smoothstep(g)
    if tone_mix is None:
        return s
    m = tone_mix.view(-1, 1, 1, 1)
    return m * s + (1 - m) * g


# --------------------------------------------------------------------------
# Colour / gains
# --------------------------------------------------------------------------
def apply_ccm(img, ccm):
    return torch.einsum("bij,bjhw->bihw", ccm, img)


def _gain_vec(p: CameraParams):
    return torch.stack([p.red_gain, torch.ones_like(p.red_gain), p.blue_gain], 1) * p.rgb_gain[:, None]


def invert_gains(img, p: CameraParams):
    return img / _gain_vec(p)[:, :, None, None]


def apply_gains(img, p: CameraParams, wb_error=None):
    g = _gain_vec(p)
    if wb_error is not None:
        g = g * wb_error
    return img * g[:, :, None, None]


# --------------------------------------------------------------------------
# Bayer mosaic / demosaic (RGGB)
# --------------------------------------------------------------------------
def mosaic(img):
    """(B,3,H,W) -> (B,1,H,W) RGGB Bayer. H, W must be even."""
    B, _, H, W = img.shape
    out = torch.zeros(B, 1, H, W, device=img.device, dtype=img.dtype)
    out[:, 0, 0::2, 0::2] = img[:, 0, 0::2, 0::2]
    out[:, 0, 0::2, 1::2] = img[:, 1, 0::2, 1::2]
    out[:, 0, 1::2, 0::2] = img[:, 1, 1::2, 0::2]
    out[:, 0, 1::2, 1::2] = img[:, 2, 1::2, 1::2]
    return out


def demosaic_bilinear(raw):
    """Bilinear demosaic of an RGGB mosaic, via masked normalised convolution."""
    B, _, H, W = raw.shape
    dev, dt = raw.device, raw.dtype
    m = torch.zeros(1, 3, H, W, device=dev, dtype=dt)
    m[:, 0, 0::2, 0::2] = 1
    m[:, 1, 0::2, 1::2] = 1
    m[:, 1, 1::2, 0::2] = 1
    m[:, 2, 1::2, 1::2] = 1
    k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=dev, dtype=dt) / 4.0
    k = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sparse = raw * m
    num = F.conv2d(F.pad(sparse, (1, 1, 1, 1), mode="reflect"), k, groups=3)
    den = F.conv2d(F.pad(m.expand(B, -1, -1, -1), (1, 1, 1, 1), mode="reflect"), k, groups=3)
    out = num / den.clamp_min(1e-6)
    return torch.where(m.bool().expand_as(out), sparse, out)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------
def add_poisson_gaussian(raw, p: CameraParams, generator=None):
    """Heteroscedastic Gaussian approximation of shot + read noise (Brooks)."""
    var = raw * p.shot.view(-1, 1, 1, 1) + p.read.view(-1, 1, 1, 1)
    n = torch.randn(raw.shape, generator=generator, device=raw.device, dtype=raw.dtype)
    return raw + n * var.clamp_min(0).sqrt()


def quantise(raw, bits=12):
    q = 2 ** bits - 1
    return torch.round(raw.clamp(0, 1) * q) / q


# --------------------------------------------------------------------------
# Full camera: linear sRGB-primaries radiance -> camera -> display sRGB
# --------------------------------------------------------------------------
def image_through_camera(lin_rgb, p: CameraParams, *, use_mosaic=True, noise=True,
                         bits=12, wb_error=None, tone_mix=None, generator=None):
    """Simulate imaging of a linear radiance image by a camera with an ISP.

    With use_mosaic=False, noise=False, wb_error=None, tone_mix=None this is the
    exact inverse of `to_linear`, so a scene without haze reproduces the input
    image (unit-tested, including saturated colours and white).
    """
    cam = invert_gains(apply_ccm(lin_rgb, p.rgb2cam), p)
    if not use_mosaic and not noise:            # no sensor effects requested: skip sensor clipping too
        return to_display(apply_ccm(apply_gains(cam, p, wb_error), p.cam2rgb), tone_mix)
    if use_mosaic:
        H, W = cam.shape[-2:]
        assert H % 2 == 0 and W % 2 == 0, "crop to even size before mosaicing"
        raw = mosaic(cam)
        if noise:
            raw = add_poisson_gaussian(raw, p, generator)
        if bits:
            raw = quantise(raw, bits)
        cam = demosaic_bilinear(raw.clamp(0, 1))
    elif noise:
        cam = add_poisson_gaussian(cam, p, generator)
        if bits:
            cam = quantise(cam, bits)
    rgb = apply_ccm(apply_gains(cam.clamp(0, 1), p, wb_error), p.cam2rgb)
    return to_display(rgb, tone_mix)


def jpeg_batch(img, quality):
    """CPU JPEG round trip per image. img (B,3,H,W) in [0,1]; quality list/tensor of ints."""
    from torchvision.io import decode_jpeg, encode_jpeg
    dev = img.device
    x = (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    out = []
    for i in range(x.shape[0]):
        q = int(quality[i])
        if q >= 100:
            out.append(x[i])
        else:
            out.append(decode_jpeg(encode_jpeg(x[i], quality=q)))
    return torch.stack(out).float().div(255).to(dev)
