"""Switchable haze image-formation pipeline (the object of the ablation).

Every arm receives the SAME clean image J, metric depth d, visibility V and
atmospheric light A for a given (image, draw) -> identical transmission t.
Only the *image-formation* stage changes between arms. This is the controlled
design that SynFog's beta-ablation lacked (it changed illumination, airlight
estimation and camera simultaneously).

Arms (see ARMS below):
  N0   sRGB ASM, neutral (grey) airlight      classical recipe with our depth and visibility
  R0   sRGB ASM, sky-coloured airlight        I = J t + A (1-t) on gamma-encoded pixels
  R1   linear-radiance ASM                    unprocess -> composite -> re-encode (no camera effects)
  R2   R1 + shot/read noise                   signal-dependent noise applied AFTER haze, in camera RAW colour space
  R3   R2 + randomised ISP + JPEG             white-balance error, tone-curve variation, JPEG Q in [60,95]
  R4   R3 + forward-scatter blur              depth-dependent blur of the direct term
  H    R0 + heterogeneous beta                sRGB, no camera (the "appearance realism" control)
  R3H  R3 + heterogeneous beta                full camera + heterogeneity
  S0   camera noise/ISP on an sRGB composite  (camera without linear compositing)
  R3M  R3 + Bayer mosaic/demosaic             optional: only meaningful at sensor resolution
Central test of the claim:  (R3H - H)  vs  (R3H - R3).

Design notes
  * Bayer mosaicing is OFF by default: the training/test images are web-resolution
    (several times below sensor resolution), where demosaicing blur is negligible.
    Simulating it at this scale would teach the network to sharpen.
  * Target: the clean image rendered with the SAME tone curve as the hazy image
    (a real hazy/clear pair is taken by one camera). Without tone randomisation the
    target is exactly J.
  * Randomness is split into independent streams (heterogeneity, blur, camera, ISP,
    JPEG, noise) derived from one draw of `generator`, so arms that share a component
    share its random numbers (e.g. R3 and R3H use the same cameras).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import isp

MOR_CONST = 2.996  # beta = 2.996 / V  (5 % contrast threshold, WMO/MOR convention)


@dataclass(frozen=True)
class Arm:
    linear: bool = False      # composite in linear radiance instead of sRGB
    camera: bool = False      # image the radiance through the camera model
    noise: bool = False       # Poisson-Gaussian sensor noise (after haze)
    mosaic: bool = False      # Bayer CFA + bilinear demosaic
    rand_isp: bool = False    # white-balance error + tone-curve variation
    jpeg: bool = False
    blur: bool = False        # forward-scatter blur of the direct term
    hetero: bool = False      # spatially heterogeneous extinction
    neutral_A: bool = False   # grey airlight (drop the sky-colour tint)


_CAM = dict(camera=True, noise=True, rand_isp=True, jpeg=True)
ARMS = {
    "N0": Arm(neutral_A=True),
    "R0": Arm(),
    "R1": Arm(linear=True),
    "R2": Arm(linear=True, camera=True, noise=True),
    "R3": Arm(linear=True, **_CAM),
    "R3N": Arm(linear=True, camera=True, rand_isp=True, jpeg=True),   # R3 without sensor noise
    "R4": Arm(linear=True, blur=True, **_CAM),
    "H": Arm(hetero=True),
    "R3H": Arm(linear=True, hetero=True, **_CAM),
    "S0": Arm(linear=False, **_CAM),
    "R3M": Arm(linear=True, mosaic=True, **_CAM),
}
STREAMS = ("het", "blur", "cam", "isp", "jpeg", "noise")


def _streams(generator, device):
    """Independent random streams derived from ONE draw of `generator`."""
    base = int(torch.randint(0, 2 ** 31 - 16, (1,), generator=generator))
    g = {k: torch.Generator().manual_seed(base + i) for i, k in enumerate(STREAMS[:-1])}
    gn = torch.Generator(device=device)
    gn.manual_seed(base + len(STREAMS) - 1)
    g["noise"] = gn
    return g


# --------------------------------------------------------------------------
# Medium
# --------------------------------------------------------------------------
def fbm_field(B, H, W, octaves=4, base=4, generator=None, device="cpu"):
    """Cheap fractal (fBm) noise, unit std per image, by summing upsampled random grids."""
    out = torch.zeros(B, 1, H, W, device=device)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = base * 2 ** o
        g = torch.randn(B, 1, s, s, generator=generator).to(device)
        out += amp * F.interpolate(g, size=(H, W), mode="bicubic", align_corners=False)
        tot += amp
        amp *= 0.5
    out = out / tot
    return out / out.flatten(1).std(1).clamp_min(1e-6).view(-1, 1, 1, 1)


def transmission(depth, V, hetero=False, strength=0.4, generator=None):
    """depth (B,1,H,W) metres, V (B,) metres -> t (B,1,H,W)."""
    beta = (MOR_CONST / V).view(-1, 1, 1, 1)
    if hetero:
        B, _, H, W = depth.shape
        n = fbm_field(B, H, W, generator=generator, device=depth.device)
        m = torch.exp(strength * n)
        beta = beta * m / m.mean(dim=(2, 3), keepdim=True)      # same mean extinction
    return torch.exp(-beta * depth)


def _gauss(x, k, s):
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-ax ** 2 / (2 * s * s))
    g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def forward_scatter_blur(direct, t, strength):
    """Approximate small-angle forward scattering: blur grows with optical depth.

    direct (B,3,H,W), t (B,1,H,W), strength (B,) ~ U[0.5,1.5]. 4 optical-depth bins;
    the batch-mean strength is used (cheap), variation enters through the bins.
    """
    tau = -torch.log(t.clamp_min(1e-4))
    edges = [0.0, 0.5, 1.0, 2.0, 1e9]
    sigmas = [0.0, 0.6, 1.2, 2.2]
    out = torch.zeros_like(direct)
    for lo, hi, s in zip(edges[:-1], edges[1:], sigmas):
        m = ((tau >= lo) & (tau < hi)).to(direct.dtype)
        if s == 0:
            out += m * direct
            continue
        sig = float(s * strength.mean())
        k = int(2 * round(3 * sig) + 1)
        out += m * _gauss(direct, k, sig)
    return out


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
@torch.no_grad()
def synthesize_pair(J, depth, V, A, arm: Arm, *, generator=None, cam: isp.CameraParams | None = None,
                    shot_range=(2e-5, 8e-4), hetero_strength=0.4, tone_range=(0.5, 1.0), wb_sigma=0.03):
    """J (B,3,H,W) clean sRGB in [0,1]; depth (B,1,H,W) m; V (B,) m; A (B,3) sRGB.

    Returns (hazy, target), both (B,3,H,W) in [0,1].
    """
    B = J.shape[0]
    dev = J.device
    g = _streams(generator, dev)
    t = transmission(depth, V, hetero=arm.hetero, strength=hetero_strength, generator=g["het"])
    if arm.neutral_A:
        A = A.mean(1, keepdim=True).expand(-1, 3)
    A4 = A.reshape(B, 3, 1, 1)
    blur_s = (0.5 + torch.rand(B, generator=g["blur"])).to(dev)

    if arm.linear:
        direct = isp.to_linear(J) * t
        if arm.blur:
            direct = forward_scatter_blur(direct, t, blur_s)
        E = direct + isp.to_linear(A4) * (1 - t)       # linear radiance at the lens
    else:
        direct = J * t
        if arm.blur:
            direct = forward_scatter_blur(direct, t, blur_s)
        I = (direct + A4 * (1 - t)).clamp(0, 1)        # classical sRGB ASM
        if not arm.camera:
            return I, J
        E = isp.to_linear(I)                           # S0: camera applied to an sRGB composite

    if not arm.camera:
        return isp.to_display(E).clamp(0, 1), J

    if cam is None:
        cam = isp.random_camera(B, generator=g["cam"], shot_range=shot_range, device=dev)
    wb_err, tone_mix, target = None, None, J
    if arm.rand_isp:
        # auto white balance is biased by the airlight in haze: small error on R and B (hazy shot only)
        e = 1.0 + wb_sigma * torch.randn(B, 3, generator=g["isp"])
        e[:, 1] = 1.0
        wb_err = e.to(dev)
        lo, hi = tone_range
        tone_mix = (lo + (hi - lo) * torch.rand(B, generator=g["isp"])).to(dev)
        target = isp.to_display(isp.to_linear(J), tone_mix).clamp(0, 1)    # same tone curve as the hazy shot
    I = isp.image_through_camera(E, cam, use_mosaic=arm.mosaic, noise=arm.noise,
                                 wb_error=wb_err, tone_mix=tone_mix, generator=g["noise"])
    if arm.jpeg:
        q = torch.randint(60, 96, (B,), generator=g["jpeg"])
        keep = torch.rand(B, generator=g["jpeg"]) < 0.3      # 30 % stay uncompressed
        q[keep] = 100
        I = isp.jpeg_batch(I, q)
    return I.clamp(0, 1), target


def synthesize(J, depth, V, A, arm: Arm, **kw):
    """Hazy image only (see synthesize_pair)."""
    return synthesize_pair(J, depth, V, A, arm, **kw)[0]
