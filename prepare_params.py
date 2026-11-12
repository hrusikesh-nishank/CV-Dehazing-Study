"""Per-image, per-draw atmosphere parameters shared by ALL arms (seeded by image name).

For each clean image, K draws of:
  V  visibility (meteorological optical range, metres), log-uniform in [v_min, v_max]
     = [10 m, 5 km] (dense smoke/fog ... light haze). A draw is re-sampled (up to 30 times)
     if the image's mean transmission exp(-2.996 d / V) falls outside [t_lo, t_hi] = [0.10, 0.85],
     so no draw is "no haze" or "all haze" whatever the scene's depth range.
  A  atmospheric light = luminance x sky colour.
     Luminance: from the far/sky region of the image (farthest 1 % of metric depth,
     brightest 10 % of those pixels), clipped to [0.55, 0.98]; if the image has no far
     region (max depth < 150 m) it is drawn from U[0.60, 0.95].
     Sky colour: a daylight-locus chromaticity with correlated colour temperature drawn
     uniformly in mired between 5500 K (warm, pollution/low sun) and 9500 K (blue
     overcast/shade), expressed relative to D65 at equal luminance. The scene itself is
     white-balanced, so this is the colour of the airlight relative to scene white.
     Arm N0 replaces A by its grey mean (the classical neutral airlight).

Where the ranges come from (measured on the HAZY INPUTS of the real sets only; no ground
truth or labels were used): dark-channel airlight estimates of O-HAZE / NH-HAZE / Dense-Haze /
250 RTTS images have B-R in [-0.12, +0.17] (5th-95th percentile; the daylight locus between
5500 K and 9500 K spans [-0.12, +0.17]), luminance 0.58-0.94, and dark-channel means that
correspond to a mean transmission of roughly 0.1-0.8.

Usage:
  python prepare_params.py --clean_dir data/reside6k_ots/GT --depth_dir data/depth/ots --out data/params_ots.npz
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MOR = 2.996
_XYZ2RGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]])
_LUM = np.array([0.2126729, 0.7151522, 0.0721750])


# ---- display <-> linear, numpy versions of haze/isp.py (kept identical; unit-tested) ----
def to_linear(x):
    y = np.clip(x, 0, 1)
    return np.maximum(0.5 - np.sin(np.arcsin(1.0 - 2.0 * y) / 3.0), 1e-8) ** 2.2


def to_display(lin):
    g = np.maximum(np.clip(lin, 0, 1), 1e-8) ** (1 / 2.2)
    return 3 * g ** 2 - 2 * g ** 3


def daylight_rgb(cct):
    """Linear-sRGB colour of CIE daylight at `cct` kelvin, relative to D65, unit luminance."""
    def raw(T):
        if T <= 7000:
            x = -4.6070e9 / T ** 3 + 2.9678e6 / T ** 2 + 0.09911e3 / T + 0.244063
        else:
            x = -2.0064e9 / T ** 3 + 1.9018e6 / T ** 2 + 0.24748e3 / T + 0.237040
        y = -3.0 * x * x + 2.87 * x - 0.275
        return _XYZ2RGB @ np.array([x / y, 1.0, (1 - x - y) / y])
    rgb = raw(float(cct)) / raw(6504.0)
    return rgb / (_LUM @ rgb)


def sky_luminance(img, d):
    lum = img.mean(-1)
    far = d >= np.quantile(d, 0.99)
    if d.max() < 150 or far.sum() < 20:
        return None
    sel = far & (lum >= np.quantile(lum[far], 0.9))
    return float(np.clip(lum[sel].mean(), 0.55, 0.98))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (sorted by name)")
    ap.add_argument("--v_min", type=float, default=10.0)
    ap.add_argument("--v_max", type=float, default=5000.0)
    ap.add_argument("--t_lo", type=float, default=0.10)
    ap.add_argument("--t_hi", type=float, default=0.85)
    ap.add_argument("--cct_min", type=float, default=5500.0)
    ap.add_argument("--cct_max", type=float, default=9500.0)
    a = ap.parse_args()
    files = sorted(p for p in Path(a.clean_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if a.limit:
        files = files[:a.limit]
    names, Vs, As, has_sky, tmeans, dmed, forced = [], [], [], [], [], [], 0
    for f in tqdm(files):
        dp = Path(a.depth_dir) / (f.stem + ".npy")
        if not dp.exists():
            continue
        rng = np.random.default_rng(zlib.crc32(f.stem.encode()))
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        d = np.load(dp).astype(np.float32)
        if d.shape != img.shape[:2] or not np.isfinite(d).all():
            print(f"skip {f.name}: bad depth map"); continue
        L0 = sky_luminance(img, d)
        has_sky.append(L0 is not None)
        ds = d[::4, ::4].astype(np.float64)
        V = np.zeros(a.K, np.float32); A = np.zeros((a.K, 3), np.float32); tm = np.zeros(a.K, np.float32)
        for k in range(a.K):
            for _ in range(30):
                v = float(np.exp(rng.uniform(np.log(a.v_min), np.log(a.v_max))))
                t = float(np.exp(-MOR * ds / v).mean())
                if a.t_lo <= t <= a.t_hi:
                    break
            else:                                   # scene too near / too far for the V range
                v = float(np.clip(MOR * np.median(ds) / np.log(2.0), 2.0, 20000.0))  # median t = 0.5
                t = float(np.exp(-MOR * ds / v).mean()); forced += 1
            V[k], tm[k] = v, t
            lum = (L0 if L0 is not None else rng.uniform(0.60, 0.95)) * rng.uniform(0.92, 1.04)
            cct = 1e6 / rng.uniform(1e6 / a.cct_max, 1e6 / a.cct_min)      # uniform in mired
            lin = to_linear(np.full(3, np.clip(lum, 0.5, 1.0))) * daylight_rgb(cct)
            Ak = to_display(lin / max(1.0, lin.max()))
            A[k] = np.clip(Ak + rng.normal(0, 0.008, 3), 0.35, 1.0)
        names.append(f.stem); Vs.append(V); As.append(A); tmeans.append(tm); dmed.append(float(np.median(d)))
    assert names, f"no depth maps found in {a.depth_dir}: run prepare_depth.py first"
    V, A, tm = np.stack(Vs), np.stack(As), np.stack(tmeans)
    np.savez(a.out, names=np.array(names), V=V, A=A, tmean=tm, has_sky=np.array(has_sky), depth_median=np.array(dmed))
    q = lambda x, p: np.percentile(x, p)
    print(f"{len(names)} images x {a.K} draws -> {a.out}")
    print(f"  far/sky region found in {np.mean(has_sky):.1%} of images")
    print(f"  median scene depth: 10/50/90 percentile = {q(dmed, 10):.0f} / {q(dmed, 50):.0f} / {q(dmed, 90):.0f} m")
    print(f"  visibility V:       10/50/90 percentile = {q(V, 10):.0f} / {q(V, 50):.0f} / {q(V, 90):.0f} m")
    print(f"  mean transmission:  10/50/90 percentile = {q(tm, 10):.2f} / {q(tm, 50):.2f} / {q(tm, 90):.2f}")
    print(f"  airlight (sRGB) mean R,G,B = {A.reshape(-1, 3).mean(0).round(3).tolist()}, "
          f"B-R range = [{(A[..., 2] - A[..., 0]).min():+.2f}, {(A[..., 2] - A[..., 0]).max():+.2f}]")
    print(f"  draws outside the V range (scene too near/far): {forced} of {V.size} ({forced / V.size:.1%})")
    if forced / V.size > 0.25:
        print("  WARNING: >25 % of draws needed a visibility outside [v_min, v_max]; check the depth previews "
              "(data/depth/ots/_preview_*.png) -- depth may be badly scaled.")
    if a.limit and len(names) < a.limit:
        print(f"  WARNING: only {len(names)} of the first {a.limit} images have a depth map")


if __name__ == "__main__":
    main()
