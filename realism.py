"""Gate-1 checks before launching training (no network training needed).

1. Visual grids: the same clean crops hazed by every arm at fixed visibility (100 m, 300 m),
   and one grid with each image's own (V, A) draw.  -> <res_dir>/grid_*.png
2. Haze-density check: dark-channel mean (a proxy for haze density) and mean colour cast of
   the synthetic training images versus real hazy crops (RTTS). The two distributions should
   overlap; if the synthetic haze is much lighter or denser, change --t_lo/--t_hi in
   prepare_params.py BEFORE training.
3. KID between each arm's synthetic hazy crops and real hazy crops (RTTS), on identical
   clean crops. Lower = closer to real haze statistics.

  python realism.py --n 500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from haze.data import CleanDepthDataset  # noqa: E402
from haze.metrics import dark_channel_mean  # noqa: E402
from haze.synth import ARMS, synthesize  # noqa: E402

# measured once on the hazy inputs of the real sets (480-px thumbnails): 5th / 50th / 95th percentile
REAL_REF = {"O-HAZE": (0.27, 0.42, 0.54), "NH-HAZE": (0.31, 0.42, 0.60), "Dense-Haze": (0.44, 0.61, 0.73),
            "RTTS": (0.24, 0.46, 0.66)}


def real_crops(rtts, n, c=256):
    files = sorted((Path(rtts) / "JPEGImages").glob("*.*"))
    rng = np.random.default_rng(0)
    out = []
    for f in rng.choice(files, min(n, len(files)), replace=False):
        im = np.asarray(Image.open(f).convert("RGB"))
        H, W = im.shape[:2]
        if min(H, W) < c:
            continue
        y, x = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
        out.append(torch.from_numpy(im[y:y + c, x:x + c].copy()).permute(2, 0, 1))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--clean_dir", default="data/reside6k_ots/GT")
    ap.add_argument("--depth_dir", default="data/depth/ots")
    ap.add_argument("--params", default="data/params_ots.npz")
    ap.add_argument("--rtts", default="data/test/RTTS")
    ap.add_argument("--res_dir", default="results")
    ap.add_argument("--arms", nargs="*", default=["N0", "R0", "R1", "R2", "R3", "R4", "H", "R3H", "S0"])
    ap.add_argument("--skip_kid", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CleanDepthDataset(a.clean_dir, a.depth_dir, a.params, 256, seed=0)
    idx = np.random.default_rng(0).choice(len(ds), min(a.n, len(ds)), replace=False)
    assert len(idx) >= 6, "need at least 6 clean images"
    batch = [ds[int(i)] for i in idx]
    J = torch.stack([b["J"] for b in batch]).to(dev)
    D = torch.stack([b["depth"] for b in batch]).to(dev)
    V = torch.stack([b["V"] for b in batch]).to(dev)
    A = torch.stack([b["A"] for b in batch]).to(dev)
    res = Path(a.res_dir)
    res.mkdir(parents=True, exist_ok=True)

    # 1. visual grids (rows: clean, then one row per arm)
    for tag, v in [("V100", 100.0), ("V300", 300.0), ("own_draw", None)]:
        rows = [J[:6]]
        for arm in a.arms:
            Vg = V[:6] if v is None else torch.full((6,), v, device=dev)
            rows.append(synthesize(J[:6], D[:6], Vg, A[:6], ARMS[arm], generator=torch.Generator().manual_seed(0)))
        save_image(torch.cat(rows), res / f"grid_{tag}.png", nrow=6)
    real = real_crops(a.rtts, 12)
    save_image(real[:12].float() / 255, res / "grid_real_rtts.png", nrow=6)
    print("grids written to", res, "| rows: clean,", ", ".join(a.arms))

    # 2. haze density and colour cast: synthetic (each image's own V, A draw) vs real hazy crops
    def pct(x):
        return " / ".join(f"{v:.2f}" for v in np.percentile(x.cpu().numpy(), [5, 50, 95]))
    real = real_crops(a.rtts, a.n).to(dev).float() / 255
    print("\nhaze-density check (dark-channel mean, 5th / 50th / 95th percentile; clear photos are ~0.10):")
    print(f"  clean training crops          : {pct(dark_channel_mean(J))}")
    print(f"  real hazy crops (RTTS, n={len(real)})  : {pct(dark_channel_mean(real))}   colour cast B-R {pct(real[:, 2].mean((1, 2)) - real[:, 0].mean((1, 2)))}")
    for arm in [x for x in ("R0", "R3") if x in a.arms]:
        I = torch.cat([synthesize(J[s:s + 50], D[s:s + 50], V[s:s + 50], A[s:s + 50], ARMS[arm],
                                  generator=torch.Generator().manual_seed(s)) for s in range(0, len(J), 50)])
        print(f"  synthetic {arm:3s} (own V, A draws) : {pct(dark_channel_mean(I))}   colour cast B-R {pct(I[:, 2].mean((1, 2)) - I[:, 0].mean((1, 2)))}")
    print("  reference, whole real images  : " + "; ".join(f"{k} {a_:.2f} / {b_:.2f} / {c_:.2f}" for k, (a_, b_, c_) in REAL_REF.items()))
    print("  -> the synthetic rows should overlap the real rows (median within about 0.1).\n", flush=True)

    # 3. KID vs real hazy (needs torch-fidelity; downloads Inception weights once)
    if a.skip_kid:
        return
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
        KernelInceptionDistance(subset_size=2)
    except Exception as e:
        print("KID skipped:", e); return
    real = (real * 255).round().to(torch.uint8)
    for arm in a.arms:
        kid = KernelInceptionDistance(subset_size=max(2, min(100, len(real) // 2, len(J) // 2))).to(dev)
        kid.update(real, real=True)
        for s in range(0, len(J), 50):
            I = synthesize(J[s:s + 50], D[s:s + 50], V[s:s + 50], A[s:s + 50], ARMS[arm],
                           generator=torch.Generator().manual_seed(s))
            kid.update((I * 255).round().to(torch.uint8), real=False)
        m, sd = kid.compute()
        print(f"KID {arm:4s}: {m.item() * 1000:.2f} +- {sd.item() * 1000:.2f} (x1e-3)", flush=True)


if __name__ == "__main__":
    main()
