"""Zero-shot evaluation on real paired benchmarks under ONE fixed protocol.

Protocol (identical for every model, ours or official):
  * hazy and ground truth are bicubic-resized so the long side = --long_side (default 1024)
    and stored as 8-bit (cached in data/cache after the first run)
  * inference through haze.models.Dehazer ([0,1] in/out, whole image, automatic tiling if
    the GPU runs out of memory); the output is quantised to 8 bit before scoring
  * PSNR on RGB in [0,1]; SSIM = Gaussian-window SSIM (sigma 1.5) on RGB, identical to skimage

  python eval_paired.py --model dehazeformer-t --ckpt runs/dehazeformer-t__R3_s0/final.pth
  python eval_paired.py --model dehazeformer-b --ckpt ckpt/outdoor/dehazeformer-b.pth --name pre__dehazeformer-b-outdoor
  python eval_paired.py --model dcp --name pre__dcp
  python eval_paired.py --model hazy-input --name pre__hazy-input
Writes <res_dir>/<name>/<SET>.csv (per image) and <res_dir>/<name>/summary.json
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from haze.data import TEST_SETS, load_test_set  # noqa: E402
from haze.metrics import psnr, ssim  # noqa: E402
from haze.models import build, infer, load_weights  # noqa: E402


def default_name(model, ckpt):
    if model in ("hazy-input", "dcp"):
        return f"pre__{model}"
    p = Path(ckpt)
    if p.name in ("final.pth", "last.pth"):
        return p.parent.name                         # our run: <runs>/<run>/final.pth
    return f"pre__{model}-{p.parent.name}"             # official: ckpt/outdoor/x.pth -> pre__x-outdoor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name, or hazy-input")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--res_dir", default="results")
    ap.add_argument("--long_side", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=0, help="tiled inference size (0 = whole image)")
    ap.add_argument("--save", type=int, default=6, help="save first N outputs per set (for figures)")
    ap.add_argument("--sets", nargs="*", default=list(TEST_SETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_partial", action="store_true", help="skip the image-count check (debugging only)")
    a = ap.parse_args()
    assert a.model in ("hazy-input", "dcp") or a.ckpt, "--ckpt is required for learned models"
    name = a.name or default_name(a.model, a.ckpt)

    model = None
    if a.model != "hazy-input":
        model = build(a.model).to(a.device).eval()
        if a.model != "dcp":
            print("load", a.ckpt, load_weights(model, a.ckpt))
    out = Path(a.res_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(out / "summary.json")) if (out / "summary.json").exists() else {}
    for s in a.sets:
        data = load_test_set(s, a.long_side, allow_partial=a.allow_partial)
        rows, t0 = [], time.time()
        for i, (fname, x8, y8) in enumerate(data):
            y = y8.astype(np.float32) / 255.0
            if model is None:
                p8 = x8
            else:
                x = torch.from_numpy(x8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(a.device)
                p = infer(model, x, tile=a.tile)[0].permute(1, 2, 0).cpu().numpy()
                p8 = np.round(p * 255).astype(np.uint8)              # score what would be saved
            p = p8.astype(np.float32) / 255.0
            rows.append({"image": fname, "psnr": psnr(p, y), "ssim": ssim(p, y, a.device)})
            if i < a.save:
                (out / s).mkdir(exist_ok=True)
                Image.fromarray(p8).save(out / s / f"{Path(fname).stem}.png")
        with open(out / f"{s}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        summary[s] = {"n": len(rows), "psnr": float(np.mean([r["psnr"] for r in rows])),
                      "ssim": float(np.mean([r["ssim"] for r in rows])), "sec": round(time.time() - t0, 1)}
        print(f"{name:40s} {s:7s} n={len(rows):3d} PSNR {summary[s]['psnr']:.3f} SSIM {summary[s]['ssim']:.4f}", flush=True)
    summary["_meta"] = {"model": a.model, "ckpt": a.ckpt, "long_side": a.long_side}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
