"""Qualitative comparison figure for the report: hazy | methods ... | GT, one row per image.

  python make_figures.py --runs pre__dcp pre__dehazeformer-b-outdoor dehazeformer-t__R0_s0 dehazeformer-t__R3_s0 \
                         --labels DCP DehazeFormer-B "Ours R0" "Ours R3" --set OHAZE --n 3
  python make_figures.py --res_dir results_quick ...          # quick protocol

Uses the outputs saved by eval_paired.py (first 6 images per set) and the cached, resized test
set, so every panel has the size the metrics were computed on.  -> <res_dir>/fig_<SET>.png / .pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from haze.data import TEST_SETS, load_test_set  # noqa: E402
from haze.metrics import psnr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="result folder names inside --res_dir")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--set", default="OHAZE", choices=list(TEST_SETS))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--res_dir", default="results")
    ap.add_argument("--long_side", type=int, default=1024)
    ap.add_argument("--allow_partial", action="store_true")
    a = ap.parse_args()
    labels = a.labels or a.runs
    assert len(labels) == len(a.runs), "--labels must have one entry per run"
    data = load_test_set(a.set, a.long_side, allow_partial=a.allow_partial)[:a.n]
    cols = 2 + len(a.runs)
    fig, axes = plt.subplots(len(data), cols, figsize=(2.2 * cols, 1.7 * len(data) + 0.5), squeeze=False)
    for i, (fname, hz8, gt8) in enumerate(data):
        gt = gt8.astype(np.float32) / 255
        ims = [hz8.astype(np.float32) / 255]
        for r in a.runs:
            p = Path(a.res_dir) / r / a.set / f"{Path(fname).stem}.png"
            o = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255 if p.exists() else None
            if o is None or o.shape != gt.shape:
                print(f"WARNING: no saved output for {r} / {fname} (run eval_paired.py with --save >= {a.n})")
                o = np.zeros_like(gt)
            ims.append(o)
        ims.append(gt)
        for j, (ax, im) in enumerate(zip(axes[i], ims)):
            ax.imshow(np.clip(im, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            if j < cols - 1:
                ax.set_xlabel(f"{psnr(im, gt):.2f} dB", fontsize=7)
            if i == 0:
                ax.set_title((["Hazy"] + labels + ["GT"])[j], fontsize=8)
    plt.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        plt.savefig(Path(a.res_dir) / f"fig_{a.set}.{ext}", dpi=200)
    print(f"wrote {Path(a.res_dir) / ('fig_' + a.set)}.png/.pdf")


if __name__ == "__main__":
    main()
