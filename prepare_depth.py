"""Metric depth for the clean images (computed once, shared by every arm).

Default: Apple Depth Pro (metric, sharp boundaries, sky -> far) via HF transformers.
Fallback: Depth Anything V2 metric-outdoor (--model dav2; saturates near 80 m, so the
          farthest pixels are treated as sky and the guided filter is applied).

Optional classical refinement (--guided): guided filter (He et al.) on log-depth with
the clean image as guide, so transmission edges follow image edges. Off for Depth Pro,
whose boundaries are already sharp; on by default for dav2.

Resumable: existing .npy files are skipped. --limit N processes only the first N images
(sorted by name), which is what the quick protocol uses.

Usage:
  python prepare_depth.py --clean_dir data/reside6k_ots/GT --out_dir data/depth/ots [--limit 1000]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def box(x, r):
    """Box filter via integral image (2D numpy), window clipped at the borders."""
    c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1)
    H, W = x.shape
    y0 = np.clip(np.arange(H) - r, 0, H); y1 = np.clip(np.arange(H) + r + 1, 0, H)
    x0 = np.clip(np.arange(W) - r, 0, W); x1 = np.clip(np.arange(W) + r + 1, 0, W)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    n = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s / n


def guided_filter(I, p, r=8, eps=1e-3):
    mI, mp = box(I, r), box(p, r)
    cov = box(I * p, r) - mI * mp
    var = box(I * I, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


def load_model(name, device):
    if name == "depthpro":
        # needs transformers >= 4.49 (DepthPro); weights ~1.9 GB download on first use
        from transformers import AutoImageProcessor, DepthProForDepthEstimation
        proc = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
        model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device).eval()
        state = {"half": device == "cuda"}
        if state["half"]:
            model.half()

        def forward(img):
            inp = proc(images=img, return_tensors="pt")
            pv = inp["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype)
            with torch.no_grad():
                out = model(pixel_values=pv)
            post = proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])
            return post[0]["predicted_depth"].float().cpu().numpy()

        def run(img):
            d = forward(img)
            if state["half"] and not np.isfinite(d).all():     # half precision overflowed: use float32 from now on
                print("depth: non-finite values in half precision -> switching to float32", flush=True)
                model.float(); state["half"] = False
                d = forward(img)
            return d
        return run
    if name == "dav2":
        from transformers import pipeline
        pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                        device=0 if device == "cuda" else -1)

        def run(img):
            d = pipe(img)["predicted_depth"].float()
            d = d.reshape(1, 1, *d.shape[-2:])
            d = torch.nn.functional.interpolate(d, size=(img.height, img.width), mode="bilinear")[0, 0].cpu().numpy()
            d[d >= 0.97 * d.max()] = 1000.0   # model saturates ~80 m: treat as sky / far
            return d
        return run
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="depthpro", choices=["depthpro", "dav2"])
    ap.add_argument("--max_depth", type=float, default=1000.0)
    ap.add_argument("--guided", action="store_true", help="guided-filter refinement (default on for dav2)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (sorted by name)")
    ap.add_argument("--preview", type=int, default=12)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.clean_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if a.limit:
        files = files[:a.limit]
    todo = [f for f in files if not (out / (f.stem + ".npy")).exists()]
    print(f"{len(files)} images, {len(todo)} to do, device {dev}, model {a.model}")
    if not todo:
        return
    guided = a.guided or a.model == "dav2"
    run = load_model(a.model, dev)
    med, bad = [], 0
    for i, f in enumerate(tqdm(todo)):
        img = Image.open(f).convert("RGB")
        d = np.asarray(run(img), dtype=np.float32)
        d = d.reshape(d.shape[-2], d.shape[-1])              # tolerate (1,H,W) outputs
        if i == 0:
            print(f"first depth map: shape {d.shape}, min {np.nanmin(d):.1f} m, median {np.nanmedian(d):.1f} m, "
                  f"max {np.nanmax(d):.1f} m", flush=True)
            assert np.nanmax(d) > 2.0, "depth looks like a relative (0-1) map, not metres: wrong model output?"
        if not np.isfinite(d).all():
            bad += 1
            d = np.nan_to_num(d, nan=a.max_depth, posinf=a.max_depth, neginf=0.3)
        d = np.clip(d, 0.3, a.max_depth)
        assert d.shape == (img.height, img.width), f"{f.name}: depth {d.shape} vs image {(img.height, img.width)}"
        if guided:
            g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
            d = np.exp(guided_filter(g.astype(np.float64), np.log(d).astype(np.float64), r=8, eps=1e-3))
            d = np.clip(d, 0.3, a.max_depth).astype(np.float32)
        tmp = out / (f.stem + ".tmp.npy")
        np.save(tmp, d.astype(np.float16))
        tmp.replace(out / (f.stem + ".npy"))            # atomic: an interrupted run never leaves a half file
        med.append(float(np.median(d)))
        if i < a.preview:                                # side-by-side: image | log-depth (bright = far)
            v = np.log(d); v = (v - v.min()) / (v.max() - v.min() + 1e-6)
            dep = np.repeat((v * 255).astype(np.uint8)[..., None], 3, 2)
            Image.fromarray(np.concatenate([np.asarray(img), dep], 1)).save(out / f"_preview_{f.stem}.png")
    print(f"done. median depth of these images: 10/50/90 percentile = "
          f"{np.percentile(med, 10):.0f} / {np.percentile(med, 50):.0f} / {np.percentile(med, 90):.0f} m; "
          f"{bad} maps had non-finite values (fixed)")


if __name__ == "__main__":
    main()
