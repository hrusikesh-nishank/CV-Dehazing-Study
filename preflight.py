"""Two-minute GPU preflight: run this BEFORE any long job.

  python preflight.py              # everything
  python preflight.py --skip_iqa   # skip the image-quality metrics (they download weights)

Checks, in order: GPU and PyTorch build, Python packages, datasets and weights on disk,
every synthesis arm on the GPU, a mixed-precision training step and inference for every
model, official checkpoints on a real hazy image, the detector, the no-reference metrics,
and (if depth + parameters exist) a real training batch through the data loader.
Exit code 0 only if no check FAILS.
"""
import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RESULTS = []


def check(name, hard=True):
    def deco(fn):
        t0 = time.time()
        try:
            msg = fn() or ""
            RESULTS.append(("PASS", name, msg))
            print(f"[PASS] {name}  {msg}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            kind = "FAIL" if hard else "WARN"
            RESULTS.append((kind, name, f"{type(e).__name__}: {e}"))
            print(f"[{kind}] {name}: {type(e).__name__}: {e}", flush=True)
            if hard:
                traceback.print_exc(limit=3)
        return fn
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_iqa", action="store_true")
    ap.add_argument("--params", default=None, help="params file to test the data loader with (default: quick, then full)")
    a = ap.parse_args()
    dev = "cuda"

    @check("GPU / PyTorch build")
    def _():
        assert torch.cuda.is_available(), "CUDA is not available in this environment"
        name = torch.cuda.get_device_name(0)
        archs = torch.cuda.get_arch_list()
        if "50" in name and "RTX" in name:
            assert "sm_120" in archs, f"this PyTorch build has no sm_120 kernels (RTX 50-series): {archs}"
        x = torch.randn(256, 256, device=dev)
        assert torch.isfinite((x @ x).sum())
        free, total = torch.cuda.mem_get_info()
        return f"{name}, torch {torch.__version__}, {free / 2 ** 30:.1f}/{total / 2 ** 30:.1f} GB free"

    @check("Python packages")
    def _():
        missing = []
        for m in ["timm", "einops", "skimage", "pandas", "scipy", "matplotlib", "tqdm", "ultralytics", "torchmetrics"]:
            try:
                importlib.import_module(m)
            except Exception as e:
                missing.append(f"{m} ({type(e).__name__})")
        assert not missing, "missing: " + ", ".join(missing) + "  -> pip install -r requirements.txt"
        return "all present"

    @check("packages for depth / quality metrics", hard=False)
    def _():
        import transformers
        v = tuple(int(p) for p in transformers.__version__.split(".")[:2])
        assert v >= (4, 49), f"transformers {transformers.__version__} is too old for Depth Pro (need >= 4.49)"
        from transformers import DepthProForDepthEstimation  # noqa: F401
        import pyiqa  # noqa: F401
        return f"transformers {transformers.__version__}, pyiqa ok"

    @check("datasets and weights on disk")
    def _():
        from haze.data import TEST_SETS, pair_files
        bad = []
        n = len(pair_files("data/reside6k_ots/hazy", "data/reside6k_ots/GT"))
        if n != 3000:
            bad.append(f"reside6k_ots pairs {n} != 3000")
        for s, (hz, gt, exp) in TEST_SETS.items():
            k = len(pair_files(hz, gt))
            if k != exp:
                bad.append(f"{s} pairs {k} != {exp}")
        for d, exp in [("data/test/RTTS/JPEGImages", 4322), ("data/test/RTTS/Annotations", 4322),
                       ("data/sota_outputs/HazeFlow/RTTS", 4322)]:
            k = len(list(Path(d).glob("*.*"))) if Path(d).exists() else 0
            if k != exp:
                bad.append(f"{d}: {k} != {exp}")
        for f in ["ckpt/outdoor/dehazeformer-t.pth", "ckpt/outdoor/dehazeformer-b.pth", "ckpt/deanet/deanet-ots.pth",
                  "ckpt/yolov8m.pt", "third_party/DehazeFormer/models/dehazeformer.py", "third_party/gUNet/models/gunet.py",
                  "third_party/DEA-Net/code/model/backbone.py"]:
            if not Path(f).exists():
                bad.append(f"missing {f}")
        assert not bad, "; ".join(bad)
        return "counts and files OK"

    @check("synthesis arms on the GPU")
    def _():
        from haze.synth import ARMS, synthesize_pair
        g = torch.Generator().manual_seed(0)
        J = (torch.rand(16, 3, 256, 256, generator=g) * 0.9 + 0.05).to(dev)
        d = (5 + 400 * torch.rand(16, 1, 256, 256, generator=g)).to(dev)
        V = torch.full((16,), 150.0, device=dev); A = torch.tensor([[0.8, 0.85, 0.92]], device=dev).repeat(16, 1)
        times = {}
        for name, arm in ARMS.items():
            synthesize_pair(J, d, V, A, arm, generator=torch.Generator().manual_seed(1))
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(3):
                I, T = synthesize_pair(J, d, V, A, arm, generator=torch.Generator().manual_seed(1))
            torch.cuda.synchronize(); times[name] = (time.time() - t0) / 3 * 1000
            assert torch.isfinite(I).all() and I.min() >= 0 and I.max() <= 1 and T.shape == J.shape, name
        return "ms per batch of 16: " + ", ".join(f"{k} {v:.0f}" for k, v in times.items())

    @check("mixed-precision training speed (network alone) + inference, every model")
    def _():
        from haze.models import build, infer, n_params
        out = []
        for n, bs in [("dehazeformer-t", 16), ("gunet-t", 16), ("dehazeformer-b", 4), ("deanet", 8)]:
            torch.cuda.reset_peak_memory_stats()
            m = build(n).to(dev).train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            x = torch.rand(bs, 3, 256, 256, device=dev)
            for k in range(13):                     # 3 warm-up steps, then 10 timed (network only, no data)
                if k == 3:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", dtype=torch.float16):
                    y = m(x)
                loss = torch.nn.functional.l1_loss(y.float(), x)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize(); its = 10 / (time.time() - t0)
            assert torch.isfinite(loss), f"{n}: loss not finite under mixed precision"
            o = infer(m.eval(), torch.rand(1, 3, 680, 1024, device=dev))
            assert torch.isfinite(o).all() and o.shape[-2:] == (680, 1024), n
            out.append(f"{n} {n_params(m) / 1e6:.1f}M, network alone {its:.1f} it/s at batch {bs}, "
                       f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GB")
            del m, opt, scaler, x, y, o
            torch.cuda.empty_cache()
        return "; ".join(out)

    @check("official checkpoints on a real hazy image")
    def _():
        from haze.data import load_test_set
        from haze.metrics import psnr, ssim
        from haze.models import build, infer, load_weights
        name, x8, y8 = load_test_set("OHAZE", 1024)[4]
        x = torch.from_numpy(x8.astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        y = y8.astype(np.float32) / 255
        res = [f"hazy {psnr(x8.astype(np.float32) / 255, y):.2f}"]
        for n, ck in [("dcp", None), ("dehazeformer-t", "ckpt/outdoor/dehazeformer-t.pth"),
                      ("dehazeformer-b", "ckpt/outdoor/dehazeformer-b.pth"), ("deanet", "ckpt/deanet/deanet-ots.pth")]:
            m = build(n).to(dev).eval()
            if ck:
                r = load_weights(m, ck)
                assert not r.missing_keys and not r.unexpected_keys, n
            p = infer(m, x)[0].permute(1, 2, 0).cpu().numpy()
            res.append(f"{n} {psnr(p, y):.2f}")
            s = ssim(p, y, dev)
            assert 0 < s <= 1
            del m
        torch.cuda.empty_cache()
        return f"PSNR on {name}: " + ", ".join(res) + "  (expected roughly: hazy 11-13, models 14-20)"

    @check("detector (YOLOv8m) + RTTS annotations")
    def _():
        from ultralytics import YOLO
        from eval_rtts import RTTS2COCO, read_voc
        p = sorted(Path("data/test/RTTS/JPEGImages").glob("*.png"))[0]
        gt = read_voc(Path("data/test/RTTS/Annotations") / (p.stem + ".xml"))
        r = YOLO("ckpt/yolov8m.pt").predict(str(p), conf=0.25, verbose=False, classes=list(RTTS2COCO.values()), device=0)[0]
        return f"{p.name}: {len(gt)} labelled objects, {len(r.boxes)} detections at conf 0.25"

    if not a.skip_iqa:
        @check("no-reference quality metrics (pyiqa; downloads weights on first use)", hard=False)
        def _():
            import pyiqa
            p = str(sorted(Path("data/test/RTTS/JPEGImages").glob("*.png"))[0])
            vals = []
            for mname in ["niqe", "brisque", "musiq", "clipiqa"]:
                f = pyiqa.create_metric(mname, device=dev)
                with torch.no_grad():
                    vals.append(f"{mname} {float(f(p).item()):.3f}")
                del f
            torch.cuda.empty_cache()
            return ", ".join(vals)

    params = a.params or next((p for p in ["data/params_quick.npz", "data/params_ots.npz"] if Path(p).exists()), None)
    if params:
        @check(f"real training batch through the data loader ({params})")
        def _():
            from torch.utils.data import DataLoader
            from haze.data import CleanDepthDataset, EpochSampler
            from haze.synth import ARMS, MOR_CONST, synthesize_pair
            ds = CleanDepthDataset("data/reside6k_ots/GT", "data/depth/ots", params, 256, 0)
            dl = DataLoader(ds, batch_size=16, sampler=EpochSampler(len(ds), 1, 0), num_workers=2, drop_last=True)
            b = next(iter(dl))
            t = torch.exp(-MOR_CONST * b["depth"] / b["V"].view(-1, 1, 1, 1))
            I, T = synthesize_pair(b["J"].to(dev), b["depth"].to(dev), b["V"].to(dev), b["A"].to(dev), ARMS["R3"],
                                   generator=torch.Generator().manual_seed(0))
            assert torch.isfinite(I).all()
            return (f"{len(ds)} images; batch transmission mean {t.mean():.2f} (min {t.mean((1, 2, 3)).min():.2f}, "
                    f"max {t.mean((1, 2, 3)).max():.2f}); hazy mean {I.mean():.2f}")
    else:
        print("[INFO] depth / parameters not computed yet: the data-loader check runs after prepare_depth + prepare_params")

    fails = [r for r in RESULTS if r[0] == "FAIL"]; warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\nPREFLIGHT: {len(RESULTS) - len(fails) - len(warns)} passed, {len(warns)} warnings, {len(fails)} failed")
    for k, n, m in fails + warns:
        print(f"  {k}: {n} -> {m}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
