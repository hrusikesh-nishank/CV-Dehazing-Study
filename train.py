"""Train one dehazer on one synthesis arm (haze is synthesised on the GPU, on the fly).

  python train.py --model dehazeformer-t --arm R3 --seed 0
  python train.py --model dehazeformer-t --arm B0 --seed 0            # RESIDE's own hazy pairs
  python train.py --model dehazeformer-t --arm R3 --time_only 100     # speed/memory probe
  # fine-tune an official checkpoint on our data:
  python train.py --model dehazeformer-b --arm R3 --init ckpt/outdoor/dehazeformer-b.pth --iters 8000 --lr 1e-4 --warmup 200 --bs 8 --tag ft

Run folder: <out>/<model>__<arm>[_<tag>]_s<seed>/   (final.pth = network state_dict)
Everything except --arm is identical across arms: same images, crops, flips, (V, A) draws,
initial weights and schedule for a given seed.

Schedule follows the official DehazeFormer recipe (AdamW, L1 loss, cosine decay to 1 % of
the peak LR, mixed precision, crop 256, horizontal flip), plus a linear warm-up.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from haze.data import CleanDepthDataset, EpochSampler, PairedDataset, pair_files  # noqa: E402
from haze.models import build, load_weights, n_params  # noqa: E402
from haze.synth import ARMS, synthesize_pair  # noqa: E402


def run_name(model, arm, tag, seed):
    return f"{model}__{arm}{('_' + tag) if tag else ''}_s{seed}"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dehazeformer-t")
    ap.add_argument("--arm", required=True, help=f"{' '.join(ARMS)} (on the fly) or B0 (RESIDE pairs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--n_train", type=int, default=0, help="use only the first N training images (0 = all)")
    ap.add_argument("--clean_dir", default="data/reside6k_ots/GT")
    ap.add_argument("--depth_dir", default="data/depth/ots")
    ap.add_argument("--params", default="data/params_ots.npz")
    ap.add_argument("--pairs_hazy", default="data/reside6k_ots/hazy")
    ap.add_argument("--pairs_gt", default="data/reside6k_ots/GT")
    ap.add_argument("--shot_min", type=float, default=2e-5)
    ap.add_argument("--shot_max", type=float, default=8e-4)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from (official or ours)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--time_only", type=int, default=0, metavar="N",
                    help="speed probe: run 20 warm-up + N iterations, print it/s and peak memory, exit")
    ap.add_argument("--timing_json", default=None, help="with --time_only: also write the result to this file")
    return ap.parse_args()


def main():
    a = get_args()
    assert a.arm == "B0" or a.arm in ARMS, f"unknown arm {a.arm}"
    torch.manual_seed(a.seed)
    dev = a.device
    on_gpu = dev.startswith("cuda")
    use_amp = on_gpu and not a.no_amp
    if on_gpu:
        torch.backends.cudnn.benchmark = True
    name = run_name(a.model, a.arm, a.tag, a.seed)
    out = Path(a.out) / name
    if (out / "final.pth").exists() and not a.time_only:
        print("done already:", out); return

    if a.arm == "B0":
        ds = PairedDataset(pair_files(a.pairs_hazy, a.pairs_gt), a.crop, a.seed, a.n_train)
        arm = None
    else:
        ds = CleanDepthDataset(a.clean_dir, a.depth_dir, a.params, a.crop, a.seed, a.n_train)
        arm = ARMS[a.arm]
    assert len(ds) >= a.bs, f"only {len(ds)} training images found"
    total = (20 + a.time_only) if a.time_only else a.iters
    n_epochs = math.ceil(total * a.bs / len(ds)) + 1
    dl = DataLoader(ds, batch_size=a.bs, sampler=EpochSampler(len(ds), n_epochs, a.seed), drop_last=True,
                    num_workers=a.workers, pin_memory=on_gpu, persistent_workers=False)

    model = build(a.model).to(dev)                    # weights initialised from torch.manual_seed(seed)
    if a.init:
        print("init from", a.init, load_weights(model, a.init))
    print(f"{name}: {len(ds)} training images, {n_params(model) / 1e6:.2f}M params, bs {a.bs}, "
          f"{total} iters (~{total * a.bs / len(ds):.1f} epochs), amp {use_amp}", flush=True)
    if not a.time_only:
        out.mkdir(parents=True, exist_ok=True)
        json.dump(vars(a), open(out / "args.json", "w"), indent=1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    def lr_factor(i):
        if i < a.warmup:
            return (i + 1) / a.warmup
        p = min(1.0, (i - a.warmup) / max(1, a.iters - a.warmup))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(10_000 + a.seed)          # one draw per iteration (see synth._streams)

    it, t0, log, run_loss, run_n, bad = 0, time.time(), [], 0.0, 0, 0
    t_probe = None
    split = {"data": 0.0, "synth": 0.0}                 # speed probe only: where the time goes

    def sync():
        if on_gpu:
            torch.cuda.synchronize()
    model.train()
    try:
        t_end = time.time()
        for b in dl:
            measuring = a.time_only and it >= 20
            if measuring:
                split["data"] += time.time() - t_end; t_s = time.time()
            J = b["J"].to(dev, non_blocking=True)
            if arm is None:
                I, T = b["hazy"].to(dev, non_blocking=True), J
            else:
                I, T = synthesize_pair(J, b["depth"].to(dev, non_blocking=True), b["V"].to(dev), b["A"].to(dev),
                                       arm, generator=gen, shot_range=(a.shot_min, a.shot_max))
            if measuring:
                sync(); split["synth"] += time.time() - t_s
            if it == 0 and not a.time_only:
                save_image(torch.cat([I[:8], T[:8]]), out / "train_samples.png", nrow=8)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(I)
            loss = F.l1_loss(pred.float(), T)
            if not torch.isfinite(loss):                          # skip a bad batch, stop if it persists
                bad += 1
                opt.zero_grad(set_to_none=True)
                assert bad < 50, "loss is not finite for 50 batches: try --no_amp or a lower --lr"
            else:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                run_loss += loss.item(); run_n += 1
            sched.step()
            it += 1
            if a.time_only:
                if it == 20:
                    if on_gpu:
                        torch.cuda.synchronize()
                    t_probe = time.time()
                if it >= total:
                    if on_gpu:
                        torch.cuda.synchronize()
                    el = time.time() - t_probe
                    r = a.time_only / el
                    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if on_gpu else 0.0
                    fd, fs = split["data"] / el, split["synth"] / el
                    print(f"TIMING model={a.model} arm={a.arm} bs={a.bs} its={r:.3f} mem_gb={mem:.2f} "
                          f"time_share: waiting_for_data {fd:.0%}, haze_synthesis {fs:.0%}, network {1 - fd - fs:.0%}",
                          flush=True)
                    if on_gpu and mem > 0.85 * torch.cuda.get_device_properties(0).total_memory / 2 ** 30:
                        print("  WARNING: peak memory is close to the card's total; on Windows the driver then moves data "
                              "to system RAM and training slows down many times. Use a smaller --bs.", flush=True)
                    if fd > 0.3:
                        print("  NOTE: the GPU waits for images more than 30 % of the time: try --workers 6 or 8", flush=True)
                    if a.timing_json:
                        Path(a.timing_json).parent.mkdir(parents=True, exist_ok=True)
                        json.dump({"model": a.model, "arm": a.arm, "bs": a.bs, "its": r, "mem_gb": mem,
                                   "share_data": fd, "share_synth": fs}, open(a.timing_json, "w"))
                    return
                t_end = time.time()
                continue
            if it % 200 == 0 or it == a.iters:
                el = time.time() - t0
                msg = (f"it {it}/{a.iters} loss(avg{run_n}) {run_loss / max(run_n, 1):.4f} lr {sched.get_last_lr()[0]:.2e} "
                       f"{it / el:.2f} it/s eta {(a.iters - it) / (it / el) / 60:.1f} min" + (f" skipped {bad}" if bad else ""))
                print(msg, flush=True); log.append(msg); run_loss, run_n = 0.0, 0
            if it % a.save_every == 0:
                torch.save(model.net.state_dict(), out / "last.pth")
            if it >= a.iters:
                break
    except torch.cuda.OutOfMemoryError:
        print(f"TIMING OOM model={a.model} arm={a.arm} bs={a.bs}: out of GPU memory, use a smaller --bs", flush=True)
        sys.exit(3)
    assert it >= a.iters, f"data ran out after {it} iterations"
    torch.save(model.net.state_dict(), out / "final.pth")
    (out / "train_log.txt").write_text("\n".join(log))
    print(f"saved {out / 'final.pth'}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
