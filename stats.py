"""Aggregate results across runs/seeds; paired bootstrap tests; the pre-registered claim.

  python stats.py                                                  # table of every result group
  python stats.py --compare dehazeformer-t__R3 dehazeformer-t__R0  # paired bootstrap per test set
  python stats.py --claim --model dehazeformer-t                   # (R3H-H) vs (R3H-R3)
  python stats.py --res_dir results_quick ...                      # same for the quick protocol

A *group* is a results folder name with the trailing _s<seed> removed, e.g.
results/dehazeformer-t__R3_s0, _s1, _s2 -> group dehazeformer-t__R3 (3 seeds).

Pre-registered decision rule: an effect counts only if the 95 % paired-bootstrap CI over
images (seed-averaged) excludes zero AND |delta| > 2 x the across-seed std of the larger arm
(the second condition needs at least 2 seeds per arm; with 1 seed only the CI is tested and
the result is marked "1 seed").
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SETS = ["OHAZE", "NHHAZE", "DENSE"]
NR = ["niqe", "brisque", "musiq", "clipiqa"]


def groups(res_dir="results"):
    g = defaultdict(dict)
    res = Path(res_dir)
    if not res.exists():
        return g
    for d in sorted(res.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"(.+)_s(\d+)$", d.name)
        key, seed = (m.group(1), int(m.group(2))) if m else (d.name, 0)
        if (d / "summary.json").exists() or (d / "rtts.json").exists():
            g[key][seed] = d
    return g


def _j(p):
    return json.load(open(p)) if p.exists() else {}


def summarize(key, seeds):
    r = {"group": key, "seeds": len(seeds)}
    for s in SETS:
        ps = [_j(d / "summary.json").get(s, {}).get("psnr") for d in seeds.values()]
        ss = [_j(d / "summary.json").get(s, {}).get("ssim") for d in seeds.values()]
        ps = [x for x in ps if x is not None]; ss = [x for x in ss if x is not None]
        r[f"{s}_psnr"] = np.mean(ps) if ps else np.nan
        r[f"{s}_psnr_std"] = np.std(ps) if len(ps) > 1 else np.nan
        r[f"{s}_ssim"] = np.mean(ss) if ss else np.nan
    rt = [_j(d / "rtts.json") for d in seeds.values()]
    m = [x["mAP50"] for x in rt if "mAP50" in x]
    r["RTTS_mAP50"] = np.mean(m) if m else np.nan
    r["RTTS_mAP50_std"] = np.std(m) if len(m) > 1 else np.nan
    for k in NR:
        v = [x[k] for x in rt if k in x]
        r[k] = np.mean(v) if v else np.nan
    return r


def table(res_dir="results", show=True):
    df = pd.DataFrame([summarize(k, v) for k, v in groups(res_dir).items()])
    if len(df) and show:
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
            print(df.drop(columns=[c for c in df.columns if c.endswith("_std")]).to_string(index=False))
        df.to_csv(Path(res_dir) / "all_results.csv", index=False)
    return df


def per_image(group, s, metric="psnr", res_dir="results"):
    G = groups(res_dir).get(group)
    assert G, f"no results for group '{group}' in {res_dir}"
    cols = [pd.read_csv(G[k] / f"{s}.csv").set_index("image")[metric] for k in sorted(G) if (G[k] / f"{s}.csv").exists()]
    assert cols, f"group '{group}' has no {s}.csv"
    return pd.concat(cols, axis=1)


def bootstrap(a, b, s, metric="psnr", B=10000, res_dir="results"):
    xa, xb = per_image(a, s, metric, res_dir), per_image(b, s, metric, res_dir)
    d = (xa.mean(axis=1) - xb.mean(axis=1)).dropna().values
    rng = np.random.default_rng(0)
    bs = rng.choice(d, (B, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    n_seeds = min(xa.shape[1], xb.shape[1])
    sd = max(xa.mean(axis=0).std(ddof=0) if xa.shape[1] > 1 else 0.0, xb.mean(axis=0).std(ddof=0) if xb.shape[1] > 1 else 0.0)
    sig = (lo > 0 or hi < 0) and (n_seeds < 2 or abs(d.mean()) > 2 * sd)
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "seed_std": float(sd),
            "n_img": len(d), "n_seeds": int(n_seeds), "significant": bool(sig)}


def fmt(r):
    seeds = f"seed-std {r['seed_std']:.3f}, {r['n_seeds']} seeds" if r["n_seeds"] > 1 else "1 seed"
    return (f"{r['delta']:+.3f} dB  CI[{r['lo']:+.3f},{r['hi']:+.3f}]  ({seeds}) "
            f"{'SIGNIFICANT' if r['significant'] else 'n.s.'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--model", default="dehazeformer-t")
    ap.add_argument("--metric", default="psnr")
    a = ap.parse_args()
    if a.compare:
        for s in SETS:
            print(f"{a.compare[0]} - {a.compare[1]}  {s:7s}: {fmt(bootstrap(*a.compare, s, a.metric, res_dir=a.res_dir))}")
    elif a.claim:
        m = a.model
        out = {}
        for s in SETS:
            cam = bootstrap(f"{m}__R3H", f"{m}__H", s, a.metric, res_dir=a.res_dir)
            het = bootstrap(f"{m}__R3H", f"{m}__R3", s, a.metric, res_dir=a.res_dir)
            out[s] = {"remove_camera": cam, "remove_heterogeneity": het}
            print(f"{s:7s} removing camera costs   {fmt(cam)}")
            print(f"{'':7s} removing heterog. costs {fmt(het)}")
            print(f"{'':7s} -> {'CAMERA matters more' if cam['delta'] > het['delta'] else 'HETEROGENEITY matters more (or equal)'}")
        json.dump(out, open(Path(a.res_dir) / f"claim_{m}.json", "w"), indent=1)
    else:
        table(a.res_dir)


if __name__ == "__main__":
    main()
