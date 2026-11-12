"""Report for the preliminary (quick) protocol, built from whatever results exist.

  python quick_report.py --res_dir results_quick

Writes into <res_dir>:
  quick_report.md    the four tables below, as Markdown, with a plain-language verdict
  quick_delta.png    bar chart: PSNR gain of our data (R3) over standard synthesis (R0), per model and test set
  tables.tex         the same numbers as LaTeX (make_tables.py)
  fig_<SET>.png      qualitative comparison (make_figures.py)

Tables
  1. Our data vs standard synthesis, same network and images (R3 - R0), with 95 % bootstrap CIs
  2. Image-formation ladder for the main model (B0, R0, R1, R2, R3, H, R3H)
  3. Published models vs ours, one protocol
  4. Camera vs heterogeneity (R3H - H against R3H - R3)
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from make_tables import ARM_DESC, ARM_ORDER, PRETRAINED  # noqa: E402
from quick import expected_runs  # noqa: E402
from stats import SETS, bootstrap, groups, summarize  # noqa: E402

NAME = {"OHAZE": "O-HAZE", "NHHAZE": "NH-HAZE", "DENSE": "Dense-Haze"}
NICE = {"dehazeformer-t": "DehazeFormer-T", "dehazeformer-s": "DehazeFormer-S", "dehazeformer-b": "DehazeFormer-B",
        "dehazeformer-m": "DehazeFormer-M", "gunet-t": "gUNet-T", "gunet-s": "gUNet-S", "deanet": "DEA-Net", "ffanet": "FFA-Net"}
NOISE = 0.3          # dB; smaller PSNR differences are treated as noise (pre-registered)


def num(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def clean_tex(s):
    return s.replace("~\\cite{he2009dcp}", "").replace("~\\cite{song2023dehazeformer}", "") \
            .replace("~\\cite{chen2024deanet}", "").replace("~\\cite{shin2025hazeflow}", "")


def metric_cells(r):
    c = []
    for s in SETS:
        c += [num(r.get(f"{s}_psnr")), num(r.get(f"{s}_ssim"), 3)]
    return c + [num(r.get("RTTS_mAP50"), 1), num(r.get("musiq")), num(r.get("brisque"))]


METRIC_HEAD = ["O-HAZE PSNR", "SSIM", "NH-HAZE PSNR", "SSIM", "Dense-Haze PSNR", "SSIM", "RTTS mAP50",
               "MUSIQ (higher = better)", "BRISQUE (lower = better)"]


def md_table(head, rows):
    return ("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
            + "".join("| " + " | ".join(r) + " |\n" for r in rows) + "\n")


def pairs_available(S):
    """(model, tag) for which both R0 and R3 results exist, main-table order."""
    out = []
    for k in S:
        if "__R3" in k:
            model, rest = k.split("__", 1)
            tag = rest[2:]                          # "" or "_ft"
            if f"{model}__R0{tag}" in S and tag in ("", "_ft"):
                out.append((model, tag))
    order = ["dehazeformer-t", "gunet-t", "dehazeformer-s", "dehazeformer-b", "dehazeformer-m", "deanet", "ffanet"]
    return sorted(out, key=lambda p: (p[1] != "", order.index(p[0]) if p[0] in order else 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results_quick")
    ap.add_argument("--runs_dir", default="runs_quick")
    ap.add_argument("--model", default="dehazeformer-t")
    a = ap.parse_args()
    res = Path(a.res_dir)
    G = groups(a.res_dir)
    assert G, f"no results in {a.res_dir} yet"
    S = {k: summarize(k, v) for k, v in G.items()}
    plan = json.load(open(Path(a.runs_dir) / "plan.json")) if (Path(a.runs_dir) / "plan.json").exists() else {}
    md = ["# Preliminary results (quick protocol)\n\n"]
    if plan:
        its = ", ".join(f"{NICE.get(m, m)} {n}" for m, n in plan["iters"].items())
        md.append(f"Settings: {plan['n_train']} training images (RESIDE-6K outdoor, first by name), one seed "
                  f"(two for the DehazeFormer-T R0/R3 comparison), "
                  f"iterations per run: {its}. Fine-tuned models start from the official RESIDE-OTS weights. "
                  f"Paired test sets are complete (45 / 55 / 55 images, long side 1024); RTTS uses a fixed subset of "
                  f"{plan['rtts_subset']} images (quality metrics on {plan['n_iqa']}). "
                  f"Total budget {plan['hours']} h on one GPU.\n\n")
    todo = [r for r in expected_runs() if not (res / r / "summary.json").exists()]
    if todo:
        md.append(f"**Not finished yet ({len(todo)} of {len(expected_runs())} trainings):** " + ", ".join(todo) + ".\n\n")
    md.append(f"Reading rule (fixed before the runs): a PSNR difference below {NOISE} dB is treated as noise; a difference "
              "counts when its 95 % paired-bootstrap interval over the test images excludes zero and, where two seeds "
              "exist, it is also larger than twice the seed-to-seed standard deviation. Short schedules and one or two "
              "seeds make these preliminary indications, not final claims.\n\n")

    # ---------------- 1. our data vs standard synthesis ----------------
    P = pairs_available(S)
    md.append("## 1. Does forming the haze through the camera help? (R3 minus R0, same network, images and haze)\n\n")
    rows, bars, wins, sig_pos, sig_neg, total = [], [], 0, 0, 0, 0
    for model, tag in P:
        a_, b_ = f"{model}__R3{tag}", f"{model}__R0{tag}"
        cells, entry, seeds, sd = [NICE.get(model, model), "fine-tuned official weights" if tag else "from scratch"], {}, 1, 0.0
        for s in SETS:
            try:
                r = bootstrap(a_, b_, s, res_dir=a.res_dir)
            except AssertionError:
                cells.append("--"); continue
            total += 1; wins += r["delta"] > 0
            pos = r["significant"] and r["delta"] >= NOISE; neg = r["significant"] and r["delta"] <= -NOISE
            seeds = max(seeds, r["n_seeds"]); sd = max(sd, r["seed_std"])
            sig_pos += pos; sig_neg += neg
            flag = " **(+)**" if pos else (" **(-)**" if neg else "")
            cells.append(f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]{flag}")
            entry[s] = r
        dm = S[a_].get("RTTS_mAP50", float("nan")) - S[b_].get("RTTS_mAP50", float("nan"))
        cells += [num(dm, 1, sign=True), f"{seeds} (seed std {sd:.2f} dB)" if seeds > 1 else "1"]
        rows.append(cells); bars.append((f"{NICE.get(model, model)}\n({'fine-tuned' if tag else 'scratch'})", entry))
    if rows:
        md.append(md_table(["Model", "Training"] + [f"{NAME[s]} dPSNR [95 % CI]" for s in SETS] + ["RTTS dmAP50", "Seeds"], rows))
        md.append(f"**(+)** / **(-)**: passes the reading rule in favour of / against our data. dPSNR and dmAP50 are R3 minus R0 "
                  "(positive = our data is better). RTTS dmAP50 has no interval (one number per run).\n\n")
        md.append(f"**Summary:** our data (R3) gives a higher PSNR than standard sRGB synthesis (R0) in {wins} of {total} "
                  f"model x test-set comparisons; {sig_pos} are clear gains, {sig_neg} are clear losses, "
                  f"{total - sig_pos - sig_neg} are within noise.\n\n")
    else:
        md.append("No model has both R0 and R3 results yet.\n\n")

    # ---------------- 1b. synthesis recipe vs RESIDE, and each camera step ----------------
    steps = [("R0", "B0", "our physically grounded haze (sRGB) vs RESIDE's own hazy images"),
             ("R0_uncal", "B0", "same, without matching the haze density to real haze (V 50 m-5 km, no rejection)"),
             ("R0_rtts", "B0", "our haze calibrated ONLY on RTTS (disjoint unpaired real hazy set) vs RESIDE"),
             ("R1_rtts", "B0", "same with linear compositing (R1), calibrated only on RTTS, vs RESIDE"),
             ("R0_rtts", "R0_uncal", "calibrated on RTTS vs not calibrated at all"),
             ("R0", "N0", "sky-coloured airlight vs grey airlight (same depth and visibility)"),
             ("N0", "B0", "our depth + visibility alone (grey airlight) vs RESIDE"),
             ("R1", "R0", "+ linear-radiance compositing"),
             ("R2", "R1", "+ sensor noise after the haze"),
             ("R3", "R2", "+ randomised ISP and JPEG"),
             ("R3N", "R1", "+ randomised ISP and JPEG, without noise"),
             ("R3", "R3N", "effect of the sensor noise inside the full camera"),
             ("R4", "R3", "+ forward-scatter blur"),
             ("R3", "B0", "full pipeline vs RESIDE's own hazy images"),
             ("R1", "B0", "best variant (R1) vs RESIDE's own hazy images")]
    rows = []
    for x, y, what in steps:
        ka, kb = f"{a.model}__{x}", f"{a.model}__{y}"
        if ka in S and kb in S:
            cells = [f"{x} - {y}", what]
            for s in SETS:
                r = bootstrap(ka, kb, s, res_dir=a.res_dir)
                flag = " **(+)**" if r["significant"] and r["delta"] >= NOISE else (
                    " **(-)**" if r["significant"] and r["delta"] <= -NOISE else "")
                cells.append(f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]{flag}")
            rows.append(cells)
    if rows:
        md.append(f"## 1b. Each step of the synthesis ({NICE.get(a.model, a.model)}, from scratch, PSNR difference [95 % CI])\n\n")
        md.append(md_table(["Step", "What changes"] + [NAME[s] for s in SETS], rows))
        md.append("Same network, same 1,000 clean images, same schedule and seed(s); only the way the hazy training images "
                  "are formed differs. B0 uses RESIDE's own hazy versions of the same clean images.\n\n")

    # ---------------- 1c. same comparisons on the other networks ----------------
    rows = []
    for model, tag, how in [("gunet-t", "", "from scratch"), ("dehazeformer-b", "_ft", "fine-tuned"),
                            ("deanet", "_ft", "fine-tuned")]:
        for x, y in [("R0", "B0"), ("R1", "R0"), ("R1", "B0"), ("R3", "B0"), ("R1_rtts", "B0")]:
            ka, kb = f"{model}__{x}{tag}", f"{model}__{y}{tag}"
            if ka in S and kb in S:
                cells = [f"{NICE.get(model, model)} ({how})", f"{x} - {y}"]
                for s in SETS:
                    r = bootstrap(ka, kb, s, res_dir=a.res_dir)
                    flag = " **(+)**" if r["significant"] and r["delta"] >= NOISE else (
                        " **(-)**" if r["significant"] and r["delta"] <= -NOISE else "")
                    cells.append(f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]{flag}")
                rows.append(cells)
    if rows:
        md.append("## 1c. The same comparisons on the other networks (PSNR difference [95 % CI])\n\n")
        md.append(md_table(["Network", "Comparison"] + [NAME[s] for s in SETS], rows))
        md.append("For fine-tuned networks, B0 means the official weights fine-tuned on RESIDE's own hazy images for the "
                  "same number of iterations: it separates the effect of our data from the effect of extra training.\n\n")

    # ---------------- 2. ladder ----------------
    md.append(f"## 2. Image-formation ladder ({NICE.get(a.model, a.model)}, from scratch)\n\n")
    rows = []
    base = S.get(f"{a.model}__R0")
    for arm in ARM_ORDER:
        k = f"{a.model}__{arm}"
        if k in S:
            d = (num(sum(S[k][f"{s}_psnr"] - base[f"{s}_psnr"] for s in SETS) / len(SETS), 2, sign=True)
                 if base and arm != "R0" else "--")
            desc = ARM_DESC[arm].replace("$\\beta$", "beta").replace("$V$", "V")
            rows.append([arm, desc] + metric_cells(S[k]) + [d])
    if rows:
        md.append(md_table(["Arm", "Synthesis"] + METRIC_HEAD + ["mean dPSNR vs R0"], rows))
        md.append("Each row adds one step of the camera model to the row above (R0 -> R1 -> R2 -> R3). B0 is RESIDE's own "
                  "hazy images for the same clean images. Rows with two seeds show the mean.\n\n")
    else:
        md.append("No from-scratch result yet.\n\n")

    # ---------------- 3. published models vs ours ----------------
    md.append("## 3. Published models and ours under one protocol\n\n")
    rows = [[clean_tex(n), tr] + metric_cells(S[k]) for k, n, tr in PRETRAINED if k in S]
    if f"{a.model}__B0" in S:
        rows.append([f"Baseline: {NICE.get(a.model, a.model)} trained on RESIDE pairs (B0)", f"{plan.get('n_train', '?')} OTS pairs"]
                    + metric_cells(S[f"{a.model}__B0"]))
    for model, tag in P:
        for arm in ["R0", "R3"]:
            k = f"{model}__{arm}{tag}"
            what = (f"{NICE.get(model, model)}, official weights fine-tuned on {arm}" if tag
                    else f"{NICE.get(model, model)} trained on {arm}")
            rows.append([("**Ours**: " if arm == "R3" else "Baseline: ") + what,
                         f"{plan.get('n_train', '?')} OTS clean images" + (" + official" if tag else "")] + metric_cells(S[k]))
    md.append(md_table(["Method", "Training data"] + METRIC_HEAD, rows))
    md.append("HazeFlow is scored from its published RTTS outputs (no paired-set outputs are published); it was trained with "
              "real unpaired hazy images, RTTS listed among them. Models trained here ran for minutes, the published ones "
              "for days: compare rows 'trained on R0' and 'trained on R3' for the effect of the data.\n\n")

    # ---------------- 4. claim ----------------
    if f"{a.model}__R3H" in S and f"{a.model}__H" in S and f"{a.model}__R3" in S:
        md.append("## 4. Camera versus heterogeneity (cost in PSNR of removing each from R3H)\n\n")
        md.append("| Test set | remove camera (R3H -> H) | remove heterogeneity (R3H -> R3) | reading |\n|---|---|---|---|\n")

        def verdict(r):
            if r["lo"] > 0:
                return "helps"
            if r["hi"] < 0:
                return "hurts"
            return "no clear effect"
        for s in SETS:
            c = bootstrap(f"{a.model}__R3H", f"{a.model}__H", s, res_dir=a.res_dir)
            h = bootstrap(f"{a.model}__R3H", f"{a.model}__R3", s, res_dir=a.res_dir)
            md.append(f"| {NAME[s]} | {c['delta']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] | {h['delta']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}] "
                      f"| camera {verdict(c)}; heterogeneity {verdict(h)} |\n")
        md.append("\nPositive = removing the component lowers PSNR (the component helps). NH-HAZE was declared in advance "
                  "as the set where heterogeneity may matter more.\n\n")

    (res / "quick_report.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {res / 'quick_report.md'}")

    # ---------------- figure ----------------
    if bars:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(max(6.5, 1.9 * len(bars) + 2.2), 4.2))
        w = 0.25
        colors = {"OHAZE": "#1F5FA8", "NHHAZE": "#E08A1E", "DENSE": "#3B8F5A"}
        for j, s in enumerate(SETS):
            xs = [i + (j - 1) * w for i, (_, e) in enumerate(bars) if s in e]
            d = [e[s]["delta"] for _, e in bars if s in e]
            err = [[e[s]["delta"] - e[s]["lo"] for _, e in bars if s in e], [e[s]["hi"] - e[s]["delta"] for _, e in bars if s in e]]
            ax.bar(xs, d, w, yerr=err, capsize=3, color=colors[s], label=NAME[s], error_kw={"lw": 1})
        ax.axhspan(-NOISE, NOISE, color="0.88", zorder=0, label=f"noise band (\u00b1{NOISE} dB)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax.set_ylabel("PSNR gain of R3 over R0 (dB)")
        ax.set_title("PSNR gain of our data (R3) over standard synthesis (R0)\n"
                     "same network, images and haze; error bars = 95 % bootstrap CI", fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(res / f"quick_delta.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote {res / 'quick_delta.png'}")

    # ---------------- LaTeX tables + qualitative figures (best effort) ----------------
    note = ""
    if plan:
        note = (f"Preliminary protocol: {plan['n_train']} training images, one seed, "
                f"{plan['iters'].get(a.model, '?')} iterations; RTTS on a {plan['rtts_subset']}-image subset.")
    subprocess.run([sys.executable, str(ROOT / "make_tables.py"), "--res_dir", a.res_dir, "--note", note,
                    "--ft_models", "dehazeformer-b", "deanet"], cwd=ROOT)
    want = [("pre__dcp", "DCP"), ("pre__dehazeformer-b-outdoor", "DehazeFormer-B"), ("pre__deanet-ots", "DEA-Net"),
            (f"{a.model}__R0_s0", "Ours R0"), (f"{a.model}__R3_s0", "Ours R3")]
    have = [(r, l) for r, l in want if (res / r).is_dir()]
    if have and not (ROOT / "data" / "test").is_dir():
        print("qualitative figures skipped: the test images (data/test) are not present")
    elif have:
        for s in SETS:
            subprocess.run([sys.executable, str(ROOT / "make_figures.py"), "--res_dir", a.res_dir, "--set", s, "--n", "3",
                            "--runs", *[r for r, _ in have], "--labels", *[l for _, l in have]], cwd=ROOT)


if __name__ == "__main__":
    main()
