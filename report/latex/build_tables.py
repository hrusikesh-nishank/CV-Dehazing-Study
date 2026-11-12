"""Regenerate the report's LaTeX tables and the effect-size figure from the stored results.

    python report/latex/build_tables.py            (from the repository root)

Reads results/per_run/<run>/{OHAZE,NHHAZE,DENSE}.csv, summary.json, rtts.json and writes
report/latex/tables/*.tex and report/latex/figs/effects.{pdf,png}.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from stats import SETS, bootstrap, groups, summarize  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "results" / "per_run")
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
G = groups(RES)
S = {k: summarize(k, v) for k, v in G.items()}
NUM = {}


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def best_marks(rows, cols):
    """Bold the best value per column (PSNR/SSIM higher better)."""
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r[c], float) and not np.isnan(r[c])]
        if vals:
            m = max(vals)
            for r in rows:
                if isinstance(r[c], float) and r[c] == m:
                    r[c] = ("B", m)


def cell(v, d):
    if isinstance(v, tuple):
        return r"\textbf{" + f(v[1], d) + "}"
    return f(v, d)


# ---------------------------------------------------------------- Table: published vs ours
pub = [("pre__hazy-input", "Hazy input", "--"),
       ("pre__dcp", r"DCP~\cite{he2011dcp}", "none (prior)"),
       ("pre__dehazeformer-t-outdoor", r"DehazeFormer-T~\cite{song2023dehazeformer}", "RESIDE-OTS"),
       ("pre__dehazeformer-b-outdoor", r"DehazeFormer-B~\cite{song2023dehazeformer}", "RESIDE-OTS"),
       ("pre__deanet-ots", r"DEA-Net~\cite{chen2024deanet}", "RESIDE-OTS")]
ours = [("dehazeformer-t__B0", "DehazeFormer-T, scratch", "B0 (RESIDE pairs)"),
        ("dehazeformer-t__R0", "DehazeFormer-T, scratch", "R0 (ours, sRGB)"),
        ("dehazeformer-t__R1", "DehazeFormer-T, scratch", "R1 (ours, linear)"),
        ("dehazeformer-t__R1_rtts", "DehazeFormer-T, scratch", "R1, RTTS-calibrated"),
        ("gunet-t__B0", "gUNet-T, scratch", "B0 (RESIDE pairs)"),
        ("gunet-t__R1", "gUNet-T, scratch", "R1 (ours, linear)"),
        ("dehazeformer-b__B0_ft", "DehazeFormer-B, fine-tuned", "B0 (RESIDE pairs)"),
        ("dehazeformer-b__R3_ft", "DehazeFormer-B, fine-tuned", "R3 (ours, camera)"),
        ("deanet__B0_ft", "DEA-Net, fine-tuned", "B0 (RESIDE pairs)"),
        ("deanet__R0_ft", "DEA-Net, fine-tuned", "R0 (ours, sRGB)")]
cols = [f"{s}_{m}" for s in SETS for m in ("psnr", "ssim")]
rows = []
for k, name, data in pub + ours:
    if k in S:
        r = {"k": k, "name": name, "data": data, "seeds": S[k]["seeds"]}
        r.update({c: float(S[k][c]) for c in cols})
        rows.append(r)
best_marks(rows, cols)
L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Method & Training data & \multicolumn{2}{c}{O-HAZE} & \multicolumn{2}{c}{NH-HAZE} & \multicolumn{2}{c}{Dense-Haze} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r" & & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\", r"\midrule"]
first_ours = True
for r in rows:
    if r["k"].startswith("pre__") is False and first_ours:
        L.append(r"\midrule \multicolumn{8}{l}{\emph{Trained here (1{,}000 clean images, short schedule)}} \\")
        first_ours = False
    seeds = f"$^{{\\times{r['seeds']}}}$" if r["seeds"] > 1 else ""
    L.append(f"{r['name']}{seeds} & {r['data']} & " + " & ".join(cell(r[c], 2 if c.endswith('psnr') else 3) for c in cols) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_main.tex").write_text("\n".join(L))

# ---------------------------------------------------------------- Table: ablation steps (DehazeFormer-T)
M = "dehazeformer-t"
steps = [("R0", "B0", "Our haze (sRGB ASM) vs.\\ RESIDE haze", "a"),
         ("N0", "B0", "\\quad same with grey airlight", "a"),
         ("R0_uncal", "B0", "\\quad same without density calibration", "a"),
         ("R0_rtts", "B0", "\\quad same, calibrated on RTTS only", "a"),
         ("R0", "N0", "Sky-coloured vs.\\ grey airlight", "b"),
         ("R1", "R0", "+ linear-radiance compositing", "c"),
         ("R2", "R1", "+ sensor noise after haze", "c"),
         ("R3", "R2", "+ randomised ISP and JPEG", "c"),
         ("R3N", "R1", "ISP and JPEG without noise", "c"),
         ("H", "R0", "+ heterogeneous extinction (sRGB)", "d"),
         ("R3H", "R3", "+ heterogeneous extinction (camera)", "d"),
         ("R1", "B0", "Best variant (R1) vs.\\ RESIDE haze", "e"),
         ("R1_rtts", "B0", "\\quad R1 calibrated on RTTS only", "e")]
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"Comparison ($A - B$) & O-HAZE & NH-HAZE & Dense-Haze & Seeds \\", r"\midrule"]
prev = None
deltas = {}
for a, b, label, grp in steps:
    ka, kb = f"{M}__{a}", f"{M}__{b}"
    if ka not in S or kb not in S:
        continue
    if prev and grp != prev:
        L.append(r"\addlinespace")
    prev = grp
    cells = []
    for s in SETS:
        r = bootstrap(ka, kb, s, res_dir=RES)
        deltas[(a, b, s)] = r
        txt = f"{r['delta']:+.2f} {{\\scriptsize[{r['lo']:+.2f}, {r['hi']:+.2f}]}}"
        if r["significant"] and abs(r["delta"]) >= 0.3:
            txt = r"\textbf{" + f"{r['delta']:+.2f}" + "}" + f" {{\\scriptsize[{r['lo']:+.2f}, {r['hi']:+.2f}]}}"
        cells.append(txt)
    ns = f"{S[ka]['seeds']}/{S[kb]['seeds']}"
    L.append(f"{label} ({a.replace('_', r'\_')} $-$ {b.replace('_', r'\_')}) & " + " & ".join(cells) + f" & {ns} \\\\")
    NUM[f"{a}-{b}"] = {s: deltas[(a, b, s)] for s in SETS}
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_steps.tex").write_text("\n".join(L))

# ---------------------------------------------------------------- Table: per-arm absolute numbers (DehazeFormer-T)
arms = [("B0", "RESIDE-OTS hazy images (official synthesis)"), ("N0", "sRGB ASM, grey airlight"),
        ("R0_uncal", "sRGB ASM, no density calibration"), ("R0_rtts", "sRGB ASM, calibrated on RTTS only"),
        ("R0", "sRGB ASM, sky-coloured airlight"), ("R1", "+ linear-radiance compositing"),
        ("R1_rtts", "R1, calibrated on RTTS only"),
        ("R2", "+ post-haze sensor noise"), ("R3", "+ randomised ISP + JPEG"), ("R3N", "R3 without sensor noise"),
        ("H", "R0 + heterogeneous extinction"), ("R3H", "R3 + heterogeneous extinction")]
rows = []
for a, desc in arms:
    k = f"{M}__{a}"
    if k in S:
        r = {"a": a, "desc": desc, "seeds": S[k]["seeds"]}
        r.update({c: float(S[k][c]) for c in cols})
        r["std"] = [S[k].get(f"{s}_psnr_std", np.nan) for s in SETS]
        rows.append(r)
best_marks(rows, cols)
L = [r"\begin{tabular}{llccccccc}", r"\toprule",
     r"Arm & Hazy-image formation & \multicolumn{2}{c}{O-HAZE} & \multicolumn{2}{c}{NH-HAZE} & \multicolumn{2}{c}{Dense-Haze} & Seeds \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r" & & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM & \\", r"\midrule"]
for r in rows:
    L.append(f"{r['a'].replace('_', r'\_')} & {r['desc']} & " + " & ".join(cell(r[c], 2 if c.endswith('psnr') else 3) for c in cols)
             + f" & {r['seeds']} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_arms.tex").write_text("\n".join(L))

# ---------------------------------------------------------------- Table: other networks
nets = [("gunet-t", "", "gUNet-T (scratch)"), ("dehazeformer-b", "_ft", "DehazeFormer-B (fine-tuned)"),
        ("deanet", "_ft", "DEA-Net (fine-tuned)")]
pairs = [("R0", "B0"), ("R1", "B0"), ("R3", "B0"), ("R1", "R0"), ("R1_rtts", "B0")]
L = [r"\begin{tabular}{llccc}", r"\toprule", r"Network & Comparison & O-HAZE & NH-HAZE & Dense-Haze \\", r"\midrule"]
first = True
for m, tag, name in nets:
    got = []
    for a, b in pairs:
        ka, kb = f"{m}__{a}{tag}", f"{m}__{b}{tag}"
        if ka in S and kb in S:
            cells = []
            for s in SETS:
                r = bootstrap(ka, kb, s, res_dir=RES)
                txt = f"{r['delta']:+.2f} {{\\scriptsize[{r['lo']:+.2f}, {r['hi']:+.2f}]}}"
                if r["significant"] and abs(r["delta"]) >= 0.3:
                    txt = r"\textbf{" + f"{r['delta']:+.2f}" + "}" + f" {{\\scriptsize[{r['lo']:+.2f}, {r['hi']:+.2f}]}}"
                cells.append(txt)
                NUM[f"{m}{tag}:{a}-{b}:{s}"] = r
            got.append(f"{a.replace('_', r'\_')} $-$ {b} & " + " & ".join(cells) + r" \\")
    if got:
        if not first:
            L.append(r"\midrule")
        first = False
        L.append(r"\multirow{" + str(len(got)) + "}{*}{" + name + "} & " + got[0])
        L += [" & " + g for g in got[1:]]
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_nets.tex").write_text("\n".join(L))

# ---------------------------------------------------------------- Table: RTTS (runs that have it)
rt = [("pre__hazy-input", "Hazy input"), ("pre__dcp", "DCP"), ("pre__dehazeformer-b-outdoor", "DehazeFormer-B (official)"),
      ("pre__deanet-ots", "DEA-Net (official)"), ("pre__hazeflow-iccv25", r"HazeFlow$^\dagger$ (published outputs)"),
      ("dehazeformer-t__B0", "DehazeFormer-T, B0"), ("dehazeformer-t__R0", "DehazeFormer-T, R0"),
      ("dehazeformer-t__R1", "DehazeFormer-T, R1"), ("dehazeformer-t__R3", "DehazeFormer-T, R3"),
      ("dehazeformer-b__R0_ft", "DehazeFormer-B ft, R0"), ("dehazeformer-b__R3_ft", "DehazeFormer-B ft, R3")]
L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r"Method & mAP$_{50}\uparrow$ & NIQE$\downarrow$ & BRISQUE$\downarrow$ & MUSIQ$\uparrow$ & CLIPIQA$\uparrow$ \\", r"\midrule"]
for k, name in rt:
    if k in S and not np.isnan(S[k]["RTTS_mAP50"]):
        x = S[k]
        L.append(f"{name} & {f(x['RTTS_mAP50'], 1)} & {f(x['niqe'])} & {f(x['brisque'], 1)} & {f(x['musiq'], 1)} & {f(x['clipiqa'], 3)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_rtts.tex").write_text("\n".join(L))

# ---------------------------------------------------------------- Figure: effect sizes
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42})
items = [(("R0", "B0"), "our haze vs RESIDE\n(R0 - B0)"), (("N0", "B0"), "grey airlight\n(N0 - B0)"),
         (("R0_uncal", "B0"), "no density calib.\n(R0$_{unc}$ - B0)"), (("R0_rtts", "B0"), "RTTS-calibrated\n(R0$_{rtts}$ - B0)"),
         (("R1", "R0"), "linear compositing\n(R1 - R0)"), (("R2", "R1"), "sensor noise\n(R2 - R1)"),
         (("R3N", "R1"), "ISP+JPEG\nno noise\n(R3N - R1)"), (("H", "R0"), "heterog. $\\beta$\n(H - R0)")]
items = [(p, lab) for p, lab in items if (p[0], p[1], "OHAZE") in deltas]
fig, ax = plt.subplots(figsize=(6.8, 2.9))
w = 0.26
colors = {"OHAZE": "#2B5C8A", "NHHAZE": "#D9822B", "DENSE": "#4E8F5B"}
names = {"OHAZE": "O-HAZE", "NHHAZE": "NH-HAZE", "DENSE": "Dense-Haze"}
for j, s in enumerate(SETS):
    xs = np.arange(len(items)) + (j - 1) * w
    d = [deltas[(p[0], p[1], s)]["delta"] for p, _ in items]
    lo = [deltas[(p[0], p[1], s)]["delta"] - deltas[(p[0], p[1], s)]["lo"] for p, _ in items]
    hi = [deltas[(p[0], p[1], s)]["hi"] - deltas[(p[0], p[1], s)]["delta"] for p, _ in items]
    ax.bar(xs, d, w, yerr=[lo, hi], capsize=1.5, color=colors[s], label=names[s], error_kw={"lw": 0.6})
ax.axhspan(-0.3, 0.3, color="0.9", zorder=0)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(np.arange(len(items)))
ax.set_xticklabels([lab.replace("\n", " ", 1) for _, lab in items], fontsize=6.2, rotation=28, ha="right", rotation_mode="anchor")
ax.set_ylabel("PSNR difference (dB)")
ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper right")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "effects.pdf", bbox_inches="tight")
plt.savefig(FIG / "effects.png", dpi=200, bbox_inches="tight")

json.dump({k: v for k, v in NUM.items()}, open(OUT / "numbers.json", "w"), indent=1, default=float)
print("groups:", len(S), "| tables:", sorted(p.name for p in OUT.glob("*.tex")))

# ---------------------------------------------------------------- Table: calibration controls (PSNR and SSIM)
cal = [("dehazeformer-t", "R0", "B0", "calibrated on test-set hazy inputs"),
       ("dehazeformer-t", "R0_uncal", "B0", "not calibrated"),
       ("dehazeformer-t", "R0_rtts", "B0", "calibrated on RTTS only"),
       ("dehazeformer-t", "R1_rtts", "B0", "R1, calibrated on RTTS only"),
       ("dehazeformer-t", "R0", "R0_rtts", "test-set vs.\\ RTTS calibration"),
       ("gunet-t", "R1", "B0", "R1, calibrated on test-set hazy inputs"),
       ("gunet-t", "R1_rtts", "B0", "R1, calibrated on RTTS only")]
L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Network & Comparison & \multicolumn{3}{c}{$\Delta$PSNR (dB)} & \multicolumn{3}{c}{$\Delta$SSIM} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r" & & O & NH & Dense & O & NH & Dense \\", r"\midrule"]
prev = None
for m, a, b, what in cal:
    ka, kb = f"{m}__{a}", f"{m}__{b}"
    if ka not in S or kb not in S:
        continue
    if prev and m != prev:
        L.append(r"\midrule")
    prev = m
    cells = []
    for metric, d in (("psnr", 2), ("ssim", 3)):
        for s in SETS:
            r = bootstrap(ka, kb, s, metric=metric, res_dir=RES)
            NUM[f"{m}:{a}-{b}:{metric}:{s}"] = r
            txt = f"{r['delta']:+.{d}f}"
            thr = 0.3 if metric == "psnr" else 0.0
            cells.append(r"\textbf{" + txt + "}" if r["significant"] and abs(r["delta"]) >= thr else txt)
    name = {"dehazeformer-t": "DehazeFormer-T", "gunet-t": "gUNet-T"}[m]
    L.append(f"{name} & {a.replace('_', r'\_')} $-$ {b.replace('_', r'\_')} ({what}) & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_calib.tex").write_text("\n".join(L))
json.dump({k: v for k, v in NUM.items()}, open(OUT / "numbers.json", "w"), indent=1, default=float)
print("calibration table written")
