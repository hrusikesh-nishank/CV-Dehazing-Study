"""Preliminary protocol: every model against our data, in a fixed time budget (default 5 hours).

  python quick.py                 # run everything (prep -> timing -> plan -> train/eval -> report)
  python quick.py --dry           # only show what would run and the time plan
  python quick.py --hours 3       # smaller budget (fewer iterations per run, same runs)
  python quick.py --report_only   # rebuild the report from whatever has finished
  python quick.py --train_extra "--no_amp"      # extra arguments for every training (rarely needed)

It is restart-safe: stop with Ctrl-C and run the same command again; finished steps are skipped
and the iteration plan (runs_quick/plan.json) is kept. Nothing here touches the full protocol:
models go to runs_quick/, numbers to results_quick/.

What it does
  A. prep     depth + (V, A) draws for the first --n_train clean images, GPU preflight, realism
              check, and a speed probe of every model (arm R3, the slowest arm)
  B. plan     the remaining time is divided over the trainings; iterations per run are set from
              the measured speed (same number of iterations for every arm of one model)
  C. run      in priority order, so stopping early still leaves a usable comparison:
              1. published models, no training      hazy input, DCP, DehazeFormer-T/B, DEA-Net, HazeFlow
              2. DehazeFormer-T from scratch         R0 (standard sRGB synthesis) vs R3 (ours)
              3. DehazeFormer-T from scratch         B0 (RESIDE's own pairs), R1, R2  (the ladder)
              4. gUNet-T from scratch                R0 vs R3  (second architecture)
              5. official checkpoints, fine-tuned    DehazeFormer-B and DEA-Net on R0 vs R3
              6. DehazeFormer-T from scratch         R0 vs R3 again with a second seed (run-to-run noise)
              7. DehazeFormer-T from scratch         H, R3H  (camera vs heterogeneity)
  D. report   results_quick/quick_report.md, tables.tex, quick_delta.png, fig_<SET>.png

Evaluation is the full-protocol code. Paired sets are evaluated completely (155 images); RTTS
uses a fixed --rtts_subset (default 500 images) to save time, so RTTS numbers from this protocol
are not comparable with the full protocol's.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from make_queues import FT_BS, FT_INIT, baseline_cmds, train_eval  # noqa: E402
from run_queue import run_queue  # noqa: E402

PARAMS, OUT, RES = "data/params_quick.npz", "runs_quick", "results_quick"
MAIN, ARCH2 = "dehazeformer-t", "gunet-t"
FT_MODELS = ["dehazeformer-b", "deanet"]
SCRATCH_STAGES = [(MAIN, ["R0", "R3"]), (MAIN, ["B0", "R1", "R2"]), (ARCH2, ["R0", "R3"])]   # then fine-tunes, then:
SEED2_STAGE = (MAIN, ["R0", "R3"])           # seed 1 of the headline comparison
LAST_STAGE = (MAIN, ["H", "R3H"])
BASELINES = [("dehazeformer-t", "outdoor"), ("dehazeformer-b", "outdoor")]
EVAL_MIN, BASELINE_MIN = 3.0, 14.0          # estimated minutes: evaluation of one run / all published models


def common(a):
    return f"--params {PARAMS} --n_train {a.n_train} --workers {a.workers}" + (f" {a.train_extra}" if a.train_extra else "")


def with_device(cmds, dev):
    """Append --device to the training / evaluation commands when it is not the default GPU."""
    if dev == "cuda":
        return cmds
    return [c + f" --device {dev}" if c.split()[1] in ("train.py", "eval_paired.py", "eval_rtts.py") else c for c in cmds]


def expected_runs():
    """Result-folder names of every training in the protocol, in priority order."""
    runs = [f"{m}__{arm}_s0" for m, arms in SCRATCH_STAGES for arm in arms]
    runs += [f"{m}__{arm}_ft_s0" for m in FT_MODELS for arm in ("R0", "R3")]
    runs += [f"{SEED2_STAGE[0]}__{arm}_s1" for arm in SEED2_STAGE[1]]
    return runs + [f"{LAST_STAGE[0]}__{arm}_s0" for arm in LAST_STAGE[1]]


def add_prep_time(minutes):
    """Preparation time accumulated over restarts (so a restart does not enlarge the training budget)."""
    f = ROOT / OUT / "_timing" / "prep_minutes.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    total = (float(f.read_text()) if f.exists() else 0.0) + minutes
    f.write_text(f"{total:.2f}")
    return total


def probe_cmd(model, a, ft=False):
    bs = FT_BS.get(model, 8) if ft else 16
    init = f" --init {FT_INIT.get(model, f'ckpt/outdoor/{model}.pth')}" if ft else ""
    return (f"python train.py --model {model} --arm R3 --bs {bs}{init} --time_only {a.probe_iters} "
            f"--timing_json {OUT}/_timing/{model}.json --out {OUT} {common(a)}")


def prep_cmds(a):
    q = ["python setup_third_party.py",
         "python tests/test_synth.py",
         "python tests/test_models.py",
         f"python prepare_depth.py --clean_dir data/reside6k_ots/GT --out_dir data/depth/ots --limit {a.n_train}",
         f"python prepare_params.py --clean_dir data/reside6k_ots/GT --depth_dir data/depth/ots --limit {a.n_train} --out {PARAMS}",
         f"python preflight.py --params {PARAMS}",
         f"python realism.py --n 300 --params {PARAMS} --res_dir {RES}"]
    q += [probe_cmd(m, a) for m in (MAIN, ARCH2)] + [probe_cmd(m, a, ft=True) for m in FT_MODELS]
    return with_device(q, a.device)


def make_plan(a, prep_minutes):
    """Divide the remaining budget: scratch runs weigh 1, fine-tunes 0.5."""
    rate = {}
    for m in [MAIN, ARCH2] + FT_MODELS:
        f = ROOT / OUT / "_timing" / f"{m}.json"
        assert f.exists(), f"speed probe result missing: {f} (did the prep queue finish?)"
        rate[m] = json.load(open(f))["its"]
    n_scratch = sum(len(arms) for _, arms in SCRATCH_STAGES) + len(SEED2_STAGE[1]) + len(LAST_STAGE[1])
    n_ft = 2 * len(FT_MODELS)
    n_runs = n_scratch + n_ft
    budget = (a.hours * 60 - prep_minutes) * 0.92 - BASELINE_MIN - EVAL_MIN * n_runs
    assert budget > 30, (f"only {budget:.0f} min left for training after preparation and evaluation: "
                         f"use a larger --hours (now {a.hours})")
    unit = budget / (n_scratch + 0.5 * n_ft)                          # minutes per scratch run

    def iters(m, minutes, lo, hi):
        # lo is a floor for usefulness; at very low speed it makes the run exceed the budget (warned below)
        return int(max(lo, min(hi, rate[m] * 60 * minutes // 100 * 100)))
    plan = {"hours": a.hours, "n_train": a.n_train, "prep_minutes": round(prep_minutes, 1),
            "train_budget_minutes": round(budget, 1), "minutes_per_scratch_run": round(unit, 1),
            "minutes_per_finetune": round(unit / 2, 1), "its_per_second": {k: round(v, 2) for k, v in rate.items()},
            "iters": {MAIN: iters(MAIN, unit, 1000, 40000), ARCH2: iters(ARCH2, unit, 1000, 40000),
                      **{m: iters(m, unit / 2, 300, 8000) for m in FT_MODELS}},
            "rtts_subset": a.rtts_subset, "n_iqa": a.n_iqa, "created": time.strftime("%Y-%m-%d %H:%M")}
    est = (sum(plan["iters"][m] / rate[m] for m, arms in SCRATCH_STAGES for _ in arms)
           + sum(plan["iters"][SEED2_STAGE[0]] / rate[SEED2_STAGE[0]] for _ in SEED2_STAGE[1])
           + sum(plan["iters"][LAST_STAGE[0]] / rate[LAST_STAGE[0]] for _ in LAST_STAGE[1])
           + sum(2 * plan["iters"][m] / rate[m] for m in FT_MODELS)) / 60
    plan["estimated_total_hours"] = round((prep_minutes + est + BASELINE_MIN + EVAL_MIN * n_runs) / 60, 2)
    (ROOT / OUT).mkdir(exist_ok=True)
    json.dump(plan, open(ROOT / OUT / "plan.json", "w"), indent=1)
    return plan


def run_cmds(plan, a):
    rt = f"--subset {plan['rtts_subset']} --n_iqa {plan['n_iqa']}"
    q = baseline_cmds(res=RES, rtts=rt, models=BASELINES)

    def scratch(model, arms, seed=0):
        it = plan["iters"][model]
        extra = f"--iters {it} --warmup {min(1000, it // 10)} {common(a)}"
        return [c for arm in arms for c in train_eval(model, arm, seed, extra, out=OUT, res=RES, rtts=rt)]

    for model, arms in SCRATCH_STAGES:
        q += scratch(model, arms)
    for m in FT_MODELS:
        init = FT_INIT.get(m, f"ckpt/outdoor/{m}.pth")
        extra = f"--init {init} --iters {plan['iters'][m]} --lr 1e-4 --warmup 100 --bs {FT_BS.get(m, 8)} {common(a)}"
        for arm in ["R0", "R3"]:
            q += train_eval(m, arm, 0, extra, tag="ft", out=OUT, res=RES, rtts=rt)
    q += scratch(*SEED2_STAGE, seed=1)
    q += scratch(*LAST_STAGE)
    return with_device(q, a.device)


def write_queue(name, cmds, header):
    p = ROOT / f"{name}.txt"
    p.write_text(f"# {header}\n" + "\n".join(cmds) + "\n", encoding="utf-8")
    return p


def report():
    """Always rebuilt (not queued), so it reflects whatever has finished."""
    rc = subprocess.run([sys.executable, str(ROOT / "quick_report.py"), "--res_dir", RES], cwd=ROOT).returncode
    if rc == 0:
        print(f"\nreport: {RES}/quick_report.md  |  LaTeX tables: {RES}/tables.tex  |  figures: {RES}/quick_delta.png, {RES}/fig_*.png")
    return rc


def show_plan(plan):
    print("\n=== quick protocol plan ===")
    print(f"  budget {plan['hours']} h | {plan['n_train']} training images | preparation took {plan['prep_minutes']} min")
    print(f"  training budget {plan['train_budget_minutes']} min -> {plan['minutes_per_scratch_run']} min per from-scratch run, "
          f"{plan['minutes_per_finetune']} min per fine-tune")
    for m, it in plan["iters"].items():
        print(f"  {m:16s} {plan['its_per_second'][m]:6.2f} it/s -> {it:6d} iterations per run")
    print("  (arm R3 is the slowest; the other arms finish earlier)")
    if "estimated_total_hours" in plan:
        print(f"  estimated total: {plan['estimated_total_hours']} h")
        if plan["estimated_total_hours"] > plan["hours"] * 1.05:
            print("  WARNING: at this speed even the minimum useful schedule does not fit the budget; "
                  "it will run longer than --hours. Check the TIMING lines in logs/quick_a_prep/ (time_share).")
    if plan["iters"][MAIN] < 3000:
        print(f"  NOTE: {plan['iters'][MAIN]} iterations is a short schedule; differences under ~0.5 dB may not show. "
              "If you can, give it more time (--hours) rather than fewer images.")
    tdir = ROOT / OUT / "_timing"
    for f in sorted(tdir.glob("*.json")) if tdir.exists() else []:
        t = json.load(open(f))
        if "share_data" in t:
            print(f"  speed probe {t['model']:16s} {t['its']:.2f} it/s: data {t['share_data']:.0%}, "
                  f"haze synthesis {t['share_synth']:.0%}, network {1 - t['share_data'] - t['share_synth']:.0%}")
    print(flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0, help="total wall-clock budget, preparation included")
    ap.add_argument("--n_train", type=int, default=1000, help="number of clean training images (first N by name)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rtts_subset", type=int, default=500)
    ap.add_argument("--n_iqa", type=int, default=100)
    ap.add_argument("--probe_iters", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_extra", default="", help='extra arguments appended to every train.py command, e.g. "--no_amp"')
    ap.add_argument("--dry", action="store_true", help="show the commands and the plan, run nothing")
    ap.add_argument("--replan", action="store_true", help="recompute the iteration plan (only before any training has run)")
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)                      # all paths in the commands are relative to the project folder
    if a.report_only:
        sys.exit(report())

    qa = write_queue("quick_a_prep", prep_cmds(a), f"quick protocol, preparation (n_train {a.n_train})")
    plan_file = ROOT / OUT / "plan.json"
    if a.dry:
        run_queue(qa, dry=True)
        if plan_file.exists():
            plan = json.load(open(plan_file)); show_plan(plan)
            run_queue(write_queue("quick_b_run", run_cmds(plan, a), "quick protocol, runs"), dry=True)
        else:
            print("\n(the iteration plan is made after the speed probes of the preparation queue have run)")
        return

    t0 = time.time()
    bad_prep = run_queue(qa, stop_on_fail=True)
    prep_minutes = add_prep_time((time.time() - t0) / 60)
    if bad_prep:
        print("\nPreparation failed: fix the failed step above (its log has the details) and run quick.py again. "
              "Nothing long has been started."); sys.exit(1)
    if plan_file.exists() and not a.replan:
        plan = json.load(open(plan_file))
        if plan["n_train"] != a.n_train or plan["hours"] != a.hours:
            print(f"NOTE: keeping the existing plan ({plan['hours']} h, n_train {plan['n_train']}); "
                  f"delete runs_quick/ and results_quick/ and use --replan to change it")
        a.n_train = plan["n_train"]
    else:
        assert not any((ROOT / OUT).glob("*/final.pth")) or not a.replan, \
            "trainings already exist in runs_quick/: delete that folder (and results_quick/) before --replan"
        plan = make_plan(a, prep_minutes)
    show_plan(plan)
    bad = run_queue(write_queue("quick_b_run", run_cmds(plan, a), f"quick protocol, runs ({json.dumps(plan['iters'])})"))
    report()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped by user; run the same command again to continue"); sys.exit(130)
