"""Sequential, restart-safe job queue for one GPU.

  python run_queue.py queues/q2_ablation_s0.txt     # run (skips commands that already finished OK)
  python run_queue.py queues/q2_ablation_s0.txt --dry      # show what would run
  python run_queue.py queues/q0_prep.txt --stop_on_fail    # stop at the first failing command

Each non-empty, non-# line of the queue file is one command. Each command writes its own log
to logs/<queue>/<n>_<hash>.log; a command whose log ends with "EXIT 0" is skipped on re-run, so
a queue can be stopped (Ctrl-C) and restarted at any time. Failed commands are retried on the
next run and listed at the end. Lines starting with "python " run with the current interpreter.

Watch a running job from a second PowerShell window:
  Get-Content (Get-ChildItem logs\\<queue>\\*.log | Sort-Object LastWriteTime | Select-Object -Last 1) -Wait -Tail 5
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


def log_path(logdir, i, cmd):
    return logdir / f"{i:03d}_{hashlib.md5(cmd.encode()).hexdigest()[:8]}.log"


def is_done(log):
    return log.exists() and log.read_text(encoding="utf-8", errors="ignore").rstrip().endswith("EXIT 0")


def run_queue(qfile, dry=False, stop_on_fail=False):
    qfile = Path(qfile)
    cmds = [l.strip() for l in qfile.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]
    logdir = Path("logs") / qfile.stem
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    failed, t_start = [], time.time()
    for i, cmd in enumerate(cmds):
        log = log_path(logdir, i, cmd)
        if is_done(log):
            print(f"[{i + 1}/{len(cmds)}] skip (done): {cmd}", flush=True)
            continue
        print(f"[{i + 1}/{len(cmds)}] {time.strftime('%d %b %H:%M')} RUN {cmd}", flush=True)
        if dry:
            continue
        real = f'"{sys.executable}" {cmd[7:]}' if cmd.startswith("python ") else cmd
        t0 = time.time()
        log.write_text(cmd + "\n\n", encoding="utf-8")
        with open(log, "ab") as f:
            rc = subprocess.run(real, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\nafter {(time.time() - t0) / 60:.1f} min\nEXIT {rc}")
        print(f"      -> exit {rc} in {(time.time() - t0) / 60:.1f} min  (log: {log})", flush=True)
        if rc != 0:
            failed.append((cmd, log))
            tail = log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-6:-2]
            print("      last lines: " + " | ".join(t.strip()[:160] for t in tail), flush=True)
            if stop_on_fail:
                print("      stopping here: the remaining steps need this one", flush=True)
                break
    print(f"queue {qfile.name}: finished in {(time.time() - t_start) / 3600:.2f} h, {len(failed)} failed", flush=True)
    for c, l in failed:
        print("  FAILED:", c, "\n     log:", l)
    return failed


if __name__ == "__main__":
    try:
        bad = run_queue(sys.argv[1], dry="--dry" in sys.argv, stop_on_fail="--stop_on_fail" in sys.argv)
    except KeyboardInterrupt:
        print("\nstopped by user; re-run the same command to continue"); sys.exit(130)
    sys.exit(1 if bad else 0)
