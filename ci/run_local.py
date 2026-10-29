"""A tiny stand-in for the GitHub runner: executes the `run:` steps of a workflow job on this machine.

    python ci/run_local.py                       # job "test" of .github/workflows/ci.yml
    python ci/run_local.py --job test --workflow .github/workflows/ci.yml

It is for learning and for a quick check before you push. It does NOT run `uses:` steps (checkout,
setup-python, artifacts), steps with an `if:` condition, installs, or Docker steps. The real run
happens on GitHub; this shows you the order of steps and which one turns red.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import yaml

SKIP_PREFIXES = ("pip install", "docker ")


def load_job(path, job):
    with open(path) as f:
        wf = yaml.safe_load(f)
    return wf["jobs"][job]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default=".github/workflows/ci.yml")
    ap.add_argument("--job", default="test")
    args = ap.parse_args(argv)

    job = load_job(args.workflow, args.job)
    print(f"job: {args.job}   ({len(job['steps'])} steps)   [local run, not GitHub]\n")
    summary = tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False)
    env = dict(os.environ, GITHUB_STEP_SUMMARY=summary.name, CI="true")
    results = []
    failed = False
    for i, step in enumerate(job["steps"], 1):
        name = step.get("name") or step.get("uses") or step.get("run", "").splitlines()[0]
        if failed:
            results.append((i, name, "not run", 0.0))
            continue
        if "uses" in step:
            results.append((i, name, "skipped (action)", 0.0))
            continue
        if "if" in step:
            results.append((i, name, "skipped (condition)", 0.0))
            continue
        cmd = step["run"].strip()
        if cmd.startswith(SKIP_PREFIXES):
            results.append((i, name, "skipped (needs runner)", 0.0))
            continue
        print(f"::group:: step {i}: {name}\n$ {cmd}")
        t0 = time.perf_counter()
        r = subprocess.run(cmd, shell=True, env=env, executable="/bin/bash")
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        print(f"-> {'ok' if ok else 'FAILED (exit code %d)' % r.returncode} in {dt:.1f}s\n")
        results.append((i, name, "passed" if ok else "FAILED", dt))
        failed = not ok

    print("=" * 64)
    for i, name, status, dt in results:
        mark = {"passed": "[ok]  ", "FAILED": "[FAIL]"}.get(status, "[--]  ")
        print(f"{mark} {i:>2}. {name:<42} {status:<24} {dt:5.1f}s" if dt else f"{mark} {i:>2}. {name:<42} {status}")
    print("=" * 64)
    print("RESULT:", "RED - the pipeline stops here" if failed else "GREEN")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
