#!/usr/bin/env python3
"""
Run a fixed clean-protocol ablation matrix via run_reeval.py.

All runs share:
  - sport=run
  - history_source=all-prior
  - val_fraction=0.15 (checkpoint on val only)
  - seq_length=128, train_stride=64
  - loss=huber, huber_delta=12, weight_decay=1e-4
  - full-workout stitched metrics
  - held-out used only for final reporting

Usage:
  python3 run_clean_ablations.py
  python3 run_clean_ablations.py --only ablation-linear-val ablation-adafs-intensity-val
  python3 run_clean_ablations.py --dry-run
  python3 run_clean_ablations.py --epochs 100 --skip-existing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
REEVAL = EXAMPLES / "reeval"
RUN_REEVAL = EXAMPLES / "run_reeval.py"

# name -> extra CLI args (beyond the shared clean protocol)
ABLATIONS: list[tuple[str, list[str], str]] = [
    (
        "ablation-linear-val",
        ["--feature-set", "basic"],
        "As-published path: linear emission, no physio, no AdaFS, basic speed features",
    ),
    (
        "ablation-physio-val",
        ["--physiological", "--feature-set", "basic"],
        "Physiological head only (no residual), basic features",
    ),
    (
        "ablation-physio-residual-val",
        ["--physiological", "--residual", "--feature-set", "basic"],
        "Physio + residual, basic features",
    ),
    (
        "ablation-physio-residual-intensity-val",
        ["--physiological", "--residual", "--feature-set", "run_intensity"],
        "Best clean stack: physio + residual + run_intensity (same recipe as intensity-val)",
    ),
    (
        "ablation-adafs-intensity-val",
        [
            "--physiological",
            "--residual",
            "--adafs",
            "--feature-set",
            "run_intensity",
        ],
        "Best stack + AdaFS (honest test of adaptive feature selection)",
    ),
    (
        "ablation-personal-val",
        ["--physiological", "--residual", "--feature-set", "run_personal"],
        "Best stack with subject-prior / relative-speed features (run_personal)",
    ),
    (
        "ablation-p2-physio-val",
        [
            "--physiological",
            "--residual",
            "--personalized-physio",
            "--feature-set",
            "run_intensity",
        ],
        "P2: subject-stable physio params + embedding-conditioned intensity",
    ),
    (
        "ablation-meanbias-val",
        [
            "--physiological",
            "--residual",
            "--feature-set",
            "run_intensity",
            "--mean-bias-weight",
            "0.5",
        ],
        "P3: step Huber + 0.5 * mean-bias term",
    ),
]


# If a prior run used the same recipe under another name, do not retrain.
ALIASES: dict[str, list[str]] = {
    "ablation-physio-residual-intensity-val": [
        "run-huber-128-delta12-intensity-val",
    ],
    "ablation-personal-val": [
        "run-huber-128-delta12-personal-val",
    ],
    "ablation-p2-physio-val": [
        "run-huber-128-delta12-intensity-p2-val",
    ],
    "ablation-meanbias-val": [
        "run-huber-128-delta12-intensity-meanbias05-val",
    ],
}


def _result_is_complete(result: Path) -> bool:
    if not result.exists():
        return False
    text = result.read_text()
    return "full_mean_workout_MAE=" in text and "val_fraction=" in text


def already_done(name: str) -> bool:
    result = REEVAL / name / "result.txt"
    ckpt = REEVAL / name / "best_model.pt"
    if result.exists() and ckpt.exists() and _result_is_complete(result):
        return True
    for alias in ALIASES.get(name, []):
        alias_result = REEVAL / alias / "result.txt"
        alias_ckpt = REEVAL / alias / "best_model.pt"
        if alias_ckpt.exists() and _result_is_complete(alias_result):
            return True
    return False


def build_cmd(name: str, extra: list[str], epochs: int) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_REEVAL),
        "--name",
        name,
        "--sport",
        "run",
        "--history-source",
        "all-prior",
        "--val-fraction",
        "0.15",
        "--seq-length",
        "128",
        "--train-stride",
        "64",
        "--loss",
        "huber",
        "--huber-delta",
        "12",
        "--weight-decay",
        "1e-4",
        "--epochs",
        str(epochs),
        "--full-workout",
    ]
    cmd.extend(extra)
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--only", nargs="*", default=None, help="Subset of ablation names")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip runs that already have full-workout result.txt (default on)")
    p.add_argument("--force", action="store_true", help="Re-run even if result exists")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true", help="List ablations and exit")
    args = p.parse_args()
    if args.force:
        args.skip_existing = False

    selected = ABLATIONS
    if args.only:
        wanted = set(args.only)
        selected = [a for a in ABLATIONS if a[0] in wanted]
        missing = wanted - {a[0] for a in selected}
        if missing:
            p.error(f"Unknown ablation names: {sorted(missing)}")

    if args.list:
        for name, extra, desc in ABLATIONS:
            status = "done" if already_done(name) else "pending"
            print(f"{status:7}  {name}\n         {desc}\n         extras: {' '.join(extra) or '(none)'}\n")
        return

    os.chdir(EXAMPLES)
    print(f"Working directory: {EXAMPLES}")
    print(f"Ablations to consider: {len(selected)}  epochs={args.epochs}\n")

    for name, extra, desc in selected:
        if args.skip_existing and already_done(name):
            print(f"[skip] {name} (existing clean full-workout result)")
            continue
        cmd = build_cmd(name, extra, args.epochs)
        print("=" * 72)
        print(f"RUN  {name}")
        print(f"     {desc}")
        print(f"     {' '.join(cmd)}")
        print("=" * 72)
        if args.dry_run:
            continue
        log_path = REEVAL / name / "ablation_train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            log.write("CMD: " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(EXAMPLES),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            print(f"[FAIL] {name} exit={proc.returncode}  see {log_path}")
            sys.exit(proc.returncode)
        print(f"[ok]   {name}  -> reeval/{name}/result.txt")

    print("\nDone. Refresh the table with: python3 summarize_ablations.py")


if __name__ == "__main__":
    main()
