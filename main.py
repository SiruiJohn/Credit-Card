from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

STAGES: dict[str, list[str]] = {
    "1": [
        "s01_prepare_data.py",
        "s01_eda.py",
    ],
    "2": [
        "s02_train_baseline.py",
        "s02_plot_baseline.py",
        "s02_ablation.py",
        "s02_lr_analysis.py",
        "s02_tune_knn.py",
        "s02_tune_knn_advanced.py",
    ],
    "3": [
        "s03_cost_threshold.py",
        "s03_temporal_robustness.py",
        "s03_error_analysis.py",
    ],
    "4": [
        "s04_external_eda.py",
        "s04_scaling_ablation.py",
        "s04_library_benchmark.py",
    ],
    "5": [
        "s05_oof_validation.py",
    ],
}

STAGE_DESCRIPTIONS = {
    "1": "Data Preparation & EDA",
    "2": "Model Training & Evaluation",
    "3": "Deep Evaluation Analysis",
    "4": "Extended Analysis (Notebook-Inspired)",
    "5": "OOF Threshold Review",
}


def run_script(script_name: str, extra_args: list[str] | None = None) -> bool:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  [SKIP] {script_name} not found")
        return True

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            print(f"  [OK] {script_name} ({elapsed:.1f}s)")
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    print(f"       {line}")
            return True
        else:
            print(f"  [FAIL] {script_name} (exit={result.returncode}, {elapsed:.1f}s)")
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines()[-8:]:
                    print(f"       {line}")
            return False
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [ERROR] {script_name}: {e} ({elapsed:.1f}s)")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — Full Pipeline Orchestrator"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "3", "4", "5"],
        help="Run a single stage only (1-5). Omit to run all stages.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 2-fold quick mode for baseline training.",
    )
    parser.add_argument(
        "--skip-stage",
        type=str,
        nargs="*",
        choices=["1", "2", "3", "4", "5"],
        help="Skip specified stages.",
    )
    args = parser.parse_args()

    extra_args = []
    if args.quick:
        extra_args.append("--quick")

    if args.stage:
        stage_ids = [args.stage]
    else:
        stage_ids = ["1", "2", "3", "4", "5"]

    if args.skip_stage:
        stage_ids = [s for s in stage_ids if s not in args.skip_stage]

    if not stage_ids:
        print("No stages to run.")
        return

    print("=" * 60)
    print("Credit Card Fraud Detection — Pipeline")
    print(f"Stages: {', '.join(stage_ids)}")
    if args.quick:
        print("Mode: quick (2-fold)")
    print("=" * 60)

    total_start = time.perf_counter()
    all_ok = True

    for stage_id in stage_ids:
        scripts = STAGES[stage_id]
        desc = STAGE_DESCRIPTIONS[stage_id]
        print(f"\n--- Stage {stage_id}: {desc} ---")

        stage_start = time.perf_counter()
        for script in scripts:
            script_extra = extra_args if "train_baseline" in script else None
            if not run_script(script, script_extra):
                all_ok = False
                print(f"\n  Pipeline stopped at Stage {stage_id} due to failure.")
                break

        stage_elapsed = time.perf_counter() - stage_start
        print(f"  Stage {stage_id} done ({stage_elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 60}")
    status = "COMPLETED" if all_ok else "COMPLETED WITH ERRORS"
    print(f"Pipeline {status} ({total_elapsed:.1f}s)")
    print(f"{'=' * 60}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
