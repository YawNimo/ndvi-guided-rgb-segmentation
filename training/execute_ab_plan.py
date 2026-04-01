"""Execute reproducible A/B experiments for NDVI-guided RGB segmentation training.

This script runs a screening matrix, captures best-metric artifacts, and writes a
summary table with accept/reject decisions using the project's A/B criteria.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_ENTRY = ROOT_DIR / "training" / "main.py"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    args: list[str]
    compare_to: str


def _run_experiment(exp: Experiment) -> dict:
    cmd = [sys.executable, str(TRAIN_ENTRY), "--run-name", exp.name, *exp.args]
    print(f"\n[AB] Running {exp.name}: {exp.description}")
    print("[AB] Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)

    metrics_path = RESULTS_DIR / f"{exp.name}_unet_best_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    best = payload["best"]
    meta = payload.get("run_metadata", {})
    return {
        "experiment": exp.name,
        "description": exp.description,
        "compare_to": exp.compare_to,
        "metrics_path": str(metrics_path),
        "val_f1_macro": float(best["val_f1_macro"]),
        "val_iou_macro": float(best["val_iou_macro"]),
        "val_pixel_acc": float(best["val_pixel_acc"]),
        "train_time_sec": float(best["train_time_sec"]),
        "val_time_sec": float(best["val_time_sec"]),
        "best_epoch": int(best["epoch"]),
        "selection_signature": meta.get("split_info", {}).get("selection_signature"),
        "sample_size": meta.get("sample_size"),
        "sample_seed": meta.get("sample_seed"),
        "loss_type": meta.get("loss_type"),
        "scheduler": meta.get("scheduler"),
        "batch_size": meta.get("batch_size", None),
    }


def _load_experiment_result(exp_name: str) -> dict:
    metrics_path = RESULTS_DIR / f"{exp_name}_unet_best_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    best = payload["best"]
    meta = payload.get("run_metadata", {})
    return {
        "experiment": exp_name,
        "description": "loaded-existing",
        "compare_to": "",
        "metrics_path": str(metrics_path),
        "val_f1_macro": float(best["val_f1_macro"]),
        "val_iou_macro": float(best["val_iou_macro"]),
        "val_pixel_acc": float(best["val_pixel_acc"]),
        "train_time_sec": float(best["train_time_sec"]),
        "val_time_sec": float(best["val_time_sec"]),
        "best_epoch": int(best["epoch"]),
        "selection_signature": meta.get("split_info", {}).get("selection_signature"),
        "sample_size": meta.get("sample_size"),
        "sample_seed": meta.get("sample_seed"),
        "loss_type": meta.get("loss_type"),
        "scheduler": meta.get("scheduler"),
        "batch_size": meta.get("batch_size", None),
    }


def _run_or_load(exp: Experiment, skip_existing: bool) -> dict:
    metrics_path = RESULTS_DIR / f"{exp.name}_unet_best_metrics.json"
    if skip_existing and metrics_path.exists():
        loaded = _load_experiment_result(exp.name)
        loaded["description"] = exp.description
        loaded["compare_to"] = exp.compare_to
        print(f"[AB] Loaded existing metrics for {exp.name}: {metrics_path}")
        return loaded
    return _run_experiment(exp)


def _decision(candidate: dict, control: dict) -> tuple[str, str]:
    df1 = candidate["val_f1_macro"] - control["val_f1_macro"]
    diou = candidate["val_iou_macro"] - control["val_iou_macro"]

    if candidate["selection_signature"] != control["selection_signature"]:
        return "reject", "Different sampled subset signature"
    if df1 > 0 and diou >= 0:
        return "accept", f"F1 +{df1:.4f}, IoU +{diou:.4f}"
    return "reject", f"F1 {df1:+.4f}, IoU {diou:+.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute A/B training experiment plan")
    parser.add_argument("--phase", choices=["all", "screening", "confirm"], default="all")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Screening matrix: same random subset for all runs to keep A/B fair.
    common = [
        "--model", "unet",
        "--epochs", "4",
        "--early-stop-patience", "4",
        "--amp",
        "--batch-size", "2",
        "--freeze-bn",
        "--sample-size", "600",
        "--sample-seed", "314159",
        "--seed", "42",
        "--num-workers", "2",
    ]

    experiments = [
        Experiment(
            name="A0_softdice_baseline",
            description="Control: weighted soft Dice + CE, plateau scheduler",
            compare_to="",
            args=[
                *common,
                "--loss-type", "soft_dice",
                "--weighted-dice",
                "--dice-weight", "1.0",
                "--scheduler", "plateau",
                "--deterministic",
                "--no-cudnn-benchmark",
            ],
        ),
        Experiment(
            name="B1_gdl",
            description="Switch to canonical generalized Dice + CE",
            compare_to="A0_softdice_baseline",
            args=[
                *common,
                "--loss-type", "gdl",
                "--dice-weight", "1.0",
                "--scheduler", "plateau",
                "--deterministic",
                "--no-cudnn-benchmark",
            ],
        ),
        Experiment(
            name="B2_gdl_dicew_07",
            description="Tune CE:GDL blend by reducing GDL weight",
            compare_to="B1_gdl",
            args=[
                *common,
                "--loss-type", "gdl",
                "--dice-weight", "0.7",
                "--scheduler", "plateau",
                "--deterministic",
                "--no-cudnn-benchmark",
            ],
        ),
        Experiment(
            name="B3_gdl_cosine",
            description="Cosine schedule with warmup for faster convergence",
            compare_to="B2_gdl_dicew_07",
            args=[
                *common,
                "--loss-type", "gdl",
                "--dice-weight", "0.7",
                "--scheduler", "cosine",
                "--warmup-epochs", "1",
                "--deterministic",
                "--no-cudnn-benchmark",
            ],
        ),
        Experiment(
            name="B4_gdl_cosine_speed",
            description="Throughput tune: cudnn benchmark + persistent workers",
            compare_to="B3_gdl_cosine",
            args=[
                *common,
                "--loss-type", "gdl",
                "--dice-weight", "0.7",
                "--scheduler", "cosine",
                "--warmup-epochs", "1",
                "--no-deterministic",
                "--cudnn-benchmark",
                "--persistent-workers",
            ],
        ),
    ]

    # Confirmation matrix: higher-fidelity runs for meaningful accuracy decisions.
    confirm_common = [
        "--model", "unet",
        "--epochs", "8",
        "--early-stop-patience", "5",
        "--amp",
        "--batch-size", "2",
        "--freeze-bn",
        "--sample-size", "2400",
        "--sample-seed", "314159",
        "--seed", "42",
        "--num-workers", "2",
        "--deterministic",
        "--no-cudnn-benchmark",
    ]

    confirm_experiments = [
        Experiment(
            name="C0_softdice_confirm",
            description="Confirmation control with larger sample and longer training",
            compare_to="",
            args=[
                *confirm_common,
                "--loss-type", "soft_dice",
                "--weighted-dice",
                "--dice-weight", "1.0",
                "--scheduler", "plateau",
            ],
        ),
        Experiment(
            name="C1_gdl_confirm",
            description="Confirmation candidate: CE + canonical GDL (dice weight 0.7)",
            compare_to="C0_softdice_confirm",
            args=[
                *confirm_common,
                "--loss-type", "gdl",
                "--dice-weight", "0.7",
                "--scheduler", "plateau",
            ],
        ),
        Experiment(
            name="C2_gdl_batch4_confirm",
            description="Confirmation candidate: CE + GDL with batch-size 4 and no BN freeze",
            compare_to="C1_gdl_confirm",
            args=[
                "--model", "unet",
                "--epochs", "8",
                "--early-stop-patience", "5",
                "--amp",
                "--batch-size", "4",
                "--sample-size", "2400",
                "--sample-seed", "314159",
                "--seed", "42",
                "--num-workers", "2",
                "--deterministic",
                "--no-cudnn-benchmark",
                "--loss-type", "gdl",
                "--dice-weight", "0.7",
                "--scheduler", "plateau",
            ],
        ),
    ]

    by_name: dict[str, dict] = {}
    rows: list[dict] = []

    def execute_matrix(matrix: list[Experiment], control_label: str) -> None:
        for exp in matrix:
            try:
                result = _run_or_load(exp, skip_existing=args.skip_existing)
            except Exception as exc:  # noqa: BLE001
                if args.continue_on_error:
                    print(f"[AB] ERROR in {exp.name}: {exc}")
                    continue
                raise

            if exp.compare_to:
                control = by_name.get(exp.compare_to)
                if control is None:
                    raise RuntimeError(f"Missing control result for {exp.name}: {exp.compare_to}")
                decision, reason = _decision(result, control)
            else:
                decision, reason = "control", control_label

            result["decision"] = decision
            result["decision_reason"] = reason
            by_name[exp.name] = result
            rows.append(result)

            print(
                f"[AB] {exp.name} -> {decision.upper()} | "
                f"F1={result['val_f1_macro']:.4f} IoU={result['val_iou_macro']:.4f} | {reason}"
            )

    if args.phase in {"all", "screening"}:
        execute_matrix(experiments, control_label="Baseline reference")

    if args.phase in {"all", "confirm"}:
            execute_matrix(confirm_experiments, control_label="Confirmation reference")

    out_json = RESULTS_DIR / "ab_screening_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    out_csv = RESULTS_DIR / "ab_screening_summary.csv"
    fields = [
        "experiment",
        "compare_to",
        "description",
        "val_f1_macro",
        "val_iou_macro",
        "val_pixel_acc",
        "train_time_sec",
        "val_time_sec",
        "best_epoch",
        "selection_signature",
        "sample_size",
        "sample_seed",
        "loss_type",
        "scheduler",
        "decision",
        "decision_reason",
        "metrics_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    print(f"\n[AB] Wrote summary: {out_json}")
    print(f"[AB] Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()
