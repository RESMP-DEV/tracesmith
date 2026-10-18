#!/usr/bin/env python3
"""Run a bounded Qwen3.5 learning-rate study on real distillation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import optuna
import yaml

ROOT = Path("/home/kearm/training")
PYTHON = ROOT / "distillkit-tracesmith/.venv/bin/python"
DISTILLKIT = ROOT / "distillkit-tracesmith/.venv/bin/distillkit"
BASE_CONFIG = ROOT / "qwen35-distill-windows.yml"
STUDY_ROOT = ROOT / "studies/qwen35-lr"
STORAGE = f"sqlite:///{STUDY_ROOT / 'study.db'}"
STUDY_NAME = "qwen35-window-ce200"


def write_config(path: Path, output_path: Path, learning_rate: float) -> None:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["project_name"] = "qwen35-lr-window-probe"
    config["output_path"] = str(output_path)
    config["dataset"]["num_samples"] = 800
    config["dataset"]["num_eval_samples"] = 200
    config["training_args"]["num_train_epochs"] = 1
    config["training_args"]["learning_rate"] = learning_rate
    config["training_args"]["warmup_steps"] = 8
    config["training_args"]["logging_steps"] = 20
    config["training_args"]["save_strategy"] = "no"
    config["training_args"]["report_to"] = "none"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def run(command: list[str], log_path: Path, cwd: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}); see {log_path}")


def final_train_loss(log_path: Path) -> float | None:
    matches = re.findall(r"'loss': ([0-9.]+)", log_path.read_text(encoding="utf-8"))
    return float(matches[-1]) if matches else None


def objective(trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 7.5e-7, 1.0e-5, log=True)
    number = trial.number
    trial_root = STUDY_ROOT / f"trial-{number:02d}"
    output = trial_root / "model"
    write_config(trial_root / "config.yml", output, learning_rate)
    trial_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    run(
        [str(DISTILLKIT), str(trial_root / "config.yml")],
        trial_root / "train.log",
        ROOT,
    )
    train_seconds = time.monotonic() - started

    evaluation = trial_root / "evaluation.json"
    eval_started = time.monotonic()
    run(
        [
            str(PYTHON),
            "evaluate_qwen35_ce.py",
            "--model",
            str(output),
            "--tokenizer",
            "models/Qwen3.5-0.8B",
            "--dataset",
            "data/qwen35",
            "--limit",
            "200",
            "--max-length",
            "1024",
            "--output",
            str(evaluation),
        ],
        trial_root / "evaluate.log",
        ROOT,
    )
    eval_seconds = time.monotonic() - eval_started
    result = json.loads(evaluation.read_text())
    cross_entropy = float(result["mean_cross_entropy"])
    trial.set_user_attr("learning_rate", learning_rate)
    trial.set_user_attr("train_seconds", train_seconds)
    trial.set_user_attr("eval_seconds", eval_seconds)
    trial.set_user_attr("perplexity", result["perplexity"])
    trial.set_user_attr("final_train_loss", final_train_loss(trial_root / "train.log"))

    shutil.rmtree(output)
    (trial_root / "config.yml").unlink()
    return cross_entropy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()

    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    optuna.logging.get_logger("optuna").addHandler(
        optuna.logging.FileHandler(STUDY_ROOT / "study.log")
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=17, n_startup_trials=3),
        load_if_exists=True,
    )
    for anchor in (3.0e-6, 1.5e-6, 5.0e-6):
        try:
            study.enqueue_trial({"learning_rate": anchor})
        except optuna.exceptions.DuplicatedStudyError:
            pass

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    best = study.best_trial
    summary = {
        "schema_version": 1,
        "study": STUDY_NAME,
        "objective": "fixed 200-row held-out mean cross-entropy after 100 optimizer steps",
        "best_trial": best.number,
        "best_learning_rate": best.params["learning_rate"],
        "best_cross_entropy": best.value,
        "best_user_attrs": best.user_attrs,
        "trials": [
            {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "user_attrs": trial.user_attrs,
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
        "non_claims": [
            "A 100-step probe ranks learning rates but does not prove final 2,500-step quality.",
            "Held-out cross-entropy is not a task-level coding-agent benchmark.",
        ],
    }
    (STUDY_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if args.apply_best:
        backup = ROOT / "qwen35-distill-windows.pre-lr-study.yml"
        if not backup.exists():
            backup.write_text(BASE_CONFIG.read_text())
        config = yaml.safe_load(BASE_CONFIG.read_text())
        config["training_args"]["learning_rate"] = best.params["learning_rate"]
        BASE_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
