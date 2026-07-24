"""Standardized 784-1000-10 comparative experiment orchestration.

This module runs one learning rule at a time so full-MNIST Colab jobs can be
saved independently and combined later.  Every run uses the repository's
shared old/new split and can collect non-mutating SNR/cosine update statistics
at selected checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from typing import Any, Iterable
import warnings

import numpy as np
import torch

from src.analysis.update_metrics import (
    analyze_update_statistics,
    canonical_rule_name,
    update_statistics_records,
)
from src.experiments.forgetting import (
    build_forgetting_loaders,
    build_model,
    evaluate_accuracy_for_loader,
)
from src.models.base import train_epoch
from src.models.hebbian import train_hebbian_epoch


FINAL_RULES = (
    "backprop",
    "feedback_alignment",
    "predictive_coding",
    "hebbian",
)
FINAL_CONDITIONS = ("sequential", "interleaved")


@dataclass(frozen=True)
class ComparativeConfig:
    """Frozen common protocol for the final learning-rule comparison."""

    num_inputs: int = 784
    num_hidden: int = 1000
    num_outputs: int = 10
    activation_type: str = "sigmoid"
    bias: bool = False
    learning_rate: float = 0.001
    batch_size: int = 32
    phase1_recorded_epochs: int = 20
    phase2_recorded_epochs: int = 20
    record_initial_baseline: bool = True
    train_prop: float = 0.8
    keep_prop: float = 1.0
    data_split_seed: int = 0
    model_seed: int = 0
    data_order_seed: int = 1000
    max_probe_batches: int = 8
    phase2_analysis_epochs: tuple[int, ...] = (1, 5, 10, 20)

    @property
    def architecture(self) -> str:
        return f"{self.num_inputs}-{self.num_hidden}-{self.num_outputs}"

    def validate(self) -> None:
        if self.num_inputs != 784 or self.num_outputs != 10:
            raise ValueError("The final MNIST protocol requires 784 inputs and 10 outputs.")
        if self.num_hidden != 1000:
            raise ValueError("The final comparison architecture requires 1000 hidden units.")
        if self.activation_type.lower() != "sigmoid":
            raise ValueError("The final comparison requires sigmoid activation.")
        if self.bias:
            raise ValueError("The final comparison requires bias=False.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size != 32:
            raise ValueError("The final comparison requires batch size 32.")
        if self.phase1_recorded_epochs <= 0 or self.phase2_recorded_epochs <= 0:
            raise ValueError("Both phases require at least one recorded epoch.")
        if self.max_probe_batches <= 0:
            raise ValueError("max_probe_batches must be positive.")
        invalid_checkpoints = [
            epoch
            for epoch in self.phase2_analysis_epochs
            if epoch < 1 or epoch > self.phase2_recorded_epochs
        ]
        if invalid_checkpoints:
            raise ValueError(
                f"Invalid phase-two analysis epochs: {invalid_checkpoints}."
            )


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without changing the JAX API."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_final_model(
    rule: str,
    config: ComparativeConfig,
    device: torch.device,
):
    rule = canonical_rule_name(rule)
    kwargs: dict[str, Any] = {
        "num_inputs": config.num_inputs,
        "num_hidden": config.num_hidden,
        "num_outputs": config.num_outputs,
        "activation_type": config.activation_type,
        "bias": config.bias,
    }
    if rule in {"hebbian", "predictive_coding"}:
        kwargs["lr"] = config.learning_rate
    if rule == "predictive_coding":
        kwargs["seed"] = config.model_seed

    model = build_model(rule, **kwargs)
    return model.to(device)


def _make_external_optimizer(
    model,
    rule: str,
    config: ComparativeConfig,
):
    rule = canonical_rule_name(rule)
    if rule in {"hebbian", "predictive_coding"}:
        return None
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def _run_recorded_epoch(
    model,
    rule: str,
    train_loader,
    valid_loader,
    optimizer,
    *,
    no_train: bool,
):
    rule = canonical_rule_name(rule)
    if rule == "hebbian":
        return train_hebbian_epoch(
            model,
            train_loader,
            valid_loader,
            no_train=no_train,
        )
    return train_epoch(
        model,
        train_loader,
        valid_loader,
        optimizer=optimizer,
        no_train=no_train,
    )


def _collect_checkpoint_metrics(
    model,
    rule: str,
    condition: str,
    checkpoint: str,
    valid_loader_old,
    valid_loader_new,
    config: ComparativeConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for probe_split, loader in (
        ("old_0_5", valid_loader_old),
        ("new_6_9", valid_loader_new),
    ):
        statistics = analyze_update_statistics(
            model,
            rule,
            loader,
            max_batches=config.max_probe_batches,
            verify_unchanged=True,
        )
        split_records = update_statistics_records(
            statistics,
            rule=rule,
            checkpoint=checkpoint,
            probe_split=probe_split,
        )
        for record in split_records:
            record["condition"] = condition
            record["architecture"] = config.architecture
        records.extend(split_records)
    return records


def run_rule_condition(
    rule: str,
    condition: str,
    loaders: dict[str, Any],
    config: ComparativeConfig,
    *,
    device: str | torch.device | None = None,
    collect_update_metrics: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one rule/condition with the final protocol."""
    config.validate()
    rule = canonical_rule_name(rule)
    if rule not in FINAL_RULES:
        raise ValueError(f"Expected one of {FINAL_RULES}; got '{rule}'.")
    if condition not in FINAL_CONDITIONS:
        raise ValueError(
            f"Expected condition in {FINAL_CONDITIONS}; got '{condition}'."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    set_all_seeds(config.model_seed)
    model = _build_final_model(rule, config, device)
    update_records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    if verbose:
        print(
            f"[{rule} | {condition}] {config.architecture} on {device}; "
            f"lr={config.learning_rate}"
        )

    if collect_update_metrics:
        update_records.extend(
            _collect_checkpoint_metrics(
                model,
                rule,
                condition,
                "initial",
                loaders["valid_loader_old"],
                loaders["valid_loader_new"],
                config,
            )
        )

    phase1_optimizer = _make_external_optimizer(model, rule, config)
    for recorded_epoch in range(1, config.phase1_recorded_epochs + 1):
        no_train = config.record_initial_baseline and recorded_epoch == 1
        epoch_results = _run_recorded_epoch(
            model,
            rule,
            loaders["train_loader_old"],
            loaders["valid_loader_old"],
            phase1_optimizer,
            no_train=no_train,
        )
        traces.append(
            {
                "condition": condition,
                "phase": 1,
                "recorded_epoch": recorded_epoch,
                "trained": not no_train or rule == "predictive_coding",
                "old_accuracy": float(epoch_results["avg_valid_accuracies"]),
                "new_accuracy": float(
                    evaluate_accuracy_for_loader(
                        model,
                        loaders["valid_loader_new"],
                    )
                ),
            }
        )
        if verbose:
            print(
                f"  phase 1 {recorded_epoch:02d}/"
                f"{config.phase1_recorded_epochs}: "
                f"old={traces[-1]['old_accuracy']:.2f}% "
                f"new={traces[-1]['new_accuracy']:.2f}%"
            )

    phase1_old_accuracy = float(
        evaluate_accuracy_for_loader(model, loaders["valid_loader_old"])
    )
    if collect_update_metrics:
        update_records.extend(
            _collect_checkpoint_metrics(
                model,
                rule,
                condition,
                "phase1_end",
                loaders["valid_loader_old"],
                loaders["valid_loader_new"],
                config,
            )
        )

    phase2_loader = (
        loaders["train_loader_new"]
        if condition == "sequential"
        else loaders["train_loader_full"]
    )
    phase2_optimizer = _make_external_optimizer(model, rule, config)
    analysis_epochs = set(config.phase2_analysis_epochs)
    for recorded_epoch in range(1, config.phase2_recorded_epochs + 1):
        no_train = config.record_initial_baseline and recorded_epoch == 1
        _run_recorded_epoch(
            model,
            rule,
            phase2_loader,
            loaders["valid_loader_old"],
            phase2_optimizer,
            no_train=no_train,
        )
        old_accuracy = float(
            evaluate_accuracy_for_loader(model, loaders["valid_loader_old"])
        )
        new_accuracy = float(
            evaluate_accuracy_for_loader(model, loaders["valid_loader_new"])
        )
        traces.append(
            {
                "condition": condition,
                "phase": 2,
                "recorded_epoch": recorded_epoch,
                "trained": not no_train or rule == "predictive_coding",
                "old_accuracy": old_accuracy,
                "new_accuracy": new_accuracy,
            }
        )
        if verbose:
            print(
                f"  phase 2 {recorded_epoch:02d}/"
                f"{config.phase2_recorded_epochs}: "
                f"old={old_accuracy:.2f}% new={new_accuracy:.2f}%"
            )

        if collect_update_metrics and recorded_epoch in analysis_epochs:
            update_records.extend(
                _collect_checkpoint_metrics(
                    model,
                    rule,
                    condition,
                    f"phase2_epoch_{recorded_epoch}",
                    loaders["valid_loader_old"],
                    loaders["valid_loader_new"],
                    config,
                )
            )

    retained_old_accuracy = float(
        evaluate_accuracy_for_loader(model, loaders["valid_loader_old"])
    )
    new_class_accuracy = float(
        evaluate_accuracy_for_loader(model, loaders["valid_loader_new"])
    )
    forgetting = phase1_old_accuracy - retained_old_accuracy
    normalized_forgetting = (
        forgetting / phase1_old_accuracy
        if phase1_old_accuracy > 0
        else float("nan")
    )

    summary = {
        "rule": rule,
        "condition": condition,
        "architecture": config.architecture,
        "phase1_old_accuracy": phase1_old_accuracy,
        "retained_old_accuracy": retained_old_accuracy,
        "forgetting": forgetting,
        "normalized_forgetting": normalized_forgetting,
        "new_class_accuracy": new_class_accuracy,
        "balanced_final_accuracy": (
            retained_old_accuracy + new_class_accuracy
        )
        / 2.0,
        "learning_rate": config.learning_rate,
        "optimizer": (
            "none"
            if rule == "hebbian"
            else "internal_optax_adam"
            if rule == "predictive_coding"
            else "pytorch_adam"
        ),
        "model_seed": config.model_seed,
        "data_split_seed": config.data_split_seed,
        "data_order_seed": config.data_order_seed,
        "device": str(device),
    }
    return {
        "summary": summary,
        "traces": traces,
        "update_metrics": update_records,
    }


def run_rule_experiment(
    rule: str,
    train_set,
    valid_set,
    config: ComparativeConfig,
    *,
    device: str | torch.device | None = None,
    collect_update_metrics: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run sequential and interleaved conditions with reset seeds/loaders."""
    combined = {
        "config": asdict(config),
        "summaries": [],
        "traces": [],
        "update_metrics": [],
    }
    for condition in FINAL_CONDITIONS:
        loaders = build_forgetting_loaders(
            train_set,
            valid_set,
            batch_size=config.batch_size,
            seed=config.data_order_seed,
        )
        result = run_rule_condition(
            rule,
            condition,
            loaders,
            config,
            device=device,
            collect_update_metrics=collect_update_metrics,
            verbose=verbose,
        )
        combined["summaries"].append(result["summary"])
        combined["traces"].extend(result["traces"])
        combined["update_metrics"].extend(result["update_metrics"])

    sequential_phase1 = combined["summaries"][0]["phase1_old_accuracy"]
    interleaved_phase1 = combined["summaries"][1]["phase1_old_accuracy"]
    combined["phase1_conditions_match"] = bool(
        np.isclose(sequential_phase1, interleaved_phase1, atol=1e-6)
    )
    if not combined["phase1_conditions_match"]:
        warnings.warn(
            "Sequential and interleaved phase-one results diverged despite "
            "reset seeds and loaders: "
            f"{sequential_phase1} versus {interleaved_phase1}.",
            RuntimeWarning,
        )
    return combined


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_rule_experiment(
    result: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Save one rule's complete machine-readable artifact bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_safe_result = _json_safe(result)
    (output_dir / "config.json").write_text(
        json.dumps(json_safe_result["config"], indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "result_bundle.json").write_text(
        json.dumps(json_safe_result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "performance_summary.csv", result["summaries"])
    _write_csv(output_dir / "performance_traces.csv", result["traces"])
    _write_csv(output_dir / "update_metrics.csv", result["update_metrics"])
    return output_dir


def _json_safe(value):
    """Recursively replace non-finite floats with JSON ``null``."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value
