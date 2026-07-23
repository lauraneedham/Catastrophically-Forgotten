"""Experiment utilities for catastrophic forgetting experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch

from src.data import restrict_classes
from src.models.base import BasicOptimizer, MultiLayerPerceptron, evaluate_accuracy_stats, train_model
from src.models.feedback_alignment import FeedbackAlignmentMLP
from src.models.hebbian import HebbianMultiLayerPerceptron
from src.models.predictive_coding import PredictiveCodingMLP

OLD_CLASSES = [0, 1, 2, 3, 4, 5]
NEW_CLASSES = [6, 7, 8, 9]

MODEL_BUILDERS = {
    "backprop": MultiLayerPerceptron,
    "feedback_alignment": FeedbackAlignmentMLP,
    "predictive_coding": PredictiveCodingMLP,
    "hebbian": HebbianMultiLayerPerceptron,
}


def build_model(model_type: str, **kwargs):
    """Instantiate a model for the forgetting experiment by ``model_type``."""
    try:
        model_cls = MODEL_BUILDERS[model_type]
    except KeyError:
        raise NotImplementedError(
            f"Model type '{model_type}' is not implemented yet. "
            f"Available: {sorted(MODEL_BUILDERS)}."
        ) from None
    return model_cls(**kwargs)


def build_forgetting_loaders(train_set, valid_set, batch_size: int = 32, old_classes=OLD_CLASSES, new_classes=NEW_CLASSES):
    """Build loaders for old-only, new-only, and interleaved training conditions."""
    train_set_old = restrict_classes(train_set, old_classes)
    valid_set_old = restrict_classes(valid_set, old_classes)
    train_set_new = restrict_classes(train_set, new_classes)
    valid_set_new = restrict_classes(valid_set, new_classes)

    train_loader_old = torch.utils.data.DataLoader(train_set_old, batch_size=batch_size, shuffle=True)
    valid_loader_old = torch.utils.data.DataLoader(valid_set_old, batch_size=batch_size, shuffle=False)
    train_loader_new = torch.utils.data.DataLoader(train_set_new, batch_size=batch_size, shuffle=True)
    valid_loader_new = torch.utils.data.DataLoader(valid_set_new, batch_size=batch_size, shuffle=False)

    train_set_full_restricted = restrict_classes(train_set, old_classes + new_classes)
    train_loader_full = torch.utils.data.DataLoader(train_set_full_restricted, batch_size=batch_size, shuffle=True)

    return {
        "train_loader_old": train_loader_old,
        "valid_loader_old": valid_loader_old,
        "train_loader_new": train_loader_new,
        "valid_loader_new": valid_loader_new,
        "train_loader_full": train_loader_full,
    }


def evaluate_accuracy_for_loader(mlp, loader):
    """Compute accuracy without training on the provided loader."""
    stats = evaluate_accuracy_stats(mlp, loader)
    return stats["accuracy"]


def run_forgetting_experiment(
    train_loader_old,
    valid_loader_old,
    train_loader_new,
    valid_loader_new,
    train_loader_full,
    model_type: str = "backprop",
    condition: str = "sequential",
    num_epochs_phase1: int = 6,
    num_epochs_phase2: int = 6,
    lr: Optional[float] = None,
    optimizer_type: Optional[str] = None,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    num_inputs: int = 784,
    num_hidden: int = 100,
    num_outputs: int = 10,
    activation_type: str = "sigmoid",
    bias: bool = True,
    device: Optional[str] = None,
    verbose: bool = False,
):
    """Run a simple forgetting experiment with a backprop model.

    This matches the reference notebook/proof-of-concept flow: the first epoch
    trains immediately so phase 1 reflects actual learning and phase 2 shows
    forgetting on the old-class validation set.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        model_type,
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        activation_type=activation_type,
        bias=bias,
    )
    model.to(device)
    if verbose:
        print(f"[forgetting] {model_type} on {device}")

    def make_optimizer():
        # No optimizer for local-update rules: return None so the training loop
        # skips optimizer.step() and the model updates itself (e.g. in forward()).
        if optimizer_type in (None, "none"):
            return None
        if optimizer_type == "adam":
            # Pure Adam: standard betas (0.9, 0.999) and no weight decay.
            return torch.optim.Adam(
                model.parameters(),
                lr=lr if lr is not None else 1e-3,
            )
        if optimizer_type == "sgd":
            return BasicOptimizer(
                model.parameters(),
                lr=lr if lr is not None else 0.01,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unknown optimizer_type '{optimizer_type}'. Use 'adam', 'sgd', or None.")

    # A fresh optimizer per phase, so optimizer state (SGD momentum / Adam moment
    # estimates) does not leak from task 1 into task 2 (per-phase reset, as in
    # the notebook).
    phase1_results = train_model(
        model,
        train_loader_old,
        valid_loader_old,
        make_optimizer(),
        num_epochs=num_epochs_phase1,
        verbose=verbose,
    )

    phase2_loader = train_loader_new if condition == "sequential" else train_loader_full
    phase2_results = train_model(
        model,
        phase2_loader,
        valid_loader_old,
        make_optimizer(),
        num_epochs=num_epochs_phase2,
        verbose=verbose,
    )

    old_class_acc_trace = phase1_results["avg_valid_accuracies"] + phase2_results["avg_valid_accuracies"]
    new_class_acc_final = evaluate_accuracy_for_loader(model, valid_loader_new)

    return {
        "model": model,
        "device": str(device),
        "old_class_acc_trace": old_class_acc_trace,
        "new_class_acc_final": new_class_acc_final,
        "phase1_epochs": num_epochs_phase1,
        "phase1_results": phase1_results,
        "phase2_results": phase2_results,
    }


def plot_forgetting_results(results: Dict[str, Any], ax=None):
    """Plot old-class accuracy across the sequential/interleaved phases."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))

    for label, metrics in results.items():
        ax.plot(range(1, len(metrics["old_class_acc_trace"]) + 1), metrics["old_class_acc_trace"], label=label)

    ax.axvline(metrics["phase1_epochs"] + 0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Accuracy on old classes across phases")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    return ax
