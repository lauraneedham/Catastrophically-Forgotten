"""Compatibility wrapper for the minimal backpropagation implementation."""
import torch.nn as nn

from backpropagation import (
    BasicOptimizer,
    MultiLayerPerceptron,
    evaluate_accuracy,
    evaluate_accuracy_stats,
    train_epoch,
    train_model,
    update_results_by_class_in_place,
)

__all__ = [
    "BasicOptimizer",
    "MultiLayerPerceptron",
    "evaluate_accuracy",
    "evaluate_accuracy_stats",
    "train_epoch",
    "train_model",
    "update_results_by_class_in_place",
]


def evaluate_accuracy(model: nn.Module, loader):
    """Compute accuracy of the MLP on a given loader without training on it."""
    return evaluate_accuracy_stats(model, loader)["accuracy"]
