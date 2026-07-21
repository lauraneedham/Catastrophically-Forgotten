"""Compatibility wrapper for the minimal backpropagation implementation."""

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
