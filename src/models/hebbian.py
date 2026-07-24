"""Layer-local Hebbian/Oja model used by the shared forgetting experiment.

The hidden layers are updated with an Oja subspace rule. Each hidden update
uses only the presynaptic input and postsynaptic activity at that layer.
The output layer uses a local supervised delta rule.  No error is
backpropagated into a hidden layer and no PyTorch optimizer is used.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.base import update_results_by_class_in_place


class HebbianMultiLayerPerceptron(nn.Module):
    """An arbitrary-depth MLP trained with explicit layer-local updates."""

    def __init__(
        self,
        num_inputs: int | None = None,
        num_hidden: int | Sequence[int] = 100,
        num_outputs: int = 10,
        num_hidden_layers: int = 1,
        activation_type: str = "sigmoid",
        bias: bool = False,
        lr: float = 0.01,
        hidden_lr: float | None = None,
        output_lr: float | None = None,
        normalize_hidden: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_inputs is None:
            num_inputs = 784
        if bias:
            raise ValueError(
                "The controlled Hebbian comparison uses bias=False; "
                "local bias learning is intentionally not mixed into this experiment."
            )

        if isinstance(num_hidden, Sequence) and not isinstance(num_hidden, (str, bytes)):
            hidden_dims = [int(width) for width in num_hidden]
        else:
            hidden_dims = [int(num_hidden)] * max(1, int(num_hidden_layers))
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("num_hidden must define one or more positive hidden widths.")

        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.hidden_dims = hidden_dims
        self.num_hidden_layers = len(hidden_dims)
        self.num_hidden = hidden_dims[0]
        self.activation_type = activation_type.lower()
        self.bias = False
        self.hidden_lr = float(lr if hidden_lr is None else hidden_lr)
        self.output_lr = float(lr if output_lr is None else output_lr)
        self.normalize_hidden = bool(normalize_hidden)
        self.eps = float(eps)

        if self.hidden_lr <= 0 or self.output_lr <= 0:
            raise ValueError("Learning rates must be positive.")

        dimensions = [self.num_inputs, *self.hidden_dims]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=False)
            for in_features, out_features in zip(dimensions[:-1], dimensions[1:])
        )
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.num_outputs, bias=False)

        if self.activation_type == "sigmoid":
            self.activation = torch.sigmoid
        elif self.activation_type == "tanh":
            self.activation = torch.tanh
        elif self.activation_type == "relu":
            self.activation = torch.relu
        elif self.activation_type == "identity":
            self.activation = lambda value: value
        else:
            raise ValueError("activation_type must be sigmoid, tanh, relu, or identity.")

        self._store_initial_weights_biases()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.normalize_hidden:
            self._normalise_hidden_rows()

    @property
    def architecture(self) -> tuple[int, ...]:
        return (self.num_inputs, *self.hidden_dims, self.num_outputs)

    @property
    def layers(self):
        """All layers in forward order, matching the repository model interface."""
        return [*self.hidden_layers, self.output_layer]

    @property
    def lin1(self):
        return self.hidden_layers[0]

    @property
    def lin2(self):
        return self.output_layer

    @property
    def trainable_weight_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _store_initial_weights_biases(self) -> None:
        self.init_weights = [layer.weight.detach().clone() for layer in self.layers]
        self.init_lin1_weight = self.lin1.weight.detach().clone()
        self.init_lin2_weight = self.lin2.weight.detach().clone()

    def _flatten(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(-1, self.num_inputs)

    def hidden_activities(self, X: torch.Tensor) -> list[torch.Tensor]:
        activity = self._flatten(X)
        values: list[torch.Tensor] = []
        for layer in self.hidden_layers:
            activity = self.activation(layer(activity))
            values.append(activity)
        return values

    def forward(self, X: torch.Tensor, y=None) -> torch.Tensor:
        hidden = self.hidden_activities(X)[-1]
        return torch.softmax(self.output_layer(hidden), dim=1)

    def list_parameters(self) -> list[str]:
        return [f"layer_{index}_weight" for index in range(len(self.layers))]

    def proposed_updates(self, X: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return local update directions without changing the model."""
        with torch.no_grad():
            presynaptic = self._flatten(X)
            batch_size = presynaptic.shape[0]
            hidden_values = self.hidden_activities(presynaptic)
            updates: dict[str, torch.Tensor] = {}

            for layer_index, (layer, postsynaptic) in enumerate(
                zip(self.hidden_layers, hidden_values)
            ):
                # Centering removes the shared sigmoid component. This is the
                # efficient symmetric Oja subspace update:
                # Y^T X - Y^T Y W = Y^T (X - YW).
                # It avoids constructing a width-by-width matrix, which is
                # critical for the 1000-unit architecture.
                learning_activity = postsynaptic - postsynaptic.mean(dim=0, keepdim=True)
                updates[f"hidden_{layer_index}_weight"] = (
                    learning_activity.T
                    @ (presynaptic - learning_activity @ layer.weight)
                    / batch_size
                ).detach().clone()
                presynaptic = postsynaptic

            targets = F.one_hot(y, num_classes=self.num_outputs).to(
                dtype=presynaptic.dtype
            )
            probabilities = torch.softmax(self.output_layer(presynaptic), dim=1)
            updates["output_weight"] = (
                (targets - probabilities).T @ presynaptic / batch_size
            ).detach().clone()
            return updates

    def proposed_parameter_deltas(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the exact one-step parameter changes without mutating.

        The training step normalizes each hidden weight row after applying the
        Oja direction. That normalization can rotate the net parameter change,
        so update-direction analysis must include it rather than inspecting the
        pre-normalization Oja term alone.
        """
        with torch.no_grad():
            local_directions = self.proposed_updates(X, y)
            deltas: dict[str, torch.Tensor] = {}
            for layer_index, layer in enumerate(self.hidden_layers):
                proposed_weight = layer.weight + (
                    self.hidden_lr
                    * local_directions[f"hidden_{layer_index}_weight"]
                )
                if self.normalize_hidden:
                    norms = proposed_weight.norm(
                        dim=1,
                        keepdim=True,
                    ).clamp_min(self.eps)
                    proposed_weight = proposed_weight / norms
                deltas[f"hidden_{layer_index}_weight"] = (
                    proposed_weight - layer.weight
                ).detach().clone()

            deltas["output_weight"] = (
                self.output_lr * local_directions["output_weight"]
            ).detach().clone()
            return deltas

    def local_update(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply one local-learning step and return the pre-update probabilities."""
        with torch.no_grad():
            probabilities = self.forward(X)
            updates = self.proposed_updates(X, y)
            for layer_index, layer in enumerate(self.hidden_layers):
                layer.weight.add_(
                    updates[f"hidden_{layer_index}_weight"], alpha=self.hidden_lr
                )
            self.output_layer.weight.add_(
                updates["output_weight"], alpha=self.output_lr
            )
            if self.normalize_hidden:
                self._normalise_hidden_rows()
            return probabilities

    def _normalise_hidden_rows(self) -> None:
        with torch.no_grad():
            for layer in self.hidden_layers:
                norms = layer.weight.norm(dim=1, keepdim=True).clamp_min(self.eps)
                layer.weight.div_(norms)


def train_hebbian_epoch(
    model: HebbianMultiLayerPerceptron,
    train_loader,
    valid_loader,
    no_train: bool = False,
):
    """Run one repository-compatible epoch without autograd or an optimizer."""
    epoch_results = {}
    for dataset in ("train", "valid"):
        for suffix in ("correct_by_class", "seen_by_class"):
            epoch_results[f"{dataset}_{suffix}"] = {
                class_id: 0 for class_id in range(model.num_outputs)
            }

    device = next(model.parameters()).device
    train_losses: list[float] = []
    train_accuracies: list[float] = []
    model.train(not no_train)

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            probabilities = model(X) if no_train else model.local_update(X, y)
            loss = F.nll_loss(torch.log(probabilities.clamp_min(model.eps)), y)
            accuracy = (probabilities.argmax(dim=1) == y).float().mean()
        train_losses.append(float(loss.item()) * len(y))
        train_accuracies.append(float(accuracy.item()) * len(y))
        update_results_by_class_in_place(
            y.detach().cpu(),
            probabilities.detach().cpu(),
            epoch_results,
            dataset="train",
            num_classes=model.num_outputs,
        )

    train_items = len(train_loader.dataset)
    epoch_results["avg_train_losses"] = float(np.sum(train_losses) / train_items)
    epoch_results["avg_train_accuracies"] = float(
        np.sum(train_accuracies) / train_items * 100
    )

    model.eval()
    valid_losses: list[float] = []
    valid_accuracies: list[float] = []
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            probabilities = model(X)
            loss = F.nll_loss(torch.log(probabilities.clamp_min(model.eps)), y)
            accuracy = (probabilities.argmax(dim=1) == y).float().mean()
            valid_losses.append(float(loss.item()) * len(y))
            valid_accuracies.append(float(accuracy.item()) * len(y))
            update_results_by_class_in_place(
                y.detach().cpu(),
                probabilities.detach().cpu(),
                epoch_results,
                dataset="valid",
                num_classes=model.num_outputs,
            )

    valid_items = len(valid_loader.dataset)
    epoch_results["avg_valid_losses"] = float(np.sum(valid_losses) / valid_items)
    epoch_results["avg_valid_accuracies"] = float(
        np.sum(valid_accuracies) / valid_items * 100
    )
    return epoch_results


def train_hebbian_model(
    model: HebbianMultiLayerPerceptron,
    train_loader,
    valid_loader,
    num_epochs: int = 5,
    verbose: bool = False,
    record_initial_baseline: bool = True,
):
    """Train with the same result layout and epoch semantics as ``train_model``."""
    results = {
        "avg_train_losses": [],
        "avg_valid_losses": [],
        "avg_train_accuracies": [],
        "avg_valid_accuracies": [],
    }

    for epoch in tqdm(range(num_epochs)):
        no_train = record_initial_baseline and epoch == 0
        epoch_results = train_hebbian_epoch(
            model, train_loader, valid_loader, no_train=no_train
        )
        for key, value in epoch_results.items():
            if key in results and isinstance(results[key], list):
                results[key].append(value)
            else:
                results[key] = value
        if verbose:
            mode = "baseline" if no_train else "trained"
            print(
                f"epoch {epoch + 1} ({mode}): "
                f"train acc = {epoch_results['avg_train_accuracies']:.2f}%, "
                f"valid acc = {epoch_results['avg_valid_accuracies']:.2f}%"
            )
    return results
