"""Minimal backpropagation baseline implementation for the forgetting experiment."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class MultiLayerPerceptron(nn.Module):
    """Multi-layer Perceptron supporting flexible depth and width."""

    def __init__(
        self,
        num_inputs: int | None = None,
        num_hidden: int | Sequence[int] = 100,
        num_outputs: int = 10,
        num_hidden_layers: int = 1,
        activation_type: str = "sigmoid",
        bias: bool = False,
    ):
        super().__init__()
        if num_inputs is None:
            num_inputs = 784

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation_type = activation_type
        self.bias = bias

        if isinstance(num_hidden, (list, tuple)):
            self.hidden_dims = list(num_hidden)
            self.num_hidden_layers = len(self.hidden_dims)
            self.num_hidden = self.hidden_dims[0] if len(self.hidden_dims) > 0 else 100
        else:
            self.num_hidden = num_hidden
            self.num_hidden_layers = max(1, num_hidden_layers)
            self.hidden_dims = [num_hidden] * self.num_hidden_layers

        layers = []
        in_dim = num_inputs
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim, bias=bias))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_outputs, bias=bias))

        self.layers = nn.ModuleList(layers)
        self.lin1 = self.layers[0]
        self.lin2 = self.layers[-1]

        self._store_initial_weights_biases()
        self._set_activation()
        self.softmax = nn.Softmax(dim=1)

    def _store_initial_weights_biases(self) -> None:
        self.init_weights = [layer.weight.data.clone() for layer in self.layers]
        self.init_lin1_weight = self.lin1.weight.data.clone()
        self.init_lin2_weight = self.lin2.weight.data.clone()
        if self.bias:
            self.init_biases = [layer.bias.data.clone() for layer in self.layers if layer.bias is not None]
            self.init_lin1_bias = self.lin1.bias.data.clone()
            self.init_lin2_bias = self.lin2.bias.data.clone()

    def _set_activation(self) -> None:
        act_str = self.activation_type.lower()
        if act_str == "sigmoid":
            self.activation = nn.Sigmoid()
        elif act_str == "tanh":
            self.activation = nn.Tanh()
        elif act_str == "relu":
            self.activation = nn.ReLU()
        elif act_str == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
                f"{self.activation_type} activation type not recognized. Only "
                "'sigmoid', 'relu', 'tanh', and 'identity' have been implemented."
            )

    def forward(self, X: torch.Tensor, y=None) -> torch.Tensor:
        out = X.reshape(-1, self.num_inputs)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return self.softmax(out)

    def forward_backprop(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    def list_parameters(self) -> list[str]:
        params_list: list[str] = []
        for i in range(len(self.layers)):
            params_list.append(f"layer_{i}_weight")
            if self.bias:
                params_list.append(f"layer_{i}_bias")
        return params_list

    def gather_gradient_dict(self) -> Dict[str, Any]:
        gradient_dict: Dict[str, Any] = {}
        for i, layer in enumerate(self.layers):
            if layer.weight.grad is None:
                raise RuntimeError("No gradient was computed")
            gradient_dict[f"layer_{i}_weight"] = layer.weight.grad.detach().cpu().clone().numpy()
            if self.bias and layer.bias is not None and layer.bias.grad is not None:
                gradient_dict[f"layer_{i}_bias"] = layer.bias.grad.detach().cpu().clone().numpy()
        return gradient_dict


class BasicOptimizer(torch.optim.Optimizer):
    """Minimal SGD optimizer used by the baseline training loop."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(parameters, defaults)

    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                if momentum != 0:
                    # Classic SGD momentum (no dampening / nesterov): keep a
                    # per-parameter velocity buffer in the optimizer state, as
                    # torch.optim.SGD does.
                    state = self.state[param]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = grad.clone().detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(grad)
                    grad = buf

                param.data.add_(grad, alpha=-lr)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)


def update_results_by_class_in_place(y, y_pred, result_dict, dataset="train", num_classes=10):
    """Track per-class counts for training and validation batches."""
    y_pred = np.argmax(y_pred, axis=1)
    if len(y) != len(y_pred):
        raise RuntimeError("Number of predictions does not match number of targets.")

    for i in result_dict[f"{dataset}_seen_by_class"].keys():
        idxs = np.where(y == int(i))[0]
        result_dict[f"{dataset}_seen_by_class"][int(i)] += len(idxs)
        num_correct = int(sum(y[idxs] == y_pred[idxs]))
        result_dict[f"{dataset}_correct_by_class"][int(i)] += num_correct


def train_epoch(model: nn.Module, train_loader, valid_loader, optimizer: BasicOptimizer, no_train: bool = False):
    """Train for one epoch and return aggregate metrics."""
    criterion = nn.NLLLoss()
    epoch_results = {}

    for dataset in ["train", "valid"]:
        for sub_str in ["correct_by_class", "seen_by_class"]:
            epoch_results[f"{dataset}_{sub_str}"] = {i: 0 for i in range(model.num_outputs)}

    device = next(model.parameters()).device

    if no_train:
        model.eval()
    else:
        model.train()
    train_losses = []
    train_acc = []
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        if no_train:
            y_pred = model(X)
        else:
            y_pred = model(X, y=y)
        loss = criterion(torch.log(y_pred), y)
        acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
        train_losses.append(loss.item() * len(y))
        train_acc.append(acc.item() * len(y))
        # per-class bookkeeping runs in NumPy, so move predictions/targets to CPU
        update_results_by_class_in_place(y.cpu(), y_pred.detach().cpu(), epoch_results, dataset="train", num_classes=model.num_outputs)

        optimizer.zero_grad()
        if not no_train:
            loss.backward()
            optimizer.step()

    num_items = len(train_loader.dataset)
    epoch_results["avg_train_losses"] = np.sum(train_losses) / num_items
    epoch_results["avg_train_accuracies"] = np.sum(train_acc) / num_items * 100

    model.eval()
    valid_losses = []
    valid_acc = []
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(torch.log(y_pred), y)
            acc = (torch.argmax(y_pred, axis=1) == y).sum() / len(y)
            valid_losses.append(loss.item() * len(y))
            valid_acc.append(acc.item() * len(y))
            update_results_by_class_in_place(y.cpu(), y_pred.detach().cpu(), epoch_results, dataset="valid", num_classes=model.num_outputs)

    num_items = len(valid_loader.dataset)
    epoch_results["avg_valid_losses"] = np.sum(valid_losses) / num_items
    epoch_results["avg_valid_accuracies"] = np.sum(valid_acc) / num_items * 100

    return epoch_results


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    num_epochs: int = 5,
    verbose: bool = False,
    record_initial_baseline: bool = True,
):
    """Train the model across epochs and aggregate notebook-style results."""
    results = {
        "avg_train_losses": [],
        "avg_valid_losses": [],
        "avg_train_accuracies": [],
        "avg_valid_accuracies": [],
    }

    for e in tqdm(range(num_epochs)):
        no_train = record_initial_baseline and e == 0
        epoch_results = train_epoch(model, train_loader, valid_loader, optimizer, no_train=no_train)

        for key, value in epoch_results.items():
            if key in results and isinstance(results[key], list):
                results[key].append(epoch_results[key])
            else:
                results[key] = value

        if verbose:
            print(
                f"epoch {e + 1}: train acc = {epoch_results['avg_train_accuracies']:.2f}%, "
                f"valid acc = {epoch_results['avg_valid_accuracies']:.2f}%"
            )

    return results


def evaluate_accuracy_stats(model: nn.Module, loader):
    """Return accuracy and loss statistics for a loader."""
    model.eval()
    correct = 0
    total = 0
    losses = []
    correct_by_class: Counter = Counter()
    seen_by_class: Counter = Counter()

    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            pred = torch.argmax(y_pred, axis=1)
            correct += (pred == y).sum().item()
            total += len(y)
            losses.append(torch.nn.NLLLoss()(torch.log(y_pred), y).item() * len(y))

            for pred_i, target_i in zip(pred.tolist(), y.tolist()):
                seen_by_class[target_i] += 1
                if pred_i == target_i:
                    correct_by_class[target_i] += 1

    return {
        "accuracy": 100 * correct / total,
        "loss": np.sum(losses) / total,
        "correct_by_class": correct_by_class,
        "seen_by_class": seen_by_class,
    }


def evaluate_accuracy(model: nn.Module, loader):
    """Compute accuracy of the MLP on a given loader without training on it."""
    return evaluate_accuracy_stats(model, loader)["accuracy"]
