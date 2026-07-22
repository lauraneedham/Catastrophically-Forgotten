"""Self-contained Hebbian MNIST continual-learning implementation.

This module intentionally does not import the repository's existing ``src``
package. The matching Colab notebook embeds this file verbatim so it can run as
a standalone upload while leaving all teammate code untouched.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset, Subset


OLD_CLASSES = (0, 1, 2, 3, 4)
NEW_CLASSES = (5, 6, 7, 8, 9)
HEBBIAN_VARIANTS = ("centered", "oja", "hard_wta", "soft_wta")
OUTPUT_RULES = ("delta", "teacher_hebb")


@dataclass(frozen=True)
class HebbianConfig:
    """Hyperparameters for the fixed-capacity local-learning MLP."""

    num_inputs: int = 784
    num_hidden: int = 100
    num_outputs: int = 10
    variant: str = "oja"
    activation_type: str = "sigmoid"
    bias: bool = False
    hidden_lr: float = 1e-3
    output_lr: float = 1e-2
    output_rule: str = "delta"
    temperature: float = 0.2
    top_k: int = 5
    normalize_hidden: bool = True
    weight_decay: float = 0.0
    eps: float = 1e-8


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared IID-selection and continual-learning protocol."""

    seed: int = 0
    batch_size: int = 64
    train_per_class: int = 5000
    iid_max_steps: int = 2000
    phase1_max_steps: int = 2000
    phase2_max_steps: int = 2000
    eval_every_steps: int = 100
    old_target_accuracy: float = 90.0
    matched_new_accuracies: tuple[float, ...] = (70.0, 80.0, 90.0)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic CuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HebbianMLP(nn.Module):
    """A 784 -> hidden -> 10 MLP trained with explicit local updates.

    Hidden updates never use a label or backpropagated error. The output update
    uses only the output prediction/target and presynaptic hidden activity.
    Parameters have autograd disabled during learning; ``local_update`` applies
    all changes under ``torch.no_grad()``.
    """

    def __init__(self, config: HebbianConfig):
        super().__init__()
        if config.variant not in HEBBIAN_VARIANTS:
            raise ValueError(f"Unknown variant {config.variant!r}; choose from {HEBBIAN_VARIANTS}.")
        if config.output_rule not in OUTPUT_RULES:
            raise ValueError(f"Unknown output rule {config.output_rule!r}; choose from {OUTPUT_RULES}.")
        if config.hidden_lr <= 0 or config.output_lr <= 0:
            raise ValueError("Learning rates must be positive.")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 1 <= config.top_k <= config.num_hidden:
            raise ValueError("top_k must be between 1 and num_hidden.")

        self.config = config
        self.num_inputs = config.num_inputs
        self.num_hidden = config.num_hidden
        self.num_outputs = config.num_outputs
        self.lin1 = nn.Linear(config.num_inputs, config.num_hidden, bias=config.bias)
        self.lin2 = nn.Linear(config.num_hidden, config.num_outputs, bias=config.bias)
        if config.activation_type == "sigmoid":
            self.activation = torch.sigmoid
        elif config.activation_type == "tanh":
            self.activation = torch.tanh
        elif config.activation_type == "relu":
            self.activation = torch.relu
        elif config.activation_type == "identity":
            self.activation = lambda value: value
        else:
            raise ValueError("activation_type must be sigmoid, tanh, relu, or identity.")

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if config.normalize_hidden:
            self._normalise_hidden_rows()

    def _flatten(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(-1, self.num_inputs)

    def hidden_activity(self, X: torch.Tensor) -> torch.Tensor:
        X = self._flatten(X)
        drive = self.lin1(X)
        base = self.activation(drive)
        if self.config.variant in ("centered", "oja"):
            return base
        if self.config.variant == "hard_wta":
            winners = torch.topk(base, k=self.config.top_k, dim=1).indices
            mask = torch.zeros_like(base)
            mask.scatter_(1, winners, 1.0)
            # Preserve an activity scale comparable with the dense variants.
            # Without this factor the readout signal is roughly H/k times
            # smaller and appeared not to learn within the screening budget.
            return base * mask * (self.num_hidden / self.config.top_k)
        responsibilities = torch.softmax(drive / self.config.temperature, dim=1)
        # Softmax responsibilities sum to one; multiply by H so that this
        # variant is not given an artificially tiny output-learning signal.
        return base * responsibilities * self.num_hidden

    def logits(self, X: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.hidden_activity(X))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(X), dim=1)

    def proposed_updates(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return local ascent/update directions without changing weights."""
        with torch.no_grad():
            X_flat = self._flatten(X)
            batch_size = X_flat.shape[0]
            hidden = self.hidden_activity(X_flat)

            if self.config.variant == "centered":
                hidden_update = hidden.T @ X_flat / batch_size
                hidden_update -= hidden_update.mean(dim=0, keepdim=True)
            elif self.config.variant == "oja":
                # Generalized Oja/Sanger update. Centering removes the common
                # sigmoid component and the triangular lateral term prevents
                # every unit from collapsing onto the first principal feature.
                learning_hidden = hidden - hidden.mean(dim=0, keepdim=True)
                correlation = learning_hidden.T @ X_flat / batch_size
                lateral = torch.tril(learning_hidden.T @ learning_hidden / batch_size)
                hidden_update = correlation - lateral @ self.lin1.weight
            else:
                correlation = hidden.T @ X_flat / batch_size
                oja_stabiliser = hidden.square().mean(dim=0).unsqueeze(1) * self.lin1.weight
                hidden_update = correlation - oja_stabiliser

            targets = F.one_hot(y, num_classes=self.num_outputs).to(dtype=X_flat.dtype)
            probabilities = torch.softmax(self.lin2(hidden), dim=1)
            if self.config.output_rule == "delta":
                output_signal = targets - probabilities
                output_update = output_signal.T @ hidden / batch_size
            else:
                output_signal = targets
                output_update = output_signal.T @ hidden / batch_size
                output_update -= output_update.mean(dim=0, keepdim=True)

            if self.config.weight_decay:
                hidden_update -= self.config.weight_decay * self.lin1.weight
                output_update -= self.config.weight_decay * self.lin2.weight

            updates: Dict[str, torch.Tensor] = {
                "lin1_weight": hidden_update,
                "lin2_weight": output_update,
            }
            if self.config.bias:
                hidden_bias_update = hidden.mean(dim=0)
                if self.config.variant == "centered":
                    hidden_bias_update -= hidden_bias_update.mean()
                updates["lin1_bias"] = hidden_bias_update
                updates["lin2_bias"] = output_signal.mean(dim=0)
            return {name: update.detach().clone() for name, update in updates.items()}

    def local_update(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        updates = self.proposed_updates(X, y)
        with torch.no_grad():
            self.lin1.weight.add_(updates["lin1_weight"], alpha=self.config.hidden_lr)
            self.lin2.weight.add_(updates["lin2_weight"], alpha=self.config.output_lr)
            if self.config.bias:
                self.lin1.bias.add_(updates["lin1_bias"], alpha=self.config.hidden_lr)
                self.lin2.bias.add_(updates["lin2_bias"], alpha=self.config.output_lr)
            if self.config.normalize_hidden:
                self._normalise_hidden_rows()

        hidden = self.hidden_activity(X)
        return {
            "lin1_update_norm": updates["lin1_weight"].norm().item(),
            "lin2_update_norm": updates["lin2_weight"].norm().item(),
            "lin1_weight_norm": self.lin1.weight.norm().item(),
            "lin2_weight_norm": self.lin2.weight.norm().item(),
            "hidden_active_fraction": (hidden.abs() > self.config.eps).float().mean().item(),
        }

    def _normalise_hidden_rows(self) -> None:
        with torch.no_grad():
            norms = self.lin1.weight.norm(dim=1, keepdim=True).clamp_min(self.config.eps)
            self.lin1.weight.div_(norms)


def dataset_targets(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        parent = dataset_targets(dataset.dataset)
        return parent[torch.as_tensor(dataset.indices, dtype=torch.long)]
    if isinstance(dataset, ConcatDataset):
        return torch.cat([dataset_targets(part) for part in dataset.datasets])
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    return torch.as_tensor([int(dataset[index][1]) for index in range(len(dataset))])


def stratified_train_valid_split(
    full_train_dataset: Dataset,
    train_per_class: int = 5000,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Make an exact 50k/10k stratified split for standard MNIST."""
    targets = dataset_targets(full_train_dataset)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    for class_id in sorted(targets.unique().tolist()):
        class_indices = torch.where(targets == class_id)[0]
        if len(class_indices) <= train_per_class:
            raise ValueError(f"Class {class_id} has too few examples for the requested split.")
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        train_indices.extend(shuffled[:train_per_class].tolist())
        valid_indices.extend(shuffled[train_per_class:].tolist())
    return Subset(full_train_dataset, train_indices), Subset(full_train_dataset, valid_indices)


def restrict_to_classes(dataset: Dataset, classes: Sequence[int]) -> Subset:
    targets = dataset_targets(dataset)
    allowed = torch.as_tensor(tuple(classes), dtype=torch.long)
    mask = (targets.unsqueeze(1) == allowed.unsqueeze(0)).any(dim=1)
    return Subset(dataset, torch.where(mask)[0].tolist())


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


class BalancedOldNewBatchSampler(BatchSampler):
    """Yield fixed-size batches containing equal numbers of old and new items."""

    def __init__(self, num_old: int, num_new: int, batch_size: int, steps_per_epoch: int, seed: int):
        if batch_size % 2:
            raise ValueError("Balanced interleaving requires an even batch size.")
        if min(num_old, num_new, steps_per_epoch) < 1:
            raise ValueError("Both datasets and steps_per_epoch must be non-empty.")
        self.num_old = num_old
        self.num_new = num_new
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

    @staticmethod
    def _stream(size: int, offset: int, generator: torch.Generator) -> Iterator[int]:
        while True:
            for index in torch.randperm(size, generator=generator).tolist():
                yield index + offset

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        old_stream = self._stream(self.num_old, 0, generator)
        new_stream = self._stream(self.num_new, self.num_old, generator)
        half = self.batch_size // 2
        for _ in range(self.steps_per_epoch):
            batch = [next(old_stream) for _ in range(half)]
            batch.extend(next(new_stream) for _ in range(half))
            order = torch.randperm(self.batch_size, generator=generator).tolist()
            yield [batch[index] for index in order]

    def __len__(self) -> int:
        return self.steps_per_epoch


def make_balanced_interleaved_loader(
    old_dataset: Dataset,
    new_dataset: Dataset,
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
) -> DataLoader:
    dataset = ConcatDataset([old_dataset, new_dataset])
    sampler = BalancedOldNewBatchSampler(
        len(old_dataset), len(new_dataset), batch_size, steps_per_epoch, seed
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def build_mnist_protocol(full_train_dataset: Dataset, test_dataset: Dataset, config: ExperimentConfig):
    train_dataset, valid_dataset = stratified_train_valid_split(
        full_train_dataset, config.train_per_class, config.seed
    )
    datasets = {
        "train_joint": train_dataset,
        "valid_joint": valid_dataset,
        "test_joint": test_dataset,
        "train_old": restrict_to_classes(train_dataset, OLD_CLASSES),
        "train_new": restrict_to_classes(train_dataset, NEW_CLASSES),
        "valid_old": restrict_to_classes(valid_dataset, OLD_CLASSES),
        "valid_new": restrict_to_classes(valid_dataset, NEW_CLASSES),
        "test_old": restrict_to_classes(test_dataset, OLD_CLASSES),
        "test_new": restrict_to_classes(test_dataset, NEW_CLASSES),
    }
    evaluation_loaders = {
        name: make_loader(dataset, config.batch_size, False, config.seed)
        for name, dataset in datasets.items()
        if name.startswith("valid_") or name.startswith("test_")
    }
    return datasets, evaluation_loaders


def evaluate(model: HebbianMLP, loader: DataLoader, device: str) -> Dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    seen_by_class: Dict[int, int] = {}
    correct_by_class: Dict[int, int] = {}
    confusion = torch.zeros(model.num_outputs, model.num_outputs, dtype=torch.long)
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            probabilities = model(X)
            predictions = probabilities.argmax(dim=1)
            loss = F.nll_loss(probabilities.clamp_min(1e-8).log(), y)
            total += y.numel()
            correct += (predictions == y).sum().item()
            loss_sum += loss.item() * y.numel()
            for target, prediction in zip(y.cpu().tolist(), predictions.cpu().tolist()):
                confusion[target, prediction] += 1
            for class_id in y.unique().tolist():
                mask = y == class_id
                seen = int(mask.sum().item())
                hits = int((predictions[mask] == y[mask]).sum().item())
                seen_by_class[int(class_id)] = seen_by_class.get(int(class_id), 0) + seen
                correct_by_class[int(class_id)] = correct_by_class.get(int(class_id), 0) + hits
    per_class = {
        class_id: 100.0 * correct_by_class.get(class_id, 0) / seen
        for class_id, seen in sorted(seen_by_class.items())
    }
    return {
        "loss": loss_sum / total,
        "micro_accuracy": 100.0 * correct / total,
        "macro_accuracy": float(np.mean(list(per_class.values()))),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
        "num_examples": total,
    }


def evaluate_named_loaders(model: HebbianMLP, loaders: Mapping[str, DataLoader], device: str) -> Dict:
    metrics: Dict = {}
    for name, loader in loaders.items():
        result = evaluate(model, loader, device)
        for key in ("loss", "micro_accuracy", "macro_accuracy", "per_class_accuracy", "confusion_matrix"):
            metrics[f"{name}_{key}"] = result[key]
    return metrics


def _infinite_batches(loader: Iterable) -> Iterator:
    while True:
        yield from loader


def train_local_steps(
    model: HebbianMLP,
    train_loader: DataLoader,
    evaluation_loaders: Mapping[str, DataLoader],
    max_steps: int,
    eval_every_steps: int,
    device: str,
    metadata: Mapping | None = None,
    stop_metric: str | None = None,
    stop_value: float | None = None,
    checkpoint_metric: str | None = None,
    checkpoint_targets: Sequence[float] = (),
) -> tuple[list[Dict], int, Dict[float, Dict[str, torch.Tensor]]]:
    """Train by local update count and capture first matched-accuracy states."""
    if min(max_steps, eval_every_steps) < 1:
        raise ValueError("max_steps and eval_every_steps must be positive.")
    model.to(device)
    metadata = dict(metadata or {})
    records: list[Dict] = []
    checkpoints: Dict[float, Dict[str, torch.Tensor]] = {}
    old_examples_seen = 0
    new_examples_seen = 0

    def record(step: int, diagnostics: Mapping | None = None) -> Dict:
        row = {
            **metadata,
            "step": step,
            "examples_seen": old_examples_seen + new_examples_seen,
            "old_examples_seen": old_examples_seen,
            "new_examples_seen": new_examples_seen,
            **evaluate_named_loaders(model, evaluation_loaders, device),
        }
        if diagnostics:
            row.update(diagnostics)
        records.append(row)
        if checkpoint_metric:
            current = float(row.get(checkpoint_metric, float("-inf")))
            for target in checkpoint_targets:
                if target not in checkpoints and current >= target:
                    checkpoints[target] = copy.deepcopy(model.state_dict())
        return row

    first = record(0)
    if stop_metric and stop_value is not None and first.get(stop_metric, float("-inf")) >= stop_value:
        return records, 0, checkpoints

    diagnostic_sums: Dict[str, float] = {}
    diagnostic_count = 0
    for step, (X, y) in enumerate(_infinite_batches(train_loader), start=1):
        X, y = X.to(device), y.to(device)
        old_mask = torch.isin(y, torch.as_tensor(OLD_CLASSES, device=y.device))
        old_examples_seen += int(old_mask.sum().item())
        new_examples_seen += int((~old_mask).sum().item())
        diagnostics = model.local_update(X, y)
        diagnostic_count += 1
        for key, value in diagnostics.items():
            diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value)

        if step % eval_every_steps == 0 or step == max_steps:
            averages = {
                f"mean_{key}": value / diagnostic_count
                for key, value in diagnostic_sums.items()
            }
            latest = record(step, averages)
            diagnostic_sums.clear()
            diagnostic_count = 0
            if stop_metric and stop_value is not None and latest.get(stop_metric, float("-inf")) >= stop_value:
                return records, step, checkpoints
        if step >= max_steps:
            return records, step, checkpoints
    raise RuntimeError("The training loader yielded no batches.")


def run_iid_candidate(
    model_config: HebbianConfig,
    experiment: ExperimentConfig,
    datasets: Mapping[str, Dataset],
    validation_loader: DataLoader,
) -> tuple[list[Dict], Dict]:
    set_seed(experiment.seed)
    model = HebbianMLP(model_config).to(experiment.device)
    train_loader = make_loader(
        datasets["train_joint"], experiment.batch_size, True, experiment.seed + 17
    )
    records, steps, _ = train_local_steps(
        model,
        train_loader,
        {"valid_joint": validation_loader},
        experiment.iid_max_steps,
        experiment.eval_every_steps,
        experiment.device,
        metadata={"seed": experiment.seed, "variant": model_config.variant, "phase": "iid"},
    )
    final = records[-1]
    per_class = final["valid_joint_per_class_accuracy"]
    summary = {
        "seed": experiment.seed,
        "variant": model_config.variant,
        "steps": steps,
        "macro_accuracy": final["valid_joint_macro_accuracy"],
        "micro_accuracy": final["valid_joint_micro_accuracy"],
        "minimum_digit_accuracy": min(per_class.values()),
        "finite_weights": all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
        "per_class_accuracy": per_class,
    }
    return records, summary


def run_continual_experiment(
    model_config: HebbianConfig,
    experiment: ExperimentConfig,
    datasets: Mapping[str, Dataset],
    validation_loaders: Mapping[str, DataLoader],
) -> Dict:
    set_seed(experiment.seed)
    model = HebbianMLP(model_config).to(experiment.device)
    old_loader = make_loader(datasets["train_old"], experiment.batch_size, True, experiment.seed + 101)
    evaluation = {
        "valid_old": validation_loaders["valid_old"],
        "valid_new": validation_loaders["valid_new"],
        "valid_joint": validation_loaders["valid_joint"],
    }
    phase1_records, phase1_steps, _ = train_local_steps(
        model,
        old_loader,
        evaluation,
        experiment.phase1_max_steps,
        experiment.eval_every_steps,
        experiment.device,
        metadata={
            "seed": experiment.seed,
            "variant": model_config.variant,
            "condition": "shared_phase1",
            "phase": 1,
        },
        stop_metric="valid_old_macro_accuracy",
        stop_value=experiment.old_target_accuracy,
    )
    old_before = phase1_records[-1]["valid_old_macro_accuracy"]
    if old_before < experiment.old_target_accuracy:
        raise RuntimeError(
            f"Seed {experiment.seed} reached only {old_before:.2f}% old validation accuracy; "
            "the forgetting experiment is inconclusive."
        )
    phase1_state = copy.deepcopy(model.state_dict())
    condition_results = {}
    for condition_index, condition in enumerate(("sequential", "interleaved")):
        branch = HebbianMLP(model_config).to(experiment.device)
        branch.load_state_dict(phase1_state)
        loader_seed = experiment.seed + 1001 + condition_index
        if condition == "sequential":
            phase2_loader = make_loader(
                datasets["train_new"], experiment.batch_size, True, loader_seed
            )
        else:
            steps_per_epoch = max(
                1, int(np.ceil(len(datasets["train_new"]) / (experiment.batch_size / 2)))
            )
            phase2_loader = make_balanced_interleaved_loader(
                datasets["train_old"], datasets["train_new"], experiment.batch_size,
                steps_per_epoch, loader_seed
            )
        records, steps, checkpoints = train_local_steps(
            branch,
            phase2_loader,
            evaluation,
            experiment.phase2_max_steps,
            experiment.eval_every_steps,
            experiment.device,
            metadata={
                "seed": experiment.seed,
                "variant": model_config.variant,
                "condition": condition,
                "phase": 2,
                "phase1_steps": phase1_steps,
            },
            checkpoint_metric="valid_new_macro_accuracy",
            checkpoint_targets=experiment.matched_new_accuracies,
        )
        condition_results[condition] = {
            "records": records,
            "steps": steps,
            "checkpoint_states": checkpoints,
            "final_state": copy.deepcopy(branch.state_dict()),
        }
    return {
        "model_config": asdict(model_config),
        "experiment_config": asdict(experiment),
        "phase1_records": phase1_records,
        "phase1_steps": phase1_steps,
        "phase1_state": phase1_state,
        "conditions": condition_results,
    }


def first_record_at_target(records: Sequence[Mapping], metric: str, target: float):
    for record in records:
        if float(record.get(metric, float("-inf"))) >= target:
            return record
    return None


def forgetting_at_target(old_before: float, records: Sequence[Mapping], target: float = 80.0) -> Dict:
    matched = first_record_at_target(records, "valid_new_macro_accuracy", target)
    if matched is None:
        return {
            "target": target,
            "target_reached": False,
            "matched_step": None,
            "retained_old_accuracy": None,
            "new_accuracy": None,
            "joint_accuracy": None,
            "forgetting": None,
        }
    retained = float(matched["valid_old_macro_accuracy"])
    return {
        "target": target,
        "target_reached": True,
        "matched_step": int(matched["step"]),
        "retained_old_accuracy": retained,
        "new_accuracy": float(matched["valid_new_macro_accuracy"]),
        "joint_accuracy": float(matched["valid_joint_macro_accuracy"]),
        "forgetting": float(old_before) - retained,
    }


def calculate_cosine_similarity(first, second, epsilon: float = 1e-12) -> float:
    first_vector = np.asarray(first).reshape(-1)
    second_vector = np.asarray(second).reshape(-1)
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator <= epsilon:
        return float("nan")
    return float(np.dot(first_vector, second_vector) / denominator)


def compute_elementwise_snr(samples, epsilon: float = 1e-7) -> float:
    values = np.asarray(samples)
    return float(np.mean(np.abs(values.mean(axis=0)) / (values.std(axis=0) + epsilon)))


def compute_vector_snr(samples, epsilon: float = 1e-7) -> float:
    values = np.asarray(samples).reshape(len(samples), -1)
    mean_vector = values.mean(axis=0)
    residuals = values - mean_vector
    rms_residual_norm = np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))
    return float(np.linalg.norm(mean_vector) / (rms_residual_norm + epsilon))


def _backprop_descent_directions(model: HebbianMLP, X: torch.Tensor, y: torch.Tensor):
    parameters = dict(model.named_parameters())
    original_flags = {name: parameter.requires_grad for name, parameter in parameters.items()}
    try:
        for parameter in parameters.values():
            parameter.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        probabilities = model(X)
        F.nll_loss(probabilities.clamp_min(1e-8).log(), y).backward()
        return {
            name.replace(".", "_"): -parameter.grad.detach().clone()
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():
            parameter.requires_grad_(original_flags[name])


def update_backprop_alignment(model: HebbianMLP, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    local = model.proposed_updates(X, y)
    backprop = _backprop_descent_directions(model, X, y)
    return {
        name: calculate_cosine_similarity(local[name].cpu().numpy(), backprop[name].cpu().numpy())
        for name in sorted(set(local).intersection(backprop))
    }


def alignment_over_loader(model: HebbianMLP, loader: DataLoader, max_batches: int = 10) -> Dict[str, float]:
    device = next(model.parameters()).device
    collected = defaultdict(list)
    for batch_index, (X, y) in enumerate(loader):
        values = update_backprop_alignment(model, X.to(device), y.to(device))
        for name, value in values.items():
            collected[name].append(value)
        if batch_index + 1 >= max_batches:
            break
    return {name: float(np.nanmean(values)) for name, values in collected.items()}


def update_snrs(
    model: HebbianMLP,
    dataset: Dataset,
    batch_size: int = 64,
    max_batches: int = 16,
) -> Dict:
    """Measure variation across minibatch update proposals.

    The Oja/Sanger rule centers activity across a minibatch, so batch size one
    would make its proposed hidden update identically zero and produce a
    meaningless SNR of zero.
    """
    device = next(model.parameters()).device
    collected = defaultdict(list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_index, (X, y) in enumerate(loader):
        updates = model.proposed_updates(X.to(device), y.to(device))
        for name, value in updates.items():
            collected[name].append(value.cpu().numpy())
        if batch_index + 1 >= max_batches:
            break
    return {
        name: {
            "elementwise_snr": compute_elementwise_snr(values),
            "vector_snr": compute_vector_snr(values),
        }
        for name, values in collected.items()
    }


def run_preflight_checks() -> None:
    """Fast tensor-only checks to run before downloading/training MNIST."""
    set_seed(0)
    X = torch.randn(8, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    for variant in HEBBIAN_VARIANTS:
        config = HebbianConfig(
            num_inputs=4, num_hidden=5, num_outputs=3, variant=variant, top_k=2
        )
        model = HebbianMLP(config)
        probabilities = model(X)
        assert probabilities.shape == (8, 3)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(8), atol=1e-6)
        assert all(not parameter.requires_grad for parameter in model.parameters())
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}
        updates = model.proposed_updates(X, y)
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, before[name])
        permuted = y.flip(0)
        assert torch.equal(
            updates["lin1_weight"], model.proposed_updates(X, permuted)["lin1_weight"]
        )
        model.local_update(X, y)
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        if variant == "hard_wta":
            assert torch.all((model.hidden_activity(X) != 0).sum(dim=1) <= 2)

    sampler = BalancedOldNewBatchSampler(5, 7, batch_size=4, steps_per_epoch=3, seed=0)
    for batch in sampler:
        assert sum(index < 5 for index in batch) == 2
        assert sum(index >= 5 for index in batch) == 2

    delta_model = HebbianMLP(
        HebbianConfig(num_inputs=4, num_hidden=5, num_outputs=3, variant="oja")
    )
    alignment = update_backprop_alignment(delta_model, X, y)
    assert alignment["lin2_weight"] > 0.999
    print("All preflight checks passed.")
