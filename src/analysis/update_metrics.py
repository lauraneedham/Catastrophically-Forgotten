"""Non-mutating update-direction analysis for the four learning rules.

The final comparison uses one hidden layer, so every rule is mapped onto the
same two weight tensors:

``hidden_weight``
    The 784 -> hidden connection.

``output_weight``
    The hidden -> 10 connection.

The functions in this module compare a rule's proposed parameter change with
the ordinary backpropagation descent direction evaluated at the same forward
weights and on the same mini-batch.  They intentionally do not assume that all
rules expose PyTorch gradients: Hebbian learning supplies explicit local
updates and predictive coding supplies a non-mutating JPC step delta.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


LAYER_KEYS = ("hidden_weight", "output_weight")
RULE_ALIASES = {
    "bp": "backprop",
    "backpropagation": "backprop",
    "fa": "feedback_alignment",
    "feedback-alignment": "feedback_alignment",
    "pc": "predictive_coding",
    "predictive-coding": "predictive_coding",
    "oja": "hebbian",
}


def canonical_rule_name(rule: str) -> str:
    """Return the repository's canonical name for a learning rule."""
    normalized = rule.strip().lower().replace(" ", "_")
    return RULE_ALIASES.get(normalized, normalized)


def _activation(value: torch.Tensor, activation_type: str) -> torch.Tensor:
    activation_type = activation_type.lower()
    if activation_type == "sigmoid":
        return torch.sigmoid(value)
    if activation_type == "tanh":
        return torch.tanh(value)
    if activation_type == "relu":
        return torch.relu(value)
    if activation_type == "identity":
        return value
    raise ValueError(f"Unsupported activation_type '{activation_type}'.")


def _weight_layers(model) -> tuple[Any, Any]:
    """Return the first hidden and output layers for the shared architecture."""
    try:
        hidden_layer = model.lin1
        output_layer = model.lin2
    except AttributeError as exc:
        raise TypeError(
            "Update analysis requires model.lin1 and model.lin2 weight layers."
        ) from exc

    if not hasattr(hidden_layer, "weight") or not hasattr(output_layer, "weight"):
        raise TypeError("Both analysis layers must expose a weight tensor.")
    return hidden_layer, output_layer


def backprop_descent_directions(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute ``-dL/dW`` at a model's current forward weights.

    This builds an ordinary functional PyTorch graph from detached copies of
    the forward weights.  It is therefore a valid backpropagation reference
    even when ``model.lin1`` and ``model.lin2`` are feedback-alignment custom
    autograd modules.
    """
    hidden_layer, output_layer = _weight_layers(model)
    num_inputs = int(model.num_inputs)

    hidden_weight = hidden_layer.weight.detach().clone().requires_grad_(True)
    output_weight = output_layer.weight.detach().clone().requires_grad_(True)

    hidden_bias = getattr(hidden_layer, "bias", None)
    output_bias = getattr(output_layer, "bias", None)
    hidden_bias_copy = (
        None
        if hidden_bias is None
        else hidden_bias.detach().clone().requires_grad_(True)
    )
    output_bias_copy = (
        None
        if output_bias is None
        else output_bias.detach().clone().requires_grad_(True)
    )

    flattened = X.reshape(-1, num_inputs)
    hidden = _activation(
        F.linear(flattened, hidden_weight, hidden_bias_copy),
        model.activation_type,
    )
    logits = F.linear(hidden, output_weight, output_bias_copy)
    loss = F.cross_entropy(logits, y)
    hidden_grad, output_grad = torch.autograd.grad(
        loss,
        (hidden_weight, output_weight),
        create_graph=False,
        retain_graph=False,
    )

    return {
        "hidden_weight": -hidden_grad.detach(),
        "output_weight": -output_grad.detach(),
    }


def _autograd_rule_directions(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Collect a rule's custom-autograd pseudo-gradient as an update."""
    hidden_layer, output_layer = _weight_layers(model)
    probabilities = model(X, y=y)
    loss = F.nll_loss(torch.log(probabilities.clamp_min(1e-8)), y)
    hidden_grad, output_grad = torch.autograd.grad(
        loss,
        (hidden_layer.weight, output_layer.weight),
        create_graph=False,
        retain_graph=False,
    )
    return {
        "hidden_weight": -hidden_grad.detach(),
        "output_weight": -output_grad.detach(),
    }


def _explicit_rule_directions(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map a model's explicit proposed updates onto the shared layer names."""
    if not hasattr(model, "proposed_updates"):
        raise TypeError(
            f"{type(model).__name__} does not expose proposed_updates(X, y)."
        )

    raw_updates: Mapping[str, torch.Tensor] = model.proposed_updates(X, y)
    hidden_keys = sorted(
        key
        for key in raw_updates
        if key.startswith("hidden_") and key.endswith("_weight")
    )
    if hidden_keys != ["hidden_0_weight"]:
        raise ValueError(
            "The final update comparison requires exactly one hidden layer; "
            f"found update keys {hidden_keys}."
        )
    try:
        output_update = raw_updates["output_weight"]
    except KeyError as exc:
        raise KeyError("proposed_updates must contain 'output_weight'.") from exc

    return {
        "hidden_weight": raw_updates["hidden_0_weight"].detach(),
        "output_weight": output_update.detach(),
    }


def proposed_rule_directions(
    model,
    rule: str,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the two proposed weight-change directions for ``rule``.

    The returned tensors are update directions, not optimizer-scaled learning
    rates.  Positive cosine with :func:`backprop_descent_directions` therefore
    means the rule proposes a change in the same direction as loss descent.
    """
    rule = canonical_rule_name(rule)
    if rule == "backprop":
        directions = backprop_descent_directions(model, X, y)
    elif rule == "feedback_alignment":
        directions = _autograd_rule_directions(model, X, y)
    elif rule in {"hebbian", "predictive_coding"}:
        if rule == "hebbian" and len(X) < 2:
            raise ValueError(
                "Batch-centred Hebbian/Oja analysis requires batch size >= 2."
            )
        directions = _explicit_rule_directions(model, X, y)
    else:
        raise ValueError(f"Unsupported learning rule '{rule}'.")

    hidden_layer, output_layer = _weight_layers(model)
    expected_shapes = {
        "hidden_weight": tuple(hidden_layer.weight.shape),
        "output_weight": tuple(output_layer.weight.shape),
    }
    for layer_key in LAYER_KEYS:
        if layer_key not in directions:
            raise KeyError(f"Missing proposed direction '{layer_key}'.")
        if tuple(directions[layer_key].shape) != expected_shapes[layer_key]:
            raise ValueError(
                f"{layer_key} has shape {tuple(directions[layer_key].shape)}, "
                f"expected {expected_shapes[layer_key]}."
            )
        if not torch.isfinite(directions[layer_key]).all():
            raise FloatingPointError(f"{layer_key} contains non-finite values.")
    return directions


def safe_cosine_similarity(
    direction: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float = 1e-12,
) -> float:
    """Return a finite cosine or ``NaN`` when either vector has zero norm."""
    direction_flat = direction.detach().reshape(-1).to(
        device="cpu", dtype=torch.float64
    )
    reference_flat = reference.detach().reshape(-1).to(
        device="cpu", dtype=torch.float64
    )
    if direction_flat.shape != reference_flat.shape:
        raise ValueError(
            f"Cosine vectors have different shapes: {direction_flat.shape} "
            f"and {reference_flat.shape}."
        )

    direction_norm = torch.linalg.vector_norm(direction_flat)
    reference_norm = torch.linalg.vector_norm(reference_flat)
    denominator = direction_norm * reference_norm
    if not torch.isfinite(denominator) or float(denominator) <= epsilon:
        return float("nan")

    cosine = torch.dot(direction_flat, reference_flat) / denominator
    return float(torch.clamp(cosine, min=-1.0, max=1.0).item())


@dataclass
class StreamingCoordinateSNR:
    """Coordinate-wise SNR moments without storing every proposed update."""

    count: int = 0
    coordinate_sum: torch.Tensor | None = None
    coordinate_squared_sum: torch.Tensor | None = None

    def update(self, direction: torch.Tensor) -> None:
        values = direction.detach().to(device="cpu", dtype=torch.float64)
        if self.coordinate_sum is None:
            self.coordinate_sum = torch.zeros_like(values)
            self.coordinate_squared_sum = torch.zeros_like(values)
        elif values.shape != self.coordinate_sum.shape:
            raise ValueError(
                f"Direction shape changed from {self.coordinate_sum.shape} "
                f"to {values.shape}."
            )

        self.coordinate_sum.add_(values)
        self.coordinate_squared_sum.add_(values.square())
        self.count += 1

    def finalize(self, epsilon: float = 1e-7) -> dict[str, float | int]:
        if (
            self.count == 0
            or self.coordinate_sum is None
            or self.coordinate_squared_sum is None
        ):
            raise RuntimeError("Cannot finalize an empty SNR accumulator.")

        mean = self.coordinate_sum / self.count
        variance = self.coordinate_squared_sum / self.count - mean.square()
        standard_deviation = variance.clamp_min(0.0).sqrt()
        coordinate_snr = mean.abs() / (standard_deviation + epsilon)
        finite = torch.isfinite(coordinate_snr)
        if not finite.any():
            average_snr = float("nan")
        else:
            average_snr = float(coordinate_snr[finite].mean().item())

        return {
            "snr": average_snr,
            "num_updates": self.count,
            "num_coordinates": int(coordinate_snr.numel()),
            "finite_coordinate_fraction": float(finite.double().mean().item()),
        }


def _parameter_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_parameter_snapshot_unchanged(
    model,
    snapshot: Mapping[str, torch.Tensor],
) -> None:
    current_names = {name for name, _ in model.named_parameters()}
    if current_names != set(snapshot):
        raise RuntimeError("Model parameter set changed during update analysis.")
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach().cpu(), snapshot[name]):
            raise RuntimeError(
                f"Analysis mutated model parameter '{name}'. "
                "Collectors must be non-mutating."
            )


def analyze_update_statistics(
    model,
    rule: str,
    loader: Iterable,
    *,
    max_batches: int | None = 32,
    snr_epsilon: float = 1e-7,
    cosine_epsilon: float = 1e-12,
    verify_unchanged: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Compute layer-wise streaming SNR and alignment to backprop.

    The model is held fixed.  Every mini-batch yields one rule update and one
    ordinary backpropagation reference update at exactly the same weights.
    Cosines are summarized across mini-batches; SNR is computed coordinate-wise
    across the same mini-batch update collection.
    """
    rule = canonical_rule_name(rule)
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None.")

    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Model has no parameters to analyze.") from exc

    snapshot = _parameter_snapshot(model) if verify_unchanged else {}
    was_training = model.training
    model.eval()

    accumulators = {
        layer_key: StreamingCoordinateSNR() for layer_key in LAYER_KEYS
    }
    cosines: dict[str, list[float]] = {layer_key: [] for layer_key in LAYER_KEYS}
    direction_norms: dict[str, list[float]] = {
        layer_key: [] for layer_key in LAYER_KEYS
    }

    try:
        for batch_index, (X, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            X, y = X.to(device), y.to(device)

            directions = proposed_rule_directions(model, rule, X, y)
            references = backprop_descent_directions(model, X, y)
            for layer_key in LAYER_KEYS:
                accumulators[layer_key].update(directions[layer_key])
                cosines[layer_key].append(
                    safe_cosine_similarity(
                        directions[layer_key],
                        references[layer_key],
                        epsilon=cosine_epsilon,
                    )
                )
                direction_norms[layer_key].append(
                    float(
                        torch.linalg.vector_norm(
                            directions[layer_key].detach().reshape(-1)
                        ).item()
                    )
                )
    finally:
        model.train(was_training)

    results: dict[str, dict[str, float | int]] = {}
    for layer_key in LAYER_KEYS:
        snr_summary = accumulators[layer_key].finalize(epsilon=snr_epsilon)
        valid_cosines = np.asarray(
            [value for value in cosines[layer_key] if np.isfinite(value)],
            dtype=np.float64,
        )
        if len(valid_cosines) == 0:
            cosine_mean = cosine_std = cosine_sem = float("nan")
        else:
            cosine_mean = float(valid_cosines.mean())
            cosine_std = (
                float(valid_cosines.std(ddof=1))
                if len(valid_cosines) > 1
                else 0.0
            )
            cosine_sem = cosine_std / math.sqrt(len(valid_cosines))

        results[layer_key] = {
            **snr_summary,
            "cosine_mean": cosine_mean,
            "cosine_std": cosine_std,
            "cosine_sem": cosine_sem,
            "cosine_valid_batches": int(len(valid_cosines)),
            "cosine_total_batches": int(len(cosines[layer_key])),
            "mean_direction_norm": float(np.mean(direction_norms[layer_key])),
            "zero_or_invalid_cosine_batches": int(
                len(cosines[layer_key]) - len(valid_cosines)
            ),
        }

    if verify_unchanged:
        _assert_parameter_snapshot_unchanged(model, snapshot)
    return results


def update_statistics_records(
    statistics: Mapping[str, Mapping[str, float | int]],
    *,
    rule: str,
    checkpoint: str,
    probe_split: str,
) -> list[dict[str, Any]]:
    """Convert a layer-statistics dictionary into CSV-friendly records."""
    records: list[dict[str, Any]] = []
    for layer_key in LAYER_KEYS:
        record: dict[str, Any] = {
            "rule": canonical_rule_name(rule),
            "checkpoint": checkpoint,
            "probe_split": probe_split,
            "layer": layer_key,
        }
        record.update(statistics[layer_key])
        records.append(record)
    return records
