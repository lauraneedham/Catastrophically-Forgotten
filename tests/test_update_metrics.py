import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from backpropagation import MultiLayerPerceptron
from src.analysis.update_metrics import (
    StreamingCoordinateSNR,
    analyze_update_statistics,
    backprop_descent_directions,
    safe_cosine_similarity,
)


def test_safe_cosine_similarity_handles_direction_and_zero_norm():
    vector = torch.tensor([1.0, 2.0, 3.0])
    assert safe_cosine_similarity(vector, vector) == pytest.approx(1.0)
    assert safe_cosine_similarity(vector, -vector) == pytest.approx(-1.0)
    assert math.isnan(safe_cosine_similarity(vector, torch.zeros_like(vector)))


def test_streaming_coordinate_snr_matches_tutorial_formula():
    accumulator = StreamingCoordinateSNR()
    for row in ([1.0, 4.0], [2.0, 4.0], [3.0, 4.0]):
        accumulator.update(torch.tensor(row))

    epsilon = 1e-7
    summary = accumulator.finalize(epsilon=epsilon)
    expected_first = 2.0 / (np.std([1.0, 2.0, 3.0]) + epsilon)
    expected_second = 4.0 / epsilon
    assert summary["snr"] == pytest.approx(
        np.mean([expected_first, expected_second])
    )
    assert summary["num_updates"] == 3
    assert summary["num_coordinates"] == 2


def test_backprop_reference_has_expected_shapes_and_does_not_mutate_model():
    torch.manual_seed(0)
    model = MultiLayerPerceptron(
        num_inputs=8,
        num_hidden=7,
        num_outputs=3,
        activation_type="sigmoid",
        bias=False,
    )
    X = torch.randn(6, 8)
    y = torch.arange(6) % 3
    hidden_before = model.lin1.weight.detach().clone()
    output_before = model.lin2.weight.detach().clone()

    directions = backprop_descent_directions(model, X, y)

    assert directions["hidden_weight"].shape == model.lin1.weight.shape
    assert directions["output_weight"].shape == model.lin2.weight.shape
    assert torch.equal(model.lin1.weight, hidden_before)
    assert torch.equal(model.lin2.weight, output_before)


def test_backprop_analysis_is_self_aligned_and_non_mutating():
    torch.manual_seed(1)
    model = MultiLayerPerceptron(
        num_inputs=8,
        num_hidden=7,
        num_outputs=3,
        activation_type="sigmoid",
        bias=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(16, 8), torch.arange(16) % 3),
        batch_size=4,
        shuffle=False,
    )
    hidden_before = model.lin1.weight.detach().clone()
    output_before = model.lin2.weight.detach().clone()

    statistics = analyze_update_statistics(
        model,
        "backprop",
        loader,
        max_batches=3,
    )

    for layer in ("hidden_weight", "output_weight"):
        assert statistics[layer]["cosine_mean"] == pytest.approx(1.0)
        assert statistics[layer]["cosine_valid_batches"] == 3
    assert np.isfinite(statistics[layer]["snr"])
    assert torch.equal(model.lin1.weight, hidden_before)
    assert torch.equal(model.lin2.weight, output_before)
