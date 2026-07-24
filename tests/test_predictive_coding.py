import numpy as np
import pytest
import torch
import jax
from torch.utils.data import DataLoader, TensorDataset

from src.models.predictive_coding import PredictiveCodingMLP
from src.models.base import BasicOptimizer, train_model
from src.experiments.comparative import _run_recorded_epoch
from src.experiments.forgetting import run_forgetting_experiment


def test_predictive_coding_mlp_initialization_and_forward():
    model = PredictiveCodingMLP(num_inputs=10, num_hidden=20, num_outputs=5, bias=False)
    X = torch.randn(8, 10)
    out = model(X)
    assert out.shape == (8, 5)
    assert torch.allclose(out.sum(dim=1), torch.ones(8), atol=1e-4)


def test_predictive_coding_step_batch_updates_weights():
    model = PredictiveCodingMLP(num_inputs=10, num_hidden=20, num_outputs=5, bias=False)
    initial_w1 = model.lin1.weight.data.clone()

    X = torch.randn(8, 10)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

    out = model.step_batch(X, y)
    assert out.shape == (8, 5)
    assert not torch.allclose(model.lin1.weight.data, initial_w1)


def test_predictive_coding_proposed_updates_do_not_mutate_model_or_state():
    model = PredictiveCodingMLP(
        num_inputs=10,
        num_hidden=20,
        num_outputs=5,
        bias=False,
    )
    X = torch.randn(8, 10)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    torch_hidden_before = model.lin1.weight.detach().clone()
    torch_output_before = model.lin2.weight.detach().clone()
    jpc_hidden_before = np.asarray(model.jpc_model[0][1].weight).copy()
    jpc_output_before = np.asarray(model.jpc_model[1][1].weight).copy()

    updates = model.proposed_updates(X, y)

    assert updates["hidden_0_weight"].shape == model.lin1.weight.shape
    assert updates["output_weight"].shape == model.lin2.weight.shape
    assert torch.isfinite(updates["hidden_0_weight"]).all()
    assert torch.isfinite(updates["output_weight"]).all()
    assert torch.equal(model.lin1.weight, torch_hidden_before)
    assert torch.equal(model.lin2.weight, torch_output_before)
    assert np.array_equal(np.asarray(model.jpc_model[0][1].weight), jpc_hidden_before)
    assert np.array_equal(np.asarray(model.jpc_model[1][1].weight), jpc_output_before)


def test_predictive_coding_recorded_baseline_does_not_update_weights():
    model = PredictiveCodingMLP(
        num_inputs=10,
        num_hidden=20,
        num_outputs=5,
        bias=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(8, 10), torch.arange(8) % 5),
        batch_size=4,
        shuffle=False,
    )
    hidden_before = np.asarray(model.jpc_model[0][1].weight).copy()
    output_before = np.asarray(model.jpc_model[1][1].weight).copy()

    results = _run_recorded_epoch(
        model,
        "predictive_coding",
        loader,
        loader,
        optimizer=None,
        no_train=True,
    )

    assert "avg_valid_accuracies" in results
    assert np.array_equal(np.asarray(model.jpc_model[0][1].weight), hidden_before)
    assert np.array_equal(np.asarray(model.jpc_model[1][1].weight), output_before)


def test_predictive_coding_optimizer_reset_preserves_weights_and_clears_moments():
    model = PredictiveCodingMLP(
        num_inputs=10,
        num_hidden=20,
        num_outputs=5,
        bias=False,
    )

    def state_arrays():
        return [
            np.asarray(leaf).copy()
            for leaf in jax.tree_util.tree_leaves(model.jpc_opt_state)
            if hasattr(leaf, "shape")
        ]

    initial_state = state_arrays()
    model.step_batch(torch.randn(8, 10), torch.arange(8) % 5)
    advanced_state = state_arrays()
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(initial_state, advanced_state)
    )

    hidden_before_reset = np.asarray(model.jpc_model[0][1].weight).copy()
    output_before_reset = np.asarray(model.jpc_model[1][1].weight).copy()
    model.reset_optimizer_state()
    reset_state = state_arrays()

    assert all(
        np.array_equal(before, after)
        for before, after in zip(initial_state, reset_state)
    )
    assert np.array_equal(
        np.asarray(model.jpc_model[0][1].weight),
        hidden_before_reset,
    )
    assert np.array_equal(
        np.asarray(model.jpc_model[1][1].weight),
        output_before_reset,
    )


def test_predictive_coding_forgetting_experiment_smoke():
    X_old = torch.randn(32, 784)
    y_old = torch.tensor([0, 1, 2, 3, 4] * 6 + [0, 1])
    old_dataset = TensorDataset(X_old, y_old)

    X_new = torch.randn(32, 784)
    y_new = torch.tensor([5, 6, 7, 8, 9] * 6 + [5, 6])
    new_dataset = TensorDataset(X_new, y_new)

    train_loader_old = DataLoader(old_dataset, batch_size=8, shuffle=True)
    valid_loader_old = DataLoader(old_dataset, batch_size=8, shuffle=False)
    train_loader_new = DataLoader(new_dataset, batch_size=8, shuffle=True)
    valid_loader_new = DataLoader(new_dataset, batch_size=8, shuffle=False)
    train_loader_full = DataLoader(TensorDataset(torch.cat([X_old, X_new]), torch.cat([y_old, y_new])), batch_size=8, shuffle=True)

    results = run_forgetting_experiment(
        train_loader_old=train_loader_old,
        valid_loader_old=valid_loader_old,
        train_loader_new=train_loader_new,
        valid_loader_new=valid_loader_new,
        train_loader_full=train_loader_full,
        model_type="predictive_coding",
        num_epochs_phase1=2,
        num_epochs_phase2=2,
        lr=2e-3,
    )

    assert "model" in results
    assert results["model"].lr == 2e-3
    assert len(results["old_class_acc_trace"]) == 4


@pytest.mark.parametrize("optimizer_type", ["adam", "sgd"])
def test_shared_runner_rejects_external_optimizer_for_predictive_coding(
    optimizer_type,
):
    with pytest.raises(ValueError, match="cannot use external optimizer"):
        run_forgetting_experiment(
            train_loader_old=None,
            valid_loader_old=None,
            train_loader_new=None,
            valid_loader_new=None,
            train_loader_full=None,
            model_type="predictive_coding",
            optimizer_type=optimizer_type,
            device="cpu",
        )
