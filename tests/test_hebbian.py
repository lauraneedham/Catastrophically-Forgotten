import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.forgetting import run_forgetting_experiment
from src.models.hebbian import HebbianMultiLayerPerceptron, train_hebbian_model


def test_hebbian_forward_supports_all_requested_architecture_shapes():
    for hidden_dims in ([100], [1000], [300, 300]):
        model = HebbianMultiLayerPerceptron(
            num_inputs=12,
            num_hidden=hidden_dims,
            num_outputs=10,
            lr=0.01,
        )
        probabilities = model(torch.randn(4, 12))
        assert probabilities.shape == (4, 10)
        assert torch.allclose(
            probabilities.sum(dim=1), torch.ones(4), atol=1e-6
        )
        assert model.architecture == (12, *hidden_dims, 10)
        assert all(not parameter.requires_grad for parameter in model.parameters())


def test_hidden_updates_are_label_independent_and_output_update_is_supervised():
    torch.manual_seed(0)
    model = HebbianMultiLayerPerceptron(
        num_inputs=6,
        num_hidden=[5, 4],
        num_outputs=3,
        lr=0.01,
    )
    X = torch.randn(8, 6)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    permuted_y = y.roll(1)

    updates = model.proposed_updates(X, y)
    permuted_updates = model.proposed_updates(X, permuted_y)

    assert torch.equal(updates["hidden_0_weight"], permuted_updates["hidden_0_weight"])
    assert torch.equal(updates["hidden_1_weight"], permuted_updates["hidden_1_weight"])
    assert not torch.equal(updates["output_weight"], permuted_updates["output_weight"])


def test_hebbian_training_updates_without_autograd_or_optimizer():
    torch.manual_seed(1)
    X = torch.randn(24, 8)
    y = torch.arange(24) % 3
    loader = DataLoader(TensorDataset(X, y), batch_size=6, shuffle=False)
    model = HebbianMultiLayerPerceptron(
        num_inputs=8, num_hidden=7, num_outputs=3, lr=0.01
    )
    initial_hidden = model.hidden_layers[0].weight.detach().clone()
    initial_output = model.output_layer.weight.detach().clone()

    results = train_hebbian_model(
        model,
        loader,
        loader,
        num_epochs=1,
        record_initial_baseline=False,
    )

    assert len(results["avg_valid_accuracies"]) == 1
    assert not torch.equal(model.hidden_layers[0].weight, initial_hidden)
    assert not torch.equal(model.output_layer.weight, initial_output)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_proposed_parameter_deltas_match_normalized_training_step():
    torch.manual_seed(4)
    model = HebbianMultiLayerPerceptron(
        num_inputs=8,
        num_hidden=7,
        num_outputs=3,
        lr=0.001,
        normalize_hidden=True,
    )
    trained_copy = copy.deepcopy(model)
    X = torch.randn(12, 8)
    y = torch.arange(12) % 3
    hidden_before = model.hidden_layers[0].weight.detach().clone()
    output_before = model.output_layer.weight.detach().clone()

    proposed = model.proposed_parameter_deltas(X, y)
    trained_copy.local_update(X, y)

    assert torch.allclose(
        proposed["hidden_0_weight"],
        trained_copy.hidden_layers[0].weight - model.hidden_layers[0].weight,
        atol=1e-7,
        rtol=1e-5,
    )
    assert torch.allclose(
        proposed["output_weight"],
        trained_copy.output_layer.weight - model.output_layer.weight,
        atol=1e-7,
        rtol=1e-5,
    )
    assert torch.equal(model.hidden_layers[0].weight, hidden_before)
    assert torch.equal(model.output_layer.weight, output_before)


def test_shared_forgetting_runner_dispatches_hebbian_local_updates():
    torch.manual_seed(2)
    X = torch.randn(24, 8)
    y = torch.arange(24) % 3
    loader = DataLoader(TensorDataset(X, y), batch_size=6, shuffle=False)

    results = run_forgetting_experiment(
        train_loader_old=loader,
        valid_loader_old=loader,
        train_loader_new=loader,
        valid_loader_new=loader,
        train_loader_full=loader,
        model_type="hebbian",
        num_epochs_phase1=2,
        num_epochs_phase2=2,
        num_inputs=8,
        num_hidden=7,
        num_outputs=3,
        lr=0.001,
        device="cpu",
    )

    model = results["model"]
    assert len(results["old_class_acc_trace"]) == 4
    assert model.hidden_lr == 0.001
    assert model.output_lr == 0.001
    assert not torch.equal(model.hidden_layers[0].weight, model.init_weights[0])
    assert not torch.equal(model.output_layer.weight, model.init_weights[-1])
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("optimizer_type", ["adam", "sgd"])
def test_shared_runner_rejects_external_optimizer_for_hebbian(optimizer_type):
    with pytest.raises(ValueError, match="cannot use external optimizer"):
        run_forgetting_experiment(
            train_loader_old=None,
            valid_loader_old=None,
            train_loader_new=None,
            valid_loader_new=None,
            train_loader_full=None,
            model_type="hebbian",
            optimizer_type=optimizer_type,
            device="cpu",
        )
