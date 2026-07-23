import torch
from torch.utils.data import DataLoader, TensorDataset

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
