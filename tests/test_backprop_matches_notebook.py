import torch

from src.models.base import BasicOptimizer, MultiLayerPerceptron, train_model


def test_basic_optimizer_matches_notebook_weight_decay_behavior():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([0.5])

    optimizer = BasicOptimizer([param], lr=0.1, weight_decay=0.2)
    optimizer.step()

    assert torch.allclose(param, torch.tensor([1.0 - 0.1 * (0.5 + 0.2 * 1.0)]))


def test_basic_optimizer_accepts_notebook_style_param_groups_without_top_level_lr():
    param1 = torch.nn.Parameter(torch.tensor([1.0]))
    param2 = torch.nn.Parameter(torch.tensor([1.0]))
    param1.grad = torch.tensor([0.5])
    param2.grad = torch.tensor([0.5])

    optimizer = BasicOptimizer(
        [
            {"params": [param1], "lr": 0.1},
            {"params": [param2], "lr": 0.2},
        ]
    )
    optimizer.step()

    assert torch.allclose(param1, torch.tensor([0.95]))
    assert torch.allclose(param2, torch.tensor([0.9]))


def test_train_model_records_notebook_baseline_before_training_by_default():
    model = MultiLayerPerceptron(num_inputs=2, num_hidden=4, num_outputs=2, bias=False)
    initial_weight = model.lin1.weight.data.clone()

    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)

    optimizer = BasicOptimizer(model.parameters(), lr=0.01)
    train_model(model, train_loader, valid_loader, optimizer, num_epochs=1, verbose=False)

    assert torch.allclose(model.lin1.weight.data, initial_weight)


def test_train_model_can_train_on_first_epoch_when_requested():
    model = MultiLayerPerceptron(num_inputs=2, num_hidden=4, num_outputs=2, bias=False)
    initial_weight = model.lin1.weight.data.clone()

    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)

    optimizer = BasicOptimizer(model.parameters(), lr=0.01)
    train_model(
        model,
        train_loader,
        valid_loader,
        optimizer,
        num_epochs=1,
        verbose=False,
        record_initial_baseline=False,
    )

    assert not torch.allclose(model.lin1.weight.data, initial_weight)
