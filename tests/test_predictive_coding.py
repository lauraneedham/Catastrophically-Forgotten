import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.predictive_coding import PredictiveCodingMLP
from src.models.base import BasicOptimizer, train_model
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
    )

    assert "model" in results
    assert len(results["old_class_acc_trace"]) == 4
