import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.forgetting import build_model, run_forgetting_experiment
from src.models.base import BasicOptimizer, MultiLayerPerceptron, train_model
from src.models.feedback_alignment import FeedbackAlignmentMLP, LinearFAModule
from src.analysis.update_metrics import (
    backprop_descent_directions,
    proposed_rule_directions,
    safe_cosine_similarity,
)


def test_feedback_alignment_forward_returns_probabilities_over_sigmoid_fa_layers():
    for num_hidden in (100, 1000):
        model = FeedbackAlignmentMLP(
            num_inputs=12, num_hidden=num_hidden, num_outputs=10, bias=False
        )
        probabilities = model(torch.randn(4, 12))
        assert probabilities.shape == (4, 10)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)

    # It is the sigmoid feedback-alignment network built from FA layers, not a
    # plain MLP nor the activation-free linear variant.
    assert isinstance(model, MultiLayerPerceptron)
    assert isinstance(model.lin1, LinearFAModule)
    assert isinstance(model.lin2, LinearFAModule)
    assert model.activation_type == "sigmoid"


def test_feedback_alignment_output_update_stays_bp_aligned_when_saturated():
    model = FeedbackAlignmentMLP(
        num_inputs=2,
        num_hidden=2,
        num_outputs=2,
        activation_type="sigmoid",
        bias=False,
    )
    with torch.no_grad():
        model.lin1.weight.zero_()
        model.lin2.weight.copy_(
            torch.tensor(
                [
                    [30.0, 30.0],
                    [-30.0, -30.0],
                ]
            )
        )

    X = torch.zeros(4, 2)
    y = torch.ones(4, dtype=torch.long)
    assert torch.all(model(X)[:, 1] < 1e-8)

    fa = proposed_rule_directions(model, "feedback_alignment", X, y)
    bp = backprop_descent_directions(model, X, y)
    assert safe_cosine_similarity(
        fa["output_weight"],
        bp["output_weight"],
    ) == pytest.approx(1.0)


def test_backward_routes_gradient_through_fixed_feedback_weights():
    torch.manual_seed(0)
    module = LinearFAModule(input_features=6, output_features=4, bias=False)
    inputs = torch.randn(3, 6, requires_grad=True)

    module(inputs).sum().backward()  # grad_output is all ones
    grad_output = torch.ones(3, 4)

    # Feedback alignment: the input gradient uses the fixed random feedback
    # weights...
    assert torch.allclose(inputs.grad, grad_output.mm(module.weight_fa), atol=1e-6)
    # ...not the transposed forward weights that standard backprop would use.
    assert not torch.allclose(inputs.grad, grad_output.mm(module.weight), atol=1e-4)


def test_training_updates_forward_weights_but_not_the_fixed_feedback_buffer():
    torch.manual_seed(1)
    X = torch.randn(24, 8)
    y = torch.arange(24) % 3
    loader = DataLoader(TensorDataset(X, y), batch_size=6, shuffle=False)
    model = FeedbackAlignmentMLP(num_inputs=8, num_hidden=7, num_outputs=3, bias=False)

    # The fixed feedback weights are buffers, never trainable parameters.
    parameter_ids = {id(p) for p in model.parameters()}
    assert id(model.lin1.weight_fa) not in parameter_ids
    assert id(model.lin2.weight_fa) not in parameter_ids

    weight_before = model.lin1.weight.detach().clone()
    feedback_before = model.lin1.weight_fa.detach().clone()

    optimizer = BasicOptimizer(model.parameters(), lr=0.1)
    train_model(model, loader, loader, optimizer, num_epochs=1, record_initial_baseline=False)

    assert not torch.equal(model.lin1.weight, weight_before)   # forward weights learn
    assert torch.equal(model.lin1.weight_fa, feedback_before)  # feedback weights stay fixed


def test_forgetting_experiment_runs_feedback_alignment_end_to_end():
    assert isinstance(
        build_model("feedback_alignment", num_inputs=8, num_hidden=4, num_outputs=10),
        FeedbackAlignmentMLP,
    )

    loader = DataLoader(
        TensorDataset(torch.randn(20, 8), torch.randint(0, 10, (20,))), batch_size=5
    )
    results = run_forgetting_experiment(
        train_loader_old=loader,
        valid_loader_old=loader,
        train_loader_new=loader,
        valid_loader_new=loader,
        train_loader_full=loader,
        model_type="feedback_alignment",
        optimizer_type="adam",
        num_epochs_phase1=1,
        num_epochs_phase2=1,
        num_inputs=8,
        num_hidden=4,
        num_outputs=10,
    )

    assert isinstance(results["model"], FeedbackAlignmentMLP)
    assert len(results["old_class_acc_trace"]) == 2


def test_external_optimizer_models_support_plain_sgd():
    torch.manual_seed(3)
    loader = DataLoader(
        TensorDataset(torch.randn(20, 8), torch.randint(0, 3, (20,))),
        batch_size=5,
        shuffle=False,
    )

    for model_type in ("backprop", "feedback_alignment"):
        results = run_forgetting_experiment(
            train_loader_old=loader,
            valid_loader_old=loader,
            train_loader_new=loader,
            valid_loader_new=loader,
            train_loader_full=loader,
            model_type=model_type,
            optimizer_type="sgd",
            lr=0.05,
            num_epochs_phase1=2,
            num_epochs_phase2=1,
            num_inputs=8,
            num_hidden=4,
            num_outputs=3,
            device="cpu",
        )

        model = results["model"]
        assert not torch.equal(model.lin1.weight, model.init_lin1_weight)
