from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import autograd

from src.models.base import MultiLayerPerceptron


class LinearFAFunction(autograd.Function):

    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_fa.to(grad_output.device))
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class LinearFAModule(nn.Module):

    def __init__(self, input_features: int, output_features: int, bias: bool = True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("weight_fa", torch.empty(output_features, input_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)

    def extra_repr(self) -> str:
        return (
            f"input_features={self.input_features}, "
            f"output_features={self.output_features}, bias={self.bias is not None}"
        )


class FeedbackAlignmentMLP(MultiLayerPerceptron):
    def __init__(
        self,
        num_inputs: int | None = None,
        num_hidden: int = 100,
        num_outputs: int = 10,
        activation_type: str = "sigmoid",
        bias: bool = False,
    ):
        super().__init__(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            activation_type=activation_type,
            bias=bias,
        )

        self.lin1 = LinearFAModule(self.num_inputs, self.num_hidden, bias=bias)
        self.lin2 = LinearFAModule(self.num_hidden, self.num_outputs, bias=bias)

        self._store_initial_weights_biases()
