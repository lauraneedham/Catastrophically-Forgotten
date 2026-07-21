"""Predictive coding model implementation using JPC (JAX)."""

from __future__ import annotations

import numpy as np
import torch

from src.models.base import MultiLayerPerceptron
import jax
import equinox as eqx
import optax
import jpc


class PredictiveCodingMLP(MultiLayerPerceptron):
    """Predictive Coding MLP using jpc (JAX)."""

    def __init__(
        self,
        num_inputs: int | None = None,
        num_hidden: int = 100,
        num_outputs: int = 10,
        activation_type: str = "sigmoid",
        bias: bool = False,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        super().__init__(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            activation_type=activation_type,
            bias=bias,
        )
        self.lr = lr
        self.seed = seed

        act_str = self.activation_type.lower()
        if act_str not in ["relu", "tanh", "sigmoid", "linear", "leaky_relu", "gelu", "selu", "silu"]:
            act_str = "sigmoid"

        key = jax.random.PRNGKey(self.seed)
        self.jpc_model = jpc.make_mlp(
            key,
            input_dim=self.num_inputs,
            width=self.num_hidden,
            depth=2,
            output_dim=self.num_outputs,
            act_fn=act_str,
            use_bias=self.bias,
        )
        self.jpc_optim = optax.adam(self.lr)
        self.jpc_opt_state = self.jpc_optim.init((eqx.filter(self.jpc_model, eqx.is_array), None))

        self.sync_jpc_to_pytorch()

    def sync_jpc_to_pytorch(self) -> None:
        """Copy weights and biases from JPC model into PyTorch parameters."""
        with torch.no_grad():
            w1 = np.array(self.jpc_model[0][1].weight)
            self.lin1.weight.copy_(torch.from_numpy(w1))
            if self.bias and getattr(self.jpc_model[0][1], "bias", None) is not None:
                b1 = np.array(self.jpc_model[0][1].bias)
                self.lin1.bias.copy_(torch.from_numpy(b1))

            w2 = np.array(self.jpc_model[1][1].weight)
            self.lin2.weight.copy_(torch.from_numpy(w2))
            if self.bias and getattr(self.jpc_model[1][1], "bias", None) is not None:
                b2 = np.array(self.jpc_model[1][1].bias)
                self.lin2.bias.copy_(torch.from_numpy(b2))

    def step_batch(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform a Predictive Coding update step on a single mini-batch."""
        X_np = X.detach().cpu().numpy().reshape(-1, self.num_inputs).astype(np.float32)
        y_np = y.detach().cpu().numpy()

        if y_np.ndim == 1:
            y_onehot = np.zeros((len(y_np), self.num_outputs), dtype=np.float32)
            y_onehot[np.arange(len(y_np)), y_np] = 1.0
        else:
            y_onehot = y_np.astype(np.float32)

        res = jpc.make_pc_step(
            model=self.jpc_model,
            optim=self.jpc_optim,
            opt_state=self.jpc_opt_state,
            input=X_np,
            output=y_onehot,
        )
        self.jpc_model = res["model"]
        self.jpc_opt_state = res["opt_state"]
        self.sync_jpc_to_pytorch()

        return super().forward(X)

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Standard forward pass. If training with y provided, runs step_batch."""
        if self.training and y is not None:
            return self.step_batch(X, y)
        return super().forward(X)
