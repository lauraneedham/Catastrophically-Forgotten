# Catastrophically Forgotten

This repository contains a modular PyTorch project for studying catastrophic forgetting in a NeuroAI setting using MNIST.

## Project goals

- Compare backpropagation against alternative learning rules in a forgetting experiment.

## Hebbian architecture experiments

The `hebbian/` folder contains the repository-integrated local Hebbian/Oja
implementation and three Colab notebooks:

- `784 -> 100 -> 10`
- `784 -> 1000 -> 10`
- `784 -> 300 -> 300 -> 10`

These notebooks import shared `src/data.py` and the unchanged
`src/experiments/forgetting.py`. They use full MNIST, batch size 32, digits
`0-5` followed by `6-9`, learning rate 0.001, and
sequential/interleaved phase-2 conditions. The Hebbian-specific training
adapter remains inside each notebook.

All three final runs learned the first and second tasks, but all retained
**0.00%** accuracy on digits 0-5 after sequential training on digits 6-9.
Interleaving old and new digits retained 69-79% old-task accuracy.

| Architecture | Old before phase 2 | Old after sequential | New after sequential | Old after interleaved |
|---|---:|---:|---:|---:|
| `784 -> 100 -> 10` | 89.09% | 0.00% | 92.45% | 79.57% |
| `784 -> 1000 -> 10` | 90.15% | 0.00% | 91.78% | 84.44% |
| `784 -> 300 -> 300 -> 10` | 83.18% | 0.00% | 91.09% | 78.66% |

The main result is therefore consistent across all three architectures: this
local Hebbian/Oja rule did not prevent catastrophic forgetting by itself.
See `hebbian/README.md` for the simple workflow, `hebbian/REPORT.md` for the
full analysis, and `hebbian/ARCHITECTURE_COMPARISON.csv` for the exact combined
metrics.

## Repository structure

- `src/data.py` – dataset download and class-restriction helpers.
- `src/models/` – model implementations and training utilities.
- `src/experiments/` – forgetting experiment harnesses.
- `src/analysis/` – plotting and metric helpers.
- `tests/` – smoke tests for the package structure.

## Getting started

1. Clone the repository, or pull the latest `main` branch if you already have it.
2. Create and activate a Python environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   On Windows, activate with:
   ```bash
   .venv\Scripts\activate
   ```
3. Install dependencies from the repository root:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the tests to confirm your setup:
   ```bash
   python -m pytest -q
   ```

## Running the baseline forgetting experiment

The backpropagation baseline is wired through `src/experiments/forgetting.py`.
Use the helpers there to build old-class, new-class, and interleaved dataloaders,
then call `run_forgetting_experiment(...)`.

This project is intentionally structured to mirror the reference notebook and the
proof-of-concept setup:

- a single hidden-layer MLP with sigmoid activations,
- a simple SGD-style optimizer (`BasicOptimizer`),
- old-class / new-class / interleaved training phases for catastrophic forgetting.

By default, the forgetting experiment starts training on epoch 0 so phase 1
reflects real learning and phase 2 shows forgetting on the old-class validation
set.

## Quick start for collaborators

If you just want the minimal backpropagation baseline, the main entry points are:

- `backpropagation.py` – the minimal implementation of `MultiLayerPerceptron`,
  `BasicOptimizer`, and `train_model`
- `src/experiments/forgetting.py` – the forgetting experiment wrapper
- `src/data.py` – dataset and class-filtering helpers

A tiny smoke test looks like this:

```python
import torch
from backpropagation import MultiLayerPerceptron, BasicOptimizer, train_model

model = MultiLayerPerceptron(num_inputs=2, num_hidden=4, num_outputs=2, bias=False)
X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
y = torch.tensor([0, 1], dtype=torch.long)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)
valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)
optimizer = BasicOptimizer(model.parameters(), lr=0.01)
results = train_model(model, train_loader, valid_loader, optimizer, num_epochs=2, verbose=False)
```

The training loop should run without errors and produce accuracy metrics in the
returned results dictionary.

## Adding new learning-rule models

Add model implementations under `src/models/`, using the existing
`MultiLayerPerceptron` interface as the compatibility target. New models should
accept batches of MNIST images and labels, expose trainable parameters in the
usual PyTorch way, and return class probabilities over MNIST digits.

When adding a new learning rule:

1. Add or update the model file in `src/models/`.
2. Export or import the model consistently with the existing `src/models/` files.
3. Add a `model_type` branch in `src/experiments/forgetting.py` so the same
   forgetting setup can run that model.
4. Add a small test or smoke check under `tests/`.
5. Run the tests:
   ```bash
   python -m pytest -q
   ```

## Suggested workflow for collaborators

- Keep experiments in `src/experiments/`.
- Keep model code in `src/models/`.
- Keep visualization helpers in `src/analysis/`.
- Add tests for new functionality under `tests/`.

## Notes

The project is intentionally modular so new learning rules can be added without rewriting the experiment harness.
