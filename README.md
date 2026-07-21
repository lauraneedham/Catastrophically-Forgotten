# Catastrophically Forgotten

This repository contains a modular PyTorch project for studying catastrophic forgetting in a NeuroAI setting using MNIST.

## Project goals

- Compare backpropagation against alternative learning rules in a forgetting experiment.

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

By default, training matches the original notebook behavior: epoch 0 records an
initial no-training baseline. If you want every epoch to perform weight updates
immediately, pass `record_initial_baseline=False`.

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
