# Hebbian learning experiments

## In simple words

These notebooks test whether a locally trained Hebbian network forgets digits
it learned earlier.

1. The model first learns MNIST digits **0-5**.
2. It then learns digits **6-9**.
3. We measure how much accuracy on digits 0-5 is lost.
4. We repeat phase 2 with all digits interleaved as a control.

The same experiment is run with three architectures:

- `784 -> 100 -> 10`
- `784 -> 1000 -> 10`
- `784 -> 300 -> 300 -> 10`

The notebooks import the shared repository data and experiment functions. They
do not carry a separate standalone data split or forgetting implementation.

## Final result

All three architectures learned digits 0-5 well enough to pass the predefined
80% competence gate. They also learned digits 6-9. However, after sequential
training on 6-9, every architecture's accuracy on 0-5 fell to **0.00%**.

| Architecture | Old before phase 2 | Old after sequential | New after sequential | Old after interleaved | New after interleaved |
|---|---:|---:|---:|---:|---:|
| `784 -> 100 -> 10` | 90.08% | 0.00% | 92.54% | 78.81% | 78.21% |
| `784 -> 1000 -> 10` | 86.92% | 0.00% | 89.66% | 77.75% | 79.91% |
| `784 -> 300 -> 300 -> 10` | 81.73% | 0.00% | 82.91% | 69.46% | 72.24% |

In simple words: changing the network's width or depth changed its accuracy,
but none of the three architectures resisted catastrophic forgetting when old
digits disappeared from training. Interleaving old and new digits prevented
complete collapse because the model continued seeing the old task.

## Learning method

- Hidden layers: local Oja subspace Hebbian updates.
- Output layer: local supervised delta update.
- Optimizer: none.
- Backpropagation into hidden layers: none.
- Activation: sigmoid.
- Bias: false.
- Learning rate: `0.01` for both local update types.

Using SGD on the hidden layers would change the method into a gradient-trained
model, so the team's SGD consensus is not applied to this learning rule.

## Shared protocol

- Full MNIST: approximately 48,000 training, 12,000 validation, and 10,000
  untouched test images. PyTorch may allocate the fractional-split remainder as
  48,001 training and 11,999 validation images.
- Old classes: `0, 1, 2, 3, 4, 5`.
- New classes: `6, 7, 8, 9`.
- Batch size: 32.
- Phase 1: 20 recorded epochs on old classes.
- Phase 2: 20 recorded epochs.
- Sequential phase 2: new classes only.
- Interleaved phase 2: the repository's full 0-9 loader.
- Seed: 0.

The current shared training interface records the first epoch of each phase as
an untrained baseline. Therefore 20 recorded epochs contain 19 update epochs.
This behavior is preserved rather than silently changing the shared protocol.

## Colab notebooks

- `Hebbian_784_100_10_Colab.ipynb`
- `Hebbian_784_1000_10_Colab.ipynb`
- `Hebbian_784_300_300_10_Colab.ipynb`

Each notebook:

1. clones branch `hebbian-learning-v2`;
2. imports `src/data.py`, the unchanged `src/experiments/forgetting.py`, and
   `src/models/hebbian.py`;
3. checks that hidden updates are label-independent;
4. loads full MNIST using the shared helper;
5. runs sequential and interleaved conditions;
6. saves the configuration, traces, model states, summary, and plot in a ZIP.

The shared forgetting file is not modified. Each notebook defines a small
Hebbian-only adapter on top of its shared loaders because its existing runner
currently accepts backpropagation only.

Use a fresh Colab GPU runtime and select **Runtime -> Run all**. Run one
architecture per runtime, then download the ZIP from the final cell.

## Hyperparameters

There is no automated hyperparameter search in the locked notebooks. The
team-agreed learning rate `0.01` is used directly. A phase-1 competence gate
stops the experiment before phase 2 if old-class validation accuracy is below
80%.

If tuning becomes necessary, select settings using phase-1 or separate IID
validation only. Forgetting and final test performance must not be used to
choose hyperparameters.

See `REPORT.md` for the full methods, results, limitations, and interpretation.
The exact combined final metrics are also available in
`ARCHITECTURE_COMPARISON.csv`.
