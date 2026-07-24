# Final standardized 784-1000-10 comparison

This directory stores the final reruns used to compare learning rules under the
same MNIST protocol.

## Shared protocol

- Architecture: `784 -> 1000 -> 10`
- Hidden activation: sigmoid
- Bias: disabled
- Learning rate: `0.001`
- Batch size: `32`
- Old task: digits `0-5`
- New task: digits `6-9`
- Conditions: sequential and interleaved
- Model, split, and loader-order seeds: fixed

`source_zips/` preserves the original Colab downloads. `runs/` contains their
extracted machine-readable files.

## Run status

| Rule | Status | Phase-one old accuracy | Sequential retained old | Sequential new | Interleaved retained old | Interleaved new |
|---|---:|---:|---:|---:|---:|---:|
| Backpropagation | Complete | 99.16% | 0.76% | 99.25% | 97.72% | 97.57% |
| Feedback alignment | Pending | — | — | — | — | — |
| Predictive coding | Pending | — | — | — | — | — |
| Hebbian/Oja | Pending standardized rerun | — | — | — | — | — |

## Backpropagation quality checks

- Correct architecture, learning rate, activation, bias setting, and batch size.
- Full retained MNIST training/validation split was used.
- Sequential and interleaved phase-one endpoints match.
- The predefined phase-one competence gate was passed.
- All update-metric coordinates were finite.
- Every cosine calculation used all eight requested probe batches.
- Backpropagation cosine similarity is approximately `1.0` by construction,
  validating the reference-update calculation.

The backpropagation run shows catastrophic forgetting under sequential
exposure, not an inability of the architecture to represent both tasks:
interleaved exposure maintained high accuracy on both old and new digits.

These are descriptive results from one seed. Cross-rule conclusions must wait
until all four standardized runs have been completed.
