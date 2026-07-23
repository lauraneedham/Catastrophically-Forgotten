# Hebbian learning and catastrophic forgetting

## Short summary

We implemented one local Hebbian/Oja learning rule and tested it on three
network architectures. All three runs use the repository's shared MNIST
loading and forgetting protocol: digits 0-5 are learned first, followed by
digits 6-9. Sequential training tests catastrophic forgetting; interleaved
training is the control.

The hidden layers learn without backpropagation or an optimizer. The output
layer uses a local supervised delta update. The locked settings are full MNIST,
batch size 32, learning rate 0.001, sigmoid activation, no biases, seed 0, and 20
recorded epochs per phase.

All three final runs are complete. Every architecture passed the phase-one
competence gate and learned the new task, but every one retained exactly 0% old
accuracy after sequential learning. Increasing width to 1000 or adding a second
300-unit hidden layer therefore did not make this isolated Hebbian/Oja network
resistant to catastrophic forgetting. Interleaving retained 79-84% old
accuracy, confirming that continued old-data exposure strongly changes the
outcome. The earlier standalone 300-300 run remains exploratory only because it
used a different class split and training protocol.

## 1. Research question

The project asks whether resistance to catastrophic forgetting comes from the
local learning rule itself or from larger systems-level mechanisms such as a
hippocampus-neocortex division.

The Hebbian part asks:

> When architecture and the forgetting task are controlled, does a local
> Hebbian/Oja learning rule retain previously learned classes after training on
> new classes?

## 2. Repository integration

The fresh `main` branch provides:

- `src/data.py` for downloading, normalizing, splitting, and filtering MNIST;
- `src/experiments/forgetting.py` for the shared old/new class task;
- `src/models/base.py` and `backpropagation.py` for the baseline interface;
- placeholder files for Hebbian, feedback alignment, and predictive coding.

The Hebbian branch replaces only the placeholder `src/models/hebbian.py` with a
real local-learning model. The shared `src/experiments/forgetting.py` is kept
byte-for-byte identical to fresh `main`, because other team members may already
depend on it.

The notebooks import its class constants, loader construction, and plotting
helper. A small Hebbian-specific training adapter is defined inside each
notebook because the unchanged shared runner currently supports backpropagation
only. The notebooks therefore share the data/task definition without changing
the team-owned experiment file.

## 3. Shared experimental protocol

### 3.1 Data

The shared `download_mnist(train_prop=0.8, keep_prop=1.0)` call uses all 70,000
MNIST images:

- approximately 48,000 images for training;
- approximately 12,000 images for validation;
- 10,000 official test images retained but not used during development.

A fixed data-split seed is set before the shared helper is called.
With the current PyTorch fractional split, the observed deterministic sizes are
48,001 training and 11,999 validation images because the one-image rounding
remainder is assigned to the first split.

### 3.2 Continual-learning task

- Old task: digits `0-5` (six classes).
- New task: digits `6-9` (four classes).
- Phase 1: train on old classes.
- Sequential phase 2: train only on new classes.
- Interleaved phase 2: train with the shared full `0-9` loader.

The shared interleaved loader is not forced to be a 50/50 old/new mixture. Since
there are six old and four new classes, it is approximately 60% old and 40% new.
This is intentionally preserved because every learning rule is expected to use
the same repository definition.

With a fixed epoch count, the two phase-2 conditions also receive different
numbers of updates: the sequential loader contains only the four new classes,
whereas the interleaved loader contains all ten classes. This is part of the
committed shared protocol, but it must be acknowledged when interpreting the
control.

### 3.3 Training budget

- Batch size: 32.
- Phase-1 recorded epochs: 20.
- Phase-2 recorded epochs: 20.
- Seed: 0.

The current base training function treats the first epoch of every call as an
untrained baseline. Consequently, 20 recorded epochs correspond to one
baseline epoch and 19 weight-update epochs. The Hebbian trainer reproduces this
semantics exactly. The team should explicitly change this setting for every
learning rule if the intended agreement was 20 actual update epochs.

## 4. Architectures

| Architecture | Hidden layers | Weight count, no biases |
|---|---:|---:|
| `784 -> 100 -> 10` | 1 | 79,400 |
| `784 -> 1000 -> 10` | 1 | 794,000 |
| `784 -> 300 -> 300 -> 10` | 2 | 328,200 |

Every architecture uses sigmoid hidden activations, ten softmax outputs, and no
biases.

Testing every learning rule on every architecture creates a crossed design. It
allows the team to separate:

- a learning-rule effect;
- an architecture effect;
- an interaction where a particular rule behaves differently in a particular
  architecture.

## 5. Hebbian learning rule

### 5.1 Hidden layers

Each hidden layer receives an Oja subspace update computed from only:

- the inputs arriving at that layer;
- the activities produced by that layer;
- the current weights at that layer.

The implementation uses the efficient symmetric form
`Y^T (X - YW)`, which avoids constructing a hidden-width by hidden-width
matrix. This matters for the 1000-unit architecture.

Labels and output errors never reach a hidden layer. For the two-hidden-layer
architecture, the same local rule is applied independently at both hidden
layers.

### 5.2 Output layer

The output layer uses the local supervised signal:

`one-hot target - predicted probability`.

This produces a local delta update between the final hidden activity and the
ten outputs. It does not propagate the output error backward.

### 5.3 Optimizer

No SGD, Adam, PyTorch optimizer, `loss.backward()`, or gradient update is used.
The weights are changed directly by the local rules.

Although SGD was discussed as a team-wide default, applying SGD to the hidden
layers would replace the Hebbian learning rule with backpropagation. The common
number `0.001` is therefore used as the scale of both direct local updates, not
as an SGD learning rate.

## 6. Hyperparameter policy

The locked notebooks do not conduct an automated search. They use:

- hidden local learning rate: `0.001`;
- output local learning rate: `0.001`;
- sigmoid activation;
- no bias;
- row normalization of hidden weights.

The final learning rate was changed from the earlier `0.01` run to `0.001`
after the team aligned the other learning-rule implementations on `0.001`.
This protocol decision was the reason for the rerun; the setting was not chosen
by inspecting which run forgot less. The complete `0.01` results are retained
as a sensitivity comparison rather than discarded.

A predefined competence gate requires at least 80% old-class validation
accuracy at the end of phase 1. If the gate fails, the notebook stops before
phase 2.

Any necessary tuning must use phase-1 or separate IID validation performance
only. Sequential forgetting, interleaved retention, and the official test set
must not be used to choose settings.

## 7. Measurements

For each architecture and condition, the notebook reports:

- old-class validation accuracy immediately before phase 2;
- retained old-class accuracy after phase 2;
- forgetting = old-before minus old-after;
- new-class validation accuracy after phase 2;
- the complete old-class accuracy trace across both phases.

The sequential condition is the primary result. Interleaving is a control
showing whether continued exposure to old examples prevents forgetting.

## 8. Final results

The completed table below was extracted from the ZIPs produced by the locked
notebooks:

| Architecture | Condition | Old before | Old after | Forgetting | New after |
|---|---|---:|---:|---:|---:|
| `784-100-10` | Sequential | 89.09% | 0.00% | 89.09 points | 92.45% |
| `784-100-10` | Interleaved | 89.09% | 79.57% | 9.52 points | 83.41% |
| `784-1000-10` | Sequential | 90.15% | 0.00% | 90.15 points | 91.78% |
| `784-1000-10` | Interleaved | 90.15% | 84.44% | 5.71 points | 84.48% |
| `784-300-300-10` | Sequential | 83.18% | 0.00% | 83.18 points | 91.09% |
| `784-300-300-10` | Interleaved | 83.18% | 78.66% | 4.52 points | 74.90% |

Do not select only the architecture that forgets least. Report all six rows.

### 8.1 Interpretation of `784-100-10`

The model passed the predefined 80% phase-one gate and therefore demonstrated
that it could learn the old task before forgetting was tested. Under sequential
phase-two training, old-class validation accuracy fell from 89.09% to 56.49%
after the first update epoch, 23.13% after the second, and reached 0.00% after
17 update epochs. At the end it classified the new digits at 92.45%, showing
that the zero old accuracy reflected replacement by new learning rather than a
general failure to train.

Interleaving prevented complete collapse: final old accuracy was 79.57% and new
accuracy was 83.41%. This is a 9.52-point loss of old-task performance, so the
control reduced but did not eliminate forgetting. It also lowered new-task
accuracy by 9.04 points relative to sequential training, illustrating the
stability-plasticity trade-off.

### 8.2 Interpretation of `784-1000-10`

The wide model reached 90.15% old-task accuracy. Sequential phase two reduced
old accuracy to 5.93% after the first update epoch, 1.74% after the second, and
0.00% after the fifth. Final new-task accuracy was 91.78%.

With interleaving it retained 84.44% old accuracy and reached 84.48% new
accuracy. Its 5.71-point interleaved forgetting was small, and it achieved the
best balanced old/new interleaved performance of the three architectures.

### 8.3 Interpretation of `784-300-300-10`

The two-hidden-layer model passed the gate at 83.18%, the lowest phase-one
accuracy of the three. Under sequential phase two, old-task accuracy reached
0.00% after the second update epoch while new-task accuracy eventually reached
91.09%.

Interleaving retained 78.66% old accuracy, produced 4.52 points of forgetting,
and reached 74.90% new accuracy. Its interleaved forgetting score was the
smallest, but its new-task control accuracy was also the lowest, so this should
not be interpreted as an unqualified win.

### 8.4 Cross-architecture conclusion

Architecture affected basic accuracy, the speed of collapse, and interleaved
performance, but it did not change the main sequential result:

- all three architectures learned phase one above the predefined 80% gate;
- all three learned the new task above 91%;
- all three retained exactly 0% old-task accuracy after sequential phase two;
- interleaving prevented total collapse in all three.

The 1000-unit model achieved the highest phase-one accuracy and the strongest
balanced interleaved result. The 100-unit model achieved the highest sequential
new-task accuracy and forgot most slowly during sequential phase two, although
it still ended at zero old accuracy. The two-layer model had the smallest
interleaved forgetting score but the weakest interleaved new-task accuracy.
Architecture therefore affected the stability-plasticity trade-off without
changing the final sequential conclusion.

The robust qualitative conclusion is that this local Hebbian/Oja rule did not,
by itself, protect an isolated network from catastrophic forgetting across any
of the tested architectures. This supports the proposal's systems-level
explanation more than a learning-rule-only explanation, subject to comparison
with the other learning rules under the same protocol.

All final runs used branch `hebbian-learning-v2`, commit
`e7c2adc382b9dad1fcaebd55504c5416147fb3d1`, CUDA, PyTorch
`2.11.0+cu128`, learning rate 0.001, no optimizer, and one baseline plus 19
update epochs per phase.

These are one-seed validation results. The shared interleaved condition is
approximately 60/40 old/new and receives a larger phase-two update budget than
the sequential condition. The fixed learning rate also fit the three
architectures unequally, so small numerical differences between architectures
should not be overinterpreted.

### 8.5 Sensitivity to the earlier `0.01` run

Lowering the common local-update scale to `0.001` improved most interleaved
metrics and substantially improved the deep model's sequential new-task
accuracy. It also slowed sequential collapse: the 100-unit model reached zero
old accuracy after 17 updates instead of four, the 1000-unit model after five
instead of three, and the two-layer model after two instead of one.

The primary endpoint was nevertheless identical at both learning rates: all
three architectures finished sequential phase two with exactly 0% old-task
accuracy. The conclusion of complete final forgetting is therefore robust to
this tenfold learning-rate change. The executed `0.001` artifacts are stored
under `FINAL (learning rate 0.001)/`; the earlier `0.01` artifacts remain in
their original architecture folders. Exact side-by-side values are provided in
`LEARNING_RATE_SENSITIVITY.csv`.

## 9. Exploratory interrupted run

An earlier standalone `784-300-300-10` notebook completed tuning and three of
five continual seeds before the Colab runtime disconnected.

- IID macro validation accuracy: approximately 84.4%.
- Old-task accuracy before phase 2: approximately 90%.
- Sequential retained-old accuracy: approximately 0% for the three completed
  seeds.
- Interleaved forgetting: approximately 5-7%.

These results suggest severe sequential forgetting, but they are not part of
the new final comparison because that run used:

- digits `0-4` followed by `5-9`;
- batch size 64;
- a step-based stopping and matched-performance protocol;
- learning rates 0.001 and 0.1;
- only three completed continual seeds.

## 10. Limitations to report

- A single seed is the default because three large architectures with two
  conditions and 20 epochs are expensive. A later multi-seed run would be
  stronger.
- The shared interleaved control is approximately 60/40 old/new, not balanced.
- Equal phase-2 epoch counts do not imply equal update or example budgets:
  interleaved phase 2 uses the larger full dataset.
- The shared epoch counter includes one non-training baseline epoch per phase.
- Different learning rules may require different numerical learning rates even
  when the same forward architecture is used.
- Conclusions must be based on comparable competence; a model that never learns
  phase 1 cannot provide an interpretable forgetting result.
