# Hebbian Learning and Catastrophic Forgetting on MNIST

## Start here: the experiment in simple words

### What did we want to find out?

We asked whether a biologically inspired **local Hebbian learning rule** can
protect a neural network from catastrophic forgetting. In simple terms: after
the network learns digits 0-4, can it learn digits 5-9 without erasing what it
already knows?

### What network did we use?

Every Hebbian candidate used the same small, fixed network:

```text
28 x 28 MNIST image
        |
        v
784 input values
        |
        |  local Hebbian update; no label or backpropagated error
        v
100 sigmoid hidden neurons
        |
        |  local supervised output update
        v
10 output neurons -> predicted digit 0-9
```

This is written compactly as **`784 -> 100 -> 10`**. There are no biases,
replay modules, task-specific heads, frozen layers, expanding layers, or
separate networks for the two tasks. The hidden layer never receives a
classification error from the output layer, so it is not trained by
backpropagation.

### Which Hebbian techniques did we try?

| Technique | Simple description | Pilot outcome |
|---|---|---|
| Centered Hebb | Strengthens connections when input and hidden neurons are active together, then recenters the update | Collapsed to one predicted digit |
| Oja/Sanger | Hebbian correlation plus normalization and competition so neurons learn different stable features | Successfully learned all ten digits |
| Hard winner-take-all | Only the five strongest hidden neurons learn for each image | Collapsed to one predicted digit |
| Soft winner-take-all | All hidden neurons compete, but stronger neurons receive larger updates | Collapsed to one predicted digit |

These pilot failures apply only to the tested implementations and
hyperparameters. They do not prove that centered or winner-take-all Hebbian
methods can never learn MNIST. Oja/Sanger was the only candidate competent
enough to continue to the forgetting experiment.

### What exactly was the workflow?

```text
Stage 1: choose a competent Hebbian rule

All digits 0-9 mixed normally (IID)
        -> screen four Hebbian candidates
        -> Oja/Sanger passes
        -> confirm Oja/Sanger over three random seeds
        -> freeze the rule and hyperparameters

Stage 2: test catastrophic forgetting

For each of five new random seeds:
fresh Oja/Sanger model
        -> train from scratch on digits 0-4 until about 90% accuracy
        -> save one checkpoint
        -> make two identical copies of that checkpoint
              |-- Sequential: train only on digits 5-9
              `-- Interleaved control: train on 50% old + 50% new digits
        -> compare both copies when they first reach 80% new-digit accuracy
        -> evaluate the frozen checkpoints once on the official test set
```

The trained IID-selection models were **not** reused for the forgetting test.
Only the chosen Oja/Sanger configuration was reused. Every continual-learning
seed started with a fresh randomly initialized model, learned digits 0-4, and
then continued updating the whole same network during phase two.

### What was the main result?

| Measurement | Sequential | Interleaved control |
|---|---:|---:|
| Old accuracy before phase two | 90.69% | 90.69% |
| New accuracy at the matched checkpoint | 82.16% | 80.30% |
| Old accuracy retained | **0.005%** | **83.61%** |
| Amount forgotten | **90.69 points** | **7.09 points** |

The official test set confirmed the same pattern: sequential learning retained
only `0.004%` old accuracy, whereas interleaving retained `84.80%`.

### What does that mean?

Oja/Sanger successfully learned MNIST, so the result is not caused by a model
that could not learn. However, when old examples disappeared, it learned the new
digits and erased almost all old knowledge. Interleaving is only a control: it
shows that continued exposure to old examples prevents the collapse; it is not
evidence that the local rule solves continual learning by itself.

The main conclusion is therefore:

> In this fixed shallow MNIST network, Oja/Sanger Hebbian learning does not
> prevent catastrophic forgetting on its own. An additional retention
> mechanism such as replay, consolidation, modularity, or systems-level
> architecture is still needed.

### Key terms

- **IID selection:** ordinary shuffled training on all digits, used only to
  verify that a candidate can learn before testing forgetting.
- **Macro accuracy:** calculate accuracy for each digit separately, then give
  every digit equal weight in the average.
- **Seed:** a controlled random starting point. Multiple seeds show whether a
  result is repeatable rather than luck.
- **Sequential condition:** learn 0-4 and then see only 5-9. This is the main
  catastrophic-forgetting test.
- **Interleaved condition:** continue seeing a balanced mixture of old and new
  digits. This is the control condition.
- **Forgetting:** old accuracy before phase two minus retained old accuracy at
  the matched new-task checkpoint.

### Which files should I open?

- `Hebbian_FULL_Experiment_Colab_EXECUTED.ipynb`: the completed Colab run with
  its outputs; start here to inspect the actual experiment.
- `Hebbian_FULL_Experiment_Colab.ipynb`: the clean locked notebook for rerunning
  the full experiment.
- `hebbian_core.py`: readable implementation of the local learning rules and
  protocol embedded in the notebooks.
- `REPORT.md`: this explanation and the complete numerical analysis.
- `results 3/`: final multi-seed CSV, JSON, checkpoints, configuration, and ZIP.
- `results 1/` and `results 2/`: preliminary quick/debugging runs, not final
  scientific results.

### Important comparison limitation

The repository's predictive-coding notebook uses a different
`784 -> 300 -> 300 -> 10` architecture and a different protocol. It also shows
complete sequential forgetting, but its raw accuracy cannot be compared
directly with this Hebbian experiment as if the learning rule were the only
difference.

---

## Detailed scientific report

## Status

The full locked experiment is complete. It ran with `QUICK_RUN=False` on CUDA,
validated Oja/Sanger over three IID seeds, completed five continual-learning
seeds, passed the predefined 80% new-task validation gate in both conditions,
and evaluated the frozen matched checkpoints once on the official MNIST test
set. The full artifact bundle is in `results 3/` beside this report.

### Preliminary quick-run evidence (not reportable)

A one-seed, 1,200-update IID screen produced:

| Variant | Macro validation accuracy | Worst digit | Outcome |
|---|---:|---:|---|
| Centered | 10.00% | 0.00% | Class collapse; eliminated |
| Oja/Sanger | 80.27% | 60.48% | Passed the preliminary gate |
| Hard WTA | 10.00% | 0.00% | Class collapse; eliminated |
| Soft WTA | 10.00% | 0.00% | Class collapse; eliminated |

The Oja/Sanger pilot reached 88.51% old-task validation accuracy. At the first
sequential checkpoint above 80% new accuracy, retained old accuracy was 0%,
showing complete forgetting. The interleaved pilot had reached 75.10% new and
83.64% old accuracy by its 800-update limit; at the 70% new checkpoint it
retained 82.23% old accuracy. These values validate the experimental pipeline
but remain non-reportable because they use one seed and relaxed quick-mode
budgets/gates.

The corrected minibatch diagnostic gave hidden-update elementwise/vector SNRs
of `0.9109`/`1.1326` and output-update SNRs of `4.2654`/`2.9357`. Hidden local
updates had cosine alignment `0.0496` with backpropagation, while the local
softmax output delta aligned essentially perfectly (`1.0`).

## Question

Does a local Hebbian learning rule reduce catastrophic forgetting on its own,
or does reliable retention require additional systems-level mechanisms such as
replay, modularity, parameter isolation, or complementary learning systems?

The primary experiment excludes replay, old data in the sequential condition,
task identity, task-specific heads, parameter freezing, network expansion, and
consolidation penalties.

## Implemented approach

All primary candidates use the same `784 -> 100 -> 10` network and one ten-way
output head. Hidden weights receive no label and no backpropagated error.

| Candidate | Hidden update | Stabilization/competition | Output update |
|---|---|---|---|
| Centered Hebb | Centered pre/post correlation | Centering | Teacher-clamped correlation |
| Oja/Sanger Hebb | Centered correlation with generalized Oja/Sanger lateral term | Feature decorrelation and row normalization | Local supervised delta |
| Hard WTA | Oja update for top-k responses | Hard competition | Local supervised delta |
| Soft WTA | Oja update weighted by soft responsibilities | Temperature-controlled competition | Local supervised delta |

The supervised output delta is local to the output unit: it uses the unit's
prediction/target difference and presynaptic hidden activity. It is not sent
back to the hidden layer.

## Protocol

- MNIST normalization: mean `0.1307`, standard deviation `0.3081`.
- Deterministic stratified split: 50,000 training and 10,000 validation
  examples from the official training set.
- Official 10,000-example test set held out until all choices are frozen.
- Old task: digits `[0,1,2,3,4]`.
- New task: digits `[5,6,7,8,9]`.
- Original digit labels and ten output units are retained throughout.
- Primary capacity: `784 -> 100 -> 10`.

### Stage 1: IID selection

Candidates are first screened using ordinary shuffled ten-class MNIST
validation performance only. Eligibility requires:

1. finite weights;
2. at least 80% mean macro validation accuracy; and
3. no digit below 50% validation accuracy.

The first one-seed screen eliminates collapsed variants. Surviving candidates
are then validated over three seeds. The highest-accuracy eligible rule is
selected. If eligible candidates are within one percentage point, the simpler
rule is preferred. The selection and hyperparameters are frozen before any
forgetting result is examined. Based on the preliminary screen, only
Oja/Sanger advances to three-seed validation.

### Stage 2: continual learning

For each of five seeds:

1. Train on digits 0-4 until 90% old-class macro validation accuracy or the
   maximum update budget.
2. Save the exact phase-one state.
3. Copy it into two conditions:
   - sequential: phase two contains only digits 5-9;
   - interleaved: every phase-two batch contains 50% old and 50% new samples.
4. Record old, new, and joint validation results every 50 updates.
5. Save the first checkpoints crossing 70%, 80%, and 90% new-class macro
   validation accuracy.
6. Evaluate the frozen 80% matched checkpoints once on the official test set.

## Metrics

Primary forgetting metric:

`old accuracy before phase two - old accuracy at 80% new validation accuracy`

A run that never reaches the new-task target is labelled `target not reached`;
it is not counted as successful retention.

Secondary measurements include:

- retained old, new, and joint macro accuracy;
- 70%, 80%, and 90% matched-learning results;
- full old-versus-new trade-off trajectories;
- fixed-update results and old/new exposure counts;
- per-digit accuracy and confusion matrices;
- weight and update norms and hidden-unit utilization;
- elementwise and vector update signal-to-noise ratios;
- cosine alignment between each local proposed update and the negative
  backpropagation gradient.

## Results

### IID selection and competence

The one-seed pilot eliminated centered Hebb, hard WTA, and soft WTA after they
collapsed to one-class prediction. Only Oja/Sanger advanced to the locked
three-seed confirmation. This is a staged IID-only selection result, not a
claim that the eliminated rule families can never learn MNIST.

| Seed | Macro validation | Micro validation | Worst digit | Finite weights |
|---:|---:|---:|---:|---|
| 0 | 84.5715% | 86.04% | 74.1093% | Yes |
| 1 | 83.7051% | 85.56% | 69.3587% | Yes |
| 2 | 83.6509% | 85.24% | 75.7720% | Yes |
| **Mean ± SD** | **83.9758 ± 0.5166%** | — | **69.3587% overall minimum** | **Yes** |

Oja/Sanger passed both predefined competence gates: mean macro accuracy was
above 80%, and every digit was above 50%. The frozen selected configuration was
`784 -> 100 -> 10`, sigmoid hidden activity, no bias, hidden learning rate
`0.001`, output learning rate `0.1`, hidden row normalization, generalized
Oja/Sanger decorrelation, and a local supervised output delta.

### Forgetting at 80% new validation accuracy

| Condition | Target reach rate | Old before | Retained old | Forgetting | New accuracy | Joint accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 5/5 | 90.6916% | 0.0046% | 90.6870 ± 0.3844 points | 82.1610% | 41.0828% |
| Interleaved | 5/5 | 90.6916% | 83.6057% | 7.0858 ± 0.6664 points | 80.2998% | 81.9528% |

The primary comparison is validation based because validation accuracy defines
the matched checkpoint. Results for every seed are:

| Seed | Condition | Matched step | Old before | Retained old | New | Forgetting | Joint |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Sequential | 100 | 90.6223% | 0.0000% | 80.7278% | 90.6223 | 40.3639% |
| 1 | Sequential | 100 | 90.6626% | 0.0000% | 82.3130% | 90.6626 | 41.1565% |
| 2 | Sequential | 100 | 90.1783% | 0.0000% | 84.5900% | 90.1783 | 42.2950% |
| 3 | Sequential | 100 | 90.7358% | 0.0230% | 81.2796% | 90.7128 | 40.6513% |
| 4 | Sequential | 150 | 91.2588% | 0.0000% | 81.8947% | 91.2588 | 40.9473% |
| 0 | Interleaved | 1,400 | 90.6223% | 84.2657% | 80.4806% | 6.3566 | 82.3732% |
| 1 | Interleaved | 2,200 | 90.6626% | 82.4864% | 80.2687% | 8.1763 | 81.3776% |
| 2 | Interleaved | 1,300 | 90.1783% | 83.2680% | 80.0566% | 6.9103 | 81.6623% |
| 3 | Interleaved | 1,350 | 90.7358% | 83.6817% | 80.0672% | 7.0540 | 81.8744% |
| 4 | Interleaved | 1,450 | 91.2588% | 84.3268% | 80.6260% | 6.9320 | 82.4764% |

Sequential learning reached 80% new accuracy after only 100-150 updates and
6,400-9,600 new examples. Interleaved learning required 1,300-2,200 updates,
with equal old/new exposure, to reach the same target. This exposure difference
is expected under matched-competence evaluation and is retained as a secondary
compute/sample-efficiency result.

### Secondary validation targets

| Target | Condition | Reached | Retained old | New accuracy | Forgetting | Joint accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 70% | Sequential | 5/5 | 5.1045% | 74.7544% | 85.5871 | 39.9294% |
| 70% | Interleaved | 5/5 | 82.4971% | 71.9436% | 8.1945 | 77.2203% |
| 80% | Sequential | 5/5 | 0.0046% | 82.1610% | 90.6870 | 41.0828% |
| 80% | Interleaved | 5/5 | 83.6057% | 80.2998% | 7.0858 | 81.9528% |
| 90% | Sequential | 5/5 | 0.0000% | 90.2390% | 90.6916 | 45.1195% |
| 90% | Interleaved | 0/5 | — | — | — | — |

The interleaved condition did not reach the secondary 90% target within 2,500
updates. This does not invalidate the experiment because 80% was the frozen
primary target and was reached by every seed in both conditions.

### One-time official test evaluation

The official test set was evaluated only at the frozen validation-selected 80%
checkpoints.

| Condition | Old macro accuracy | New macro accuracy | Joint macro accuracy |
|---|---:|---:|---:|
| Sequential | 0.0035 ± 0.0079% | 82.8646 ± 0.4003% | 41.4341 ± 0.1983% |
| Interleaved | 84.7973 ± 0.7467% | 80.1091 ± 1.2457% | 82.4532 ± 0.6223% |

In the sequential condition, all old digits had 0% test accuracy in four seeds;
seed 3 retained only 0.09% on digit 1. In the interleaved condition, no digit
collapsed: per-digit test accuracy ranged from 64.52% to 96.92% across seeds.

### Update diagnostics

- Hidden Oja/Sanger update alignment with backpropagation descent: `-0.0489`.
- Local output-delta alignment with backpropagation descent: approximately
  `1.0000`, as expected mathematically.
- Hidden elementwise/vector update SNR: `0.6758` / `1.0248`.
- Output elementwise/vector update SNR: `2.3716` / `2.8160`.

The near-zero hidden alignment confirms that the representation was not trained
by backpropagating the classification error. Perfect output alignment reflects
the local softmax delta used only at the readout.

## Relation to current team results

The current predictive-coding notebook reports:

- 98.9648% on digits 0-4 after Task 1;
- 97.6458% on digits 5-9 after Task 2; and
- 0.0000% retained accuracy on digits 0-4 after Task 2.

That is complete forgetting in that predictive-coding implementation, but it is
not a controlled direct comparison. It uses a larger `784 -> 300 -> 300 -> 10`
network, different software and optimization, one seed, repeated test-set
evaluation, and no interleaved control. It belongs in a legacy/exploratory table
unless rerun under the shared protocol.

The repository's existing backpropagation experiment also uses an unbalanced
`0-5`/`6-9` split, half of MNIST, fixed epochs, and unmatched interleaving. Its
results should not be placed in the primary causal comparison without a
protocol-matched rerun.

The original Neuromatch Hebbian POC also does not provide a competing
full-MNIST pure-Hebbian success. Its target-clamped basic Hebbian model succeeds
on the two-class 0-versus-1 task, but its three-class runs collapse. Its
ten-class result is a hybrid with Hebbian hidden learning and backpropagation at
the output and reaches only roughly 40-42% validation accuracy. The POC's added
forgetting experiment runs backpropagation only.

## Interpretation and conclusion

The experiment produced a competent local-learning network, so its retention
result cannot be dismissed as failure to learn MNIST. Nevertheless, once old
examples disappeared, every seed lost essentially all old-class knowledge by
the time it reached the new-task target. The same starting checkpoints retained
about 84% old accuracy when old and new examples remained interleaved.

Therefore, within this fixed-capacity MNIST model, Oja/Sanger local learning
does **not** provide resistance to catastrophic forgetting by itself. The
result supports the need for an additional retention mechanism such as replay,
continued exposure, modularity, parameter isolation, synaptic consolidation,
or a complementary-learning-systems-style architecture.

This is not proof that complementary learning systems are the only biological
explanation, nor that every Hebbian model must forget. It is a controlled result
for one selected Hebbian rule, one shallow architecture, one task split, and
one dataset. The pilot failures of centered and WTA variants reflect their
tested implementations and hyperparameters, not a general impossibility
theorem. Direct numerical comparison with predictive coding, feedback
alignment, or backpropagation requires those models to be rerun under the same
architecture, data split, seeds, stopping gates, and matched-accuracy protocol.

## Reproduction

1. Upload `Hebbian_FULL_Experiment_Colab.ipynb` to Google Colab.
2. Select a GPU runtime and run every cell in order without changing the frozen
   configuration. The file has `QUICK_RUN=False` and `RUN_FINAL_TEST=True`.
3. The notebook validates Oja/Sanger over three IID seeds, runs five continual
   seeds, and blocks official-test access if a required gate fails.
4. Download the generated `hebbian_results_final.zip`. The reported run used
   PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, and CUDA.

## Files

- `hebbian/Hebbian_Catastrophic_Forgetting_Colab.ipynb`: standalone runnable
  quick/debugging experiment.
- `hebbian/Hebbian_FULL_Experiment_Colab.ipynb`: locked full experiment to run
  for the final report.
- `hebbian/Hebbian_FULL_Experiment_Colab_EXECUTED.ipynb`: completed Colab run
  with the full experiment outputs.
- `hebbian/hebbian_core.py`: readable copy of the implementation embedded in
  the notebook.
- `hebbian/build_colab_notebook.ps1`: deterministic notebook builder.
- `hebbian/results 3/`: complete full-run artifact bundle.
- `hebbian/results 1/` and `hebbian/results 2/`: non-reportable quick/debugging
  artifacts retained for provenance.

## References

- Neuromatch NeuroAI Course: https://github.com/neuromatch/NeuroAI_Course
- SoftHebb: https://arxiv.org/abs/2107.05747
- Official SoftHebb implementation: https://github.com/NeuromorphicComputing/SoftHebb
- Krotov and Hopfield on competing hidden units: https://pubmed.ncbi.nlm.nih.gov/30926658/
- JPC predictive-coding library: https://github.com/thebuckleylab/jpc
