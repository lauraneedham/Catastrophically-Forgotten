param(
    [string]$OutputPath = (Join-Path $PSScriptRoot 'Hebbian_Catastrophic_Forgetting_Colab.ipynb'),
    [switch]$FullRun,
    [switch]$EnableFinalTest
)

$ErrorActionPreference = 'Stop'

function Convert-ToSourceLines([string]$Text) {
    $normalised = $Text.Replace("`r`n", "`n").Replace("`r", "`n")
    $matches = [regex]::Matches($normalised, '.*?(?:\n|$)')
    $lines = @()
    foreach ($match in $matches) {
        if ($match.Value.Length -gt 0) {
            $lines += $match.Value
        }
    }
    return $lines
}

function New-MarkdownCell([string]$Text) {
    return [ordered]@{
        cell_type = 'markdown'
        metadata = [ordered]@{}
        source = @(Convert-ToSourceLines $Text)
    }
}

function New-CodeCell([string]$Text) {
    return [ordered]@{
        cell_type = 'code'
        execution_count = $null
        metadata = [ordered]@{}
        outputs = @()
        source = @(Convert-ToSourceLines $Text)
    }
}

$corePath = Join-Path $PSScriptRoot 'hebbian_core.py'
$coreSource = Get-Content -Raw -LiteralPath $corePath

$cells = @()
$cells += New-MarkdownCell @'
# Hebbian Learning and Catastrophic Forgetting on MNIST

This is the self-contained Colab notebook for the Hebbian part of the team project. It does not import or alter the repository's existing model code.

The workflow has two locked stages:

1. Select a competent Hebbian rule using ordinary shuffled MNIST validation performance only.
2. Freeze that choice, then measure forgetting on digits **0-4 followed by 5-9** under sequential and interleaved conditions.

The main comparison is made when models reach the same new-task validation accuracy. A model that fails to learn the new task is not considered resistant to forgetting.
'@

$cells += New-MarkdownCell @'
## 1. Runtime check

Use a Colab GPU runtime if available. `QUICK_RUN=True` is only a smoke test; its numbers must not be used in the report.
'@

$cells += New-CodeCell @'
!pip -q install tabulate

import json
import platform
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms

print('Python:', platform.python_version())
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
'@

$cells += New-MarkdownCell @'
## 2. Embedded Hebbian implementation

The following cell contains the complete local-learning model, deterministic data protocol, balanced interleaving, evaluation, matched-checkpoint, and diagnostic code. Hidden-layer updates never receive labels or backpropagated errors.
'@

$cells += New-CodeCell $coreSource

$cells += New-CodeCell @'
run_preflight_checks()
'@

$cells += New-MarkdownCell @'
## 3. Experiment configuration

The four primary candidates have the same `784 -> 100 -> 10` capacity. They differ in local learning dynamics, not network width. The centered tutorial-like rule is expected to be a useful reference or negative control; Oja and winner-take-all variants add stabilization or specialization.
'@

$cells += New-CodeCell @'
QUICK_RUN = True
RUN_FINAL_TEST = False
ALLOW_DIAGNOSTIC_CONTINUAL_RUN_IF_GATE_FAILS = QUICK_RUN

BASE_EXPERIMENT = ExperimentConfig(
    seed=0,
    batch_size=64,
    train_per_class=5000,
    iid_max_steps=1200 if QUICK_RUN else 4000,
    phase1_max_steps=800 if QUICK_RUN else 2000,
    phase2_max_steps=800 if QUICK_RUN else 2500,
    eval_every_steps=100 if QUICK_RUN else 50,
    old_target_accuracy=10.0 if QUICK_RUN else 90.0,
    matched_new_accuracies=(70.0, 80.0, 90.0),
)

SCREEN_SEEDS = [0] if QUICK_RUN else [0, 1, 2]
FINAL_SEEDS = [0] if QUICK_RUN else [0, 1, 2, 3, 4]

ALL_CANDIDATES = [
    HebbianConfig(
        variant='centered', output_rule='teacher_hebb',
        hidden_lr=1e-4, output_lr=2e-2, normalize_hidden=False,
    ),
    HebbianConfig(
        variant='oja', output_rule='delta',
        hidden_lr=1e-3, output_lr=1e-1, normalize_hidden=True,
    ),
    HebbianConfig(
        variant='hard_wta', output_rule='delta', top_k=5,
        hidden_lr=1e-4, output_lr=5e-2, normalize_hidden=True,
    ),
    HebbianConfig(
        variant='soft_wta', output_rule='delta', temperature=0.2,
        hidden_lr=1e-4, output_lr=5e-2, normalize_hidden=True,
    ),
]

# The first quick IID screen eliminated centered, hard-WTA, and soft-WTA due to
# complete class collapse. Full mode validates only the surviving Oja/Sanger
# candidate over three seeds, following the predefined staged sweep.
SHORTLISTED_VARIANTS = {'oja'}
CANDIDATES = (
    ALL_CANDIDATES
    if QUICK_RUN
    else [config for config in ALL_CANDIDATES if config.variant in SHORTLISTED_VARIANTS]
)

print('Mode:', 'QUICK CHECK - NOT REPORTABLE' if QUICK_RUN else 'FULL LOCKED RUN')
display(pd.DataFrame([asdict(config) for config in CANDIDATES]))
'@

$cells += New-MarkdownCell @'
## 4. Download MNIST and create deterministic splits

For each shared seed, the official 60,000 training examples are split into exactly 50,000 training and 10,000 validation examples using 5,000 training examples per digit. The official 10,000 test examples are not evaluated during model selection.
'@

$cells += New-CodeCell @'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train_dataset = datasets.MNIST('/content/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('/content/data', train=False, download=True, transform=transform)

check_experiment = replace(BASE_EXPERIMENT, seed=SCREEN_SEEDS[0])
check_datasets, check_loaders = build_mnist_protocol(
    full_train_dataset, test_dataset, check_experiment
)
sizes = {name: len(dataset) for name, dataset in check_datasets.items()}
display(pd.Series(sizes, name='examples'))

assert sizes['train_joint'] == 50_000
assert sizes['valid_joint'] == 10_000
assert sizes['test_joint'] == 10_000
assert set(dataset_targets(check_datasets['train_old']).unique().tolist()) == set(OLD_CLASSES)
assert set(dataset_targets(check_datasets['train_new']).unique().tolist()) == set(NEW_CLASSES)
print('Deterministic split checks passed.')
'@

$cells += New-MarkdownCell @'
## 5. IID candidate screening

This is the only stage allowed to choose the Hebbian rule. Eligibility requires finite weights, at least 80% mean macro validation accuracy, and no digit below 50% accuracy. Forgetting performance is deliberately unavailable during selection.
'@

$cells += New-CodeCell @'
iid_records = []
iid_summaries = []

for model_config in CANDIDATES:
    for seed in SCREEN_SEEDS:
        experiment = replace(BASE_EXPERIMENT, seed=seed)
        datasets_for_seed, loaders_for_seed = build_mnist_protocol(
            full_train_dataset, test_dataset, experiment
        )
        records, summary = run_iid_candidate(
            model_config,
            experiment,
            datasets_for_seed,
            loaders_for_seed['valid_joint'],
        )
        iid_records.extend(records)
        iid_summaries.append(summary)
        print(
            f"{model_config.variant:>10} seed={seed}: "
            f"macro={summary['macro_accuracy']:.2f}%, "
            f"worst digit={summary['minimum_digit_accuracy']:.2f}%"
        )

iid_summary_df = pd.DataFrame(iid_summaries)
display(iid_summary_df)
'@

$cells += New-CodeCell @'
selection_table = (
    iid_summary_df.groupby('variant', as_index=False)
    .agg(
        mean_macro_accuracy=('macro_accuracy', 'mean'),
        std_macro_accuracy=('macro_accuracy', 'std'),
        worst_digit_accuracy=('minimum_digit_accuracy', 'min'),
        all_finite=('finite_weights', 'all'),
    )
)
selection_table['std_macro_accuracy'] = selection_table['std_macro_accuracy'].fillna(0.0)
selection_table['eligible'] = (
    (selection_table['mean_macro_accuracy'] >= 80.0)
    & (selection_table['worst_digit_accuracy'] >= 50.0)
    & selection_table['all_finite']
)
selection_table = selection_table.sort_values('mean_macro_accuracy', ascending=False)
display(selection_table)

eligible = selection_table[selection_table['eligible']].copy()
if len(eligible):
    best_accuracy = eligible['mean_macro_accuracy'].max()
    near_best = eligible[eligible['mean_macro_accuracy'] >= best_accuracy - 1.0].copy()
    simplicity = {'centered': 0, 'oja': 1, 'hard_wta': 2, 'soft_wta': 3}
    near_best['simplicity_rank'] = near_best['variant'].map(simplicity)
    selected_variant = near_best.sort_values('simplicity_rank').iloc[0]['variant']
elif ALLOW_DIAGNOSTIC_CONTINUAL_RUN_IF_GATE_FAILS:
    selected_variant = selection_table.iloc[0]['variant']
    print('WARNING: no candidate passed the gate; later output is diagnostic only.')
else:
    selected_variant = None

SELECTED_CONFIG = next((config for config in CANDIDATES if config.variant == selected_variant), None)
print('Frozen selection:', SELECTED_CONFIG)
'@

$cells += New-CodeCell @'
iid_curve_df = pd.DataFrame([
    {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
    for row in iid_records
])

fig, ax = plt.subplots(figsize=(9, 4.5))
for (variant, seed), group in iid_curve_df.groupby(['variant', 'seed']):
    ax.plot(
        group['step'], group['valid_joint_macro_accuracy'],
        marker='o', alpha=0.8, label=f'{variant}, seed {seed}',
    )
ax.axhline(80, color='black', linestyle='--', linewidth=1, label='eligibility target')
ax.set(
    xlabel='Local update step',
    ylabel='Validation macro accuracy (%)',
    title='IID competence screening',
)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.2)
plt.show()
'@

$cells += New-MarkdownCell @'
## 6. Frozen continual-learning experiment

The selected model learns digits 0-4 once. That exact state is copied into both phase-two conditions:

- **Sequential:** only digits 5-9 are available.
- **Interleaved control:** every batch contains 50% old and 50% new examples.

The primary result is forgetting at the first checkpoint reaching 80% new-class macro validation accuracy.
'@

$cells += New-CodeCell @'
if SELECTED_CONFIG is None:
    raise RuntimeError(
        'No candidate passed the IID gate. Adjust hyperparameters using IID validation only, '
        'then restart and rerun the notebook. Do not use forgetting to choose a model.'
    )

continual_runs = []
for seed in FINAL_SEEDS:
    experiment = replace(BASE_EXPERIMENT, seed=seed)
    datasets_for_seed, loaders_for_seed = build_mnist_protocol(
        full_train_dataset, test_dataset, experiment
    )
    result = run_continual_experiment(
        SELECTED_CONFIG, experiment, datasets_for_seed, loaders_for_seed
    )
    continual_runs.append(result)
    old_before = result['phase1_records'][-1]['valid_old_macro_accuracy']
    print(f'\nSeed {seed}: old accuracy before phase two = {old_before:.2f}%')
    for condition, condition_result in result['conditions'].items():
        print(condition, forgetting_at_target(old_before, condition_result['records'], 80.0))
'@

$cells += New-CodeCell @'
continual_rows = []
matched_rows = []

for result in continual_runs:
    seed = result['experiment_config']['seed']
    old_before = result['phase1_records'][-1]['valid_old_macro_accuracy']
    for condition, condition_result in result['conditions'].items():
        continual_rows.extend(condition_result['records'])
        for target in BASE_EXPERIMENT.matched_new_accuracies:
            matched_rows.append({
                'seed': seed,
                'variant': SELECTED_CONFIG.variant,
                'condition': condition,
                'old_accuracy_before_phase2': old_before,
                **forgetting_at_target(old_before, condition_result['records'], target),
            })

continual_scalar_df = pd.DataFrame([
    {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
    for row in continual_rows
])
matched_df = pd.DataFrame(matched_rows)
display(matched_df)

primary_summary = (
    matched_df[matched_df['target'] == 80.0]
    .groupby('condition', as_index=False)
    .agg(
        target_reach_rate=('target_reached', 'mean'),
        mean_old_before=('old_accuracy_before_phase2', 'mean'),
        mean_retained_old=('retained_old_accuracy', 'mean'),
        mean_forgetting=('forgetting', 'mean'),
        std_forgetting=('forgetting', 'std'),
        mean_new_accuracy=('new_accuracy', 'mean'),
        mean_joint_accuracy=('joint_accuracy', 'mean'),
    )
)
primary_summary['std_forgetting'] = primary_summary['std_forgetting'].fillna(0.0)
display(primary_summary)
'@

$cells += New-CodeCell @'
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for (condition, seed), group in continual_scalar_df.groupby(['condition', 'seed']):
    axes[0].plot(
        group['valid_new_macro_accuracy'], group['valid_old_macro_accuracy'],
        marker='o', alpha=0.75, label=f'{condition}, seed {seed}',
    )
    axes[1].plot(
        group['step'], group['valid_old_macro_accuracy'],
        marker='o', alpha=0.75, label=f'{condition}, seed {seed}',
    )
axes[0].axvline(80, color='black', linestyle='--', linewidth=1)
axes[0].set(
    xlabel='New-class validation macro accuracy (%)',
    ylabel='Old-class validation macro accuracy (%)',
    title='Stability-plasticity trade-off',
)
axes[1].set(
    xlabel='Phase-two local update step',
    ylabel='Old-class validation macro accuracy (%)',
    title='Old knowledge during phase two',
)
for ax in axes:
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
'@

$cells += New-MarkdownCell @'
## 7. Per-digit and confusion-matrix validation results

These plots use the first seed's first checkpoint crossing 80% new validation accuracy. Aggregate conclusions must still use all seeds.
'@

$cells += New-CodeCell @'
first_run = continual_runs[0]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for column, condition in enumerate(('sequential', 'interleaved')):
    records = first_run['conditions'][condition]['records']
    matched = first_record_at_target(records, 'valid_new_macro_accuracy', 80.0)
    if matched is None:
        axes[0, column].text(0.5, 0.5, '80% target not reached', ha='center', va='center')
        axes[1, column].text(0.5, 0.5, '80% target not reached', ha='center', va='center')
        continue

    old_scores = matched['valid_old_per_class_accuracy']
    new_scores = matched['valid_new_per_class_accuracy']
    scores = {**old_scores, **new_scores}
    axes[0, column].bar(list(scores.keys()), list(scores.values()))
    axes[0, column].set(
        title=f'{condition}: per-digit validation accuracy',
        xlabel='Digit', ylabel='Accuracy (%)', ylim=(0, 105),
    )

    confusion = np.asarray(matched['valid_joint_confusion_matrix'])
    image = axes[1, column].imshow(confusion, cmap='Blues')
    axes[1, column].set(
        title=f'{condition}: validation confusion matrix',
        xlabel='Predicted digit', ylabel='True digit',
    )
    fig.colorbar(image, ax=axes[1, column], fraction=0.046)

plt.tight_layout()
plt.show()
'@

$cells += New-MarkdownCell @'
## 8. Local-update diagnostics

Hebbian quantities are called **proposed updates**, not gradients. We measure their signal-to-noise ratio and cosine alignment with the negative backpropagation gradient that the same network would receive.
'@

$cells += New-CodeCell @'
diagnostic_run = continual_runs[0]
diagnostic_seed = diagnostic_run['experiment_config']['seed']
diagnostic_experiment = replace(BASE_EXPERIMENT, seed=diagnostic_seed)
diagnostic_datasets, diagnostic_loaders = build_mnist_protocol(
    full_train_dataset, test_dataset, diagnostic_experiment
)
diagnostic_model = HebbianMLP(SELECTED_CONFIG).to(diagnostic_experiment.device)
diagnostic_model.load_state_dict(diagnostic_run['phase1_state'])

alignment_metrics = alignment_over_loader(
    diagnostic_model, diagnostic_loaders['valid_joint'], max_batches=10
)
snr_metrics = update_snrs(
    diagnostic_model,
    diagnostic_datasets['valid_joint'],
    batch_size=BASE_EXPERIMENT.batch_size,
    max_batches=16,
)

print('Cosine alignment with backpropagation descent:', alignment_metrics)
print('Local-update signal-to-noise:', snr_metrics)
'@

$cells += New-MarkdownCell @'
## 9. One-time official test evaluation

Only set `RUN_FINAL_TEST=True` after `QUICK_RUN=False`, IID selection is complete, the winner is frozen, and you will not change any hyperparameter afterward.
'@

$cells += New-CodeCell @'
if not RUN_FINAL_TEST:
    print('Official test evaluation skipped. This is correct during development and selection.')
    test_rows = []
    test_summary_df = pd.DataFrame()
else:
    if QUICK_RUN:
        raise RuntimeError('Do not evaluate the official test set in QUICK_RUN mode.')
    missing_checkpoints = [
        (result['experiment_config']['seed'], condition)
        for result in continual_runs
        for condition, condition_result in result['conditions'].items()
        if condition_result['checkpoint_states'].get(80.0) is None
    ]
    if missing_checkpoints:
        raise RuntimeError(
            'Official test evaluation blocked because the 80% new-validation target '
            f'was not reached for: {missing_checkpoints}. The forgetting comparison is inconclusive.'
        )
    test_rows = []
    for result in continual_runs:
        seed = result['experiment_config']['seed']
        experiment = replace(BASE_EXPERIMENT, seed=seed)
        _, loaders_for_seed = build_mnist_protocol(full_train_dataset, test_dataset, experiment)
        test_loaders = {
            name: loader for name, loader in loaders_for_seed.items()
            if name.startswith('test_')
        }
        for condition, condition_result in result['conditions'].items():
            state = condition_result['checkpoint_states'].get(80.0)
            if state is None:
                test_rows.append({'seed': seed, 'condition': condition, 'target_reached': False})
                continue
            model = HebbianMLP(SELECTED_CONFIG).to(experiment.device)
            model.load_state_dict(state)
            metrics = evaluate_named_loaders(model, test_loaders, experiment.device)
            test_rows.append({
                'seed': seed,
                'condition': condition,
                'target_reached': True,
                **metrics,
            })
    test_summary_df = pd.DataFrame([
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in test_rows
    ])
    display(test_summary_df)
'@

$cells += New-MarkdownCell @'
## 10. Export results and checkpoints

Quick outputs are clearly separated from final outputs. The generated Markdown summary can be copied into `hebbian/REPORT.md` after the locked full run.
'@

$cells += New-CodeCell @'
OUTPUT_DIR = Path('/content/hebbian_results_quick' if QUICK_RUN else '/content/hebbian_results_final')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

iid_export = iid_summary_df.copy()
iid_export['per_class_accuracy'] = iid_export['per_class_accuracy'].map(json.dumps)
iid_export.to_csv(OUTPUT_DIR / 'iid_selection.csv', index=False)
selection_table.to_csv(OUTPUT_DIR / 'selection_table.csv', index=False)
iid_curve_df.to_csv(OUTPUT_DIR / 'iid_trajectories.csv', index=False)
continual_scalar_df.to_csv(OUTPUT_DIR / 'continual_trajectories.csv', index=False)
matched_df.to_csv(OUTPUT_DIR / 'matched_forgetting.csv', index=False)
primary_summary.to_csv(OUTPUT_DIR / 'primary_summary.csv', index=False)

with (OUTPUT_DIR / 'continual_details.json').open('w') as handle:
    json.dump(continual_rows, handle, indent=2)
with (OUTPUT_DIR / 'update_alignment.json').open('w') as handle:
    json.dump(alignment_metrics, handle, indent=2)
with (OUTPUT_DIR / 'update_snr.json').open('w') as handle:
    json.dump(snr_metrics, handle, indent=2)
if len(test_summary_df):
    test_summary_df.to_csv(OUTPUT_DIR / 'final_test_metrics.csv', index=False)
    with (OUTPUT_DIR / 'final_test_details.json').open('w') as handle:
        json.dump(test_rows, handle, indent=2)

run_config = {
    'quick_run': QUICK_RUN,
    'run_final_test': RUN_FINAL_TEST,
    'screen_seeds': SCREEN_SEEDS,
    'final_seeds': FINAL_SEEDS,
    'experiment': asdict(BASE_EXPERIMENT),
    'all_candidates': [asdict(config) for config in ALL_CANDIDATES],
    'active_candidates': [asdict(config) for config in CANDIDATES],
    'selected': asdict(SELECTED_CONFIG),
    'torch_version': torch.__version__,
    'torchvision_version': torchvision.__version__,
}
with (OUTPUT_DIR / 'run_config.json').open('w') as handle:
    json.dump(run_config, handle, indent=2)

for result in continual_runs:
    seed = result['experiment_config']['seed']
    torch.save(result['phase1_state'], OUTPUT_DIR / f'seed_{seed}_phase1.pt')
    for condition, condition_result in result['conditions'].items():
        for target, state in condition_result['checkpoint_states'].items():
            torch.save(state, OUTPUT_DIR / f'seed_{seed}_{condition}_new_{int(target)}.pt')

summary_markdown = f"""# Hebbian run summary

Mode: **{'QUICK / NOT REPORTABLE' if QUICK_RUN else 'FULL LOCKED RUN'}**

Selected configuration:

```json
{json.dumps(asdict(SELECTED_CONFIG), indent=2)}
```

## IID selection

{selection_table.to_markdown(index=False)}

## Forgetting at 80% new validation accuracy

{primary_summary.to_markdown(index=False)}

The official test set was {'evaluated' if RUN_FINAL_TEST else 'not evaluated'} in this run.
"""
(OUTPUT_DIR / 'RESULTS_SUMMARY.md').write_text(summary_markdown)

archive_path = shutil.make_archive(str(OUTPUT_DIR), 'zip', OUTPUT_DIR)
print('Saved:', OUTPUT_DIR)
print('Downloadable archive:', archive_path)
'@

$cells += New-CodeCell @'
from google.colab import files
files.download(str(OUTPUT_DIR) + '.zip')
'@

$cells += New-MarkdownCell @'
## Interpretation rules

- Less forgetting in the fixed-capacity Hebbian model at matched new competence supports a contribution from the local learning rule in this MNIST model.
- Similar or worse forgetting suggests that the local rule alone is insufficient here and additional systems-level mechanisms may be required.
- Failure to pass the IID, old-task, or new-task competence gates makes the forgetting comparison inconclusive.
- A wider SoftHebb model, replay, freezing, expansion, task-specific heads, or consolidation must be reported separately as architecture-plus-rule evidence.
- Do not compare these results directly with the current predictive-coding notebook's `98.96% -> 0%` result unless predictive coding is rerun under the same data, architecture, seed, and matched-accuracy protocol.
'@

if ($FullRun) {
    foreach ($cell in $cells) {
        $cellText = ($cell.source -join '')
        $cellText = $cellText.Replace('QUICK_RUN = True', 'QUICK_RUN = False')
        $cellText = $cellText.Replace(
            'Use a Colab GPU runtime if available. `QUICK_RUN=True` is only a smoke test; its numbers must not be used in the report.',
            'This file is locked for the full experiment. Use a Colab GPU runtime, run every cell in order, and do not change the frozen configuration.'
        )
        $cell.source = @(Convert-ToSourceLines $cellText)
    }
}

if ($EnableFinalTest) {
    foreach ($cell in $cells) {
        $cellText = ($cell.source -join '')
        $cellText = $cellText.Replace('RUN_FINAL_TEST = False', 'RUN_FINAL_TEST = True')
        $cell.source = @(Convert-ToSourceLines $cellText)
    }
}

$notebook = [ordered]@{
    cells = $cells
    metadata = [ordered]@{
        accelerator = 'GPU'
        colab = [ordered]@{
            name = if ($FullRun) {
                'Hebbian Catastrophic Forgetting - FULL LOCKED RUN'
            } else {
                'Hebbian Catastrophic Forgetting - Self Contained'
            }
            provenance = @()
        }
        kernelspec = [ordered]@{
            display_name = 'Python 3'
            language = 'python'
            name = 'python3'
        }
        language_info = [ordered]@{
            name = 'python'
            version = '3.x'
        }
    }
    nbformat = 4
    nbformat_minor = 5
}

$json = $notebook | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($OutputPath, $json, [System.Text.UTF8Encoding]::new($false))
Write-Output "Built $OutputPath"
