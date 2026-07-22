# Hebbian run summary

Mode: **QUICK / NOT REPORTABLE**

Selected configuration:

```json
{
  "num_inputs": 784,
  "num_hidden": 100,
  "num_outputs": 10,
  "variant": "oja",
  "activation_type": "sigmoid",
  "bias": false,
  "hidden_lr": 0.001,
  "output_lr": 0.1,
  "output_rule": "delta",
  "temperature": 0.2,
  "top_k": 5,
  "normalize_hidden": true,
  "weight_decay": 0.0,
  "eps": 1e-08
}
```

## IID selection

| variant   |   mean_macro_accuracy |   std_macro_accuracy |   worst_digit_accuracy | all_finite   | eligible   |
|:----------|----------------------:|---------------------:|-----------------------:|:-------------|:-----------|
| oja       |               80.2736 |                    0 |                60.4847 | True         | True       |
| centered  |               10      |                    0 |                 0      | True         | False      |
| hard_wta  |               10      |                    0 |                 0      | True         | False      |
| soft_wta  |               10      |                    0 |                 0      | True         | False      |

## Forgetting at 80% new validation accuracy

| condition   |   target_reach_rate |   mean_old_before |   mean_retained_old |   mean_forgetting |   std_forgetting |   mean_new_accuracy |   mean_joint_accuracy |
|:------------|--------------------:|------------------:|--------------------:|------------------:|-----------------:|--------------------:|----------------------:|
| interleaved |                   0 |           88.5102 |                 nan |          nan      |                0 |            nan      |              nan      |
| sequential  |                   1 |           88.5102 |                   0 |           88.5102 |                0 |             83.3403 |               41.6702 |

The official test set was not evaluated in this run.
