# Hebbian run summary

Mode: **FULL LOCKED RUN**

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
| oja       |               83.9758 |             0.516575 |                69.3587 | True         | True       |

## Forgetting at 80% new validation accuracy

| condition   |   target_reach_rate |   mean_old_before |   mean_retained_old |   mean_forgetting |   std_forgetting |   mean_new_accuracy |   mean_joint_accuracy |
|:------------|--------------------:|------------------:|--------------------:|------------------:|-----------------:|--------------------:|----------------------:|
| interleaved |                   1 |           90.6916 |         83.6057     |           7.08585 |         0.666379 |             80.2998 |               81.9528 |
| sequential  |                   1 |           90.6916 |          0.00459242 |          90.687   |         0.384436 |             82.161  |               41.0828 |

The official test set was evaluated in this run.
