import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import download_mnist
from src.experiments.forgetting import build_forgetting_loaders, run_forgetting_experiment

SEED = 0
DATA_SPLIT_SEED = 0
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPES = ["backprop", "feedback_alignment", "predictive_coding"]
CONDITIONS = ["sequential", "interleaved"]

torch.manual_seed(DATA_SPLIT_SEED)
np.random.seed(DATA_SPLIT_SEED)
train_set, valid_set, test_set = download_mnist(train_prop=0.8, keep_prop=1.0)

all_results = {}
start_all = time.time()

for model_type in MODEL_TYPES:
    all_results[model_type] = {}
    for condition in CONDITIONS:
        print(f"\n=== {model_type} / {condition} ===", flush=True)
        loaders = build_forgetting_loaders(train_set, valid_set, batch_size=32)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        t0 = time.time()
        res = run_forgetting_experiment(
            **loaders,
            model_type=model_type,
            condition=condition,
            num_hidden=1000,
            lr=0.001,
            num_epochs_phase1=20,
            num_epochs_phase2=20,
            bias=False,
            verbose=True,
        )
        elapsed = time.time() - t0

        phase1_old_accuracy = float(res["phase1_results"]["avg_valid_accuracies"][-1])
        retained_old_accuracy = float(res["phase2_results"]["avg_valid_accuracies"][-1])
        forgetting = phase1_old_accuracy - retained_old_accuracy
        new_class_accuracy = float(res["new_class_acc_final"])

        serializable = {
            "model_type": model_type,
            "condition": condition,
            "architecture": [784, 1000, 10],
            "learning_rate": 0.001,
            "optimizer": "adam" if model_type != "predictive_coding" else "none (jpc predictive coding update)",
            "seed": SEED,
            "phase1_old_accuracy": phase1_old_accuracy,
            "retained_old_accuracy": retained_old_accuracy,
            "forgetting": forgetting,
            "new_class_accuracy": new_class_accuracy,
            "old_class_acc_trace": [float(x) for x in res["old_class_acc_trace"]],
            "phase1_results": {
                k: v for k, v in res["phase1_results"].items() if isinstance(v, list)
            },
            "phase2_results": {
                k: v for k, v in res["phase2_results"].items() if isinstance(v, list)
            },
            "wall_time_seconds": elapsed,
        }
        all_results[model_type][condition] = serializable

        print(
            f"{model_type}/{condition}: phase1={phase1_old_accuracy:.2f}%, "
            f"retained={retained_old_accuracy:.2f}%, forgetting={forgetting:.2f}, "
            f"new={new_class_accuracy:.2f}%, took {elapsed:.1f}s",
            flush=True,
        )

        with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

print(f"\nALL DONE in {time.time() - start_all:.1f}s total", flush=True)
