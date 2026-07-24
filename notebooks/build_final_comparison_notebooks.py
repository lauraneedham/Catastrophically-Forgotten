"""Generate the two Colab notebooks used for the final comparison.

Run from the repository root:

    py notebooks/build_final_comparison_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def source(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "Final catastrophic-forgetting comparison",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


RUNNER_CELLS = [
    markdown(
        """
        # Final 784→1000→10 learning-rule run

        This Colab runs **one learning rule** under the frozen final protocol,
        including both sequential and interleaved conditions.

        Run it once for each rule:

        1. `backprop`
        2. `feedback_alignment`
        3. `predictive_coding`
        4. `hebbian`

        Download the ZIP produced by the last cell immediately. The second
        notebook combines the four ZIP files into the final tables and plots.

        The primary outcome is catastrophic forgetting. SNR and cosine are
        secondary, layer-wise update diagnostics. Analysis probes never update
        model weights.
        """
    ),
    markdown(
        """
        ## 1. Get the repository and install dependencies

        This currently targets the `post-merge-fixes` branch containing the
        standardized runner and analysis code.
        """
    ),
    code(
        """
        import os
        from pathlib import Path
        import subprocess
        import sys

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        REPO_URL = "https://github.com/lauraneedham/Catastrophically-Forgotten.git"
        REPO_BRANCH = "post-merge-fixes"
        REPO_DIR = Path("/content/Catastrophically-Forgotten")

        if (REPO_DIR / ".git").exists():
            subprocess.check_call(["git", "-C", str(REPO_DIR), "fetch", "origin"])
            subprocess.check_call(
                ["git", "-C", str(REPO_DIR), "switch", REPO_BRANCH]
            )
            subprocess.check_call(
                ["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", REPO_BRANCH]
            )
        else:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--branch",
                    REPO_BRANCH,
                    REPO_URL,
                    str(REPO_DIR),
                ]
            )

        # The full repository requirements include development and marimo
        # dependencies that are not needed by this runner. Installing them
        # unpinned makes pip backtrack across incompatible recent JAX
        # ecosystem releases in Colab. JPC currently requires JAX <= 0.5.2,
        # so keep this runtime environment explicit and reproducible.
        colab_runtime_dependencies = [
            "numpy>=1.26,<2.3",
            "pandas>=2.0",
            "matplotlib>=3.8",
            "scipy>=1.11",
            "tqdm>=4.66",
            "jax==0.5.2",
            "jaxlib==0.5.1",
            "equinox==0.13.8",
            "optax==0.2.5",
            "diffrax==0.7.2",
            "lineax==0.0.8",
            (
                "jpc @ git+https://github.com/thebuckleylab/jpc.git"
                "@a7015be6249c05ced833ecbf36491bd5d6b9c0db"
            ),
        ]
        print("Installing the pinned Colab runtime dependencies...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--retries",
                "5",
                "--timeout",
                "120",
                "--upgrade-strategy",
                "only-if-needed",
                *colab_runtime_dependencies,
            ]
        )
        os.chdir(REPO_DIR)
        sys.path.insert(0, str(REPO_DIR))
        print("Repository:", REPO_DIR)
        print(
            "Commit:",
            subprocess.check_output(
                ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
        )
        """
    ),
    markdown(
        """
        ## 2. Select exactly one rule

        Do not change the architecture or common protocol. You may reduce
        `max_probe_batches` only if PC probing is too slow; record the change.
        """
    ),
    code(
        """
        import torch

        from src.data import download_mnist
        from src.experiments.comparative import (
            ComparativeConfig,
            FINAL_RULES,
            run_rule_experiment,
            save_rule_experiment,
        )

        RULE_TO_RUN = "backprop"  # change to one FINAL_RULES value per session
        COLLECT_UPDATE_METRICS = True

        config = ComparativeConfig(
            num_inputs=784,
            num_hidden=1000,
            num_outputs=10,
            activation_type="sigmoid",
            bias=False,
            learning_rate=0.001,
            batch_size=32,
            phase1_recorded_epochs=20,
            phase2_recorded_epochs=20,
            record_initial_baseline=True,
            train_prop=0.8,
            keep_prop=1.0,
            data_split_seed=0,
            model_seed=0,
            data_order_seed=1000,
            max_probe_batches=8,
            phase2_analysis_epochs=(1, 5, 10, 20),
        )
        config.validate()
        assert RULE_TO_RUN in FINAL_RULES
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("Rule:", RULE_TO_RUN)
        print("Device:", DEVICE)
        print("Architecture:", config.architecture)
        """
    ),
    markdown(
        """
        ## 3. Load the single shared MNIST split

        The private split seed makes the train/validation indices independent
        of model initialization and previous random-number use.
        """
    ),
    code(
        """
        train_set, valid_set, test_set = download_mnist(
            train_prop=config.train_prop,
            keep_prop=config.keep_prop,
            seed=config.data_split_seed,
        )
        assert len(train_set) + len(valid_set) == 60_000
        assert len(test_set) == 10_000
        print("Shared split verified.")
        """
    ),
    markdown(
        """
        ## 3.1 Fast integration smoke test

        Keep this enabled the first time each learning rule is run. It uses the
        final 1,000-unit architecture but only a small deterministic subset and
        two recorded epochs per phase. It catches model/collector/dependency
        problems before the long full-MNIST cell.
        """
    ),
    code(
        """
        from dataclasses import replace
        from torch.utils.data import Subset

        RUN_SMOKE_TEST = True
        if RUN_SMOKE_TEST:
            smoke_train = Subset(
                train_set.dataset,
                list(train_set.indices[:512]),
            )
            smoke_valid = Subset(
                valid_set.dataset,
                list(valid_set.indices[:256]),
            )
            smoke_config = replace(
                config,
                phase1_recorded_epochs=2,
                phase2_recorded_epochs=2,
                max_probe_batches=1,
                phase2_analysis_epochs=(1, 2),
            )
            smoke_result = run_rule_experiment(
                RULE_TO_RUN,
                smoke_train,
                smoke_valid,
                smoke_config,
                device=DEVICE,
                collect_update_metrics=True,
                verbose=False,
            )
            print("Smoke test passed.")
            print(smoke_result["summaries"])
        """
    ),
    markdown(
        """
        ## 4. Run sequential and interleaved conditions

        This is the long cell. It prints accuracy after every recorded epoch.
        Both conditions restart from the same seed and private loader order.
        """
    ),
    code(
        """
        result = run_rule_experiment(
            RULE_TO_RUN,
            train_set,
            valid_set,
            config,
            device=DEVICE,
            collect_update_metrics=COLLECT_UPDATE_METRICS,
            verbose=True,
        )
        assert result["phase1_conditions_match"], (
            "Sequential and interleaved phase-one endpoints did not match."
        )
        print("Run complete.")
        """
    ),
    markdown("## 5. Inspect the performance summary"),
    code(
        """
        import pandas as pd
        from IPython.display import display

        summary_df = pd.DataFrame(result["summaries"])
        trace_df = pd.DataFrame(result["traces"])
        update_df = pd.DataFrame(result["update_metrics"])
        display(summary_df)

        assert set(summary_df["condition"]) == {"sequential", "interleaved"}
        assert (summary_df["architecture"] == "784-1000-10").all()
        assert (summary_df["learning_rate"] == 0.001).all()
        assert summary_df["phase1_old_accuracy"].min() >= 80.0, (
            "The rule did not pass the predefined phase-one competence gate."
        )
        """
    ),
    code(
        """
        import matplotlib.pyplot as plt

        trace_df["global_epoch"] = trace_df["recorded_epoch"]
        trace_df.loc[trace_df["phase"] == 2, "global_epoch"] += (
            config.phase1_recorded_epochs
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        for axis, condition in zip(axes, ("sequential", "interleaved")):
            subset = trace_df[trace_df["condition"] == condition]
            axis.plot(
                subset["global_epoch"],
                subset["old_accuracy"],
                label="Old digits 0-5",
                linewidth=2,
            )
            axis.plot(
                subset["global_epoch"],
                subset["new_accuracy"],
                label="New digits 6-9",
                linewidth=2,
            )
            axis.axvline(
                config.phase1_recorded_epochs + 0.5,
                color="black",
                linestyle="--",
                alpha=0.6,
            )
            axis.set_title(f"{RULE_TO_RUN}: {condition}")
            axis.set_xlabel("Recorded epoch")
            axis.set_ylim(-2, 102)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Validation accuracy (%)")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 6. Inspect update SNR and cosine

        Values are kept separate by layer, checkpoint, probe split, and
        condition. Do not average the hidden and output layers together.
        """
    ),
    code(
        """
        if COLLECT_UPDATE_METRICS:
            display(
                update_df[
                    [
                        "condition",
                        "checkpoint",
                        "probe_split",
                        "layer",
                        "snr",
                        "cosine_mean",
                        "cosine_sem",
                        "cosine_valid_batches",
                    ]
                ]
            )

            final_update = update_df[
                update_df["checkpoint"] == "phase2_epoch_20"
            ].copy()
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for axis, metric, title in (
                (axes[0], "snr", "Update SNR"),
                (axes[1], "cosine_mean", "Cosine to BP descent"),
            ):
                labels = (
                    final_update["condition"]
                    + "\\n"
                    + final_update["probe_split"]
                    + "\\n"
                    + final_update["layer"]
                )
                axis.bar(range(len(final_update)), final_update[metric])
                axis.set_xticks(range(len(final_update)))
                axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
                axis.set_title(title)
                axis.grid(axis="y", alpha=0.25)
            axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
        """
    ),
    markdown(
        """
        ## 7. Save and download this rule's artifact bundle

        Keep the ZIP unchanged. The combining notebook expects its CSV/JSON
        files and directory structure.
        """
    ),
    code(
        """
        import json
        import platform
        import shutil
        from google.colab import files

        OUTPUT_ROOT = Path("/content/final_rule_outputs")
        rule_output_dir = OUTPUT_ROOT / RULE_TO_RUN
        rule_output_dir.mkdir(parents=True, exist_ok=True)

        result["environment"] = {
            "repository": REPO_URL,
            "branch": REPO_BRANCH,
            "commit": subprocess.check_output(
                ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": DEVICE,
        }
        save_rule_experiment(result, rule_output_dir)
        (rule_output_dir / "README.txt").write_text(
            "Final 784-1000-10 run. Keep all files together for the "
            "combining notebook.\\n",
            encoding="utf-8",
        )

        zip_base = Path("/content") / f"final_784_1000_10_{RULE_TO_RUN}"
        zip_path = Path(
            shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_ROOT, base_dir=RULE_TO_RUN)
        )
        print("Saved:", zip_path)
        files.download(str(zip_path))
        """
    ),
]


COMBINER_CELLS = [
    markdown(
        """
        # Combine the four final learning-rule runs

        Upload the four ZIP files produced by the runner notebook. This notebook
        verifies protocol compatibility, combines performance/SNR/cosine
        tables, creates final figures, and downloads one final analysis ZIP.
        """
    ),
    code(
        """
        from pathlib import Path
        import shutil
        from google.colab import files

        uploaded = files.upload()
        INPUT_ROOT = Path("/content/final_comparison_inputs")
        if INPUT_ROOT.exists():
            shutil.rmtree(INPUT_ROOT)
        INPUT_ROOT.mkdir(parents=True)

        for filename, content in uploaded.items():
            archive_path = INPUT_ROOT / filename
            archive_path.write_bytes(content)
            shutil.unpack_archive(str(archive_path), str(INPUT_ROOT / archive_path.stem))

        print("Uploaded archives:", sorted(uploaded))
        """
    ),
    markdown("## 1. Load and validate every artifact"),
    code(
        """
        import pandas as pd
        import numpy as np
        from IPython.display import display

        def load_csv_family(filename):
            paths = sorted(INPUT_ROOT.rglob(filename))
            if not paths:
                raise FileNotFoundError(f"No {filename} files were found.")
            frames = []
            for path in paths:
                frame = pd.read_csv(path)
                frame["source_file"] = str(path)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True)

        performance_df = load_csv_family("performance_summary.csv")
        traces_df = load_csv_family("performance_traces.csv")
        update_df = load_csv_family("update_metrics.csv")

        required_rules = {
            "backprop",
            "feedback_alignment",
            "predictive_coding",
            "hebbian",
        }
        found_rules = set(performance_df["rule"])
        missing_rules = required_rules - found_rules
        if missing_rules:
            raise RuntimeError(f"Missing rule artifacts: {sorted(missing_rules)}")

        for rule in required_rules:
            conditions = set(
                performance_df.loc[performance_df["rule"] == rule, "condition"]
            )
            if conditions != {"sequential", "interleaved"}:
                raise RuntimeError(
                    f"{rule} has conditions {conditions}, expected both final conditions."
                )

        if not (performance_df["architecture"] == "784-1000-10").all():
            raise RuntimeError("At least one run used the wrong architecture.")
        if not np.allclose(performance_df["learning_rate"], 0.001):
            raise RuntimeError("At least one run used the wrong learning rate.")

        duplicates = performance_df.duplicated(["rule", "condition"], keep=False)
        if duplicates.any():
            display(performance_df.loc[duplicates])
            raise RuntimeError(
                "Duplicate rule/condition rows found. Upload exactly one final ZIP per rule."
            )

        print("All four rules and both conditions passed protocol validation.")
        """
    ),
    markdown("## 2. Final performance table"),
    code(
        """
        performance_columns = [
            "rule",
            "condition",
            "phase1_old_accuracy",
            "retained_old_accuracy",
            "forgetting",
            "normalized_forgetting",
            "new_class_accuracy",
            "balanced_final_accuracy",
            "optimizer",
        ]
        final_performance = performance_df[performance_columns].sort_values(
            ["condition", "rule"]
        )
        display(
            final_performance.style.format(
                {
                    "phase1_old_accuracy": "{:.2f}%",
                    "retained_old_accuracy": "{:.2f}%",
                    "forgetting": "{:.2f} pp",
                    "normalized_forgetting": "{:.3f}",
                    "new_class_accuracy": "{:.2f}%",
                    "balanced_final_accuracy": "{:.2f}%",
                }
            )
        )
        """
    ),
    markdown("## 3. Forgetting trajectories and endpoint comparison"),
    code(
        """
        import matplotlib.pyplot as plt

        traces_df["global_epoch"] = traces_df["recorded_epoch"]
        traces_df.loc[traces_df["phase"] == 2, "global_epoch"] += 20

        OUTPUT_DIR = Path("/content/final_comparative_analysis")
        FIGURE_DIR = OUTPUT_DIR / "figures"
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for axis, condition in zip(axes, ("sequential", "interleaved")):
            condition_df = traces_df[traces_df["condition"] == condition]
            for rule, rule_df in condition_df.groupby("rule"):
                axis.plot(
                    rule_df["global_epoch"],
                    rule_df["old_accuracy"],
                    label=rule,
                    linewidth=2,
                )
            axis.axvline(20.5, color="black", linestyle="--", alpha=0.6)
            axis.set_title(f"Old-task retention: {condition}")
            axis.set_xlabel("Recorded epoch")
            axis.set_ylim(-2, 102)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Accuracy on digits 0-5 (%)")
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(FIGURE_DIR / "old_task_retention_curves.png", dpi=180)
        plt.show()

        endpoint = performance_df.sort_values(["condition", "rule"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for axis, condition in zip(axes, ("sequential", "interleaved")):
            subset = endpoint[endpoint["condition"] == condition]
            x = np.arange(len(subset))
            axis.bar(x - 0.2, subset["retained_old_accuracy"], width=0.4, label="Old")
            axis.bar(x + 0.2, subset["new_class_accuracy"], width=0.4, label="New")
            axis.set_xticks(x)
            axis.set_xticklabels(subset["rule"], rotation=30, ha="right")
            axis.set_title(f"Final performance: {condition}")
            axis.set_ylim(0, 105)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Accuracy (%)")
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(FIGURE_DIR / "final_old_new_accuracy.png", dpi=180)
        plt.show()
        """
    ),
    markdown(
        """
        ## 4. Layer-wise SNR and cosine

        Statistical variation is summarized across fixed probe batches.
        Hidden and output layers remain separate.
        """
    ),
    code(
        """
        checkpoint_order = [
            "initial",
            "phase1_end",
            "phase2_epoch_1",
            "phase2_epoch_5",
            "phase2_epoch_10",
            "phase2_epoch_20",
        ]
        layer_styles = {
            "hidden_weight": ("-", "o"),
            "output_weight": ("--", "s"),
        }

        def plot_update_metric(metric, condition, probe_split, ylabel, filename):
            subset = update_df[
                (update_df["condition"] == condition)
                & (update_df["probe_split"] == probe_split)
            ].copy()
            subset["checkpoint_index"] = subset["checkpoint"].map(
                {name: i for i, name in enumerate(checkpoint_order)}
            )
            subset = subset.dropna(subset=["checkpoint_index"])

            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
            for axis, rule in zip(axes.ravel(), sorted(required_rules)):
                rule_df = subset[subset["rule"] == rule]
                for layer, (linestyle, marker) in layer_styles.items():
                    layer_df = rule_df[rule_df["layer"] == layer].sort_values(
                        "checkpoint_index"
                    )
                    axis.plot(
                        layer_df["checkpoint_index"],
                        layer_df[metric],
                        linestyle=linestyle,
                        marker=marker,
                        label=layer,
                    )
                    if metric == "cosine_mean":
                        sem = layer_df["cosine_sem"].fillna(0)
                        axis.fill_between(
                            layer_df["checkpoint_index"],
                            layer_df[metric] - sem,
                            layer_df[metric] + sem,
                            alpha=0.15,
                        )
                axis.set_title(rule)
                axis.grid(alpha=0.25)
                if metric == "cosine_mean":
                    axis.axhline(0, color="black", linestyle=":", alpha=0.5)
                    axis.set_ylim(-1.05, 1.05)
            for axis in axes[-1]:
                axis.set_xticks(range(len(checkpoint_order)))
                axis.set_xticklabels(checkpoint_order, rotation=50, ha="right")
            axes[0, 0].set_ylabel(ylabel)
            axes[1, 0].set_ylabel(ylabel)
            axes[0, 1].legend()
            fig.suptitle(f"{ylabel}: {condition}, {probe_split}")
            plt.tight_layout()
            fig.savefig(FIGURE_DIR / filename, dpi=180)
            plt.show()

        for condition in ("sequential", "interleaved"):
            for probe_split in ("old_0_5", "new_6_9"):
                slug = f"{condition}_{probe_split}"
                plot_update_metric(
                    "snr",
                    condition,
                    probe_split,
                    "Update SNR",
                    f"snr_{slug}.png",
                )
                plot_update_metric(
                    "cosine_mean",
                    condition,
                    probe_split,
                    "Cosine to BP descent",
                    f"cosine_{slug}.png",
                )
        """
    ),
    markdown("## 5. Export the complete final analysis"),
    code(
        """
        final_performance.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
        traces_df.to_csv(OUTPUT_DIR / "performance_traces.csv", index=False)
        update_df.to_csv(OUTPUT_DIR / "update_metrics.csv", index=False)

        sequential = final_performance[
            final_performance["condition"] == "sequential"
        ].sort_values("normalized_forgetting")
        interleaved = final_performance[
            final_performance["condition"] == "interleaved"
        ].sort_values("normalized_forgetting")
        report_lines = [
            "# Final comparative analysis",
            "",
            "Architecture: 784-1000-10; full MNIST; old 0-5, new 6-9; "
            "batch 32; learning rate 0.001; seed 0.",
            "",
            "## Sequential ranking by normalized forgetting",
        ]
        for row in sequential.itertuples():
            report_lines.append(
                f"- {row.rule}: normalized forgetting "
                f"{row.normalized_forgetting:.3f}, retained old "
                f"{row.retained_old_accuracy:.2f}%, new "
                f"{row.new_class_accuracy:.2f}%."
            )
        report_lines.extend(["", "## Interleaved control"])
        for row in interleaved.itertuples():
            report_lines.append(
                f"- {row.rule}: normalized forgetting "
                f"{row.normalized_forgetting:.3f}, retained old "
                f"{row.retained_old_accuracy:.2f}%, new "
                f"{row.new_class_accuracy:.2f}%."
            )
        report_lines.extend(
            [
                "",
                "## Limitations",
                "- Results use one seed and support descriptive conclusions only.",
                "- The first recorded epoch is a baseline for BP, FA, and Hebbian; "
                "PC updates internally during that pass.",
                "- PC update cosine uses the non-mutating JPC/Optax parameter delta; "
                "the other rules expose pre-optimizer learning directions.",
                "- SNR and cosine are signal-consistency and directional-alignment "
                "proxies, not literal statistical variance and bias.",
            ]
        )
        (OUTPUT_DIR / "ANALYSIS_REPORT.md").write_text(
            "\\n".join(report_lines) + "\\n",
            encoding="utf-8",
        )

        final_zip = Path(
            shutil.make_archive(
                "/content/final_comparative_analysis_784_1000_10",
                "zip",
                root_dir=OUTPUT_DIR,
            )
        )
        print("Final analysis:", final_zip)
        files.download(str(final_zip))
        """
    ),
]


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    runner_path = NOTEBOOK_DIR / "final_rule_run_784_1000_10_colab.ipynb"
    combiner_path = NOTEBOOK_DIR / "final_comparative_analysis_784_1000_10_colab.ipynb"
    write_notebook(runner_path, RUNNER_CELLS)
    write_notebook(combiner_path, COMBINER_CELLS)
    print(runner_path)
    print(combiner_path)


if __name__ == "__main__":
    main()
