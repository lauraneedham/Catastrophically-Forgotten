import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from src.data import download_mnist
    from src.experiments.forgetting import build_forgetting_loaders, run_forgetting_experiment

    return (
        build_forgetting_loaders,
        download_mnist,
        mo,
        plt,
        run_forgetting_experiment,
    )


@app.cell
def _(mo):
    mo.md("""
    # Catastrophic Forgetting Comparison: Predictive Coding vs Backprop

    This interactive notebook compares **Predictive Coding (`jpc`)** and **Backpropagation** in a sequential class-learning experiment on MNIST.

    - **Phase 1**: Train model on Digits 0–5.
    - **Phase 2**: Train model on Digits 6–9.
    - **Metric**: Track retainment of Digits 0–5 knowledge and learning speed on Digits 6–9.
    """)
    return


@app.cell
def _(mo):
    epochs_slider = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Epochs per Phase")
    keep_prop_slider = mo.ui.slider(start=0.01, stop=0.5, step=0.02, value=0.05, label="MNIST Subset Proportion")
    batch_size_slider = mo.ui.slider(start=16, stop=128, step=16, value=64, label="Batch Size")
    run_btn = mo.ui.run_button(label="Run Experiment")

    mo.hstack([epochs_slider, keep_prop_slider, batch_size_slider, run_btn], justify="start")
    return batch_size_slider, epochs_slider, keep_prop_slider, run_btn


@app.cell
def _(
    batch_size_slider,
    build_forgetting_loaders,
    download_mnist,
    epochs_slider,
    keep_prop_slider,
    mo,
    plt,
    run_btn,
    run_forgetting_experiment,
):
    mo.stop(not run_btn.value, mo.md("*Adjust parameters above and click **Run Experiment** to execute training.*"))

    train_set, valid_set, test_set = download_mnist(train_prop=0.8, keep_prop=keep_prop_slider.value)
    loaders = build_forgetting_loaders(train_set, valid_set, batch_size=batch_size_slider.value)

    num_epochs = epochs_slider.value

    # Run Backprop
    bp_results = run_forgetting_experiment(
        train_loader_old=loaders["train_loader_old"],
        valid_loader_old=loaders["valid_loader_old"],
        train_loader_new=loaders["train_loader_new"],
        valid_loader_new=loaders["valid_loader_new"],
        train_loader_full=loaders["train_loader_full"],
        model_type="backprop",
        num_epochs_phase1=num_epochs,
        num_epochs_phase2=num_epochs,
        verbose=False,
    )

    # Run Predictive Coding
    pc_results = run_forgetting_experiment(
        train_loader_old=loaders["train_loader_old"],
        valid_loader_old=loaders["valid_loader_old"],
        train_loader_new=loaders["train_loader_new"],
        valid_loader_new=loaders["valid_loader_new"],
        train_loader_full=loaders["train_loader_full"],
        model_type="predictive_coding",
        num_epochs_phase1=num_epochs,
        num_epochs_phase2=num_epochs,
        verbose=False,
    )

    bp_trace = bp_results["old_class_acc_trace"]
    pc_trace = pc_results["old_class_acc_trace"]
    bp_p1_final = float(bp_trace[num_epochs - 1])
    bp_p2_final = float(bp_trace[-1])
    bp_new_final = float(bp_results["new_class_acc_final"])

    pc_p1_final = float(pc_trace[num_epochs - 1])
    pc_p2_final = float(pc_trace[-1])
    pc_new_final = float(pc_results["new_class_acc_final"])

    # Stat cards
    bp_stats = mo.hstack([
        mo.stat(value=f"{bp_p1_final:.1f}%", label="Phase 1 Accuracy", caption="Digits 0-5"),
        mo.stat(value=f"{bp_p2_final:.1f}%", label="Retained Accuracy", caption="Digits 0-5 after Phase 2"),
        mo.stat(value=f"{bp_new_final:.1f}%", label="New Task Accuracy", caption="Digits 6-9"),
    ], justify="space-between")

    pc_stats = mo.hstack([
        mo.stat(value=f"{pc_p1_final:.1f}%", label="Phase 1 Accuracy", caption="Digits 0-5"),
        mo.stat(value=f"{pc_p2_final:.1f}%", label="Retained Accuracy", caption="Digits 0-5 after Phase 2"),
        mo.stat(value=f"{pc_new_final:.1f}%", label="New Task Accuracy", caption="Digits 6-9"),
    ], justify="space-between")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)

    epochs_axis = list(range(1, len(bp_trace) + 1))
    ax.axvspan(0.5, num_epochs + 0.5, color="#e6f2ff", alpha=0.5, label="Phase 1 (Digits 0-5)")
    ax.axvspan(num_epochs + 0.5, len(bp_trace) + 0.5, color="#fff0e6", alpha=0.5, label="Phase 2 (Digits 6-9)")

    ax.plot(epochs_axis, bp_trace, "o-", label="Backpropagation", color="#1f77b4", linewidth=2.2, markersize=6)
    ax.plot(epochs_axis, pc_trace, "s-", label="Predictive Coding (JPC)", color="#d62728", linewidth=2.2, markersize=6)

    ax.axvline(num_epochs + 0.5, color="#7f7f7f", linestyle="--", linewidth=1.2)

    ax.set_title("Catastrophic Forgetting Comparison", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Training Epoch", fontsize=10)
    ax.set_ylabel("Accuracy on Digits 0-5 (%)", fontsize=10)
    ax.set_xlim(0.5, len(bp_trace) + 0.5)
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout()

    mo.vstack([
        mo.md("### Metric Overview"),
        mo.md("#### Backpropagation Baseline"),
        bp_stats,
        mo.md("#### Predictive Coding (JPC)"),
        pc_stats,
        mo.md("---"),
        mo.md("### Accuracy Trace & Forgetting Curve"),
        fig,
    ])
    return (
        ax,
        bp_new_final,
        bp_p1_final,
        bp_p2_final,
        bp_results,
        bp_stats,
        bp_trace,
        epochs_axis,
        fig,
        loaders,
        num_epochs,
        pc_new_final,
        pc_p1_final,
        pc_p2_final,
        pc_results,
        pc_stats,
        pc_trace,
        test_set,
        train_set,
        valid_set,
    )


if __name__ == "__main__":
    app.run()
