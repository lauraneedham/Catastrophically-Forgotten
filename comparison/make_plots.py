import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMPARISON_DIR = Path(__file__).resolve().parent
REPO = COMPARISON_DIR.parent
OUT_DIR = COMPARISON_DIR

# Fixed categorical order/colors (validated default palette, first 4 slots).
COLORS = {
    "Backprop": "#2a78d6",           # blue
    "Feedback Alignment": "#008300",  # green
    "Predictive Coding": "#e87ba4",   # magenta
    "Hebbian": "#eda100",             # yellow
}
METHOD_ORDER = ["Backprop", "Feedback Alignment", "Predictive Coding", "Hebbian"]

# --- Load data ---
new_results = json.load(open(COMPARISON_DIR / "results.json"))
hebbian_results = json.load(
    open(REPO / "hebbian" / "FINAL (learning rate 0.001)" / "1000" / "results.json")
)

label_to_key = {
    "Backprop": ("new", "backprop"),
    "Feedback Alignment": ("new", "feedback_alignment"),
    "Predictive Coding": ("new", "predictive_coding"),
}

data = {}
for label in METHOD_ORDER:
    if label == "Hebbian":
        data[label] = {
            "sequential": hebbian_results["sequential"],
            "interleaved": hebbian_results["interleaved"],
        }
    else:
        _, key = label_to_key[label]
        data[label] = {
            "sequential": new_results[key]["sequential"],
            "interleaved": new_results[key]["interleaved"],
        }


def trace(label, condition):
    d = data[label][condition]
    if "old_class_acc_trace" in d:
        return np.asarray(d["old_class_acc_trace"], dtype=float)
    raise KeyError(condition)


def summary_row(label, condition):
    d = data[label][condition]
    if label == "Hebbian":
        phase1 = d["phase1_accuracy"]
        retained = d["retained_old_accuracy"]
        new = d["new_class_acc_final"]
    else:
        phase1 = d["phase1_old_accuracy"]
        retained = d["retained_old_accuracy"]
        new = d["new_class_accuracy"]
    forgetting = phase1 - retained
    relative_forgetting = forgetting / phase1 * 100 if phase1 > 0 else float("nan")
    return {
        "phase1": phase1,
        "retained": retained,
        "forgetting": forgetting,
        "relative_forgetting": relative_forgetting,
        "new": new,
    }


PHASE1_EPOCHS = 20

# ============================================================
# Figure 1: forgetting curves, sequential | interleaved
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, condition, title in zip(
    axes, ["sequential", "interleaved"], ["Sequential (old task dropped)", "Interleaved (old task retained in training)"]
):
    for label in METHOD_ORDER:
        y = trace(label, condition)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, label=label, color=COLORS[label], linewidth=2, alpha=0.9)

    ax.axvline(PHASE1_EPOCHS + 0.5, color="#8a8a86", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlabel("Epoch")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-3, 103)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=-1)

axes[0].set_ylabel("Old-class (digits 0-5) validation accuracy (%)")
axes[1].legend(loc="lower left", frameon=False, fontsize=9)
fig.suptitle(
    "Catastrophic forgetting across learning rules\n"
    "784 → 1000 → 10, batch 32, lr 0.001, seed 0",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT_DIR / "forgetting_curves.png", dpi=200)
plt.close(fig)

# ============================================================
# Figure 2: forgetting magnitude bar chart (sequential vs interleaved)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))

x = np.arange(len(METHOD_ORDER))
width = 0.35

seq_vals = [summary_row(label, "sequential")["relative_forgetting"] for label in METHOD_ORDER]
int_vals = [summary_row(label, "interleaved")["relative_forgetting"] for label in METHOD_ORDER]

bars_seq = ax.bar(
    x - width / 2, seq_vals, width, label="Sequential",
    color=[COLORS[l] for l in METHOD_ORDER], alpha=0.95,
)
bars_int = ax.bar(
    x + width / 2, int_vals, width, label="Interleaved",
    color=[COLORS[l] for l in METHOD_ORDER], alpha=0.45,
)

for bars in (bars_seq, bars_int):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h),
            textcoords="offset points", xytext=(0, 3),
            ha="center", fontsize=8, color="#0b0b0b",
        )

ax.set_xticks(x)
ax.set_xticklabels(METHOD_ORDER, rotation=10)
ax.set_ylabel("Relative forgetting\n(% of phase-1 accuracy lost)")
ax.set_title("Forgetting magnitude by learning rule and condition\n(normalized to each method's own phase-1 accuracy)", fontsize=12)
ax.set_ylim(0, 112)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=-1)

# Legend explaining alpha (condition), separate from color (method identity).
from matplotlib.patches import Patch
condition_legend = [
    Patch(facecolor="#52514e", alpha=0.95, label="Sequential"),
    Patch(facecolor="#52514e", alpha=0.45, label="Interleaved"),
]
ax.legend(
    handles=condition_legend, loc="upper left", bbox_to_anchor=(1.01, 1.0),
    frameon=False, fontsize=9, title="Condition",
)

fig.tight_layout()
fig.savefig(OUT_DIR / "forgetting_magnitude_bars.png", dpi=200)
plt.close(fig)

# ============================================================
# Table image
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")

columns = ["Method", "Condition", "Old before", "Old after", "Forgetting (relative)", "New after"]
rows = []
row_colors = []
for label in METHOD_ORDER:
    for condition, cond_label in [("sequential", "Sequential"), ("interleaved", "Interleaved")]:
        s = summary_row(label, condition)
        rows.append(
            [
                label,
                cond_label,
                f"{s['phase1']:.2f}%",
                f"{s['retained']:.2f}%",
                f"{s['relative_forgetting']:.1f}%",
                f"{s['new']:.2f}%",
            ]
        )
        row_colors.append(COLORS[label])

table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

for i in range(len(columns)):
    cell = table[0, i]
    cell.set_text_props(weight="bold", color="white")
    cell.set_facecolor("#3a3a38")

for r, color in enumerate(row_colors, start=1):
    swatch_cell = table[r, 0]
    swatch_cell.set_text_props(color=color, weight="bold")

ax.set_title(
    "784 → 1000 → 10, batch 32, lr 0.001, seed 0, Adam optimizer where used",
    fontsize=10, pad=14,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "comparison_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved figures to", OUT_DIR)
for p in sorted(OUT_DIR.iterdir()):
    print(" -", p.name)
