from __future__ import annotations

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def get_plotting_color(model_idx):
    """Returns a color based on the model index for consistent plotting."""
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    return colors[model_idx % len(colors)]


def compute_SNR(data, epsilon=1e-7):
    """
    Calculates the average SNR of data across the first axis.

    Arguments:
    - data (np array): items x gradients
    - epsilon (float, optional): value added to the denominator to avoid
      division by zero.

    Returns:
    - avg_SNR (float): average SNR across data items
    """
    absolute_mean = np.abs(np.mean(data, axis=0))
    std = np.std(data, axis=0)
    SNR_by_item = absolute_mean / (std + epsilon)
    avg_SNR = np.mean(SNR_by_item)
    return avg_SNR


def compute_gradient_SNR(model, loader, param_names, collect_gradients_fn):
    """
    Computes the gradient SNR for each requested parameter of `model`, using
    the model's own forward/backward pass (i.e. its own learning rule).

    Arguments:
    - model: a trained model with a `.named_parameters()` method.
    - loader: a DataLoader to draw examples from.
    - param_names (list[str]): which parameters to compute SNR for.
    - collect_gradients_fn (callable): a function with signature
      `(model, loader, param_names) -> dict[str, np.ndarray]` that returns
      per-example flattened gradients for each requested parameter. This is
      left as an injected dependency since gradient collection is rule-specific
      (e.g. feedback alignment needs to swap in the true backprop weight to
      get a comparison gradient; other rules may not).

    Returns:
    - dict[str, float]: parameter name -> average SNR
    """
    grads = collect_gradients_fn(model, loader, param_names)
    return {name: compute_SNR(grads[name]) for name in param_names}


def plot_gradient_SNRs(SNR_dict, width=0.5, ax=None):
    """
    Plot gradient SNRs for various learning rules.

    Arguments:
    - SNR_dict (dict): Gradient SNRs for each learning rule.
    - width (float, optional): Width of the bars.
    - ax (plt subplot, optional): Axis on which to plot gradient SNRs. If None, a
      new axis will be created.

    Returns:
    - ax (plt subplot): Axis on which gradient SNRs were plotted.
    """
    if ax is None:
        wid = min(8, len(SNR_dict) * 1.5)
        _, ax = plt.subplots(figsize=(wid, 4))

    xlabels = list()
    for m, (model_type, SNRs) in enumerate(SNR_dict.items()):
        xlabels.append(model_type)
        color = get_plotting_color(model_idx=m)
        ax.bar(
            m, np.mean(SNRs), yerr=scipy.stats.sem(SNRs),
            alpha=0.5, width=width, capsize=5, color=color
        )
        s = [20 + i * 30 for i in range(len(SNRs))]
        ax.scatter([m] * len(SNRs), SNRs, alpha=0.8, s=s, color=color, zorder=5)

    x = np.arange(len(xlabels))
    ax.set_xticks(x)
    x_pad = (x.max() - x.min() + width) * 0.3
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_xlabel("Learning rule")
    ax.set_ylabel("SNR")
    ax.set_title("SNR of the gradients")

    return ax


def calculate_cosine_similarity(data1, data2):
    """Cosine similarity between two flattened gradient arrays."""
    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)

    numerator = np.dot(data1, data2)
    denominator = (
        np.sqrt(np.dot(data1, data1)) * np.sqrt(np.dot(data2, data2))
    )

    return numerator / denominator


# train_and_calculate_cosine_sim is left not implemented; unsure what it means.
def train_and_calculate_cosine_sim(*args, **kwargs):
    raise NotImplementedError("Cosine-sim training analysis is not implemented yet.")


def plot_gradient_cosine_sims(cosine_sim_dict, ax=None):
    """
    Plot gradient cosine similarities to backprop (bias) across epochs, for
    each learning rule.

    Arguments:
    - cosine_sim_dict (dict): model_type -> list of per-parameter cosine
      similarity trajectories (params x epochs).
    - ax (plt subplot, optional): Axis on which to plot. If None, a new axis
      will be created.

    Returns:
    - ax (plt subplot): Axis on which cosine similarities were plotted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    max_num_epochs = 0
    for m, (model_type, cosine_sims) in enumerate(cosine_sim_dict.items()):
        cosine_sims = np.asarray(cosine_sims)  # params x epochs
        num_epochs = cosine_sims.shape[1]
        x = np.arange(num_epochs)
        cosine_sim_means = np.nanmean(cosine_sims, axis=0)
        cosine_sim_sems = scipy.stats.sem(cosine_sims, axis=0, nan_policy="omit")

        ax.plot(x, cosine_sim_means, label=model_type, alpha=0.8)

        color = get_plotting_color(model_idx=m)
        ax.fill_between(
            x,
            cosine_sim_means - cosine_sim_sems,
            cosine_sim_means + cosine_sim_sems,
            alpha=0.3, lw=0, color=color
        )

        for i, param_cosine_sims in enumerate(cosine_sims):
            s = 20 + i * 30
            ax.scatter(x, param_cosine_sims, color=color, s=s, alpha=0.6)

        max_num_epochs = max(max_num_epochs, num_epochs)

    if max_num_epochs > 0:
        x = np.arange(max_num_epochs)
        xlabels = [f"{int(e)}" for e in x]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)

    ymin = ax.get_ylim()[0]
    ymin = min(-0.1, ymin)
    ax.set_ylim(ymin, 1.1)

    ax.axhline(0, ls="dashed", color="k", zorder=-5, alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity to backprop gradients")
    ax.legend()

    return ax
