"""Weight visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_weights(mlp, shared_colorbar=False):
    """Plot initial weights, final weights, and weight difference for each parameter."""
    param_names = mlp.list_parameters()

    params_images = {}
    pre_means = {}
    post_means = {}
    vmin = np.inf
    vmax = -np.inf

    for param_name in param_names:
        layer, param_type = param_name.split("_")
        init_params = getattr(mlp, f"init_{layer}_{param_type}").detach().cpu().numpy()
        last_params = getattr(getattr(mlp, layer), param_type).detach().cpu().numpy()
        diff_params = last_params - init_params
        separator = np.full((1, init_params.shape[-1]), np.nan)
        params_image = np.vstack([init_params, separator, last_params, separator, diff_params])
        vmin = min(vmin, np.nanmin(params_image))
        vmax = max(vmax, np.nanmax(params_image))
        params_images[param_name] = params_image
        pre_means[param_name] = init_params.mean()
        post_means[param_name] = last_params.mean()

    nrows = len(param_names)
    gridspec_kw = {}
    if len(param_names) == 4:
        gridspec_kw["height_ratios"] = [5, 1, 5, 1]
        cbar_label = "Weight/bias values"
    elif len(param_names) == 2:
        gridspec_kw["height_ratios"] = [5, 5]
        cbar_label = "Weight values"
    else:
        raise NotImplementedError(f"Expected 2 or 4 parameters, found {len(param_names)}")

    if shared_colorbar:
        nrows += 1
        gridspec_kw["height_ratios"].append(1)
    else:
        vmin, vmax = None, None

    fig, axes = plt.subplots(nrows, 1, figsize=(6, nrows + 3), gridspec_kw=gridspec_kw)

    for i, (param_name, params_image) in enumerate(params_images.items()):
        layer, param_type = param_name.split("_")
        layer_str = "First" if layer == "lin1" else "Second"
        param_str = "weights" if param_type == "weight" else "biases"
        axes[i].set_title(f"{layer_str} linear layer {param_str} (pre, post and diff)")
        im = axes[i].imshow(params_image, aspect="auto", vmin=vmin, vmax=vmax)
        if not shared_colorbar:
            cbar = fig.colorbar(im, ax=axes[i], aspect=10)
            cbar.ax.axhline(pre_means[param_name], ls="dotted", color="k", alpha=0.5)
            cbar.ax.axhline(post_means[param_name], color="k", alpha=0.5)
        if param_type == "weight":
            axes[i].set_xlabel("Input dim.")
            axes[i].set_ylabel("Output dim.")
        axes[i].spines[["left", "bottom"]].set_visible(False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    if shared_colorbar:
        cax = axes[-1]
        fig.colorbar(im, cax=cax, orientation="horizontal", location="bottom")
        cax.set_xlabel(cbar_label)

    return axes
