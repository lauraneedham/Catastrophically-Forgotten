"""Dataset helpers for the forgetting experiment."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def download_mnist(
    train_prop: float = 0.8,
    keep_prop: float = 1.0,
    seed: int | None = None,
):
    """Download MNIST and return train/validation/test splits.

    ``seed=None`` preserves the original global-RNG behavior. Passing a seed
    creates a private split generator so every learning rule can use exactly
    the same train/validation indices without depending on model
    initialization or other random-number consumption.
    """
    valid_prop = 1 - train_prop
    discard_prop = 1 - keep_prop
    split_generator = (
        None if seed is None else torch.Generator().manual_seed(int(seed))
    )

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        full_train_set = torchvision.datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
        full_test_set = torchvision.datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=transform
        )

    if discard_prop > 0:
        train_set, valid_set, _ = torch.utils.data.random_split(
            full_train_set,
            [train_prop * keep_prop, valid_prop * keep_prop, discard_prop],
            generator=split_generator,
        )
        test_set, _ = torch.utils.data.random_split(
            full_test_set,
            [keep_prop, discard_prop],
            generator=split_generator,
        )
    else:
        # keep_prop >= 1.0: use all of MNIST (the notebook's setting). Only split
        # train into train/valid; avoids random_split's empty-split warnings for
        # a zero-length discard, and keeps the full test set.
        train_set, valid_set = torch.utils.data.random_split(
            full_train_set,
            [train_prop, valid_prop],
            generator=split_generator,
        )
        test_set = full_test_set

    print("Number of examples retained:")
    print(f"  {len(train_set)} (training)")
    print(f"  {len(valid_set)} (validation)")
    print(f"  {len(test_set)} (test)")

    return train_set, valid_set, test_set


def restrict_classes(dataset, classes=(6,), keep=True):
    """Keep or remove a subset of classes from a dataset or subset."""
    if hasattr(dataset, "dataset"):
        indices = np.asarray(dataset.indices)
        targets = dataset.dataset.targets[indices]
        dataset = dataset.dataset
    else:
        indices = np.arange(len(dataset))
        targets = dataset.targets

    specified_idxs = np.isin(targets, np.asarray(classes))
    retain_indices = indices[specified_idxs] if keep else indices[~specified_idxs]
    return torch.utils.data.Subset(dataset, retain_indices)


def get_plotting_color(dataset="train", model_idx=None):
    """Return a consistent color for plots."""
    if model_idx is not None:
        dataset = None

    if model_idx == 0 or dataset == "train":
        return "#1F77B4"
    if model_idx == 1 or dataset == "valid":
        return "#FF7F0E"
    if model_idx == 2 or dataset == "test":
        return "#2CA02C"
    raise NotImplementedError("Colors only implemented for up to 3 models.")


def plot_class_distribution(train_set, valid_set=None, test_set=None, num_classes=10, ax=None):
    """Plot class frequency histograms for train/valid/test splits."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    bins = np.arange(num_classes + 1) - 0.5

    for dataset_name, dataset in [("train", train_set), ("valid", valid_set), ("test", test_set)]:
        if dataset is None:
            continue
        if hasattr(dataset, "dataset"):
            targets = dataset.dataset.targets[dataset.indices]
        else:
            targets = dataset.targets

        ax.hist(targets, bins=bins, alpha=0.3, color=get_plotting_color(dataset_name), label=dataset_name)
        per_class = len(targets) / num_classes
        ax.axhline(per_class, ls="dashed", color=get_plotting_color(dataset_name), alpha=0.8)

    ax.set_xticks(range(num_classes))
    ax.set_title("Counts per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.legend(loc="center right")
    return ax


def plot_examples(subset, num_examples_per_class=8, mlp=None, seed=None, batch_size=32, num_classes=10, ax=None):
    """Visualize sample images grouped by label or model prediction."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if seed is not None:
        np.random.seed(seed)

    if hasattr(subset, "dataset"):
        targets = subset.dataset.targets[subset.indices]
        images = subset.dataset.data[subset.indices]
    else:
        targets = subset.targets
        images = subset.data

    classes = sorted(np.unique(targets))
    selected = []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        if len(cls_indices) == 0:
            continue
        pick = np.random.choice(cls_indices, size=min(num_examples_per_class, len(cls_indices)), replace=False)
        selected.extend(list(pick))

    selected = np.array(selected)
    images = images[selected]
    labels = targets[selected]

    if len(images) == 0:
        return ax

    grid_h = int(np.ceil(len(images) / num_examples_per_class))
    fig, axes = plt.subplots(grid_h, num_examples_per_class, figsize=(2.0 * num_examples_per_class, 2.0 * grid_h))
    axes = np.atleast_1d(axes).reshape(grid_h, num_examples_per_class)

    for idx, (image, label) in enumerate(zip(images, labels)):
        ax_i = axes[idx // num_examples_per_class, idx % num_examples_per_class]
        ax_i.imshow(image.squeeze(), cmap="gray")
        ax_i.set_title(int(label))
        ax_i.axis("off")

    for ax_i in axes.flat[len(images):]:
        ax_i.axis("off")

    return ax
