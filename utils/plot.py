# utils/plot.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from utils.data import unnormalize


def plot_client_label_distribution(
    distribution_matrix: list[list[int]],
    class_names: list[str],
    *,
    iid_rate: float,
) -> None:
    matrix = np.asarray(distribution_matrix)

    num_clients = len(distribution_matrix)

    plt.figure(figsize=(10, 4))

    bottom = np.zeros(num_clients)

    for class_idx, class_name in enumerate(class_names):
        plt.bar(
            np.arange(num_clients),
            matrix[:, class_idx],
            bottom=bottom,
            label=class_name,
        )

        bottom += matrix[:, class_idx]

    plt.xticks(
        np.arange(num_clients),
        [f"Client {idx}" for idx in range(num_clients)],
    )

    plt.ylabel("Samples")

    plt.title(
        f"Client label distribution (iid_rate={iid_rate})"
    )

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    plt.tight_layout()
    plt.show()


def plot_metric_summary(rows: list[dict], title: str) -> None:
    labels = [f"{row['defense']}\n{row['backdoor']}\nIID={row['iid_rate']}" for row in rows]
    ma_values = [row["final_MA"] for row in rows]
    ba_values = [row["final_BA"] for row in rows]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(8, len(labels) * 1.4), 4.5))
    plt.bar(x - width / 2, ma_values, width, label="MA")
    plt.bar(x + width / 2, ba_values, width, label="BA")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dataset_samples(
        dataset,
        title="Dataset samples",
        num_samples=12,
        nrow=4,
        figsize=(8, 6),
):
    """
    Plot a grid of sample images from a dataset.

    Args:
        dataset: PyTorch dataset returning (image, label)
        unnormalize_fn: Function used to unnormalize images
        title: Plot title
        num_samples: Number of images to display
        nrow: Number of images per row
        figsize: Matplotlib figure size
    """

    sample_items = [dataset[i][0] for i in range(num_samples)]

    sample_grid = make_grid(
        torch.stack([unnormalize(image) for image in sample_items]),
        nrow=nrow,
    )

    plt.figure(figsize=figsize)
    plt.imshow(sample_grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()
