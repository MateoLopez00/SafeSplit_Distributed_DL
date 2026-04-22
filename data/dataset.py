from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_datasets(data_dir: Path):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def get_targets(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if isinstance(dataset, Subset):
        parent_targets = get_targets(dataset.dataset)
        return [parent_targets[i] for i in dataset.indices]
    raise TypeError(f"Dataset targets not supported for {type(dataset)!r}")


def _assign_main_labels(num_clients: int, num_classes: int, rng: np.random.Generator) -> list[int]:
    return [int(rng.integers(0, num_classes)) for _ in range(num_clients)]


def partition_data(
    dataset,
    num_clients: int,
    iid_rate: float,
    num_classes: int,
    seed: int,
    max_samples_per_client: int | None = None,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    targets = np.asarray(get_targets(dataset), dtype=np.int64)
    all_indices = list(range(len(targets)))
    rng.shuffle(all_indices)

    main_labels = _assign_main_labels(num_clients, num_classes, rng)
    label_to_indices = defaultdict(list)
    for idx in all_indices:
        label_to_indices[int(targets[idx])].append(int(idx))

    for label in label_to_indices:
        rng.shuffle(label_to_indices[label])

    per_client = len(dataset) // num_clients
    if max_samples_per_client is not None:
        per_client = min(per_client, max_samples_per_client)

    partitions: list[list[int]] = []
    remaining = set(all_indices)
    for client_id in range(num_clients):
        main_label = main_labels[client_id]
        iid_count = int(round(per_client * iid_rate))
        non_iid_count = per_client - iid_count

        client_indices: list[int] = []
        if iid_count > 0:
            available = np.asarray(sorted(remaining), dtype=np.int64)
            iid_take = min(iid_count, len(available))
            iid_choices = rng.choice(available, size=iid_take, replace=False)
            client_indices.extend(int(i) for i in iid_choices)
            remaining.difference_update(int(i) for i in iid_choices)

        main_pool = [idx for idx in label_to_indices[main_label] if idx in remaining]
        if len(main_pool) < non_iid_count:
            filler = [idx for idx in sorted(remaining) if idx not in main_pool]
            main_pool.extend(filler)
        main_pool = np.asarray(main_pool, dtype=np.int64)
        if non_iid_count > 0:
            main_take = min(non_iid_count, len(main_pool))
            main_choices = rng.choice(main_pool, size=main_take, replace=False)
            client_indices.extend(int(i) for i in main_choices)
            remaining.difference_update(int(i) for i in main_choices)

        if len(client_indices) < per_client and remaining:
            filler_pool = np.asarray(sorted(remaining), dtype=np.int64)
            filler_take = min(per_client - len(client_indices), len(filler_pool))
            filler_choices = rng.choice(filler_pool, size=filler_take, replace=False)
            client_indices.extend(int(i) for i in filler_choices)
            remaining.difference_update(int(i) for i in filler_choices)

        rng.shuffle(client_indices)
        partitions.append(client_indices)

    return partitions


def build_client_subsets(dataset, partitions: Iterable[list[int]]) -> list[Subset]:
    return [Subset(dataset, indices) for indices in partitions]
