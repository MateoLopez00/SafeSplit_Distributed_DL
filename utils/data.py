from typing import Counter

import torch

from data.backdoor import MEAN, STD


def unnormalize(image: torch.Tensor) -> torch.Tensor:
    return ((image.detach().cpu() * STD) + MEAN).clamp(0.0, 1.0)


def summarize_counter(counter: Counter, class_names: list[str]) -> dict[str, int]:
    return {class_names[idx]: int(counter.get(idx, 0)) for idx in range(len(class_names)) if counter.get(idx, 0) > 0}
