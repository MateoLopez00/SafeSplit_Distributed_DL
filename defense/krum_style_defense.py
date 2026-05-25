from __future__ import annotations

import copy

import torch

from defense.common import Checkpoint, flatten_state_dict

from .interface import DefenseInterface


class KrumStyleDefense(DefenseInterface):
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        if len(history) <= 1:
            return copy.deepcopy(history[-1])

        window = history[-min(self.window_size, len(history)):]
        majority_size = max(1, len(window) // 2)
        update_vectors = [flatten_state_dict(cp.update_state) for cp in window]
        scores = []
        for i in range(len(window)):
            distances = []
            for j in range(len(window)):
                if i == j:
                    continue
                distances.append(torch.dist(update_vectors[i], update_vectors[j], p=2).item())
            scores.append(sum(sorted(distances)[:majority_size]))

        best_idx = min(range(len(window)), key=lambda idx: scores[idx])
        return copy.deepcopy(window[best_idx])
