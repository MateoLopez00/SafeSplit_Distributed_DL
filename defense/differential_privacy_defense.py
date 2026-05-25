from __future__ import annotations

import copy

import torch

from defense.common import Checkpoint, flatten_state_dict

from .interface import DefenseInterface


class DifferentialPrivacyDefense(DefenseInterface):
    def __init__(self, clip_norm: float, noise_scale: float) -> None:
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        latest = copy.deepcopy(history[-1])
        update_vec = flatten_state_dict(latest.update_state)
        update_norm = torch.norm(update_vec, p=2)
        scale = min(1.0, self.clip_norm / (update_norm.item() + 1e-8))

        noisy_backbone = {}
        for name, tensor in latest.update_state.items():
            clipped = tensor * scale
            noise = torch.randn_like(clipped) * self.noise_scale
            noisy_backbone[name] = latest.backbone_state[name] - tensor + clipped + noise
        latest.backbone_state = noisy_backbone
        return latest
