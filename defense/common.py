import copy
import math
from dataclasses import dataclass

import torch


@dataclass
class Checkpoint:
    step: int
    round_id: int
    client_id: int
    head_state: dict[str, torch.Tensor]
    backbone_state: dict[str, torch.Tensor]
    tail_state: dict[str, torch.Tensor]
    update_state: dict[str, torch.Tensor]


def clone_state_dict(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for key in sorted(state_dict):
        parts.append(state_dict[key].reshape(-1).float())
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts)


def load_state_dict(module, state_dict: dict[str, torch.Tensor], device: torch.device) -> None:
    module.load_state_dict({k: v.to(device) for k, v in state_dict.items()})


def diff_state_dict(new_state: dict[str, torch.Tensor], old_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: new_state[key].float() - old_state[key].float() for key in new_state}


def state_to_matrix(vector: torch.Tensor, width: int) -> torch.Tensor:
    width = max(8, width)
    height = math.ceil(vector.numel() / width)
    padded = torch.zeros(height * width, dtype=torch.float32)
    padded[: vector.numel()] = vector.float()
    return padded.reshape(height, width)
