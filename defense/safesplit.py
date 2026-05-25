from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from scipy.fft import dctn

from defense.common import Checkpoint, flatten_state_dict, state_to_matrix

from .interface import DefenseInterface


def dct_low_frequency(update_vector: torch.Tensor, width: int, low_freq_frac: float) -> torch.Tensor:
    matrix = state_to_matrix(update_vector, width).numpy()
    coeffs = dctn(matrix, type=2, norm="ortho")
    keep_h = max(1, int(math.ceil(coeffs.shape[0] * low_freq_frac)))
    keep_w = max(1, int(math.ceil(coeffs.shape[1] * low_freq_frac)))
    return torch.tensor(coeffs[:keep_h, :keep_w].reshape(-1), dtype=torch.float32)


def rotational_signature(backbone_vector: torch.Tensor, width: int) -> torch.Tensor:
    matrix = state_to_matrix(backbone_vector, width)
    row_mean = matrix.mean(dim=1, keepdim=True)
    col_mean = matrix.mean(dim=0, keepdim=True)
    bx = (row_mean * matrix).reshape(-1)
    by = (matrix * col_mean).reshape(-1)
    theta = torch.atan2(by, bx + 1e-8)
    omega = torch.zeros_like(theta)
    omega[1:] = theta[1:] - theta[:-1]
    return omega / (2 * math.pi)


def smallest_majority_sum(values: list[float], majority_size: int) -> float:
    return float(sum(sorted(values)[:majority_size]))


@dataclass
class SafeSplitAnalysis:
    window: list[Checkpoint]
    frequency_scores: list[float]
    rotation_scores: list[float]
    frequency_majority: set[int]
    rotation_majority: set[int]
    benign_indices: set[int]
    selected_checkpoint: Checkpoint
    latest_is_benign: bool
    latest_suspiciousness: float


class SafeSplitDefense(DefenseInterface):
    def __init__(self, window_size: int, low_freq_frac: float, matrix_width: int) -> None:
        self.window_size = window_size
        self.low_freq_frac = low_freq_frac
        self.matrix_width = matrix_width

    def analyze(self, history: list[Checkpoint]) -> SafeSplitAnalysis:
        if not history:
            raise ValueError("History must contain at least one checkpoint.")

        if len(history) < self.window_size:
            latest = copy.deepcopy(history[-1])
            return SafeSplitAnalysis(
                window=list(history),
                frequency_scores=[],
                rotation_scores=[],
                frequency_majority=set(range(len(history))),
                rotation_majority=set(range(len(history))),
                benign_indices=set(range(len(history))),
                selected_checkpoint=latest,
                latest_is_benign=True,
                latest_suspiciousness=0.0,
            )

        window = history[-self.window_size:]
        majority_size = self.window_size // 2 + 1

        freq_signatures = [
            dct_low_frequency(flatten_state_dict(cp.update_state), self.matrix_width, self.low_freq_frac)
            for cp in window
        ]
        rotation_signatures = [
            rotational_signature(flatten_state_dict(cp.backbone_state), self.matrix_width) for cp in window
        ]

        frequency_scores = []
        rotation_scores = []
        for i in range(len(window)):
            freq_distances = []
            rot_distances = []
            for j in range(len(window)):
                if i == j:
                    continue
                freq_distances.append(torch.dist(freq_signatures[i], freq_signatures[j], p=2).item())
                rot_distances.append(torch.mean(torch.abs(rotation_signatures[i] - rotation_signatures[j])).item())
            frequency_scores.append(smallest_majority_sum(freq_distances, majority_size))
            rotation_scores.append(smallest_majority_sum(rot_distances, majority_size))

        frequency_majority = set(sorted(range(len(window)), key=lambda idx: frequency_scores[idx])[:majority_size])
        rotation_majority = set(sorted(range(len(window)), key=lambda idx: rotation_scores[idx])[:majority_size])
        benign_indices = frequency_majority.intersection(rotation_majority)

        selected = copy.deepcopy(window[-1])
        for local_idx in range(len(window) - 1, -1, -1):
            if local_idx in benign_indices:
                selected = copy.deepcopy(window[local_idx])
                break

        latest_idx = len(window) - 1
        latest_suspiciousness = self._normalized_suspiciousness(
            latest_idx, frequency_scores, rotation_scores
        )
        return SafeSplitAnalysis(
            window=window,
            frequency_scores=frequency_scores,
            rotation_scores=rotation_scores,
            frequency_majority=frequency_majority,
            rotation_majority=rotation_majority,
            benign_indices=benign_indices,
            selected_checkpoint=selected,
            latest_is_benign=latest_idx in benign_indices,
            latest_suspiciousness=latest_suspiciousness,
        )

    @staticmethod
    def _percentile_rank(index: int, values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        ordered = sorted(range(len(values)), key=lambda idx: values[idx])
        rank = ordered.index(index)
        return rank / (len(values) - 1)

    @classmethod
    def _normalized_suspiciousness(
        cls, index: int, frequency_scores: list[float], rotation_scores: list[float]
    ) -> float:
        if not frequency_scores or not rotation_scores:
            return 0.0
        freq_rank = cls._percentile_rank(index, frequency_scores)
        rot_rank = cls._percentile_rank(index, rotation_scores)
        return float((freq_rank + rot_rank) / 2.0)

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        return self.analyze(history).selected_checkpoint


class TemporalTrustSafeSplitDefense(SafeSplitDefense):
    def __init__(
        self,
        window_size: int,
        low_freq_frac: float,
        matrix_width: int,
        num_clients: int,
        initial_trust: float,
        reward: float,
        penalty: float,
        soft_threshold: float,
        low_threshold: float,
    ) -> None:
        super().__init__(window_size, low_freq_frac, matrix_width)
        self.trust_scores = {client_id: float(initial_trust) for client_id in range(num_clients)}
        self.reward = reward
        self.penalty = penalty
        self.soft_threshold = soft_threshold
        self.low_threshold = low_threshold
        self.decision_log: list[dict[str, object]] = []
        self.latest_decision: dict[str, object] | None = None

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        analysis = self.analyze(history)
        latest = history[-1]
        trust_before = self.trust_scores.get(latest.client_id, 1.0)
        suspiciousness = analysis.latest_suspiciousness

        if suspiciousness < self.soft_threshold and analysis.latest_is_benign:
            trust_after = min(1.0, trust_before + self.reward)
        else:
            trust_after = max(0.0, trust_before - self.penalty * max(suspiciousness, 0.25))
        self.trust_scores[latest.client_id] = trust_after

        low_trust = trust_after <= self.low_threshold
        selected = analysis.selected_checkpoint
        if low_trust:
            selected = self._select_trusted_checkpoint(analysis)

        decision = {
            "step": latest.step,
            "round": latest.round_id + 1,
            "client_id": latest.client_id,
            "suspiciousness": round(float(suspiciousness), 6),
            "trust_before": round(float(trust_before), 6),
            "trust_after": round(float(trust_after), 6),
            "latest_is_benign": analysis.latest_is_benign,
            "low_trust": low_trust,
            "selected_step": selected.step,
            "selected_client": selected.client_id,
            "frequency_score": self._score_for_latest(analysis.frequency_scores),
            "rotation_score": self._score_for_latest(analysis.rotation_scores),
        }
        self.latest_decision = decision
        self.decision_log.append(decision)
        return copy.deepcopy(selected)

    def _select_trusted_checkpoint(self, analysis: SafeSplitAnalysis) -> Checkpoint:
        for local_idx in range(len(analysis.window) - 1, -1, -1):
            checkpoint = analysis.window[local_idx]
            if local_idx not in analysis.benign_indices:
                continue
            if self.trust_scores.get(checkpoint.client_id, 1.0) > self.low_threshold:
                return copy.deepcopy(checkpoint)
        return copy.deepcopy(analysis.selected_checkpoint)

    @staticmethod
    def _score_for_latest(scores: list[float]) -> float | None:
        if not scores:
            return None
        return round(float(scores[-1]), 6)

    def get_trust_snapshot(self) -> dict[int, float]:
        return {client_id: round(float(score), 6) for client_id, score in sorted(self.trust_scores.items())}

    def get_latest_decision(self) -> dict[str, object] | None:
        return self.latest_decision
