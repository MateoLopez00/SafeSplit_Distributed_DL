from __future__ import annotations

import copy

from .safesplit import (
    Checkpoint,
    SafeSplitAnalysis,
    TemporalTrustSafeSplitDefense,
)


class ContinuousRatchetSafeSplitDefense(TemporalTrustSafeSplitDefense):
    """
    Continuous suspiciousness accumulation variant of the ratchet defense.

    Unlike the binary ratchet (which only increments when suspiciousness >= 0.60),
    this variant adds the raw suspiciousness score to each client's running total
    every round, regardless of threshold. The penalty floor is then proportional
    to that total.

    This directly catches slow poisoning: a client ramping PDR gradually will
    accumulate a higher total than a consistently benign client, so its floor
    rises across rounds even while individual scores stay below the 0.60 flag
    threshold.

    Decay: on clearly-clean updates (suspiciousness < ratchet_decay_threshold AND
    latest_is_benign), a fixed amount is subtracted from the total so genuinely
    benign clients do not drift upward indefinitely.

    Net per-round effect:
      - Benign at susp~0.15, clearly_clean:  0.15 - 0.20 = -0.05  (floor drifts down)
      - Slow poisoner at susp~0.35, barely clean: 0.35 - 0.20 = +0.15  (floor creeps up)
      - Slow poisoner at susp~0.45, not clearly clean: +0.45  (floor rises faster)
      - Flagged at susp~0.80:  +0.80  (floor spikes)
    """

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
        floor_base: float,
        ratchet_step: float,
        ratchet_max_floor: float,
        ratchet_decay: float,
        ratchet_decay_threshold: float,
    ) -> None:
        super().__init__(
            window_size=window_size,
            low_freq_frac=low_freq_frac,
            matrix_width=matrix_width,
            num_clients=num_clients,
            initial_trust=initial_trust,
            reward=reward,
            penalty=penalty,
            soft_threshold=soft_threshold,
            low_threshold=low_threshold,
        )
        self.floor_base = floor_base
        self.ratchet_step = ratchet_step
        self.ratchet_max_floor = ratchet_max_floor
        self.ratchet_decay = ratchet_decay
        self.ratchet_decay_threshold = ratchet_decay_threshold
        self.suspiciousness_totals: dict[int, float] = {cid: 0.0 for cid in range(num_clients)}

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        analysis = self.analyze(history)
        latest = history[-1]
        client_id = latest.client_id
        trust_before = self.trust_scores.get(client_id, 1.0)
        suspiciousness = analysis.latest_suspiciousness

        # Always accumulate raw suspiciousness — no binary threshold.
        self.suspiciousness_totals[client_id] += suspiciousness

        # Subtract fixed decay only when clearly clean, so benign clients don't drift up.
        clearly_clean = suspiciousness < self.ratchet_decay_threshold and analysis.latest_is_benign
        if clearly_clean:
            self.suspiciousness_totals[client_id] = max(
                0.0, self.suspiciousness_totals[client_id] - self.ratchet_decay
            )

        floor = min(
            self.ratchet_max_floor,
            self.floor_base + self.ratchet_step * self.suspiciousness_totals[client_id],
        )

        flagged = suspiciousness >= self.soft_threshold or not analysis.latest_is_benign
        if not flagged:
            trust_after = min(1.0, trust_before + self.reward)
        else:
            trust_after = max(0.0, trust_before - self.penalty * max(suspiciousness, floor))

        self.trust_scores[client_id] = trust_after

        low_trust = trust_after <= self.low_threshold
        selected = analysis.selected_checkpoint
        if low_trust:
            selected = self._select_trusted_checkpoint(analysis)

        decision = {
            "step": latest.step,
            "round": latest.round_id + 1,
            "client_id": client_id,
            "suspiciousness": round(float(suspiciousness), 6),
            "trust_before": round(float(trust_before), 6),
            "trust_after": round(float(trust_after), 6),
            "latest_is_benign": analysis.latest_is_benign,
            "low_trust": low_trust,
            "selected_step": selected.step,
            "selected_client": selected.client_id,
            "frequency_score": self._score_for_latest(analysis.frequency_scores),
            "rotation_score": self._score_for_latest(analysis.rotation_scores),
            "suspiciousness_total": round(float(self.suspiciousness_totals[client_id]), 4),
            "ratchet_floor": round(float(floor), 4),
        }
        self.latest_decision = decision
        self.decision_log.append(decision)
        return copy.deepcopy(selected)

    def get_trust_snapshot(self) -> dict[int, float]:
        return {cid: round(float(score), 6) for cid, score in sorted(self.trust_scores.items())}
