from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from .continuous_ratchet import ContinuousRatchetSafeSplitDefense
from .safesplit import Checkpoint, load_state_dict


class ActivationClusteringDefense(ContinuousRatchetSafeSplitDefense):
    """
    Augments ContinuousRatchetSafeSplitDefense with backbone activation fingerprints.

    Each checkpoint is fingerprinted by running a fixed set of clean reference images
    through the client's head+backbone, global-avg-pooling the output → [C], then
    mean-averaging across images → a single [C] vector cached by checkpoint.step.

    A window of fingerprints is clustered (k=2) in PCA-reduced space.  Three signals
    are computed for the latest checkpoint in the window:

      A — L2 distance from its cluster centroid (normalized by window max)
      B — intra-cluster total variance for its assigned cluster (normalized)
      C — inverted silhouette score: 1 − normalized_silhouette

    activation_suspiciousness = mean(A, B, C)
    combined_suspiciousness   = max(dct_rotation_suspiciousness, activation_suspiciousness)

    The combined value replaces the raw DCT/rotation suspiciousness in all downstream
    continuous-ratchet trust and floor update logic.
    """

    def __init__(
        self,
        *args,
        reference_images: torch.Tensor,
        head_template,
        backbone_template,
        device: str,
        n_pca_components: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reference_images = reference_images          # [N, C, H, W] already on device
        self.head_template = head_template                # nn.Module, eval, on CPU
        self.backbone_template = backbone_template        # nn.Module, eval, on CPU
        self._device = torch.device(device)
        self.n_pca_components = n_pca_components
        self._fingerprint_cache: dict[int, np.ndarray] = {}  # keyed by checkpoint.step

    # ------------------------------------------------------------------
    # Fingerprint helpers
    # ------------------------------------------------------------------

    def _extract_fingerprint(self, checkpoint: Checkpoint) -> np.ndarray:
        """Forward reference images through this checkpoint's head+backbone → 1-D vector."""
        if checkpoint.step in self._fingerprint_cache:
            return self._fingerprint_cache[checkpoint.step]

        head = copy.deepcopy(self.head_template).to(self._device)
        backbone = copy.deepcopy(self.backbone_template).to(self._device)
        load_state_dict(head, checkpoint.head_state, self._device)
        load_state_dict(backbone, checkpoint.backbone_state, self._device)
        head.eval()
        backbone.eval()

        with torch.no_grad():
            smashed = head(self.reference_images)
            out = backbone(smashed)
            if out.dim() == 4:
                out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)  # [N, C]
            fingerprint = out.mean(dim=0).cpu().numpy()  # [C]

        self._fingerprint_cache[checkpoint.step] = fingerprint
        return fingerprint

    def _compute_activation_suspiciousness(
        self, window: list[Checkpoint], latest_idx: int
    ) -> float:
        """Return activation suspiciousness ∈ [0, 1] for the latest checkpoint in window."""
        fingerprints = np.stack([self._extract_fingerprint(cp) for cp in window])  # [W, C]

        n_components = min(self.n_pca_components, len(window) - 1)
        if n_components < 1:
            return 0.0

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(fingerprints)  # [W, n_components]

        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(reduced)

        # Signal A: distance from assigned cluster centroid, normalized
        centroids = km.cluster_centers_
        dists = np.linalg.norm(reduced - centroids[labels], axis=1)
        signal_a = dists / (dists.max() + 1e-8)

        # Signal B: total intra-cluster variance for each point's assigned cluster, normalized
        cluster_vars = np.zeros(len(window))
        for k in range(2):
            mask = labels == k
            if mask.sum() > 1:
                var = float(reduced[mask].var(axis=0).sum())
            else:
                var = 0.0
            cluster_vars[mask] = var
        signal_b = cluster_vars / (cluster_vars.max() + 1e-8)

        # Signal C: inverted silhouette — requires 2 <= n_labels < n_samples
        n_unique = len(np.unique(labels))
        if n_unique >= 2 and n_unique < len(window):
            sil = silhouette_samples(reduced, labels)   # [-1, 1]
            sil_norm = (sil + 1.0) / 2.0               # [0, 1]
            signal_c = 1.0 - sil_norm
        else:
            signal_c = np.zeros(len(window))

        combined = float(np.mean([signal_a[latest_idx], signal_b[latest_idx], signal_c[latest_idx]]))
        return float(np.clip(combined, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        analysis = self.analyze(history)
        window = history[-self.window_size:] if len(history) >= self.window_size else list(history)
        latest_idx = len(window) - 1

        dct_rot_susp = analysis.latest_suspiciousness

        if len(window) >= 2:
            activation_susp = self._compute_activation_suspiciousness(window, latest_idx)
        else:
            activation_susp = 0.0

        # Conservative fusion: take the stronger signal
        suspiciousness = max(dct_rot_susp, activation_susp)

        # Continuous-ratchet trust + floor logic (mirrors ContinuousRatchetSafeSplitDefense)
        latest = history[-1]
        client_id = latest.client_id
        trust_before = self.trust_scores.get(client_id, 1.0)

        self.suspiciousness_totals[client_id] += suspiciousness

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
            "dct_rot_suspiciousness": round(float(dct_rot_susp), 6),
            "activation_suspiciousness": round(float(activation_susp), 6),
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
