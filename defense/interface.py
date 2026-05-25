from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .safesplit import Checkpoint


class DefenseInterface(ABC):
    @abstractmethod
    def select_checkpoint(self, history: list[Checkpoint]) -> Checkpoint:
        """Select the checkpoint to keep after a client update."""

    def get_latest_decision(self) -> dict[str, object] | None:
        """Optional structured debug/analysis record for the latest selection."""
        return None

    def get_trust_snapshot(self) -> dict[int, float] | None:
        """Optional trust-state snapshot for defenses that track client trust."""
        return None

