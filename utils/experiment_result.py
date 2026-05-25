from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExperimentResult:
    config: dict[str, Any]
    history: list[dict[str, Any]]
    final_MA: float
    final_BA: float
    results_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "history": self.history,
            "final_MA": float(self.final_MA),
            "final_BA": float(self.final_BA),
            "results_path": self.results_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentResult":
        return cls(
            config=dict(payload.get("config", {})),
            history=list(payload.get("history", [])),
            final_MA=float(payload.get("final_MA", 0.0)),
            final_BA=float(payload.get("final_BA", 0.0)),
            results_path=payload.get("results_path"),
        )

    # Compatibility helpers so existing notebook/dict-style code keeps working.
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def ensure_experiment_result(value: ExperimentResult | dict[str, Any]) -> ExperimentResult:
    if isinstance(value, ExperimentResult):
        return value
    return ExperimentResult.from_dict(value)
