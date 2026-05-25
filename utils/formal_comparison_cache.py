from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runner.experiment_runner import run_experiment
from utils.display import result_to_row, show_rows
from utils.experiment_request import ExperimentRequest
from utils.experiment_result import ExperimentResult, ensure_experiment_result
from utils.plot import plot_metric_summary


@dataclass(frozen=True)
class FormalCase:
    label: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class SeedSelectionSummary:
    seed: int
    max_safesplit_ba: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "max_safesplit_BA": round(float(self.max_safesplit_ba), 4),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class CacheEntry:
    key: str
    path: str
    created_at: str
    schema_version: str
    preset: str
    seed: int
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "preset": self.preset,
            "seed": int(self.seed),
            "overrides": _normalize_for_json(self.overrides),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CacheEntry":
        return cls(
            key=str(payload["key"]),
            path=str(payload["path"]),
            created_at=str(payload["created_at"]),
            schema_version=str(payload["schema_version"]),
            preset=str(payload["preset"]),
            seed=int(payload["seed"]),
            overrides=dict(payload.get("overrides", {})),
        )


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, set):
        return [_normalize_for_json(item) for item in sorted(value, key=repr)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def make_cache_key(seed: int, overrides: dict[str, Any], preset: str, schema_version: str) -> str:
    payload = {
        "seed": int(seed),
        "preset": str(preset),
        "schema_version": str(schema_version),
        "overrides": _normalize_for_json(overrides),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class FormalExperimentCache:
    def __init__(
        self,
        cache_dir: Path | str,
        preset: str,
        schema_version: str = "v1",
        *,
        device: str | None = None,
        write_json: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.entries_dir = self.cache_dir / "entries"
        self.index_path = self.cache_dir / "index.json"
        self.preset = str(preset)
        self.schema_version = str(schema_version)
        self.device = device
        self.write_json = bool(write_json)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.entries_dir.mkdir(parents=True, exist_ok=True)

        self._index: dict[str, CacheEntry] = self._load_index()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "recomputed": 0,
            "corrupt_recovered": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def clear_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0

    def clear(self) -> None:
        self._index = {}
        self._write_index()

    def prefill(self, cases: list[FormalCase], seeds: tuple[int, ...] | list[int]) -> None:

        total = len(cases) * len(seeds)
        print(f"Prefilling formal experiment cache ({total} experiments)...")

        completed = 0
        initial_stats = self.stats

        for seed in seeds:

            print(f"\nSeed {seed}")

            for case in cases:

                already_cached = self.has_entry(case=case, seed=int(seed))

                status = "HIT" if already_cached else "RUN"
                print(f"[{completed + 1}/{total}] {status:<4} {case.label}, seed={seed}")

                self.get_or_run(case=case, seed=int(seed))
                completed += 1

        final_stats = self.stats

        new_hits = final_stats["hits"]- initial_stats["hits"]
        new_misses = final_stats["misses"]- initial_stats["misses"]
        new_recomputed = final_stats["recomputed"] - initial_stats["recomputed"]

        print("\nPrefill completed. Summary:"
            f"\n  hits:        {new_hits}"
            f"\n  misses:      {new_misses}"
            f"\n  recomputed:  {new_recomputed}"
        )
    def get_or_run(self, case: FormalCase, seed: int) -> ExperimentResult:
        key = make_cache_key(seed=seed, overrides=case.overrides, preset=self.preset, schema_version=self.schema_version)

        cached = self._read_entry_if_valid(key)
        if cached is not None:
            self._stats["hits"] += 1
            return cached

        self._stats["misses"] += 1
        result = self._execute_case(seed=seed, overrides=case.overrides)
        self._stats["recomputed"] += 1
        self._write_entry(key=key, case=case, seed=seed, result=result)
        return result

    def _execute_case(self, seed: int, overrides: dict[str, Any]) -> ExperimentResult:
        request = ExperimentRequest(
            preset=self.preset,
            seed=int(seed),
            device=self.device,
            write_json=self.write_json,
            **overrides,
        )
        return ensure_experiment_result(run_experiment(request))

    def _cache_key_for(self, seed: int, overrides: dict[str, Any]) -> str:
        return make_cache_key(seed=seed, overrides=overrides, preset=self.preset, schema_version=self.schema_version)

    def has_entry(self, case: FormalCase, seed: int) -> bool:
        return self._read_entry_if_valid(self._cache_key_for(seed=seed, overrides=case.overrides)) is not None

    def get_cached(self, case: FormalCase, seed: int) -> ExperimentResult | None:
        return self._read_entry_if_valid(self._cache_key_for(seed=seed, overrides=case.overrides))

    def _load_index(self) -> dict[str, CacheEntry]:
        if not self.index_path.exists():
            return {}
        try:
            raw = json.loads(self.index_path.read_text())
            entries = raw.get("entries", {})
            if isinstance(entries, dict):
                parsed: dict[str, CacheEntry] = {}
                for key, payload in entries.items():
                    if isinstance(payload, dict):
                        parsed[str(key)] = CacheEntry.from_dict(payload)
                return parsed
        except Exception:
            pass
        return {}

    def _write_index(self) -> None:
        serialized = {key: entry.to_dict() for key, entry in self._index.items()}
        self._atomic_write_json(self.index_path, {"entries": serialized})

    def _read_entry_if_valid(self, key: str) -> ExperimentResult | None:
        entry = self._index.get(key)
        if not entry:
            return None

        entry_path = self.cache_dir / entry.path
        try:
            raw = json.loads(entry_path.read_text())
            if raw.get("schema_version") != self.schema_version:
                return None
            if raw.get("preset") != self.preset:
                return None
            if raw.get("key") != key:
                self._stats["corrupt_recovered"] += 1
                return None
            result = raw.get("result")
            if not isinstance(result, dict):
                self._stats["corrupt_recovered"] += 1
                return None
            return ensure_experiment_result(result)
        except Exception:
            self._stats["corrupt_recovered"] += 1
            return None

    def _write_entry(self, key: str, case: FormalCase, seed: int, result: ExperimentResult) -> None:
        rel_path = Path("entries") / f"{key}.json"
        abs_path = self.cache_dir / rel_path

        payload = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": self.schema_version,
            "preset": self.preset,
            "seed": int(seed),
            "label": case.label,
            "overrides": _normalize_for_json(case.overrides),
            "result": result.to_dict(),
        }
        self._atomic_write_json(abs_path, payload)

        self._index[key] = CacheEntry(
            key=key,
            path=rel_path.as_posix(),
            created_at=payload["created_at"],
            schema_version=self.schema_version,
            preset=self.preset,
            seed=int(seed),
            overrides=payload["overrides"],
        )
        self._write_index()

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(path)


def run_formal_experiment(
    cache: FormalExperimentCache,
    label: str,
    seed: int,
    overrides: dict[str, Any],
) -> tuple[ExperimentResult, dict[str, Any]]:
    result = cache.get_or_run(FormalCase(label=label, overrides=overrides), seed=seed)
    row = {"label": label, **result_to_row(result)}
    return result, row


def select_reporting_seed(
    cache: FormalExperimentCache,
    reporting_cases: list[FormalCase],
    seed_candidates: tuple[int, ...] | list[int],
    ba_limit: float,
    default_seed: int,
) -> int:
    seed_summary: list[dict[str, Any]] = []
    total_expected = len(reporting_cases) * len(seed_candidates)
    available_total = 0

    for seed in seed_candidates:
        seed_rows = []
        for case in reporting_cases:
            cached = cache.get_cached(case=case, seed=seed)
            if cached is None:
                continue
            row = {"label": f"{case.label}-seed-{seed}", **result_to_row(cached)}
            seed_rows.append(row)
            available_total += 1

        if not seed_rows:
            continue

        max_ba = max(float(row["final_BA"]) for row in seed_rows)
        passed = max_ba <= ba_limit
        summary = SeedSelectionSummary(seed=int(seed), max_safesplit_ba=max_ba, passed=passed)
        seed_summary.append(summary.to_dict())
        if passed:
            print(f"Selected reporting seed: {seed}")
            show_rows(seed_summary, max_rows=len(seed_summary))
            return int(seed)

    print(
        f"Warning: select_reporting_seed has {available_total}/{total_expected} cached cases available."
    )
    if available_total == 0:
        print("No cached reporting cases found. Returning default seed.")
        return int(default_seed)

    print("No reporting seed passed the SafeSplit BA limit. Showing candidates for inspection.")
    show_rows(seed_summary, max_rows=len(seed_summary))
    return int(default_seed)


def run_case_group(
    cache: FormalExperimentCache,
    cases: list[FormalCase],
    title: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        _, row = run_formal_experiment(
            cache=cache,
            label=case.label,
            seed=seed,
            overrides=case.overrides,
        )
        rows.append(row)

    print(title)
    show_rows(rows, max_rows=len(rows))
    plot_metric_summary(rows, title)
    print(f"Cache stats: {cache.stats}")
    return rows
