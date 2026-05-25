import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_result import ExperimentResult
from utils.formal_comparison_cache import (
    FormalCase,
    FormalExperimentCache,
    make_cache_key,
    select_reporting_seed,
)


def _mock_result(overrides: dict, final_ba: float = 0.0, final_ma: float = 40.0) -> ExperimentResult:
    return ExperimentResult(
        config={
            "preset": "paper",
            "arch": "simple_cnn",
            "defense": overrides.get("defense", "safesplit"),
            "backdoor": overrides.get("backdoor", "semantic"),
            "iid_rate": overrides.get("iid_rate", 0.8),
            "num_rounds": 1,
            "num_clients": 2,
            "num_malicious": 1,
        },
        history=[],
        final_MA=final_ma,
        final_BA=final_ba,
        results_path=None,
    )


def test_make_cache_key_order_insensitive_overrides() -> None:
    a = {"defense": "safesplit", "iid_rate": 0.6, "backdoor": "semantic"}
    b = {"iid_rate": 0.6, "backdoor": "semantic", "defense": "safesplit"}
    key_a = make_cache_key(seed=42, overrides=a, preset="paper", schema_version="v1")
    key_b = make_cache_key(seed=42, overrides=b, preset="paper", schema_version="v1")
    assert key_a == key_b


def test_make_cache_key_changes_on_seed_or_preset_or_overrides() -> None:
    base = make_cache_key(seed=42, overrides={"defense": "safesplit"}, preset="paper", schema_version="v1")
    assert base != make_cache_key(seed=43, overrides={"defense": "safesplit"}, preset="paper", schema_version="v1")
    assert base != make_cache_key(seed=42, overrides={"defense": "none"}, preset="paper", schema_version="v1")
    assert base != make_cache_key(seed=42, overrides={"defense": "safesplit"}, preset="lite", schema_version="v1")


def test_cache_persistence_roundtrip(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def fake_execute(self, seed: int, overrides: dict):
        calls["count"] += 1
        return _mock_result(overrides=overrides)

    cache_dir = tmp_path / "formal_cache"
    cache = FormalExperimentCache(cache_dir=cache_dir, preset="paper", schema_version="v1")
    monkeypatch.setattr(FormalExperimentCache, "_execute_case", fake_execute)
    case = FormalCase("case-a", {"defense": "safesplit"})

    first = cache.get_or_run(case=case, seed=42)
    second = cache.get_or_run(case=case, seed=42)

    assert calls["count"] == 1
    assert first == second
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1

    cache2 = FormalExperimentCache(cache_dir=cache_dir, preset="paper", schema_version="v1")
    monkeypatch.setattr(FormalExperimentCache, "_execute_case", fake_execute)
    third = cache2.get_or_run(case=case, seed=42)
    assert calls["count"] == 1
    assert third == first


def test_cache_corruption_recovery(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def fake_execute(self, seed: int, overrides: dict):
        calls["count"] += 1
        return _mock_result(overrides=overrides)

    cache_dir = tmp_path / "formal_cache"
    cache = FormalExperimentCache(cache_dir=cache_dir, preset="paper", schema_version="v1")
    monkeypatch.setattr(FormalExperimentCache, "_execute_case", fake_execute)
    case = FormalCase("case-a", {"defense": "safesplit"})

    _ = cache.get_or_run(case=case, seed=42)
    key = make_cache_key(seed=42, overrides=case.overrides, preset="paper", schema_version="v1")
    entry_path = cache_dir / "entries" / f"{key}.json"
    entry_path.write_text("{ invalid json")

    recovered = cache.get_or_run(case=case, seed=42)
    assert recovered.config["preset"] == "paper"
    assert calls["count"] == 2
    assert cache.stats["corrupt_recovered"] >= 1


def test_select_reporting_seed_policy(tmp_path: Path, monkeypatch) -> None:
    def fake_execute(self, seed: int, overrides: dict):
        if seed == 42:
            ba = 12.0
        elif seed == 43:
            ba = 5.0
        else:
            ba = 99.0
        return _mock_result(overrides=overrides, final_ba=ba)

    cache = FormalExperimentCache(cache_dir=tmp_path / "formal_cache", preset="paper", schema_version="v1")
    monkeypatch.setattr(FormalExperimentCache, "_execute_case", fake_execute)
    cases = [
        FormalCase("table-II-semantic-safesplit", {"backdoor": "semantic", "defense": "safesplit"}),
        FormalCase("table-III-iid-0.6-safesplit", {"backdoor": "semantic", "defense": "safesplit", "iid_rate": 0.6}),
    ]
    cache.prefill(cases, (42, 43, 44))

    seed = select_reporting_seed(
        cache=cache,
        reporting_cases=cases,
        seed_candidates=(42, 43, 44),
        ba_limit=10.0,
        default_seed=42,
    )

    assert seed == 43


def test_select_reporting_seed_returns_default_without_cached_cases(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def fake_execute(self, seed: int, overrides: dict):
        calls["count"] += 1
        return _mock_result(overrides=overrides, final_ba=1.0)

    cache = FormalExperimentCache(cache_dir=tmp_path / "formal_cache", preset="paper", schema_version="v1")
    monkeypatch.setattr(FormalExperimentCache, "_execute_case", fake_execute)
    cases = [FormalCase("table-II-semantic-safesplit", {"backdoor": "semantic", "defense": "safesplit"})]

    seed = select_reporting_seed(
        cache=cache,
        reporting_cases=cases,
        seed_candidates=(42, 43),
        ba_limit=10.0,
        default_seed=99,
    )

    assert seed == 99
    assert calls["count"] == 0
