from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from config import PresetType
from main import build_experiment_request, run_experiment


TABLE_II_RUNS = [
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", "none"],
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", "safesplit"],
    ["--arch", "resnet18", "--backdoor", "pixel", "--defense", "none"],
    ["--arch", "resnet18", "--backdoor", "pixel", "--defense", "safesplit"],
]

TABLE_III_RUNS = [
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", defense, "--iid-rate", str(iid)]
    for iid in (0.6, 0.8, 1.0)
    for defense in ("none", "safesplit")
]

BASELINE_RUNS = [
    ["--arch", "resnet18", "--backdoor", "semantic", "--defense", defense, "--iid-rate", "0.6"]
    for defense in ("none", "dp", "krum", "safesplit")
]


def cli_args_to_overrides(args: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    idx = 0
    while idx < len(args):
        key = args[idx]
        if key == "--arch":
            overrides["arch"] = args[idx + 1]
            idx += 2
        elif key == "--backdoor":
            overrides["backdoor"] = args[idx + 1]
            idx += 2
        elif key == "--defense":
            overrides["defense"] = args[idx + 1]
            idx += 2
        elif key == "--iid-rate":
            overrides["iid_rate"] = float(args[idx + 1])
            idx += 2
        elif key == "--num-rounds":
            overrides["num_rounds"] = int(args[idx + 1])
            idx += 2
        elif key == "--num-clients":
            overrides["num_clients"] = int(args[idx + 1])
            idx += 2
        elif key == "--num-malicious":
            overrides["num_malicious"] = int(args[idx + 1])
            idx += 2
        elif key == "--max-samples-per-client":
            overrides["max_samples_per_client"] = int(args[idx + 1])
            idx += 2
        elif key == "--local-epochs":
            overrides["local_epochs"] = int(args[idx + 1])
            idx += 2
        elif key == "--batch-size":
            overrides["batch_size"] = int(args[idx + 1])
            idx += 2
        elif key == "--eval-batch-size":
            overrides["eval_batch_size"] = int(args[idx + 1])
            idx += 2
        elif key == "--pdr":
            overrides["pdr"] = float(args[idx + 1])
            idx += 2
        elif key == "--device":
            overrides["device"] = args[idx + 1]
            idx += 2
        elif key == "--seed":
            overrides["seed"] = int(args[idx + 1])
            idx += 2
        elif key == "--preset":
            preset_str = args[idx + 1]
            overrides["preset"] = PresetType(preset_str)
            idx += 2
        elif key == "--fast-dev-run":
            overrides["preset"] = PresetType.LITE
            idx += 1
        else:
            raise ValueError(f"Unsupported experiment argument: {key}")
    return overrides


def run_one(args: list[str], preset: PresetType | None = None) -> dict[str, object]:
    overrides = cli_args_to_overrides(args)
    effective_preset = cast(PresetType | None, overrides.pop("preset", preset))
    request = build_experiment_request(preset=effective_preset, **overrides)
    return run_experiment(request)


def main():
    parser = argparse.ArgumentParser(description="Run the SafeSplit midterm experiment matrix.")
    parser.add_argument("--out", default="results/experiment_index.json")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--preset", choices=[p.value for p in PresetType], default=None)
    args = parser.parse_args()

    experiment_sets = {
        "table_ii": TABLE_II_RUNS,
        "table_iii": TABLE_III_RUNS,
        "baseline_panel": BASELINE_RUNS,
    }

    outputs: dict[str, list[dict]] = {}
    for name, run_list in experiment_sets.items():
        outputs[name] = []
        for run_args in run_list:
            effective_args = list(run_args)
            if args.fast_dev_run:
                effective_args.append("--fast-dev-run")
            result = run_one(effective_args, preset=args.preset)
            outputs[name].append({"args": effective_args, "result": result})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2))
    print(json.dumps({"index": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
