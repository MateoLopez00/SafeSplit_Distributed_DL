from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Summarize SafeSplit experiment outputs.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "experiment_index.json":
            continue
        payload = load_json(path)
        cfg = payload["config"]
        rows.append(
            {
                "file": path.name,
                "defense": cfg["defense"],
                "backdoor": cfg["backdoor"],
                "iid_rate": cfg["iid_rate"],
                "final_MA": round(payload["final_MA"], 3),
                "final_BA": round(payload["final_BA"], 3),
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
