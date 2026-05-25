from __future__ import annotations

import argparse
import json

import config as cfg
from runner.experiment_runner import run_experiment
from utils.experiment_request import ExperimentRequest


def parse_args():
    parser = argparse.ArgumentParser(description="SafeSplit midterm-scope reproduction")
    parser.add_argument("--preset", choices=sorted(cfg.EXPERIMENT_PRESETS), default=None)
    parser.add_argument("--arch", default=None, choices=["resnet18", "simple_cnn"])
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-malicious", type=int, default=None)
    parser.add_argument("--iid-rate", type=float, default=None)
    parser.add_argument("--defense", default="safesplit",
                        choices=["none", "safesplit", "safesplit_trust", "dp", "krum"])
    parser.add_argument("--backdoor", default=cfg.BACKDOOR_TYPE, choices=["pixel", "semantic", "none"])
    parser.add_argument("--attack-schedule", default=None, choices=["static", "slow"])
    parser.add_argument("--slow-pdr-start", type=float, default=None)
    parser.add_argument("--slow-pdr-end", type=float, default=None)
    parser.add_argument("--slow-ramp-rounds", type=int, default=None)
    parser.add_argument("--pdr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--out-dir", default=str(cfg.RESULTS_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    preset = "lite" if args.fast_dev_run else args.preset

    request = ExperimentRequest(
        preset=preset,
        arch=args.arch,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        num_malicious=args.num_malicious,
        iid_rate=args.iid_rate,
        defense=args.defense,
        backdoor=args.backdoor,
        pdr=args.pdr,
        device=args.device,
        seed=args.seed,
        max_samples_per_client=args.max_samples_per_client,
        out_dir=args.out_dir,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        attack_schedule=args.attack_schedule,
        slow_pdr_start=args.slow_pdr_start,
        slow_pdr_end=args.slow_pdr_end,
        slow_ramp_rounds=args.slow_ramp_rounds,
    )

    result = run_experiment(request)
    print(
        json.dumps(
            {
                "final_MA": result.final_MA,
                "final_BA": result.final_BA,
                "results": result.results_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
