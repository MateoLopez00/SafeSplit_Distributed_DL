from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from data import (
    PixelTriggerAttack,
    PoisonedDataset,
    ScheduledPoisonedDataset,
    SemanticTriggerAttack,
    build_client_subsets,
    load_datasets,
    partition_data,
)
from defense import DifferentialPrivacyDefense, KrumStyleDefense, SafeSplitDefense, TemporalTrustSafeSplitDefense
from evaluate import evaluate_backdoor, evaluate_model
from models import get_split_model
from training import SplitLearningTrainer


@dataclass(slots=True)
class ExperimentRequest:
    preset: str = cfg.DEFAULT_PRESET
    arch: str = cfg.ARCH
    num_rounds: int = cfg.NUM_ROUNDS
    num_clients: int = cfg.NUM_CLIENTS
    num_malicious: int = cfg.NUM_MALICIOUS
    iid_rate: float = cfg.IID_RATE
    defense: str = "safesplit"
    backdoor: str = cfg.BACKDOOR_TYPE
    pdr: float = cfg.POISONED_DATA_RATE
    device: str | None = None
    seed: int = cfg.SEED
    max_samples_per_client: int | None = None
    out_dir: str = str(cfg.RESULTS_DIR)
    write_json: bool = True
    local_epochs: int = cfg.LOCAL_EPOCHS
    batch_size: int = cfg.BATCH_SIZE
    eval_batch_size: int = cfg.EVAL_BATCH_SIZE
    attack_schedule: str = cfg.ATTACK_SCHEDULE
    slow_pdr_start: float = cfg.SLOW_POISON_START_PDR
    slow_pdr_end: float = cfg.SLOW_POISON_END_PDR
    slow_ramp_rounds: int = cfg.SLOW_POISON_RAMP_ROUNDS


def experiment_request_to_dict(request: ExperimentRequest) -> dict[str, object]:
    return asdict(request)


def build_experiment_request(preset: str | None = None, **overrides) -> ExperimentRequest:
    preset_values = cfg.resolve_preset(preset)
    request_data = {
        "preset": str(preset_values["name"]),
        "arch": str(preset_values["arch"]),
        "num_rounds": int(preset_values["num_rounds"]),
        "num_clients": int(preset_values["num_clients"]),
        "num_malicious": int(preset_values["num_malicious"]),
        "iid_rate": float(preset_values["iid_rate"]),
        "defense": "safesplit",
        "backdoor": cfg.BACKDOOR_TYPE,
        "pdr": cfg.POISONED_DATA_RATE,
        "device": None,
        "seed": cfg.SEED,
        "max_samples_per_client": preset_values["max_samples_per_client"],
        "out_dir": str(cfg.RESULTS_DIR),
        "write_json": True,
        "local_epochs": int(preset_values["local_epochs"]),
        "batch_size": int(preset_values["batch_size"]),
        "eval_batch_size": int(preset_values["eval_batch_size"]),
        "attack_schedule": cfg.ATTACK_SCHEDULE,
        "slow_pdr_start": cfg.SLOW_POISON_START_PDR,
        "slow_pdr_end": cfg.SLOW_POISON_END_PDR,
        "slow_ramp_rounds": int(preset_values["num_rounds"]),
    }
    for key, value in overrides.items():
        if value is not None:
            request_data[key] = value
    return ExperimentRequest(**request_data)


def parse_args():
    parser = argparse.ArgumentParser(description="SafeSplit midterm-scope reproduction")
    parser.add_argument("--preset", choices=sorted(cfg.EXPERIMENT_PRESETS), default=None)
    parser.add_argument("--arch", default=None, choices=["resnet18", "simple_cnn"])
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-malicious", type=int, default=None)
    parser.add_argument("--iid-rate", type=float, default=None)
    parser.add_argument("--defense", default="safesplit", choices=["none", "safesplit", "safesplit_trust", "dp", "krum"])
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_attack(name: str):
    if name == "pixel":
        return PixelTriggerAttack(
            trigger_size=cfg.TRIGGER_SIZE,
            position=cfg.TRIGGER_POS,
            target_label=cfg.PIXEL_TARGET_LABEL,
        )
    if name == "semantic":
        return SemanticTriggerAttack(
            source_label=cfg.SEMANTIC_SOURCE_LABEL,
            target_label=cfg.SEMANTIC_TARGET_LABEL,
        )
    return None


def build_defense(name: str, num_clients: int):
    if name == "safesplit":
        return SafeSplitDefense(
            window_size=num_clients,
            low_freq_frac=cfg.DCT_LOW_FREQ_FRAC,
            matrix_width=cfg.ROTATION_MATRIX_WIDTH,
        )
    if name == "dp":
        return DifferentialPrivacyDefense(cfg.DP_CLIP_NORM, cfg.DP_NOISE_SCALE)
    if name == "krum":
        return KrumStyleDefense(window_size=num_clients)
    if name == "safesplit_trust":
        return TemporalTrustSafeSplitDefense(
            window_size=num_clients,
            low_freq_frac=cfg.DCT_LOW_FREQ_FRAC,
            matrix_width=cfg.ROTATION_MATRIX_WIDTH,
            num_clients=num_clients,
            initial_trust=cfg.TRUST_INITIAL,
            reward=cfg.TRUST_REWARD,
            penalty=cfg.TRUST_PENALTY,
            soft_threshold=cfg.TRUST_SOFT_THRESHOLD,
            low_threshold=cfg.TRUST_LOW_THRESHOLD,
        )
    return None


def build_experiment_request_from_args(args: argparse.Namespace) -> ExperimentRequest:
    preset = "lite" if args.fast_dev_run else args.preset
    return build_experiment_request(
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


def build_output_path(request: ExperimentRequest) -> Path:
    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iid_token = str(request.iid_rate).replace(".", "p")
    out_name = (
        f"{cfg.DATASET.lower()}_{request.arch}_{request.defense}_{request.backdoor}_{request.attack_schedule}_"
        f"{request.preset}_iid{iid_token}_clients{request.num_clients}_mal{request.num_malicious}.json"
    )
    return out_dir / out_name


def run_experiment(request: ExperimentRequest) -> dict[str, object]:
    set_seed(request.seed)
    device = cfg.resolve_device(request.device)
    train_dataset, test_dataset = load_datasets(cfg.DATA_DIR)
    partitions = partition_data(
        train_dataset,
        num_clients=request.num_clients,
        iid_rate=request.iid_rate,
        num_classes=cfg.NUM_CLASSES,
        seed=request.seed,
        max_samples_per_client=request.max_samples_per_client,
    )
    client_subsets = build_client_subsets(train_dataset, partitions)

    attack = build_attack(request.backdoor)
    malicious_ids = (
        set(random.sample(range(request.num_clients), request.num_malicious)) if attack is not None else set()
    )

    client_loaders = []
    for client_id, subset in enumerate(client_subsets):
        dataset = subset
        if client_id in malicious_ids and attack is not None:
            if request.attack_schedule == "slow":
                dataset = ScheduledPoisonedDataset(subset, attack, request.slow_pdr_end, seed=request.seed + client_id)
                dataset.set_poisoned_data_rate(request.slow_pdr_start)
            else:
                dataset = PoisonedDataset(subset, attack, request.pdr, seed=request.seed + client_id)
        client_loaders.append(DataLoader(dataset, batch_size=request.batch_size, shuffle=True, num_workers=0))

    test_loader = DataLoader(test_dataset, batch_size=request.eval_batch_size, shuffle=False, num_workers=0)
    trigger_set = [] if attack is None else attack.build_backdoor_test_set(test_dataset)

    head0, backbone0, tail0 = get_split_model(request.arch, cfg.NUM_CLASSES)
    defense = build_defense(request.defense, request.num_clients)
    trainer = SplitLearningTrainer(
        head0=head0,
        backbone0=backbone0,
        tail0=tail0,
        client_loaders=client_loaders,
        device=device,
        lr=cfg.LR,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
        local_epochs=request.local_epochs,
        defense=defense,
        poison_schedule={
            "start_pdr": request.slow_pdr_start,
            "end_pdr": request.slow_pdr_end,
            "ramp_rounds": request.slow_ramp_rounds,
        }
        if request.attack_schedule == "slow"
        else None,
    )

    history = trainer.run(num_rounds=request.num_rounds, test_loader=test_loader, trigger_set=trigger_set)
    final_ma = evaluate_model(trainer.current_head, trainer.current_backbone, trainer.current_tail, test_loader, device)
    final_ba = evaluate_backdoor(trainer.current_head, trainer.current_backbone, trainer.current_tail, trigger_set, device)

    config_payload = experiment_request_to_dict(request)
    config_payload.update(
        {
            "dataset": cfg.DATASET,
            "device": device,
            "malicious_ids": sorted(malicious_ids),
            "trust_parameters": {
                "initial": cfg.TRUST_INITIAL,
                "reward": cfg.TRUST_REWARD,
                "penalty": cfg.TRUST_PENALTY,
                "soft_threshold": cfg.TRUST_SOFT_THRESHOLD,
                "low_threshold": cfg.TRUST_LOW_THRESHOLD,
            },
        }
    )
    output = {
        "config": config_payload,
        "history": history,
        "final_MA": final_ma,
        "final_BA": final_ba,
    }
    if request.write_json:
        out_path = build_output_path(request)
        out_path.write_text(json.dumps(output, indent=2))
        output["results_path"] = str(out_path)
    else:
        output["results_path"] = None
    return output


def main():
    args = parse_args()
    request = build_experiment_request_from_args(args)
    result = run_experiment(request)
    print(
        json.dumps(
            {
                "final_MA": result["final_MA"],
                "final_BA": result["final_BA"],
                "results": result["results_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
