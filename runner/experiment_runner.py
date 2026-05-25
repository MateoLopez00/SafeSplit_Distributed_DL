from __future__ import annotations

import json
import random
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
from utils.experiment_request import ExperimentRequest
from utils.experiment_result import ExperimentResult


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


def build_output_path(request: ExperimentRequest) -> Path:
    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iid_token = str(request.iid_rate).replace(".", "p")
    out_name = (
        f"{cfg.DATASET.lower()}_{request.arch}_{request.defense}_{request.backdoor}_{request.attack_schedule}_"
        f"{request.preset}_iid{iid_token}_clients{request.num_clients}_mal{request.num_malicious}.json"
    )
    return out_dir / out_name


def run_experiment(request: ExperimentRequest) -> ExperimentResult:
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
                dataset = ScheduledPoisonedDataset(
                    subset,
                    attack,
                    request.slow_pdr_end,
                    seed=request.seed + client_id
                )
                dataset.set_poisoned_data_rate(request.slow_pdr_start)
            else:
                dataset = PoisonedDataset(subset, attack, request.pdr, seed=request.seed + client_id)
        client_loaders.append(DataLoader(dataset, batch_size=request.batch_size, shuffle=True, num_workers=0))

    test_loader = DataLoader(test_dataset, batch_size=request.eval_batch_size, shuffle=False, num_workers=0)
    trigger_set = [] if attack is None else attack.build_backdoor_test_set(test_dataset)

    head0, backbone0, tail0 = get_split_model(request.arch, cfg.NUM_CLASSES)

    defense = build_defense(request.defense, request.num_clients)

    trainer = SplitLearningTrainer(
        head_initial=head0,
        backbone_initial=backbone0,
        tail_initial=tail0,
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
    history_payload = [item.to_dict() for item in history]
    final_ma = evaluate_model(trainer.current_head, trainer.current_backbone, trainer.current_tail, test_loader, device)
    final_ba = evaluate_backdoor(trainer.current_head, trainer.current_backbone,
                                 trainer.current_tail, trigger_set, device)

    config_payload = request.to_dict()
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
    output_dict = {
        "config": config_payload,
        "history": history_payload,
        "final_MA": final_ma,
        "final_BA": final_ba,
        "results_path": None,
    }
    if request.write_json:
        out_path = build_output_path(request)
        output_dict["results_path"] = str(out_path)
        out_path.write_text(json.dumps(output_dict, indent=2))
    return ExperimentResult.from_dict(output_dict)
