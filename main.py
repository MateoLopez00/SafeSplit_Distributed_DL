"""
main.py – Entry point for SafeSplit experiments.

Usage examples
--------------
# Default: CIFAR-10, ResNet-18, SafeSplit, semantic backdoor
python main.py

# Pixel trigger, no defense
python main.py --dataset CIFAR10 --backdoor pixel --defense none

# MNIST, pixel trigger, SafeSplit
python main.py --dataset MNIST --backdoor pixel --defense safesplit

# Vary IID rate
python main.py --iid_rate 0.6

# Vary number of clients
python main.py --num_clients 20 --num_malicious 4

# Compare baselines
python main.py --defense krum
python main.py --defense dp
python main.py --defense freqfed

Experiment results are printed to stdout and written to results/<run_name>.json.
"""


# Run python main.py --help to see all experiment options, or see README.md for commands to reproduce each table/figure from the paper.

import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
from defense import ActivationClusteringDefense, ContinuousRatchetSafeSplitDefense, DifferentialPrivacyDefense, KrumStyleDefense, RatchetTemporalTrustSafeSplitDefense, SafeSplitDefense, TemporalTrustSafeSplitDefense
from evaluate import evaluate_backdoor, evaluate_model
from models import get_split_model
from training import SplitLearningTrainer
from evaluate import evaluate_model, evaluate_backdoor


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SafeSplit midterm-scope reproduction")
    parser.add_argument("--preset", choices=sorted(cfg.EXPERIMENT_PRESETS), default=None)
    parser.add_argument("--arch", default=None, choices=["resnet18", "simple_cnn"])
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-malicious", type=int, default=None)
    parser.add_argument("--iid-rate", type=float, default=None)
    parser.add_argument("--defense", default="safesplit", choices=["none", "safesplit", "safesplit_trust", "safesplit_trust_ratchet", "safesplit_continuous_ratchet", "safesplit_activation_clustering", "dp", "krum"])
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


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available – using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_defense(defense_name: str, num_clients: int):
    if defense_name == "safesplit":
        return SafeSplit(num_clients, low_freq_frac=cfg.DCT_LOW_FREQ_FRAC)
    elif defense_name == "krum":
        return KrumDefense(num_clients)
    elif defense_name == "dp":
        return DifferentialPrivacyDefense(cfg.DP_CLIP_NORM, cfg.DP_NOISE_SCALE)
    elif defense_name == "freqfed":
        return FreqFedDefense(num_clients, low_freq_frac=cfg.DCT_LOW_FREQ_FRAC)
    else:
        return None


def build_attack(backdoor_type: str, dataset: str, pdr: float, target_label: int):
    if backdoor_type == "pixel":
        return PixelTriggerAttack(
            trigger_size=cfg.TRIGGER_SIZE,
            position=cfg.TRIGGER_POS,
            dataset=dataset,
            target_label=target_label,
            poisoned_data_rate=pdr,
        )
    elif backdoor_type == "semantic":
        return SemanticTriggerAttack(
            source_label=cfg.SEMANTIC_SOURCE_LABEL,
            target_label=cfg.SEMANTIC_TARGET_LABEL,
            poisoned_data_rate=pdr,
        )
    else:
        return None


def _build_reference_images(dataset, n: int, device: str) -> torch.Tensor:
    imgs = torch.stack([dataset[i][0] for i in range(min(n, len(dataset)))])
    return imgs.to(device)


def build_defense(name: str, num_clients: int, **kwargs):
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
    if name == "safesplit_continuous_ratchet":
        return ContinuousRatchetSafeSplitDefense(
            window_size=num_clients,
            low_freq_frac=cfg.DCT_LOW_FREQ_FRAC,
            matrix_width=cfg.ROTATION_MATRIX_WIDTH,
            num_clients=num_clients,
            initial_trust=cfg.TRUST_INITIAL,
            reward=cfg.TRUST_REWARD,
            penalty=cfg.TRUST_PENALTY,
            soft_threshold=cfg.TRUST_SOFT_THRESHOLD,
            low_threshold=cfg.TRUST_LOW_THRESHOLD,
            floor_base=cfg.TRUST_RATCHET_FLOOR_BASE,
            ratchet_step=cfg.TRUST_RATCHET_STEP,
            ratchet_max_floor=cfg.TRUST_RATCHET_MAX_FLOOR,
            ratchet_decay=cfg.TRUST_RATCHET_DECAY,
            ratchet_decay_threshold=cfg.TRUST_RATCHET_DECAY_THRESHOLD,
        )
    if name == "safesplit_trust_ratchet":
        return RatchetTemporalTrustSafeSplitDefense(
            window_size=num_clients,
            low_freq_frac=cfg.DCT_LOW_FREQ_FRAC,
            matrix_width=cfg.ROTATION_MATRIX_WIDTH,
            num_clients=num_clients,
            initial_trust=cfg.TRUST_INITIAL,
            reward=cfg.TRUST_REWARD,
            penalty=cfg.TRUST_PENALTY,
            soft_threshold=cfg.TRUST_SOFT_THRESHOLD,
            low_threshold=cfg.TRUST_LOW_THRESHOLD,
            floor_base=cfg.TRUST_RATCHET_FLOOR_BASE,
            ratchet_step=cfg.TRUST_RATCHET_STEP,
            ratchet_max_floor=cfg.TRUST_RATCHET_MAX_FLOOR,
            ratchet_decay=cfg.TRUST_RATCHET_DECAY,
            ratchet_decay_threshold=cfg.TRUST_RATCHET_DECAY_THRESHOLD,
        )
    if name == "safesplit_activation_clustering":
        return ActivationClusteringDefense(
            window_size=num_clients,
            low_freq_frac=cfg.DCT_LOW_FREQ_FRAC,
            matrix_width=cfg.ROTATION_MATRIX_WIDTH,
            num_clients=num_clients,
            initial_trust=cfg.TRUST_INITIAL,
            reward=cfg.TRUST_REWARD,
            penalty=cfg.TRUST_PENALTY,
            soft_threshold=cfg.TRUST_SOFT_THRESHOLD,
            low_threshold=cfg.TRUST_LOW_THRESHOLD,
            floor_base=cfg.TRUST_RATCHET_FLOOR_BASE,
            ratchet_step=cfg.TRUST_RATCHET_STEP,
            ratchet_max_floor=cfg.TRUST_RATCHET_MAX_FLOOR,
            ratchet_decay=cfg.TRUST_RATCHET_DECAY,
            ratchet_decay_threshold=cfg.TRUST_RATCHET_DECAY_THRESHOLD,
            reference_images=kwargs["reference_images"],
            head_template=kwargs["head_template"],
            backbone_template=kwargs["backbone_template"],
            device=kwargs["device"],
            n_pca_components=cfg.ACTIVATION_PCA_COMPONENTS,
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

    # ── Attack ────────────────────────────────────────────────────────────
    attack = build_attack(args.backdoor, args.dataset, args.pdr, cfg.BACKDOOR_TARGET)

    malicious_ids = set(random.sample(range(args.num_clients), args.num_malicious))
    print(f"Malicious clients: {sorted(malicious_ids)}")

    # Build client dataloaders (poisoned for malicious, clean for benign)
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
    ref_images = _build_reference_images(test_dataset, cfg.ACTIVATION_N_REFERENCE, device)
    defense_kwargs = {
        "reference_images": ref_images,
        "head_template": copy.deepcopy(head0).eval(),
        "backbone_template": copy.deepcopy(backbone0).eval(),
        "device": device,
    }
    defense = build_defense(request.defense, request.num_clients, **defense_kwargs)
    trainer = SplitLearningTrainer(
        head0=H0, backbone0=B0, tail0=T0,
        client_loaders=client_loaders,
        malicious_ids=malicious_ids,
        attack=attack,
        defense=defense,
        device=device,
        cfg=cfg,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n=== SafeSplit Experiment ===")
    print(f"  Dataset={args.dataset}  Arch={arch}  Defense={args.defense}")
    print(f"  Clients={args.num_clients}  Malicious={args.num_malicious}")
    print(f"  IID-rate={args.iid_rate}  PDR={args.pdr}  Rounds={args.num_rounds}")
    print(f"  Backdoor={args.backdoor}  Adaptive={args.adaptive}\n")

    history = trainer.run(
        num_rounds=args.num_rounds,
        test_loader=test_loader,
        trigger_loader=DataLoader(
            torch.utils.data.TensorDataset(
                torch.stack([t[0] for t in trigger_set]) if trigger_set else torch.zeros(1,3,32,32),
                torch.tensor([t[1] for t in trigger_set], dtype=torch.long) if trigger_set else torch.zeros(1,dtype=torch.long),
            ),
            batch_size=256, shuffle=False
        ) if trigger_set else None,
    )

    # ── Final evaluation ──────────────────────────────────────────────────
    final_ma = evaluate_model(
        trainer.client_heads[0], trainer.backbone, trainer.client_tails[0],
        test_loader, device
    )
    final_ba = evaluate_backdoor(
        trainer.client_heads[0], trainer.backbone, trainer.client_tails[0],
        trigger_set, device
    )

    print(f"\n{'='*50}")
    print(f"Final  MA = {final_ma:.2f}%   BA = {final_ba:.2f}%")
    print(f"{'='*50}\n")

    # ── Save results ──────────────────────────────────────────────────────
    run_name = (f"{args.dataset}_{arch}_{args.defense}"
                f"_N{args.num_clients}_M{args.num_malicious}"
                f"_iid{args.iid_rate}_pdr{args.pdr}"
                f"_{args.backdoor}")
    out_path = os.path.join(args.out_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "final_MA": final_ma,
            "final_BA": final_ba,
            "history": history,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
