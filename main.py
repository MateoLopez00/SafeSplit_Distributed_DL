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
from models  import get_split_model
from data    import load_datasets, partition_data
from data    import PixelTriggerAttack, SemanticTriggerAttack
from data.backdoor import PoisonedDataset
from defense import SafeSplit, KrumDefense, DifferentialPrivacyDefense, FreqFedDefense
from training import SplitLearningTrainer
from evaluate import evaluate_model, evaluate_backdoor


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SafeSplit – NDSS 2025 reimplementation")
    p.add_argument("--dataset",       default=cfg.DATASET,
                   choices=["CIFAR10","MNIST","FMNIST","CIFAR100","GTSRB"])
    p.add_argument("--arch",          default=None,
                   choices=["resnet18","simple_cnn","vgg11","googlenet",
                            "wide_resnet50","micronnet"])
    p.add_argument("--backdoor",      default=cfg.BACKDOOR_TYPE,
                   choices=["pixel","semantic","none"])
    p.add_argument("--defense",       default="safesplit",
                   choices=["safesplit","krum","dp","freqfed","none"])
    p.add_argument("--num_clients",   type=int, default=cfg.NUM_CLIENTS)
    p.add_argument("--num_malicious", type=int, default=cfg.NUM_MALICIOUS)
    p.add_argument("--num_rounds",    type=int, default=cfg.NUM_ROUNDS)
    p.add_argument("--iid_rate",      type=float, default=cfg.IID_RATE)
    p.add_argument("--pdr",           type=float, default=cfg.POISONED_DATA_RATE,
                   help="Poisoned Data Rate")
    p.add_argument("--seed",          type=int, default=cfg.SEED)
    p.add_argument("--device",        default=cfg.DEVICE)
    p.add_argument("--out_dir",       default="results")
    p.add_argument("--bb_blocks",     type=int, default=cfg.RESNET_BACKBONE_BLOCKS,
                   help="ResNet-18 backbone blocks (2|3|4)")
    p.add_argument("--adaptive",      action="store_true",
                   help="Enable adaptive attack (loss-constrain on rotational metric)")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Architecture ───────────────────────────────────────────────────────
    arch = args.arch or cfg.ARCH_MAP.get(args.dataset.upper(), "simple_cnn")
    num_classes = cfg.NUM_CLASSES_MAP[args.dataset.upper()]

    # ── Datasets ──────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} …")
    train_ds, test_ds = load_datasets(args.dataset, cfg.DATA_DIR)

    # ── Non-IID partition ─────────────────────────────────────────────────
    client_indices = partition_data(
        train_ds, args.num_clients, args.iid_rate, num_classes, seed=args.seed
    )

    # ── Attack ────────────────────────────────────────────────────────────
    attack = build_attack(args.backdoor, args.dataset, args.pdr, cfg.BACKDOOR_TARGET)

    malicious_ids = set(random.sample(range(args.num_clients), args.num_malicious))
    print(f"Malicious clients: {sorted(malicious_ids)}")

    # Build client dataloaders (poisoned for malicious, clean for benign)
    client_loaders = []
    for cid in range(args.num_clients):
        subset = Subset(train_ds, client_indices[cid])
        if attack is not None and cid in malicious_ids:
            # Determine source label for pixel attacks
            src = (cfg.SEMANTIC_SOURCE_LABEL
                   if args.backdoor == "semantic" else None)
            poisoned = PoisonedDataset(subset, attack, args.pdr,
                                       attack.target_label, source_label=src)
            loader_ds = poisoned
        else:
            loader_ds = subset
        client_loaders.append(
            DataLoader(loader_ds, batch_size=cfg.BATCH_SIZE,
                       shuffle=True, num_workers=0, pin_memory=False,
                       drop_last=True)
        )

    # Clean test loader
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Trigger test set for BA evaluation
    trigger_set = []
    if attack is not None:
        if args.backdoor == "semantic":
            trigger_set = attack.build_backdoor_test_set(test_ds)
        else:
            trigger_set = attack.build_backdoor_test_set(test_ds, source_label=None)

    # ── Models ────────────────────────────────────────────────────────────
    print(f"Building {arch} split model …")
    H0, B0, T0 = get_split_model(arch, args.dataset, num_classes,
                                  num_bb_blocks=args.bb_blocks)
    H0.to(device); B0.to(device); T0.to(device)

    # ── Defense ───────────────────────────────────────────────────────────
    defense = build_defense(args.defense, args.num_clients)

    # Inject adaptive-attack flag into cfg for trainer
    cfg.ADAPTIVE_ATTACK = args.adaptive

    # ── Trainer ───────────────────────────────────────────────────────────
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
