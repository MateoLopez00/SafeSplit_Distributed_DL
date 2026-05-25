"""
Sweep num_malicious over a range to visualise performance degradation.

Default: paper preset, safesplit defense, semantic backdoor, mal in 3..10.
Quick smoke-test: --preset lite --malicious 1 2 3
"""
from __future__ import annotations

# Python 3.14 on Windows may fail SSL cert verification for torchvision downloads.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from main import build_experiment_request, run_experiment

DEFAULT_SWEEP = list(range(3, 11))  # 3..10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep num_malicious and plot degradation curves.")
    parser.add_argument("--preset", choices=sorted(cfg.EXPERIMENT_PRESETS), default="paper")
    parser.add_argument("--defense", default="safesplit",
                        choices=["none", "safesplit", "safesplit_trust", "safesplit_trust_ratchet", "safesplit_continuous_ratchet", "safesplit_activation_clustering", "dp", "krum"])
    parser.add_argument("--backdoor", default=cfg.BACKDOOR_TYPE, choices=["pixel", "semantic", "none"])
    parser.add_argument("--attack-schedule", default="static", choices=["static", "slow"],
                        help="'static' uses a fixed PDR; 'slow' ramps from slow-pdr-start to slow-pdr-end")
    parser.add_argument("--slow-pdr-start", type=float, default=cfg.SLOW_POISON_START_PDR)
    parser.add_argument("--slow-pdr-end", type=float, default=cfg.SLOW_POISON_END_PDR)
    parser.add_argument("--num-rounds", type=int, default=None,
                        help="Override num_rounds from preset")
    parser.add_argument("--slow-ramp-rounds", type=int, default=None,
                        help="Rounds to ramp PDR over (default: num_rounds from preset)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(cfg.RESULTS_DIR))
    parser.add_argument("--malicious", nargs="+", type=int, default=DEFAULT_SWEEP, metavar="N",
                        help="num_malicious values to sweep (default: 3 4 5 6 7 8 9 10)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> list[dict]:
    preset_cfg = cfg.resolve_preset(args.preset)
    num_clients = int(preset_cfg["num_clients"])

    sweep_values = sorted(args.malicious)
    invalid = [m for m in sweep_values if m > num_clients]
    if invalid:
        raise ValueError(
            f"num_malicious values {invalid} exceed num_clients={num_clients} for preset '{args.preset}'."
        )

    preset_rounds = int(cfg.resolve_preset(args.preset)["num_rounds"])
    resolved_rounds = args.num_rounds if args.num_rounds is not None else preset_rounds
    slow_ramp_rounds = args.slow_ramp_rounds if args.slow_ramp_rounds is not None else resolved_rounds

    results = []
    for num_malicious in sweep_values:
        print(f"\n{'='*62}")
        print(f" num_malicious={num_malicious}/{num_clients}  |  defense={args.defense}  |  "
              f"preset={args.preset}  |  schedule={args.attack_schedule}")
        print(f"{'='*62}")

        request = build_experiment_request(
            preset=args.preset,
            num_malicious=num_malicious,
            num_rounds=args.num_rounds,
            defense=args.defense,
            backdoor=args.backdoor,
            attack_schedule=args.attack_schedule,
            slow_pdr_start=args.slow_pdr_start,
            slow_pdr_end=args.slow_pdr_end,
            slow_ramp_rounds=slow_ramp_rounds,
            device=args.device,
            out_dir=args.out_dir,
        )
        result = run_experiment(request)

        print(f"  Final MA : {result['final_MA']:.2f}%")
        print(f"  Final BA : {result['final_BA']:.2f}%")

        results.append({
            "num_malicious": num_malicious,
            "num_clients": num_clients,
            "final_MA": result["final_MA"],
            "final_BA": result["final_BA"],
            "history": result["history"],
            "results_path": result.get("results_path"),
        })

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_summary(results: list[dict], out_dir: Path, prefix: str) -> Path:
    payload = {
        "sweep": [
            {
                "num_malicious": r["num_malicious"],
                "num_clients": r["num_clients"],
                "final_MA": r["final_MA"],
                "final_BA": r["final_BA"],
            }
            for r in results
        ]
    }
    path = out_dir / f"{prefix}_summary.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ax_style(ax: plt.Axes, title: str, ylabel: str, xticks: list[int]) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Number of malicious clients", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(xticks)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linewidth=0.6, linestyle=":")
    ax.grid(axis="y", linestyle="--", alpha=0.45)


def plot_summary(results: list[dict], out_dir: Path, prefix: str) -> Path:
    """Two-panel bar+line chart: final MA and BA vs num_malicious."""
    xs = [r["num_malicious"] for r in results]
    mas = [r["final_MA"] for r in results]
    bas = [r["final_BA"] for r in results]

    fig, (ax_ma, ax_ba) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Final accuracy vs. malicious client count\n"
        f"(preset={prefix.split('_')[2]}  defense={results[0]['num_clients']-results[0]['num_malicious']} honest clients shown)",
        fontsize=12,
    )

    # MA panel
    ax_ma.bar(xs, mas, color="steelblue", alpha=0.65, width=0.6, label="MA")
    ax_ma.plot(xs, mas, marker="o", color="steelblue", linewidth=2, markersize=7)
    _ax_style(ax_ma, "Main Accuracy (MA)", "Accuracy (%)", xs)

    # BA panel
    ax_ba.bar(xs, bas, color="tomato", alpha=0.65, width=0.6, label="BA")
    ax_ba.plot(xs, bas, marker="o", color="tomato", linewidth=2, markersize=7)
    _ax_style(ax_ba, "Backdoor Success Rate (BA)", "Backdoor accuracy (%)", xs)

    # Annotate data labels
    for ax, vals in [(ax_ma, mas), (ax_ba, bas)]:
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = out_dir / f"{prefix}_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rounds(results: list[dict], out_dir: Path, prefix: str) -> Path:
    """Two-panel line chart: per-round MA and BA, one curve per num_malicious."""
    fig, (ax_ma, ax_ba) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Per-round metrics by malicious client count  (preset={prefix.split('_')[2]})",
        fontsize=12,
    )

    n = len(results)
    colors = cm.plasma(np.linspace(0.1, 0.9, n))

    for color, r in zip(colors, results):
        history = r["history"]
        rounds = [h["round"] for h in history]
        mas = [h["clean_ma"] for h in history]
        bas = [h["backdoor_ba"] for h in history]
        label = f"mal={r['num_malicious']}"
        ax_ma.plot(rounds, mas, marker="o", color=color, label=label, linewidth=1.8, markersize=5)
        ax_ba.plot(rounds, bas, marker="o", color=color, label=label, linewidth=1.8, markersize=5)

    for ax, title, ylabel in [
        (ax_ma, "Main Accuracy per Round", "Main Accuracy (%)"),
        (ax_ba, "Backdoor Accuracy per Round", "Backdoor Accuracy (%)"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.legend(fontsize=8, ncol=2, loc="best")

    plt.tight_layout()
    path = out_dir / f"{prefix}_rounds.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_combined(results: list[dict], out_dir: Path, prefix: str) -> Path:
    """Single dual-axis chart: MA (blue, left) and BA (red, right) vs num_malicious."""
    xs = [r["num_malicious"] for r in results]
    mas = [r["final_MA"] for r in results]
    bas = [r["final_BA"] for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(xs, mas, marker="o", color="steelblue", linewidth=2.2, markersize=8, label="Main Accuracy (MA)")
    ax1.fill_between(xs, mas, alpha=0.10, color="steelblue")

    l2, = ax2.plot(xs, bas, marker="s", color="tomato", linewidth=2.2, markersize=8, label="Backdoor Accuracy (BA)")
    ax2.fill_between(xs, bas, alpha=0.10, color="tomato")

    ax1.set_xlabel("Number of malicious clients", fontsize=11)
    ax1.set_ylabel("Main Accuracy (%)", color="steelblue", fontsize=11)
    ax2.set_ylabel("Backdoor Accuracy (%)", color="tomato", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax1.set_xticks(xs)
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax1.set_title(
        f"MA vs BA degradation as malicious clients increase\n"
        f"(preset={prefix.split('_')[2]}  defense={prefix.split('_')[3]})",
        fontsize=12,
    )
    ax1.legend(handles=[l1, l2], loc="center left", fontsize=10)

    plt.tight_layout()
    path = out_dir / f"{prefix}_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"malicious_sweep_{args.preset}_{args.defense}_{args.backdoor}_{args.attack_schedule}"

    results = run_sweep(args)

    summary_json = save_summary(results, out_dir, prefix)
    summary_png = plot_summary(results, out_dir, prefix)
    rounds_png = plot_rounds(results, out_dir, prefix)
    combined_png = plot_combined(results, out_dir, prefix)

    print(f"\n{'='*62}")
    print("Sweep complete.")
    print(f"  Summary JSON    : {summary_json}")
    print(f"  Summary plot    : {summary_png}")
    print(f"  Rounds plot     : {rounds_png}")
    print(f"  Combined plot   : {combined_png}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
