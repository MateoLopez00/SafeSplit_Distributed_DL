from enum import Enum
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cache"
RESULTS_DIR = BASE_DIR / "results"

# Reproduction scope
DATASET = "CIFAR10"
NUM_CLASSES = 10
SEED = 42


class PresetType(Enum):
    """Experiment preset configurations."""
    LITE = "lite"
    MEDIUM = "medium"
    PAPER = "paper"

# Split learning setup (defaults, overridden by presets)
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# Attack setup
BACKDOOR_TYPE = "semantic"
PIXEL_TARGET_LABEL = 0
SEMANTIC_SOURCE_LABEL = 1
SEMANTIC_TARGET_LABEL = 2
POISONED_DATA_RATE = 0.5
TRIGGER_SIZE = 4
TRIGGER_POS = "bottom-right"

# SafeSplit setup
# SAFE_SPLIT_WINDOW = NUM_CLIENTS
DCT_LOW_FREQ_FRAC = 0.25
ROTATION_MATRIX_WIDTH = 128
EPS = 1e-8

# Baselines
DP_CLIP_NORM = 1.0
DP_NOISE_SCALE = 1e-3

# Shared experiment presets
DEFAULT_PRESET = PresetType.PAPER
EXPERIMENT_PRESETS = {
    PresetType.LITE: {
        "arch": "simple_cnn",
        "num_clients": 4,
        "num_malicious": 1,
        "num_rounds": 1,
        "iid_rate": 0.8,
        "max_samples_per_client": 256,
        "local_epochs": 1,
        "batch_size": 64,
        "eval_batch_size": 128,
    },
    PresetType.MEDIUM: {
        "arch": "resnet18",
        "num_clients": 6,
        "num_malicious": 1,
        "num_rounds": 2,
        "iid_rate": 0.8,
        "max_samples_per_client": 1024,
        "local_epochs": 1,
        "batch_size": 96,
        "eval_batch_size": 128,
    },
    PresetType.PAPER: {
        "arch": "resnet18",
        "num_clients": 10,
        "num_malicious": 2,
        "num_rounds": 5,
        "iid_rate": 0.8,
        "max_samples_per_client": None,
        "local_epochs": 1,
        "batch_size": 128,
        "eval_batch_size": 256,
    },
}


def resolve_preset(preset: PresetType | None = None) -> dict:
    """Resolve a preset enum to its configuration dictionary."""
    if preset is None:
        preset = DEFAULT_PRESET
    if not isinstance(preset, PresetType):
        raise TypeError(f"Expected PresetType enum, got {type(preset)}")
    return {
        "name": preset.value,
        **EXPERIMENT_PRESETS[preset],
    }


def resolve_device(requested: str | None = None) -> str:
    import torch

    if requested:
        try:
            device = torch.device(requested)
        except Exception:
            return "cpu"

        if device.type == "cuda" and not torch.cuda.is_available():
            return "cpu"

        if device.type == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                return "cpu"

        return str(device)

    return "cuda" if torch.cuda.is_available() else "cpu"