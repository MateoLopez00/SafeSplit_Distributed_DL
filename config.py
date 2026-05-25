from dataclasses import dataclass
from enum import Enum
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = CACHE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# ------------------------------------------------------------------
# Reproduction scope
# ------------------------------------------------------------------
DATASET = "CIFAR10"
ARCH = "resnet18"
NUM_CLASSES = 10


# ------------------------------------------------------------------
# Split learning setup
# ------------------------------------------------------------------
NUM_CLIENTS = 10
NUM_MALICIOUS = 2
NUM_ROUNDS = 5

LOCAL_EPOCHS = 1

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

SEED = 42

IID_RATE = 0.8

# ------------------------------------------------------------------
# Attack setup
# ------------------------------------------------------------------
BACKDOOR_TYPE = "semantic"

PIXEL_TARGET_LABEL = 0

SEMANTIC_SOURCE_LABEL = 1
SEMANTIC_TARGET_LABEL = 2

POISONED_DATA_RATE = 0.5

TRIGGER_SIZE = 4
TRIGGER_POS = "bottom-right"

# ------------------------------------------------------------------
# SafeSplit setup
# ------------------------------------------------------------------
SAFE_SPLIT_WINDOW = NUM_CLIENTS
DCT_LOW_FREQ_FRAC = 0.25
ROTATION_MATRIX_WIDTH = 128
EPS = 1e-8


# ------------------------------------------------------------------
# Temporal trust extension
# ------------------------------------------------------------------
TRUST_INITIAL = 1.0

TRUST_REWARD = 0.02
TRUST_PENALTY = 0.10

TRUST_SOFT_THRESHOLD = 0.60
TRUST_LOW_THRESHOLD = 0.70

# ------------------------------------------------------------------
# Slow poisoning extension
# ------------------------------------------------------------------
ATTACK_SCHEDULE = "static"

SLOW_POISON_START_PDR = 0.05
SLOW_POISON_END_PDR = POISONED_DATA_RATE

SLOW_POISON_RAMP_ROUNDS = NUM_ROUNDS

# ------------------------------------------------------------------
# Differential privacy config
# ------------------------------------------------------------------
DP_CLIP_NORM = 1.0
DP_NOISE_SCALE = 1e-3


# ==================================================================
# Experiment presets
# ==================================================================

class Preset(str, Enum):

    LITE = "lite"
    MEDIUM = "medium"
    PAPER = "paper"


@dataclass(frozen=True, slots=True)
class ExperimentPreset:

    name: str

    arch: str

    num_clients: int
    num_malicious: int
    num_rounds: int

    iid_rate: float

    max_samples_per_client: int | None

    local_epochs: int

    batch_size: int
    eval_batch_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arch": self.arch,
            "num_clients": self.num_clients,
            "num_malicious": self.num_malicious,
            "num_rounds": self.num_rounds,
            "iid_rate": self.iid_rate,
            "max_samples_per_client": self.max_samples_per_client,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
        }


DEFAULT_PRESET = Preset.PAPER

PRESET_ALIASES: dict[str, Preset] = {
    "fast-dev": Preset.LITE,
}

EXPERIMENT_PRESETS: dict[Preset, ExperimentPreset] = {

    Preset.LITE: ExperimentPreset(
        name="lite",
        arch="simple_cnn",
        num_clients=4,
        num_malicious=1,
        num_rounds=1,
        iid_rate=IID_RATE,
        max_samples_per_client=256,
        local_epochs=1,
        batch_size=64,
        eval_batch_size=128,
    ),

    Preset.MEDIUM: ExperimentPreset(
        name="medium",
        arch="resnet18",
        num_clients=6,
        num_malicious=1,
        num_rounds=2,
        iid_rate=IID_RATE,
        max_samples_per_client=1024,
        local_epochs=1,
        batch_size=96,
        eval_batch_size=128,
    ),

    Preset.PAPER: ExperimentPreset(
        name="paper",
        arch=ARCH,
        num_clients=NUM_CLIENTS,
        num_malicious=NUM_MALICIOUS,
        num_rounds=NUM_ROUNDS,
        iid_rate=IID_RATE,
        max_samples_per_client=None,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    ),
}

# ------------------------------------------------------------------
# Optional reduced-compute compatibility flags
# ------------------------------------------------------------------
FAST_DEV_RUN = False
FAST_DEV_NUM_CLIENTS = EXPERIMENT_PRESETS[Preset.LITE].num_clients
FAST_DEV_NUM_MALICIOUS = EXPERIMENT_PRESETS[Preset.LITE].num_malicious
FAST_DEV_NUM_ROUNDS = EXPERIMENT_PRESETS[Preset.LITE].num_rounds
FAST_DEV_MAX_SAMPLES_PER_CLIENT = EXPERIMENT_PRESETS[Preset.LITE].max_samples_per_client


# ==================================================================
# Preset resolution
# ==================================================================
def normalize_preset(
    preset: Preset | str | None,
) -> Preset:

    if preset is None:
        return DEFAULT_PRESET

    if isinstance(preset, Preset):
        return preset

    if preset in PRESET_ALIASES:
        return PRESET_ALIASES[preset]

    try:
        return Preset(preset)

    except ValueError as exc:

        valid_presets = ", ".join(p.value for p in Preset)
        raise ValueError(f"Unknown preset '{preset}'. Expected one of: {valid_presets}.") from exc


def resolve_preset(
    preset: Preset | str | None = None,
) -> ExperimentPreset:

    normalized = normalize_preset(preset)

    return EXPERIMENT_PRESETS[normalized]


# ==================================================================
# Device resolution
# ==================================================================

def resolve_device(requested: str | None = None) -> str:
    import torch

    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
