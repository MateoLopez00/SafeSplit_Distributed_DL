"""
SafeSplit Configuration
Mirrors the experimental setup from the NDSS 2025 paper.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_cache")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ─── Training Setup ───────────────────────────────────────────────────────────
NUM_CLIENTS       = 10       # default number of clients (N)
NUM_MALICIOUS     = 2        # malicious clients (< N/2)
NUM_ROUNDS        = 50       # training rounds (R)
BATCH_SIZE        = 64
LOCAL_EPOCHS      = 1        # epochs per client per round
LR                = 0.01
MOMENTUM          = 0.9
WEIGHT_DECAY      = 1e-4

# ─── Data / IID ───────────────────────────────────────────────────────────────
IID_RATE          = 0.8      # fraction of each client's data drawn uniformly
                             # 1.0 = full IID, 0.0 = fully non-IID

# ─── Backdoor Attack ──────────────────────────────────────────────────────────
BACKDOOR_TYPE     = "pixel"  # "pixel" | "semantic"
POISONED_DATA_RATE = 0.75   # fraction of malicious client's data that is poisoned
TRIGGER_SIZE      = 5        # pixel-trigger square side (pixels)
TRIGGER_POS       = "bottom-right"
BACKDOOR_TARGET   = 0        # target class for backdoor (0 = airplane for CIFAR-10)

# Semantic backdoor (CIFAR-10): cars (label 1) with striped background → birds (label 2)
SEMANTIC_SOURCE_LABEL = 1    # "car"
SEMANTIC_TARGET_LABEL = 2    # "bird"

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET           = "CIFAR10" # CIFAR10 | MNIST | FMNIST | CIFAR100 | GTSRB
NUM_CLASSES_MAP   = {
    "CIFAR10":  10,
    "MNIST":    10,
    "FMNIST":   10,
    "CIFAR100": 100,
    "GTSRB":    43,
}
NUM_CLASSES       = NUM_CLASSES_MAP[DATASET]

# ─── Model Architecture ───────────────────────────────────────────────────────
# Mapping from dataset to default architecture (Table I of the paper)
ARCH_MAP = {
    "CIFAR10":  "resnet18",    # primary
    "MNIST":    "simple_cnn",
    "FMNIST":   "simple_cnn",
    "CIFAR100": "wide_resnet50",
    "GTSRB":    "micronnet",
}
ARCHITECTURE      = ARCH_MAP[DATASET]

# Cut-layer configuration: how many ResNet-18 blocks go to the backbone
# 2 | 3 | 4 (default = 4, i.e. all blocks on the server)
RESNET_BACKBONE_BLOCKS = 4

# ─── SafeSplit Hyperparameters ────────────────────────────────────────────────
# FIFO window = N clients; majority = N//2 + 1
# Low-frequency fraction for 2-D DCT (fraction of coefficients kept per dimension)
DCT_LOW_FREQ_FRAC = 0.5      # keep top-left 50% in each DCT dimension

# ─── Adaptive-Attack Hyperparameters ─────────────────────────────────────────
ADAPTIVE_ALPHA    = 0.5      # weight for backdoor vs. anomaly-evasion loss

# ─── Differential Privacy Baseline ───────────────────────────────────────────
DP_CLIP_NORM      = 1.0
DP_NOISE_SCALE    = 0.001

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE            = "cuda"   # overridden to "cpu" automatically if no GPU

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_INTERVAL      = 10       # print stats every N rounds
SEED              = 42
