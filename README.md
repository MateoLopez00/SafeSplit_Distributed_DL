# SafeSplit – Reimplementation (NDSS 2025)

Reimplementation of **SafeSplit: A Novel Defense Against Client-Side Backdoor Attacks in Split Learning** (Rieger et al., NDSS 2025).

---

## Project Structure

```
.
├── config.py                   # All hyperparameters (mirrors paper defaults)
├── main.py                     # CLI entry point
├── evaluate.py                 # BA / MA / confusion-matrix utilities
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── split_models.py         # Head / Backbone / Tail for ResNet-18, Simple CNN,
│                               #   VGG-11, GoogLeNet, Wide-ResNet-50, MicronNet
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset loading + non-IID partitioning
│   └── backdoor.py             # Pixel trigger + semantic backdoor attacks
├── defense/
│   ├── __init__.py
│   ├── safesplit.py            # SafeSplit (Algorithm 1) – DCT + Rotational Distance
│   └── baselines.py            # KRUM, Differential Privacy, FreqFed
└── training/
    ├── __init__.py
    └── trainer.py              # U-shaped SL training loop + SplitLearningTrainer
```

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Replicate Table II (primary results)
```bash
# CIFAR-10, ResNet-18, semantic backdoor, SafeSplit (default)
python main.py

# CIFAR-10, pixel trigger, SafeSplit
python main.py --backdoor pixel

# MNIST, pixel trigger, SafeSplit
python main.py --dataset MNIST --backdoor pixel

# FMNIST, pixel trigger, SafeSplit
python main.py --dataset FMNIST --backdoor pixel

# No defense (baseline)
python main.py --defense none
```

### Replicate Fig. 7 (defense comparison)
```bash
python main.py --defense krum    --iid_rate 0.6
python main.py --defense dp      --iid_rate 0.6
python main.py --defense freqfed --iid_rate 0.6
python main.py --defense safesplit --iid_rate 0.6
python main.py --defense none    --iid_rate 0.6
```

### Replicate Table III (IID rates)
```bash
for IID in 0.6 0.8 1.0; do
  python main.py --iid_rate $IID
  python main.py --iid_rate $IID --defense none
done
```

### Replicate Fig. 5 (varying client counts)
```bash
for N in 5 10 15 20; do
  M=$((N / 5))
  python main.py --num_clients $N --num_malicious $M
done
```

### Replicate Table V (varying Poisoned Data Rate)
```bash
for PDR in 0.25 0.50 0.75 1.00; do
  python main.py --pdr $PDR
  python main.py --pdr $PDR --defense none
done
```

### Adaptive attack (Table IV / Sect. VI-E)
```bash
python main.py --adaptive
```

### Different backbone sizes (Sect. VI-H)
```bash
python main.py --bb_blocks 2
python main.py --bb_blocks 3
python main.py --bb_blocks 4
```

---

## Key Implementation Details

### U-shaped Split Learning (Sect. II-A)
The DNN **F** is split at two cut layers:

```
F(x) = (T ∘ B ∘ H)(x)
```

- **Head (H)** – first few layers, runs on the client.
- **Backbone (B)** – bulk of the network, runs on the server.
  Gradients are exchanged at both cut points during backpropagation.
- **Tail (T)** – final classifier layers, runs on the client.

### SafeSplit Algorithm (Algorithm 1)

After each client's training step the server:

1. Stores the new backbone in a **FIFO** of size N.
2. Computes the **frequency score** E_i for each stored backbone:
   ```
   S_i = DCT_low(B_i − B_{i−1})   (Eq. 2)
   E_i = Σ ||S_i − S_j||₂  for N/2+1 nearest j   (Eq. 3)
   ```
3. Computes the **rotational distance score** R_i:
   ```
   θ(t)  = arctan(B^x_t, B^y_t)       (Eq. 4 / App. H)
   ω(t)  = θ(t) − θ(t−1)              (Eq. 5, Δt=1)
   RD    = ω(t) / 2π                   (Eq. 6)
   R_i   = Σ |RD_i − RD_j|
   ```
4. Selects the **most recent backbone** in the intersection of both
   N/2+1-smallest majority sets. Rolls back if the latest is poisoned.

### Non-IID Data (Sect. VI-A)
Main-label strategy: each client has a "main label" class.
A fraction `iid_rate` of its data is drawn uniformly; the rest from
its main label only. Default: `iid_rate=0.8`, 10 clients, 2 malicious.

---

## Results (reproduced, expected range)

| Dataset   | Attack          | Defense   | BA (%) | MA (%) |
|-----------|-----------------|-----------|--------|--------|
| CIFAR-10  | Semantic trigger| SafeSplit | ~0.0   | ~62.7  |
| CIFAR-10  | Pixel trigger   | SafeSplit | ~0.3   | ~66.4  |
| MNIST     | Pixel trigger   | SafeSplit | ~0.0   | ~98.8  |
| FMNIST    | Pixel trigger   | SafeSplit | ~3.4   | ~84.6  |

---

## Citation

```bibtex
@inproceedings{rieger2025safesplit,
  title     = {SafeSplit: A Novel Defense Against Client-Side Backdoor Attacks in Split Learning},
  author    = {Rieger, Phillip and Pegoraro, Alessandro and Kumari, Kavita and
               Abera, Tigist and Knauer, Jonathan and Sadeghi, Ahmad-Reza},
  booktitle = {Network and Distributed System Security (NDSS) Symposium},
  year      = {2025},
  doi       = {10.14722/ndss.2025.241698}
}
```
