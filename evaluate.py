"""
evaluate.py – Compute Backdoor Accuracy (BA) and Main Task Accuracy (MA).

Provides:
  evaluate_model(head, backbone, tail, loader, device) → accuracy %
  evaluate_backdoor(head, backbone, tail, trigger_set, device) → BA %
  confusion_matrix(head, backbone, tail, loader, device, num_classes) → ndarray
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def evaluate_model(
    head: nn.Module,
    backbone: nn.Module,
    tail: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Main Task Accuracy (MA): fraction of clean samples correctly classified.
    """
    head.eval(); backbone.eval(); tail.eval()
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        smashed = head(x)
        mid     = backbone(smashed)
        logits  = tail(mid)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_backdoor(
    head: nn.Module,
    backbone: nn.Module,
    tail: nn.Module,
    trigger_set: list,           # list of (image_tensor, target_label) tuples
    device: torch.device,
    batch_size: int = 128,
) -> float:
    """
    Backdoor Accuracy (BA): fraction of triggered samples classified as the
    backdoor target label.
    """
    if not trigger_set:
        return 0.0

    head.eval(); backbone.eval(); tail.eval()

    imgs   = torch.stack([t[0] for t in trigger_set])
    labels = torch.tensor([t[1] for t in trigger_set], dtype=torch.long)
    ds     = TensorDataset(imgs, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        smashed = head(x)
        mid     = backbone(smashed)
        logits  = tail(mid)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def confusion_matrix(
    head: nn.Module,
    backbone: nn.Module,
    tail: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """Return a num_classes × num_classes confusion matrix (rows=true, cols=pred)."""
    head.eval(); backbone.eval(); tail.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x       = x.to(device)
        smashed = head(x)
        mid     = backbone(smashed)
        logits  = tail(mid)
        preds   = logits.argmax(1).cpu().numpy()
        true    = y.numpy()
        for t, p in zip(true, preds):
            cm[t, p] += 1
    return cm
