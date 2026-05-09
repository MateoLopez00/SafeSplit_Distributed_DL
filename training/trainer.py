from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.optim import SGD

from defense import Checkpoint, clone_state_dict, diff_state_dict, load_state_dict
from evaluate import evaluate_backdoor, evaluate_model


class SplitLearningTrainer:
    def __init__(
        self,
        head0,
        backbone0,
        tail0,
        client_loaders,
        device: str,
        lr: float,
        momentum: float,
        weight_decay: float,
        local_epochs: int,
        defense=None,
        poison_schedule: dict[str, float | int] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.current_head = copy.deepcopy(head0).to(self.device)
        self.current_backbone = copy.deepcopy(backbone0).to(self.device)
        self.current_tail = copy.deepcopy(tail0).to(self.device)
        self.client_loaders = client_loaders
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.defense = defense
        self.poison_schedule = poison_schedule or {}
        self.history: list[Checkpoint] = []
        self.last_selected_checkpoint: Checkpoint | None = None
        self.defense_decisions: list[dict[str, object]] = []

    def _make_optimizers(self, head, backbone, tail):
        head_opt = SGD(head.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        backbone_opt = SGD(
            backbone.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        tail_opt = SGD(tail.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return head_opt, backbone_opt, tail_opt

    def train_one_client(self, client_id: int):
        head = copy.deepcopy(self.current_head).to(self.device)
        backbone = copy.deepcopy(self.current_backbone).to(self.device)
        tail = copy.deepcopy(self.current_tail).to(self.device)
        head.train()
        backbone.train()
        tail.train()

        head_opt, backbone_opt, tail_opt = self._make_optimizers(head, backbone, tail)

        for _ in range(self.local_epochs):
            for inputs, targets in self.client_loaders[client_id]:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                head_opt.zero_grad()
                backbone_opt.zero_grad()
                tail_opt.zero_grad()

                smashed = head(inputs)
                smashed_detached = smashed.detach().requires_grad_(True)

                backbone_output = backbone(smashed_detached)
                backbone_output_detached = backbone_output.detach().requires_grad_(True)

                logits = tail(backbone_output_detached)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                tail_opt.step()

                backbone_output.backward(backbone_output_detached.grad)
                backbone_opt.step()

                smashed.backward(smashed_detached.grad)
                head_opt.step()

        return head, backbone, tail

    def _store_checkpoint(self, step: int, round_id: int, client_id: int, head, backbone, tail) -> Checkpoint:
        previous_backbone = clone_state_dict(self.current_backbone)
        head_state = clone_state_dict(head)
        backbone_state = clone_state_dict(backbone)
        tail_state = clone_state_dict(tail)
        checkpoint = Checkpoint(
            step=step,
            round_id=round_id,
            client_id=client_id,
            head_state=head_state,
            backbone_state=backbone_state,
            tail_state=tail_state,
            update_state=diff_state_dict(backbone_state, previous_backbone),
        )
        self.history.append(checkpoint)
        return checkpoint

    def _select_checkpoint(self) -> Checkpoint:
        if self.defense is None:
            return copy.deepcopy(self.history[-1])
        checkpoint = self.defense.select_checkpoint(self.history)
        decision_getter = getattr(self.defense, "get_latest_decision", None)
        if callable(decision_getter):
            decision = decision_getter()
            if decision is not None:
                self.defense_decisions.append(dict(decision))
        return checkpoint

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        load_state_dict(self.current_head, checkpoint.head_state, self.device)
        load_state_dict(self.current_backbone, checkpoint.backbone_state, self.device)
        load_state_dict(self.current_tail, checkpoint.tail_state, self.device)
        self.last_selected_checkpoint = checkpoint

    def _apply_poison_schedule(self, round_id: int) -> None:
        if not self.poison_schedule:
            return
        start_pdr = float(self.poison_schedule.get("start_pdr", 0.0))
        end_pdr = float(self.poison_schedule.get("end_pdr", start_pdr))
        ramp_rounds = max(1, int(self.poison_schedule.get("ramp_rounds", 1)))
        progress = min(1.0, round_id / max(1, ramp_rounds - 1))
        current_pdr = start_pdr + (end_pdr - start_pdr) * progress
        for loader in self.client_loaders:
            dataset = getattr(loader, "dataset", None)
            setter = getattr(dataset, "set_poisoned_data_rate", None)
            if callable(setter):
                setter(current_pdr)

    def _trust_snapshot(self) -> dict[int, float] | None:
        snapshot_getter = getattr(self.defense, "get_trust_snapshot", None)
        if callable(snapshot_getter):
            return snapshot_getter()
        return None

    def run(self, num_rounds: int, test_loader, trigger_set=None):
        metrics = []
        global_step = 0
        for round_id in range(num_rounds):
            self._apply_poison_schedule(round_id)
            for client_id in range(len(self.client_loaders)):
                head, backbone, tail = self.train_one_client(client_id)
                global_step += 1
                self._store_checkpoint(global_step, round_id, client_id, head, backbone, tail)
                selected = self._select_checkpoint()
                self._load_checkpoint(selected)

            clean_acc = evaluate_model(self.current_head, self.current_backbone, self.current_tail, test_loader, self.device)
            backdoor_acc = evaluate_backdoor(
                self.current_head, self.current_backbone, self.current_tail, trigger_set or [], self.device
            )
            metrics.append(
                {
                    "round": round_id + 1,
                    "clean_ma": clean_acc,
                    "backdoor_ba": backdoor_acc,
                    "selected_step": None if self.last_selected_checkpoint is None else self.last_selected_checkpoint.step,
                    "selected_client": None
                    if self.last_selected_checkpoint is None
                    else self.last_selected_checkpoint.client_id,
                    "trust_scores": self._trust_snapshot(),
                    "defense_decisions": list(self.defense_decisions),
                }
            )
        return metrics
