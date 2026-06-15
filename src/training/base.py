"""Shared training infrastructure for all variants.

Responsibilities common to every experiment:
- train/val loop orchestration
- loss curve logging (history dict, saved to results/)
- best-checkpoint saving by validation metric
- resuming from checkpoint
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


class BaseTrainer:
    """Base class with the train/val loop, logging and checkpointing.

    Subclasses build their networks/optimizers in ``__init__`` (registering
    them via :meth:`register_network` and ``self.optimizers``) and implement
    ``training_step`` / ``validation_step``, each returning a dict of scalar
    logs (e.g. ``{"train_l1": 0.12}``).

    Args:
        train_loader / val_loader: dataloaders yielding dict batches.
        checkpoint_dir: where to save best/last checkpoints.
        monitor: validation metric used to select the best checkpoint
            (e.g. "val_l1").
        device: torch device string.
    """

    def __init__(self, train_loader, val_loader, checkpoint_dir: str | Path,
                 monitor: str = "val_l1", device: str = "cuda"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.device = torch.device(device)

        self.networks: dict[str, nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.history: dict[str, list[float]] = {}
        self.best_metric = math.inf
        self.epoch = 0

    # Subclass interface
    def register_network(self, name: str, module: nn.Module) -> nn.Module:
        """Move a network to the device and track it for checkpointing."""
        module = module.to(self.device)
        self.networks[name] = module
        return module

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        raise NotImplementedError

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        raise NotImplementedError

    # Train/val loop
    def fit(self, epochs: int) -> dict[str, list[float]]:
        """Run the train/val loop up to ``epochs`` total epochs."""
        for epoch in range(self.epoch + 1, epochs + 1):
            self.epoch = epoch

            self._set_train_mode(True)
            train_logs = self._run_epoch(self.train_loader, self.training_step,
                                         desc=f"epoch {epoch}/{epochs} [train]")

            self._set_train_mode(False)
            with torch.no_grad():
                val_logs = self._run_epoch(self.val_loader, self.validation_step,
                                           desc=f"epoch {epoch}/{epochs} [val]")

            # Merge the dicts
            logs = {**train_logs, **val_logs}
            for key, value in logs.items():
                self.history.setdefault(key, []).append(value)

            self.save_checkpoint("last.pt")
            monitored = logs.get(self.monitor) # e.g., logs.get("val_l1")
            if monitored is not None and monitored < self.best_metric:
                # Save the new record
                self.best_metric = monitored
                self.save_checkpoint("best.pt")

            summary = " | ".join(f"{k}={v:.4f}" for k, v in logs.items())
            print(f"epoch {epoch:3d}/{epochs} | {summary}")
        return self.history

    def _run_epoch(self, loader, step_fn, desc: str) -> dict[str, float]:
        """Average the per-batch logs of ``step_fn`` over one epoch."""
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in tqdm(loader, desc=desc, leave=False):
            # Execute the single training or validation step
            logs = step_fn(self._batch_to_device(batch))
            # Accumulate loss and metric values to compute the epoch average later
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1
        return {key: value / n_batches for key, value in totals.items()}

    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _set_train_mode(self, training: bool) -> None:
        """Set to train mode every module of the trainer (e.g., Pix2PixTrainer has gen. and discr.)"""
        for module in self.networks.values():
            module.train(training)

    # Checkpointing
    def save_checkpoint(self, filename: str) -> None:
        torch.save({
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "history": self.history,
            "networks": {k: m.state_dict() for k, m in self.networks.items()},
            "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
        }, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str = "last.pt") -> None:
        state = torch.load(self.checkpoint_dir / filename,
                           map_location=self.device, weights_only=True)
        self.epoch = state["epoch"]
        self.best_metric = state["best_metric"]
        self.history = state["history"]
        for key, module in self.networks.items():
            module.load_state_dict(state["networks"][key])
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][key])
