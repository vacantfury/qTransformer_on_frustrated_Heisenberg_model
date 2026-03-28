"""
VMC training callbacks.

Callbacks are called during the VMC training loop to perform
logging, checkpointing, early stopping, and evaluation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class CallbackList:
    """Container that runs a list of callbacks in order."""

    def __init__(self, callbacks: List[Callback] | None = None):
        self.callbacks = callbacks or []

    def on_step(self, step: int, log_data: Dict[str, Any]) -> bool:
        """
        Called after each VMC step.

        Returns:
            True if training should stop (early stopping triggered).
        """
        for cb in self.callbacks:
            if cb.on_step(step, log_data):
                return True
        return False

    def on_train_end(self, log_data: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(log_data)


class Callback:
    """Base callback interface."""

    def on_step(self, step: int, log_data: Dict[str, Any]) -> bool:
        return False

    def on_train_end(self, log_data: Dict[str, Any]) -> None:
        pass


class EnergyLogger(Callback):
    """Log energy and variance to a list for later saving."""

    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.steps: list[int] = []
        self.energies: list[float] = []
        self.variances: list[float] = []

    def on_step(self, step: int, log_data: Dict[str, Any]) -> bool:
        if step % self.log_every == 0:
            energy = log_data.get("energy", float("nan"))
            variance = log_data.get("variance", float("nan"))
            self.steps.append(step)
            self.energies.append(float(np.real(energy)))
            self.variances.append(float(np.real(variance)))
            logger.info(
                f"Step {step:5d} | E = {np.real(energy):.6f} | "
                f"Var = {np.real(variance):.6f}"
            )
        return False


class EarlyStopping(Callback):
    """Stop training if energy variance stays below threshold."""

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        patience: int = 50,
        min_steps: int = 100,
    ):
        self.variance_threshold = variance_threshold
        self.patience = patience
        self.min_steps = min_steps
        self._below_count = 0

    def on_step(self, step: int, log_data: Dict[str, Any]) -> bool:
        if step < self.min_steps:
            return False

        variance = float(np.real(log_data.get("variance", float("inf"))))
        if variance < self.variance_threshold:
            self._below_count += 1
            if self._below_count >= self.patience:
                logger.info(
                    f"Early stopping at step {step}: "
                    f"variance {variance:.2e} < {self.variance_threshold:.2e} "
                    f"for {self.patience} consecutive steps"
                )
                return True
        else:
            self._below_count = 0
        return False


class CheckpointSaver(Callback):
    """Save model parameters periodically."""

    def __init__(self, experiment_dir: str | Path, save_every: int = 100):
        self.experiment_dir = Path(experiment_dir)
        self.save_every = save_every

    def on_step(self, step: int, log_data: Dict[str, Any]) -> bool:
        if step > 0 and step % self.save_every == 0:
            params = log_data.get("params")
            if params is not None:
                import jax
                ckpt_dir = self.experiment_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                path = ckpt_dir / f"params_step{step}.npz"
                flat_params = jax.tree.leaves(params)
                np.savez(path, *flat_params)
        return False
