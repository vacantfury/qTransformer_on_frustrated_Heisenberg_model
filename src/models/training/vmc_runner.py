"""
VMC training runner.

Orchestrates Variational Monte Carlo training for any BaseModel:
1. Build sampler (MCMC over spin configs)
2. Build SR preconditioner + optimizer
3. Run VMC iterations
4. Evaluate and save results to experiment directory
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base_model import BaseModel
from src.models.training.sr_optimizer import SRConfig, build_sr_preconditioner
from src.models.training.callbacks import (
    CallbackList, EnergyLogger, EarlyStopping, CheckpointSaver,
)
from src.evaluation.energy import save_energy_results, save_energy_history
from src.evaluation.entanglement import half_chain_entropy, save_entanglement_results

logger = logging.getLogger(__name__)


@dataclass
class VMCConfig:
    """Configuration for a VMC training run."""
    # Training
    n_steps: int = 500
    learning_rate: float = 0.01
    # Sampling
    n_samples: int = 1024
    n_chains: int = 1
    n_discard_per_chain: int = 16
    # SR
    sr: SRConfig = field(default_factory=SRConfig)
    # Callbacks
    log_every: int = 10
    checkpoint_every: int = 100
    early_stop_variance: float = 1e-6
    early_stop_patience: int = 50
    early_stop_min_steps: int = 100


def train(
    model: BaseModel,
    hilbert: Any,
    hamiltonian: Any,
    experiment_dir: str | Path,
    config: VMCConfig = VMCConfig(),
    E_exact: float | None = None,
) -> dict:
    """
    Run VMC training for a given model and Hamiltonian.

    Args:
        model: Any BaseModel subclass (RBM, ViT, QSANN, etc.).
        hilbert: NetKet Hilbert space.
        hamiltonian: NetKet Hamiltonian operator.
        experiment_dir: Path to save results.
        config: VMC training configuration.
        E_exact: Exact ground-state energy (for relative error).

    Returns:
        Dict with final energy, variance, n_params, relative_error.
    """
    import jax
    import netket as nk
    import optax

    experiment_dir = Path(experiment_dir)

    # Build sampler
    sampler = nk.sampler.MetropolisExchange(
        hilbert,
        n_chains=config.n_chains,
        d_max=1,
    )

    # Build variational state
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=config.n_samples,
        n_discard_per_chain=config.n_discard_per_chain,
    )

    n_params = vstate.n_parameters
    logger.info(f"Model: {model.__class__.__name__} | Parameters: {n_params}")

    # Build optimizer
    optimizer = optax.sgd(learning_rate=config.learning_rate)
    sr = build_sr_preconditioner(config.sr)

    # Build VMC driver
    gs = nk.driver.VMC(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        preconditioner=sr,
        variational_state=vstate,
    )

    # Set up callbacks
    energy_logger = EnergyLogger(log_every=config.log_every)
    callbacks = CallbackList([
        energy_logger,
        EarlyStopping(
            variance_threshold=config.early_stop_variance,
            patience=config.early_stop_patience,
            min_steps=config.early_stop_min_steps,
        ),
        CheckpointSaver(experiment_dir, save_every=config.checkpoint_every),
    ])

    # Run training
    logger.info(f"Starting VMC training for {config.n_steps} steps...")
    for step in range(config.n_steps):
        gs.advance()
        log_data = gs.energy

        step_data = {
            "energy": float(np.real(log_data.mean)),
            "variance": float(np.real(log_data.variance)),
            "params": vstate.parameters,
        }

        if callbacks.on_step(step + 1, step_data):
            logger.info(f"Training stopped early at step {step + 1}")
            break

    # Final energy
    final_energy = float(np.real(gs.energy.mean))
    final_variance = float(np.real(gs.energy.variance))
    N_sites = hilbert.size

    logger.info(
        f"Training complete | E = {final_energy:.8f} | "
        f"Var = {final_variance:.2e} | E/N = {final_energy/N_sites:.8f}"
    )

    # Save energy results
    save_energy_results(
        experiment_dir,
        energy=final_energy,
        variance=final_variance,
        E_exact=E_exact,
        N_sites=N_sites,
        extra={
            "method": "vmc",
            "model": model.__class__.__name__,
            "n_params": n_params,
            "n_steps": len(energy_logger.steps),
            "learning_rate": config.learning_rate,
        },
    )

    # Save energy history
    if energy_logger.steps:
        save_energy_history(
            experiment_dir,
            steps=energy_logger.steps,
            energies=energy_logger.energies,
            variances=energy_logger.variances,
        )

    return {
        "energy": final_energy,
        "variance": final_variance,
        "n_params": n_params,
        "relative_error": (
            abs(final_energy - E_exact) / abs(E_exact) if E_exact else None
        ),
    }
