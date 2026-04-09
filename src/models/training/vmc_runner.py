"""
VMC training runner.

Orchestrates Variational Monte Carlo training for any BaseModel:
1. Build sampler (MCMC over spin configs)
2. Build SR preconditioner + optimizer
3. Run VMC iterations
4. Evaluate and save results to experiment directory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import time

from src.models.base_model import BaseModel
from src.models.training.sr_optimizer import SRConfig, build_sr_preconditioner
from src.models.training.callbacks import (
    CallbackList, EnergyLogger, EarlyStopping, CheckpointSaver,
    find_latest_checkpoint,
)
from src.evaluation.results import (
    save_results, build_energy_results, build_entanglement_results,
    save_energy_history, load_energy_history,
)
from src.evaluation.entanglement import half_chain_entropy
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    # Sampler
    d_max: int = 1  # Max exchange distance for MetropolisExchange. Increase for frustrated systems.
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
    vmc_config: VMCConfig = VMCConfig(),
    E_exact: float | None = None,
    resume_experiment_dir: str | Path | None = None,
    graph: Any = None,
    full_config: dict | None = None,
) -> dict:
    """
    Run VMC training for a given model and Hamiltonian.

    Args:
        model: Any BaseModel subclass (RBM, ViT, QSANN, etc.).
        hilbert: NetKet Hilbert space.
        hamiltonian: NetKet Hamiltonian operator.
        experiment_dir: Path to save results.
        vmc_config: VMC training configuration.
        E_exact: Exact ground-state energy (for relative error).
        resume_experiment_dir: If set and contains checkpoints, resume from latest.
        graph: NetKet graph for MetropolisExchange sampler.
        full_config: Full resolved task config (solution, hamiltonian, training)
            saved to results.json for reproducibility.

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
        graph=graph,
        n_chains=vmc_config.n_chains,
        d_max=vmc_config.d_max,
    )

    # Build variational state
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=vmc_config.n_samples,
        n_discard_per_chain=vmc_config.n_discard_per_chain,
    )

    n_params = vstate.n_parameters
    logger.info(f"Model: {model.__class__.__name__} | Parameters: {n_params}")

    # Build optimizer
    optimizer = optax.sgd(learning_rate=vmc_config.learning_rate)

    # Build VMC driver with SR
    # VMC_SR auto-detects holomorphic via nk.utils.is_probably_holomorphic().
    # We log the configured value for diagnostics (it's set per solution type).
    logger.info(f"SR config: diag_shift={vmc_config.sr.diag_shift}, holomorphic={vmc_config.sr.holomorphic}")
    gs = nk.driver.VMC_SR(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        diag_shift=vmc_config.sr.diag_shift,
        variational_state=vstate,
    )

    # Set up callbacks
    energy_logger = EnergyLogger(log_every=vmc_config.log_every)
    callbacks = CallbackList([
        energy_logger,
        EarlyStopping(
            variance_threshold=vmc_config.early_stop_variance,
            patience=vmc_config.early_stop_patience,
            min_steps=vmc_config.early_stop_min_steps,
        ),
        CheckpointSaver(experiment_dir, save_every=vmc_config.checkpoint_every),
    ])

    # Check for resume
    start_step = 0
    if resume_experiment_dir:
        checkpoint = find_latest_checkpoint(resume_experiment_dir)
        if checkpoint:
            vstate.parameters = checkpoint["params"]
            start_step = checkpoint["step"]
            logger.info(f"Resumed from step {start_step}")
        else:
            logger.info(f"No checkpoint found in {resume_experiment_dir}, starting fresh")

    # Run training
    remaining_steps = vmc_config.n_steps - start_step
    train_start_time = time.time()
    if remaining_steps <= 0:
        logger.info(f"Already completed {start_step}/{vmc_config.n_steps} steps, skipping training")
    else:
        logger.info(f"Starting VMC training: steps {start_step+1}..{vmc_config.n_steps}")
        for step_idx in range(remaining_steps):
            gs.advance()
            step = start_step + step_idx + 1
            log_data = vstate.expect(hamiltonian)

            step_data = {
                "energy": float(np.real(log_data.mean)),
                "variance": float(np.real(log_data.variance)),
                "params": vstate.parameters,
            }

            if callbacks.on_step(step, step_data):
                logger.info(f"Training stopped early at step {step}")
                break
    train_elapsed = time.time() - train_start_time

    # Final energy
    final_log_data = vstate.expect(hamiltonian)
    final_energy = float(np.real(final_log_data.mean))
    final_variance = float(np.real(final_log_data.variance))
    final_error_of_mean = float(np.real(final_log_data.error_of_mean))
    N_sites = hilbert.size

    logger.info(
        f"Training complete | E = {final_energy:.8f} ± {final_error_of_mean:.2e} | "
        f"Var = {final_variance:.2e} | E/N = {final_energy/N_sites:.8f} | "
        f"Time = {train_elapsed:.1f}s"
    )

    # Build parameters section
    steps_this_run = len(energy_logger.steps)
    total_steps = start_step + steps_this_run

    # Save unified results.json
    save_results(
        experiment_dir,
        task_config=full_config,
        energy_results=build_energy_results(
            energy=final_energy,
            variance=final_variance,
            error_of_mean=final_error_of_mean,
            E_exact=E_exact,
            N_sites=N_sites,
            # Runtime stats
            model=model.__class__.__name__,
            n_params=n_params,
            n_steps=vmc_config.n_steps,
            total_steps=total_steps,
            steps_this_run=steps_this_run,
            training_time_seconds=round(train_elapsed, 2),
        ),
    )

    # Save energy history (merge with previous run if resumed)
    if energy_logger.steps:
        steps = energy_logger.steps
        energies = energy_logger.energies
        variances = energy_logger.variances

        # Prepend history from the resumed run
        if resume_experiment_dir and start_step > 0:
            try:
                prev = load_energy_history(resume_experiment_dir)
                steps = list(prev["steps"]) + steps
                energies = list(prev["energies"]) + energies
                if "variances" in prev:
                    variances = list(prev["variances"]) + variances
                logger.info(f"Merged energy history: {len(prev['steps'])} prev + {len(energy_logger.steps)} new steps")
            except Exception as e:
                logger.warning(f"Could not load previous energy history: {e}")

        save_energy_history(experiment_dir, steps=steps, energies=energies,
                            variances=variances, wall_times=energy_logger.wall_times)

    return {
        "energy": final_energy,
        "variance": final_variance,
        "n_params": n_params,
        "relative_error": (
            abs(final_energy - E_exact) / abs(E_exact) if E_exact else None
        ),
    }

