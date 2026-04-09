"""
Unified experiment results I/O.

All experiment outputs are saved to a single `results.json` file with
three top-level sections:

    {
        "parameters": { ... },           # Experiment config / metadata
        "energy_results": { ... },        # Energy, variance, relative error
        "entanglement_results": { ... }   # Entropy, partitions
    }

Binary data (energy_history.npz, wavefunction.npy) remain as separate files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


RESULTS_FILE = "results.json"


# ==================== Save / Load ====================

def save_results(
    experiment_dir: str | Path,
    task_config: Dict[str, Any] | None = None,
    energy_results: Dict[str, Any] | None = None,
    entanglement_results: Dict[str, Any] | None = None,
) -> Path:
    """
    Save or update the unified results.json in experiment_dir.

    If results.json already exists, sections are merged (updated).

    Sections:
        task_config: Full resolved config for reproducibility
            (solution, hamiltonian, training, evaluation dicts).
        energy_results: Computed outputs (energy, variance, n_params, timing).
        entanglement_results: Entropy and partition info (ED only).
    """
    experiment_dir = Path(experiment_dir)
    path = experiment_dir / RESULTS_FILE

    # Load existing if present
    existing: Dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    # Merge sections
    if task_config is not None:
        existing["task_config"] = task_config
    if energy_results is not None:
        existing.setdefault("energy_results", {}).update(energy_results)
    if entanglement_results is not None:
        existing.setdefault("entanglement_results", {}).update(entanglement_results)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=_json_default)

    return path


def load_results(experiment_dir: str | Path) -> Dict[str, Any]:
    """Load the unified results.json from experiment_dir."""
    path = Path(experiment_dir) / RESULTS_FILE
    with open(path) as f:
        return json.load(f)


# ==================== Convenience builders ====================

def build_energy_results(
    energy: float,
    N_sites: int | None = None,
    variance: float | None = None,
    error_of_mean: float | None = None,
    E_exact: float | None = None,
    **extra,
) -> Dict[str, Any]:
    """Build the energy_results section."""
    results: Dict[str, Any] = {"energy": float(energy)}

    if N_sites is not None:
        results["energy_per_site"] = float(energy) / N_sites
    if variance is not None:
        results["variance"] = float(variance)
    if error_of_mean is not None:
        results["error_of_mean"] = float(error_of_mean)
    if E_exact is not None:
        results["E_exact"] = float(E_exact)
        results["relative_error"] = abs(energy - E_exact) / abs(E_exact)

    results.update(extra)
    return results


def build_entanglement_results(
    entropy: float,
    subsystem_sites: list[list[int]] | None = None,
    N_sites: int | None = None,
) -> Dict[str, Any]:
    """Build the entanglement_results section."""
    results: Dict[str, Any] = {"entropy": float(entropy)}
    if subsystem_sites is not None:
        results["subsystem_sites"] = subsystem_sites
    if N_sites is not None:
        results["N_sites"] = N_sites
    return results


# ==================== Energy history (binary) ====================

def save_energy_history(
    experiment_dir: str | Path,
    steps: list[int],
    energies: list[float],
    variances: list[float] | None = None,
    wall_times: list[float] | None = None,
) -> Path:
    """Save per-step convergence history as energy_history.npz."""
    data = {"steps": np.array(steps), "energies": np.array(energies)}
    if variances is not None:
        data["variances"] = np.array(variances)
    if wall_times is not None:
        data["wall_times"] = np.array(wall_times)

    path = Path(experiment_dir) / "energy_history.npz"
    np.savez(path, **data)
    return path


def load_energy_history(experiment_dir: str | Path) -> Dict[str, np.ndarray]:
    """Load energy convergence history."""
    path = Path(experiment_dir) / "energy_history.npz"
    return dict(np.load(path))


# ==================== Internal ====================

def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
