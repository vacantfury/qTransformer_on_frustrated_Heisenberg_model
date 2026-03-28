"""
Energy evaluation utilities.

Per-experiment functions to compute and save energy metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np


def relative_error(E_var: float, E_exact: float) -> float:
    """Relative energy error: |E_var - E_exact| / |E_exact|."""
    return abs(E_var - E_exact) / abs(E_exact)


def energy_per_site(E: float, N: int) -> float:
    """Energy per lattice site."""
    return E / N


def save_energy_results(
    experiment_dir: str | Path,
    energy: float,
    variance: float | None = None,
    E_exact: float | None = None,
    N_sites: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """
    Save energy results to an experiment directory.

    Creates `energy_results.json` in the experiment dir with:
    - energy, variance, energy_per_site
    - relative_error (if E_exact provided)
    - any extra metadata

    Args:
        experiment_dir: Path to experiment output directory.
        energy: Variational or exact energy.
        variance: Energy variance (for VMC results).
        E_exact: Exact reference energy (for computing relative error).
        N_sites: Number of lattice sites.
        extra: Additional metadata to save.

    Returns:
        Path to the saved JSON file.
    """
    results: Dict[str, Any] = {"energy": float(energy)}

    if variance is not None:
        results["variance"] = float(variance)
    if N_sites is not None:
        results["energy_per_site"] = energy_per_site(energy, N_sites)
    if E_exact is not None:
        results["E_exact"] = float(E_exact)
        results["relative_error"] = relative_error(energy, E_exact)
    if extra:
        results.update(extra)

    path = Path(experiment_dir) / "energy_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    return path


def save_energy_history(
    experiment_dir: str | Path,
    steps: list[int],
    energies: list[float],
    variances: list[float] | None = None,
) -> Path:
    """
    Save per-step energy convergence history.

    Creates `energy_history.npz` with arrays: steps, energies, variances.
    """
    data = {"steps": np.array(steps), "energies": np.array(energies)}
    if variances is not None:
        data["variances"] = np.array(variances)

    path = Path(experiment_dir) / "energy_history.npz"
    np.savez(path, **data)
    return path


def load_energy_results(experiment_dir: str | Path) -> Dict[str, Any]:
    """Load energy results from an experiment directory."""
    path = Path(experiment_dir) / "energy_results.json"
    with open(path) as f:
        return json.load(f)


def load_energy_history(experiment_dir: str | Path) -> Dict[str, np.ndarray]:
    """Load energy convergence history."""
    path = Path(experiment_dir) / "energy_history.npz"
    return dict(np.load(path))
