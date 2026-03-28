"""
Entanglement entropy evaluation.

Computes subsystem Von Neumann entropy from wavefunctions.
Per-experiment: results saved in the experiment directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np


def von_neumann_entropy(
    psi: np.ndarray,
    N: int,
    subsystem_sites: List[int],
) -> float:
    """
    Compute the Von Neumann entanglement entropy S = -Tr(ρ_A log ρ_A)
    for a subsystem A of a pure state |ψ⟩.

    Args:
        psi: Full wavefunction as a vector of length 2^N.
        N: Total number of sites.
        subsystem_sites: List of site indices in subsystem A.

    Returns:
        Von Neumann entropy of subsystem A.
    """
    N_A = len(subsystem_sites)
    N_B = N - N_A

    if N_A == 0 or N_A == N:
        return 0.0

    # Build the list of all site indices, subsystem A first then B
    all_sites = list(range(N))
    subsystem_B = [s for s in all_sites if s not in subsystem_sites]
    perm = subsystem_sites + subsystem_B

    # Reshape psi into a matrix (A × B)
    psi_tensor = psi.reshape([2] * N)
    psi_tensor = psi_tensor.transpose(perm)
    psi_matrix = psi_tensor.reshape(2**N_A, 2**N_B)

    # SVD to get Schmidt values
    s = np.linalg.svd(psi_matrix, compute_uv=False)
    s = s[s > 1e-15]  # Remove numerical zeros

    # S = -Σ |λ|² log(|λ|²)
    probs = s**2
    entropy = -np.sum(probs * np.log(probs))

    return float(entropy)


def half_chain_entropy(psi: np.ndarray, N: int) -> float:
    """Entanglement entropy for a half-chain bipartition (sites 0..N//2-1)."""
    return von_neumann_entropy(psi, N, list(range(N // 2)))


def save_entanglement_results(
    experiment_dir: str | Path,
    entropy: float,
    subsystem_sites: List[int] | None = None,
    N_sites: int | None = None,
) -> Path:
    """Save entanglement entropy results to experiment directory."""
    results = {"entropy": float(entropy)}
    if subsystem_sites is not None:
        results["subsystem_sites"] = subsystem_sites
    if N_sites is not None:
        results["N_sites"] = N_sites

    path = Path(experiment_dir) / "entanglement_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def load_entanglement_results(experiment_dir: str | Path) -> dict:
    """Load entanglement results from experiment directory."""
    path = Path(experiment_dir) / "entanglement_results.json"
    with open(path) as f:
        return json.load(f)
