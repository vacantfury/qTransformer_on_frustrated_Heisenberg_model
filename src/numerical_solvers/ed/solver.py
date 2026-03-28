"""
Exact Diagonalisation via QuSpin.

Thin wrapper that takes a Hamiltonian specification, builds the QuSpin
operator, and returns the ground-state energy and wavefunction. Results
are saved to the experiment directory via evaluation functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.evaluation.energy import save_energy_results
from src.evaluation.entanglement import half_chain_entropy, save_entanglement_results


def solve(
    geometry: str,
    J1: float = 1.0,
    g: float = 0.0,
    pbc: bool = True,
    # Chain params
    L: int | None = None,
    # Square params
    Lx: int | None = None,
    Ly: int | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Run exact diagonalisation and return ground-state energy + wavefunction.

    Args:
        geometry: "chain" or "square".
        J1: Nearest-neighbour coupling (sets the energy scale).
        g: Frustration ratio J2/J1.
        pbc: Periodic boundary conditions.
        L: Chain length (for geometry="chain").
        Lx, Ly: Lattice dimensions (for geometry="square").

    Returns:
        (E0, psi0) — ground-state energy and wavefunction vector.
    """
    J2 = g * J1

    if geometry == "chain":
        assert L is not None, "Must provide L for chain geometry"
        from src.hamiltonians.j1j2_chain import build_quspin_hamiltonian
        H = build_quspin_hamiltonian(L, J1=J1, J2=J2, pbc=pbc)
        N = L
    elif geometry == "square":
        assert Lx is not None and Ly is not None, "Must provide Lx, Ly for square geometry"
        from src.hamiltonians.j1j2_square import build_quspin_hamiltonian
        H = build_quspin_hamiltonian(Lx, Ly, J1=J1, J2=J2, pbc=pbc)
        N = Lx * Ly
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Compute ground state
    E0, psi0 = H.eigsh(k=1, which="SA")
    E0 = E0[0]
    psi0 = psi0[:, 0]

    return E0, psi0


def run_and_save(
    experiment_dir: str | Path,
    geometry: str,
    J1: float = 1.0,
    g: float = 0.0,
    pbc: bool = True,
    L: int | None = None,
    Lx: int | None = None,
    Ly: int | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Run ED, save energy and entanglement results to experiment directory.

    Returns:
        (E0, psi0)
    """
    E0, psi0 = solve(
        geometry=geometry, J1=J1, g=g, pbc=pbc,
        L=L, Lx=Lx, Ly=Ly,
    )

    N = L if geometry == "chain" else Lx * Ly

    # Save energy
    save_energy_results(
        experiment_dir,
        energy=E0,
        N_sites=N,
        extra={"method": "exact_diag", "geometry": geometry, "g": g},
    )

    # Save entanglement (half-chain bipartition)
    entropy = half_chain_entropy(psi0, N)
    save_entanglement_results(
        experiment_dir,
        entropy=entropy,
        subsystem_sites=list(range(N // 2)),
        N_sites=N,
    )

    # Save wavefunction
    psi_path = Path(experiment_dir) / "wavefunction.npy"
    np.save(psi_path, psi0)

    return E0, psi0
