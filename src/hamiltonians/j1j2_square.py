"""
2D J1-J2 Heisenberg model on a square lattice.

H = J1 Σ_{<i,j>} S_i · S_j  +  J2 Σ_{<<i,j>>} S_i · S_j

Provides both QuSpin (for ED) and NetKet (for VMC) representations.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.hamiltonians.lattice_utils import square_neighbours


# ---------- QuSpin Hamiltonian ----------

def build_quspin_hamiltonian(
    Lx: int,
    Ly: int,
    J1: float = 1.0,
    J2: float = 0.0,
    pbc: bool = True,
) -> Any:
    """
    Build the 2D J1-J2 Heisenberg model as a QuSpin hamiltonian.

    Args:
        Lx: Number of columns.
        Ly: Number of rows.
        J1: Nearest-neighbour coupling.
        J2: Next-nearest-neighbour (diagonal) coupling.
        pbc: Periodic boundary conditions.

    Returns:
        QuSpin hamiltonian object.
    """
    from quspin.operators import hamiltonian as quspin_hamiltonian
    from quspin.basis import spin_basis_1d  # 1d basis works for any geometry

    N = Lx * Ly
    basis = spin_basis_1d(N, pauli=False)
    nn, nnn = square_neighbours(Lx, Ly, pbc=pbc)

    J_zz_nn = [[J1, i, j] for i, j in nn]
    J_pm_nn = [[0.5 * J1, i, j] for i, j in nn]

    J_zz_nnn = [[J2, i, j] for i, j in nnn]
    J_pm_nnn = [[0.5 * J2, i, j] for i, j in nnn]

    static = [
        ["zz", J_zz_nn + J_zz_nnn],
        ["+-", J_pm_nn + J_pm_nnn],
        ["-+", J_pm_nn + J_pm_nnn],
    ]
    dynamic = []

    return quspin_hamiltonian(static, dynamic, basis=basis, dtype=np.float64)


# ---------- NetKet Hamiltonian ----------

def build_netket_hamiltonian(
    Lx: int,
    Ly: int,
    J1: float = 1.0,
    J2: float = 0.0,
    pbc: bool = True,
) -> Any:
    """
    Build the 2D J1-J2 Heisenberg model as a NetKet operator.

    Args:
        Lx: Number of columns.
        Ly: Number of rows.
        J1: Nearest-neighbour coupling.
        J2: Next-nearest-neighbour coupling.
        pbc: Periodic boundary conditions.

    Returns:
        (hilbert, hamiltonian) — NetKet Hilbert space and operator.
    """
    import netket as nk

    N = Lx * Ly
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    nn, nnn = square_neighbours(Lx, Ly, pbc=pbc)

    ha = _heisenberg_sum(hilbert, nn, J1) + _heisenberg_sum(hilbert, nnn, J2)

    return hilbert, ha


def _heisenberg_sum(hilbert, pairs, J: float):
    """Sum of J * S_i · S_j over given pairs."""
    import netket as nk

    if abs(J) < 1e-15 or len(pairs) == 0:
        return 0.0 * nk.operator.spin.sigmax(hilbert, 0)

    ha = None
    for i, j in pairs:
        bond = (
            J * nk.operator.spin.sigmaz(hilbert, i) * nk.operator.spin.sigmaz(hilbert, j)
            + J * nk.operator.spin.sigmax(hilbert, i) * nk.operator.spin.sigmax(hilbert, j)
            + J * nk.operator.spin.sigmay(hilbert, i) * nk.operator.spin.sigmay(hilbert, j)
        ) * 0.25  # Factor of 1/4: σ = 2S
        ha = bond if ha is None else ha + bond

    return ha


# ---------- Convenience ----------

def hamiltonian_id(Lx: int, Ly: int, g: float) -> str:
    """Return a string identifier like 'square_4x4_g0.5'."""
    return f"square_{Lx}x{Ly}_g{g:.1f}"
