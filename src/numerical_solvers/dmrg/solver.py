"""
DMRG via TeNPy.

Thin wrapper for density matrix renormalization group ground-state
calculations. For larger systems (6×6 and beyond) where ED is intractable.

All TeNPy imports are lazy (deferred to function calls) so that the
module can be imported without TeNPy installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from src.evaluation.results import save_results, build_energy_results


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
    # DMRG params
    chi_max: int = 256,
    n_sweeps: int = 20,
    mixer: bool = True,
) -> Dict[str, Any]:
    """
    Run DMRG and return ground-state energy and info.

    Args:
        geometry: "chain" or "square".
        J1: Nearest-neighbour coupling.
        g: Frustration ratio J2/J1.
        pbc: Periodic boundary conditions.
        L: Chain length (for geometry="chain").
        Lx, Ly: Lattice dimensions (for geometry="square").
        chi_max: Maximum bond dimension.
        n_sweeps: Number of DMRG sweeps.
        mixer: Whether to use a subspace expansion (mixer).

    Returns:
        Dict with keys: energy, chi_max, n_sweeps.
    """
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg as tenpy_dmrg

    J2 = g * J1

    if geometry == "chain":
        assert L is not None
        model = _build_chain_model(L, J1, J2, pbc)
        N = L
    elif geometry == "square":
        assert Lx is not None and Ly is not None
        model = _build_square_model(Lx, Ly, J1, J2, pbc)
        N = Lx * Ly
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Initial state: Néel order
    init_state = ["up", "down"] * (N // 2)
    if N % 2 == 1:
        init_state.append("up")
    psi = MPS.from_desired_bond_dimension(
        model.lat.mps_sites(), init_state[:N], bc="finite", bond_dim=4
    )

    # DMRG params
    dmrg_params = {
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-10},
        "mixer": mixer,
        "max_sweeps": n_sweeps,
        "min_sweeps": 4,
    }

    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0, _ = eng.run()

    return {
        "energy": float(E0),
        "chi_max": chi_max,
        "n_sweeps": n_sweeps,
    }


def run_and_save(
    experiment_dir: str | Path,
    geometry: str,
    J1: float = 1.0,
    g: float = 0.0,
    pbc: bool = True,
    L: int | None = None,
    Lx: int | None = None,
    Ly: int | None = None,
    chi_max: int = 256,
    n_sweeps: int = 20,
    mixer: bool = True,
    config: dict | None = None,
) -> Dict[str, Any]:
    """Run DMRG and save unified results.json."""
    results = solve(
        geometry=geometry, J1=J1, g=g, pbc=pbc,
        L=L, Lx=Lx, Ly=Ly,
        chi_max=chi_max, n_sweeps=n_sweeps, mixer=mixer,
    )

    N = L if geometry == "chain" else Lx * Ly

    save_results(
        experiment_dir,
        task_config=config,
        energy_results=build_energy_results(energy=results["energy"], N_sites=N),
    )

    return results


# ---------- Internal model builders (all imports inside functions) ----------

def _build_chain_model(L, J1, J2, pbc):
    """Build TeNPy J1-J2 chain model."""
    from tenpy.models.spins import SpinChain

    if abs(J2) < 1e-15:
        model_params = {
            "L": L,
            "Jx": J1,
            "Jy": J1,
            "Jz": J1,
            "bc_MPS": "finite",
            "conserve": "Sz",
        }
        return SpinChain(model_params)

    # J2 != 0: build custom J1-J2 model
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Chain
    from tenpy.networks.site import SpinHalfSite

    site = SpinHalfSite(conserve="Sz")
    lat = Chain(L, site, bc="periodic" if pbc else "open", bc_MPS="finite")

    model = CouplingMPOModel(lat)

    # J1 terms (nearest neighbour)
    model.add_coupling(J1, 0, "Sz", 0, "Sz", 1)
    model.add_coupling(0.5 * J1, 0, "Sp", 0, "Sm", 1)
    model.add_coupling(0.5 * J1, 0, "Sm", 0, "Sp", 1)

    # J2 terms (next-nearest neighbour)
    model.add_coupling(J2, 0, "Sz", 0, "Sz", 2)
    model.add_coupling(0.5 * J2, 0, "Sp", 0, "Sm", 2)
    model.add_coupling(0.5 * J2, 0, "Sm", 0, "Sp", 2)

    model.init_H_from_terms()
    return model


def _build_square_model(Lx, Ly, J1, J2, pbc):
    """Build TeNPy J1-J2 square lattice model."""
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Square
    from tenpy.networks.site import SpinHalfSite

    site = SpinHalfSite(conserve="Sz")
    bc = "periodic" if pbc else "open"
    lat = Square(Lx, Ly, site, bc=[bc, bc], bc_MPS="finite")

    model = CouplingMPOModel(lat)

    # J1: nearest neighbours
    for u1, u2, dx in lat.pairs["nearest_neighbors"]:
        model.add_coupling(J1, u1, "Sz", u2, "Sz", dx)
        model.add_coupling(0.5 * J1, u1, "Sp", u2, "Sm", dx)
        model.add_coupling(0.5 * J1, u1, "Sm", u2, "Sp", dx)

    # J2: next-nearest neighbours
    if abs(J2) > 1e-15:
        for u1, u2, dx in lat.pairs["next_nearest_neighbors"]:
            model.add_coupling(J2, u1, "Sz", u2, "Sz", dx)
            model.add_coupling(0.5 * J2, u1, "Sp", u2, "Sm", dx)
            model.add_coupling(0.5 * J2, u1, "Sm", u2, "Sp", dx)

    model.init_H_from_terms()
    return model
