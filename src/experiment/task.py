"""
Task runner for qTransformer experiments.
Dispatches to appropriate solver based on solution type.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.paths import OUTPUTS_DIR
from src.utils.experiment import create_experiment_dir

logger = logging.getLogger(__name__)


def _build_hamiltonian(ham_cfg: dict):
    """Build NetKet Hamiltonian from resolved config."""
    geometry = ham_cfg["geometry"]
    g = ham_cfg.get("g", 0.0)
    J1 = ham_cfg.get("J1", 1.0)
    J2 = g * J1

    if geometry == "chain":
        from src.hamiltonians.j1j2_chain import build_netket_hamiltonian, hamiltonian_id
        L = ham_cfg["L"]
        hilbert, H = build_netket_hamiltonian(L=L, J1=J1, J2=J2, pbc=ham_cfg.get("pbc", True))
        return hilbert, H, hamiltonian_id(L, g), L
    elif geometry == "square":
        from src.hamiltonians.j1j2_square import build_netket_hamiltonian, hamiltonian_id
        Lx, Ly = ham_cfg["Lx"], ham_cfg["Ly"]
        hilbert, H = build_netket_hamiltonian(Lx=Lx, Ly=Ly, J1=J1, J2=J2, pbc=ham_cfg.get("pbc", True))
        return hilbert, H, hamiltonian_id(Lx, Ly, g), Lx * Ly
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def _build_model(sol_cfg: dict):
    """Instantiate an NQS model from solution config."""
    sol_type = sol_cfg["type"]

    if sol_type == "rbm":
        from src.models.classical_models.rbm.model import RBM
        return RBM(alpha=sol_cfg.get("alpha", 1))
    elif sol_type == "cnn_resnet":
        from src.models.classical_models.cnn_resnet.model import CNNResNet
        return CNNResNet(
            features=sol_cfg.get("features", 32),
            n_res_blocks=sol_cfg.get("n_res_blocks", 4),
        )
    elif sol_type == "classical_vit":
        from src.models.classical_models.classical_vit.model import ClassicalViT
        return ClassicalViT(
            d_model=sol_cfg.get("d_model", 64),
            n_heads=sol_cfg.get("n_heads", 4),
            n_layers=sol_cfg.get("n_layers", 2),
        )
    elif sol_type == "simplified_vit":
        from src.models.classical_models.simplified_vit.model import SimplifiedViT
        return SimplifiedViT(
            d_model=sol_cfg.get("d_model", 64),
            n_heads=sol_cfg.get("n_heads", 4),
            n_layers=sol_cfg.get("n_layers", 2),
        )
    elif sol_type == "qsann":
        from src.models.quantum_models.qsann.model import QSANN
        return QSANN(
            n_qubits_per_token=sol_cfg.get("n_qubits_per_token", 4),
            n_pqc_layers=sol_cfg.get("n_pqc_layers", 2),
        )
    elif sol_type == "qmsan":
        from src.models.quantum_models.qmsan.model import QMSAN
        return QMSAN(
            n_qubits_per_token=sol_cfg.get("n_qubits_per_token", 4),
            n_pqc_layers=sol_cfg.get("n_pqc_layers", 2),
        )
    else:
        raise ValueError(f"Unknown solution type: {sol_type}")


def _get_ed_reference(ham_cfg: dict) -> float | None:
    """Get ED ground-state energy if system is small enough."""
    geometry = ham_cfg["geometry"]
    N = ham_cfg.get("L", 0) if geometry == "chain" else ham_cfg.get("Lx", 0) * ham_cfg.get("Ly", 0)
    if N > 20:
        logger.info(f"System too large for ED ({N} sites), skipping.")
        return None
    try:
        from src.numerical_solvers.ed.solver import solve
        E0, _ = solve(
            geometry=geometry, J1=ham_cfg.get("J1", 1.0), g=ham_cfg.get("g", 0.0),
            pbc=ham_cfg.get("pbc", True),
            L=ham_cfg.get("L"), Lx=ham_cfg.get("Lx"), Ly=ham_cfg.get("Ly"),
        )
        logger.info(f"ED reference: E0 = {E0:.8f}, E0/N = {E0/N:.8f}")
        return E0
    except Exception as e:
        logger.warning(f"ED failed: {e}")
        return None


def _ham_id(ham_cfg: dict) -> str:
    """Generate a human-readable Hamiltonian identifier."""
    geometry = ham_cfg["geometry"]
    g = ham_cfg.get("g", 0.0)
    if geometry == "chain":
        return f"chain_{ham_cfg['L']}_g{g:.1f}"
    else:
        return f"square_{ham_cfg['Lx']}x{ham_cfg['Ly']}_g{g:.1f}"


def run_task(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run a single task based on the solution type.

    Args:
        config: Resolved config dict with 'solution', 'hamiltonian',
                'training', 'evaluation' sub-dicts.

    Returns:
        Dict with results including energy and experiment_dir.
    """
    sol_cfg = config["solution"]
    ham_cfg = config["hamiltonian"]
    sol_type = sol_cfg["type"]
    sol_name = sol_cfg["name"]

    logger.info(f"Running task: solution={sol_name}, type={sol_type}")

    if sol_type == "ed":
        return _run_ed(config)
    elif sol_type == "dmrg":
        return _run_dmrg(config)
    else:
        return _run_vmc(config)


def _run_ed(config: dict) -> dict:
    """Run exact diagonalisation."""
    from src.numerical_solvers.ed.solver import run_and_save

    ham_cfg = config["hamiltonian"]
    hid = _ham_id(ham_cfg)
    experiment_dir = create_experiment_dir(OUTPUTS_DIR, hamiltonian=hid, method="ed")
    logger.info(f"Running ED → {experiment_dir}")

    E0, psi0 = run_and_save(
        experiment_dir=experiment_dir,
        geometry=ham_cfg["geometry"], J1=ham_cfg.get("J1", 1.0), g=ham_cfg.get("g", 0.0),
        pbc=ham_cfg.get("pbc", True),
        L=ham_cfg.get("L"), Lx=ham_cfg.get("Lx"), Ly=ham_cfg.get("Ly"),
    )
    geometry = ham_cfg["geometry"]
    N = ham_cfg.get("L") if geometry == "chain" else ham_cfg["Lx"] * ham_cfg["Ly"]
    logger.info(f"ED done: E0 = {E0:.8f}, E0/N = {E0/N:.8f}")
    return {"energy": E0, "experiment_dir": str(experiment_dir)}


def _run_dmrg(config: dict) -> dict:
    """Run DMRG."""
    from src.numerical_solvers.dmrg.solver import run_and_save

    ham_cfg = config["hamiltonian"]
    sol_cfg = config["solution"]
    hid = _ham_id(ham_cfg)
    experiment_dir = create_experiment_dir(OUTPUTS_DIR, hamiltonian=hid, method="dmrg")
    logger.info(f"Running DMRG → {experiment_dir}")

    results = run_and_save(
        experiment_dir=experiment_dir,
        geometry=ham_cfg["geometry"], J1=ham_cfg.get("J1", 1.0), g=ham_cfg.get("g", 0.0),
        pbc=ham_cfg.get("pbc", True),
        L=ham_cfg.get("L"), Lx=ham_cfg.get("Lx"), Ly=ham_cfg.get("Ly"),
        chi_max=sol_cfg.get("chi_max", 256),
        n_sweeps=sol_cfg.get("n_sweeps", 20),
    )
    E0 = results["energy"]
    geometry = ham_cfg["geometry"]
    N = ham_cfg.get("L") if geometry == "chain" else ham_cfg["Lx"] * ham_cfg["Ly"]
    logger.info(f"DMRG done: E0 = {E0:.8f}, E0/N = {E0/N:.8f}")
    return {"energy": E0, "experiment_dir": str(experiment_dir)}


def _run_vmc(config: dict) -> dict:
    """Run VMC training for a single NQS model."""
    import mlflow
    from src.models.training.vmc_runner import VMCConfig, train
    from src.models.training.sr_optimizer import SRConfig

    sol_cfg = config["solution"]
    ham_cfg = config["hamiltonian"]
    train_cfg = config["training"]

    sol_name = sol_cfg["name"]
    hilbert, H, hid, N = _build_hamiltonian(ham_cfg)
    E_exact = _get_ed_reference(ham_cfg)

    experiment_dir = create_experiment_dir(OUTPUTS_DIR, hamiltonian=hid, method=sol_name)
    logger.info(f"Running VMC: {sol_name} on {hid} → {experiment_dir}")

    model = _build_model(sol_cfg)
    vmc_config = VMCConfig(
        n_steps=train_cfg.get("n_steps", 500),
        learning_rate=train_cfg.get("learning_rate", 0.01),
        n_samples=train_cfg.get("n_samples", 1024),
        sr=SRConfig(diag_shift=train_cfg.get("diag_shift", 0.01)),
    )

    mlflow.set_experiment(f"vmc_{hid}")
    with mlflow.start_run(run_name=f"{sol_name}_g{ham_cfg.get('g', 0.0)}"):
        mlflow.log_params({
            "solution": sol_name, "g": ham_cfg.get("g", 0.0),
            "geometry": ham_cfg["geometry"], "N": N,
        })
        results = train(model, hilbert, H, experiment_dir, vmc_config, E_exact)
        mlflow.log_metrics({
            "final_energy": results["energy"],
            "final_variance": results["variance"],
            "n_params": results["n_params"],
        })
        if results["relative_error"] is not None:
            mlflow.log_metric("relative_error", results["relative_error"])

    logger.info(f"VMC done: E = {results['energy']:.8f}, params = {results['n_params']}")
    return results
