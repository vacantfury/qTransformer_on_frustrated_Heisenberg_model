"""
Task runner for qTransformer experiments.
Dispatches to appropriate solver based on solution type.

Receives Hydra DictConfig with typed dot access:
    config.solution.type, config.training.n_steps, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from omegaconf import OmegaConf, DictConfig

from src.paths import OUTPUTS_DIR, MLRUNS_DIR
from src.utils.experiment import create_experiment_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set MLflow to use local file store — must be set before any mlflow calls
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

def _build_hamiltonian(ham_cfg: DictConfig):
    """Build NetKet Hamiltonian and graph from resolved config."""
    import netket as nk

    geometry = ham_cfg.geometry
    g = ham_cfg.get("g", 0.0)
    J1 = ham_cfg.get("J1", 1.0)
    J2 = g * J1
    pbc = ham_cfg.get("pbc", True)

    if geometry == "chain":
        from src.hamiltonians.j1j2_chain import build_netket_hamiltonian, hamiltonian_id
        from src.hamiltonians.lattice_utils import chain_neighbours
        L = ham_cfg.L
        hilbert, H = build_netket_hamiltonian(L=L, J1=J1, J2=J2, pbc=pbc)
        nn, nnn = chain_neighbours(L, pbc=pbc)
        edges = nn + nnn
        graph = nk.graph.Graph(edges=edges)
        return hilbert, H, graph, hamiltonian_id(L, g), L
    elif geometry == "square":
        from src.hamiltonians.j1j2_square import build_netket_hamiltonian, hamiltonian_id
        from src.hamiltonians.lattice_utils import square_neighbours
        Lx, Ly = ham_cfg.Lx, ham_cfg.Ly
        hilbert, H = build_netket_hamiltonian(Lx=Lx, Ly=Ly, J1=J1, J2=J2, pbc=pbc)
        nn, nnn = square_neighbours(Lx, Ly, pbc=pbc)
        edges = nn + nnn
        graph = nk.graph.Graph(edges=edges)
        return hilbert, H, graph, hamiltonian_id(Lx, Ly, g), Lx * Ly
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def _build_model(sol_cfg: DictConfig):
    """Instantiate an NQS model from solution config."""
    from src.models.factory import create_model
    # Factory expects a dict-like object — DictConfig supports dict access
    return create_model(sol_cfg)



def _get_ed_reference(ham_cfg: DictConfig) -> float | None:
    """Get ED ground-state energy if system is small enough."""
    geometry = ham_cfg.geometry
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


def _ham_id(ham_cfg: DictConfig) -> str:
    """Generate a human-readable Hamiltonian identifier."""
    geometry = ham_cfg.geometry
    g = ham_cfg.get("g", 0.0)
    if geometry == "chain":
        return f"chain_{ham_cfg.L}_g{g:.1f}"
    else:
        return f"square_{ham_cfg.Lx}x{ham_cfg.Ly}_g{g:.1f}"


def run_task(config: DictConfig) -> dict[str, Any]:
    """
    Run a single task based on the solution type.

    Args:
        config: Hydra-resolved DictConfig with solution, hamiltonian,
                training, evaluation sub-configs.

    Returns:
        Dict with results including energy and experiment_dir.
    """
    sol_cfg = config.solution
    ham_cfg = config.hamiltonian
    sol_type = sol_cfg.type
    sol_name = sol_cfg.name

    logger.info(f"Running task: solution={sol_name}, type={sol_type}")

    if sol_type == "ed":
        return _run_ed(config)
    elif sol_type == "dmrg":
        return _run_dmrg(config)
    else:
        return _run_vmc(config)


# ==================== MLflow helpers ====================

def _mlflow_experiment_name(ham_cfg: DictConfig) -> str:
    """Consistent MLflow experiment name per Hamiltonian system."""
    return _ham_id(ham_cfg)


def _mlflow_common_params(config: DictConfig) -> dict:
    """Common parameters logged for ALL task types."""
    ham_cfg = config.hamiltonian
    sol_cfg = config.solution
    geometry = ham_cfg.geometry
    N = ham_cfg.get("L", 0) if geometry == "chain" else ham_cfg.get("Lx", 0) * ham_cfg.get("Ly", 0)
    return {
        "solution": sol_cfg.name,
        "method": sol_cfg.type,
        "geometry": geometry,
        "g": ham_cfg.get("g", 0.0),
        "N": N,
    }


# ==================== Task runners ====================

def _run_ed(config: DictConfig) -> dict:
    """Run exact diagonalisation."""
    from src.numerical_solvers.ed.solver import run_and_save

    ham_cfg = config.hamiltonian
    experiment_dir = create_experiment_dir(OUTPUTS_DIR, method="ed")
    logger.info(f"Running ED → {experiment_dir}")

    # Convert DictConfig to plain dict for saving in results.json
    task_config_dict = OmegaConf.to_container(config, resolve=True)

    mlflow.set_experiment(_mlflow_experiment_name(ham_cfg))
    with mlflow.start_run(run_name=f"ed_g{ham_cfg.get('g', 0.0)}"):
        mlflow.log_params(_mlflow_common_params(config))
        mlflow.log_param("experiment_dir", str(experiment_dir))

        E0, psi0 = run_and_save(
            experiment_dir=experiment_dir,
            geometry=ham_cfg.geometry, J1=ham_cfg.get("J1", 1.0), g=ham_cfg.get("g", 0.0),
            pbc=ham_cfg.get("pbc", True),
            L=ham_cfg.get("L"), Lx=ham_cfg.get("Lx"), Ly=ham_cfg.get("Ly"),
            config=task_config_dict,
        )

        geometry = ham_cfg.geometry
        N = ham_cfg.get("L") if geometry == "chain" else ham_cfg.Lx * ham_cfg.Ly
        mlflow.log_metrics({
            "energy": E0,
            "energy_per_site": E0 / N,
        })

    logger.info(f"ED done: E0 = {E0:.8f}, E0/N = {E0/N:.8f}")
    return {"energy": E0, "experiment_dir": str(experiment_dir)}


def _run_dmrg(config: DictConfig) -> dict:
    """Run DMRG."""
    from src.numerical_solvers.dmrg.solver import run_and_save

    ham_cfg = config.hamiltonian
    sol_cfg = config.solution
    experiment_dir = create_experiment_dir(OUTPUTS_DIR, method="dmrg")
    logger.info(f"Running DMRG → {experiment_dir}")

    task_config_dict = OmegaConf.to_container(config, resolve=True)

    mlflow.set_experiment(_mlflow_experiment_name(ham_cfg))
    with mlflow.start_run(run_name=f"dmrg_g{ham_cfg.get('g', 0.0)}"):
        mlflow.log_params({
            **_mlflow_common_params(config),
            "chi_max": sol_cfg.get("chi_max", 256),
            "n_sweeps": sol_cfg.get("n_sweeps", 20),
        })
        mlflow.log_param("experiment_dir", str(experiment_dir))

        results = run_and_save(
            experiment_dir=experiment_dir,
            geometry=ham_cfg.geometry, J1=ham_cfg.get("J1", 1.0), g=ham_cfg.get("g", 0.0),
            pbc=ham_cfg.get("pbc", True),
            L=ham_cfg.get("L"), Lx=ham_cfg.get("Lx"), Ly=ham_cfg.get("Ly"),
            chi_max=sol_cfg.get("chi_max", 256),
            n_sweeps=sol_cfg.get("n_sweeps", 20),
            config=task_config_dict,
        )

        E0 = results["energy"]
        geometry = ham_cfg.geometry
        N = ham_cfg.get("L") if geometry == "chain" else ham_cfg.Lx * ham_cfg.Ly
        mlflow.log_metrics({
            "energy": E0,
            "energy_per_site": E0 / N,
        })

    logger.info(f"DMRG done: E0 = {E0:.8f}, E0/N = {E0/N:.8f}")
    return {"energy": E0, "experiment_dir": str(experiment_dir)}


def _run_vmc(config: DictConfig) -> dict:
    """Run VMC training for a single NQS model."""
    from src.models.training.vmc_runner import VMCConfig, train
    from src.models.training.sr_optimizer import SRConfig

    sol_cfg = config.solution
    ham_cfg = config.hamiltonian
    train_cfg = config.training

    sol_name = sol_cfg.name
    sol_type = sol_cfg.type
    hilbert, H, graph, hid, N = _build_hamiltonian(ham_cfg)
    E_exact = _get_ed_reference(ham_cfg)

    experiment_dir = create_experiment_dir(OUTPUTS_DIR, method=sol_name)
    logger.info(f"Running VMC: {sol_name} on {hid} → {experiment_dir}")

    model = _build_model(sol_cfg)

    # Auto-detect holomorphic: RBM has complex params, everything else is real
    sr_cfg = train_cfg.get("sr", {})
    is_holomorphic = sr_cfg.get("holomorphic", sol_type == "rbm")

    vmc_config = VMCConfig(
        n_steps=train_cfg.n_steps,
        learning_rate=train_cfg.learning_rate,
        n_samples=train_cfg.n_samples,
        n_chains=train_cfg.get("n_chains", 1),
        n_discard_per_chain=train_cfg.get("n_discard_per_chain", 16),
        d_max=train_cfg.get("d_max", 1),
        sr=SRConfig(
            diag_shift=sr_cfg.get("diag_shift", 0.01),
            holomorphic=is_holomorphic,
        ),
        log_every=train_cfg.get("log_every", 10),
        checkpoint_every=train_cfg.get("checkpoint_every", 100),
        early_stop_variance=train_cfg.get("early_stop_variance", 1e-6),
        early_stop_patience=train_cfg.get("early_stop_patience", 50),
        early_stop_min_steps=train_cfg.get("early_stop_min_steps", 100),
    )

    # Convert DictConfig to plain dict for saving in results.json
    task_config_dict = OmegaConf.to_container(config, resolve=True)

    mlflow.set_experiment(_mlflow_experiment_name(ham_cfg))
    with mlflow.start_run(run_name=f"{sol_name}_g{ham_cfg.get('g', 0.0)}"):
        mlflow.log_params({
            **_mlflow_common_params(config),
            "n_steps": vmc_config.n_steps,
            "learning_rate": vmc_config.learning_rate,
            "n_samples": vmc_config.n_samples,
        })
        mlflow.log_param("experiment_dir", str(experiment_dir))

        results = train(model, hilbert, H, experiment_dir, vmc_config, E_exact,
                        resume_experiment_dir=OmegaConf.to_container(config, resolve=True).get("resume_experiment_dir"),
                        graph=graph,
                        full_config=task_config_dict)

        mlflow.log_metrics({
            "energy": results["energy"],
            "variance": results["variance"],
            "energy_per_site": results["energy"] / N,
            "n_params": results["n_params"],
        })
        if results["relative_error"] is not None:
            mlflow.log_metric("relative_error", results["relative_error"])

    logger.info(f"VMC done: E = {results['energy']:.8f}, params = {results['n_params']}")
    return results
