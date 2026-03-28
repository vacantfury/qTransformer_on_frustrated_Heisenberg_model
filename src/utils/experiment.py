"""
Experiment directory utilities.
"""
import os
import time
from pathlib import Path


def create_experiment_dir(
    base_dir: str | Path,
    hamiltonian: str,
    method: str,
    tag: str | None = None,
) -> Path:
    """
    Create a structured experiment output directory.

    Layout: {base_dir}/{hamiltonian}/{method}_{timestamp}/
    With optional tag: {base_dir}/{hamiltonian}/{method}_{tag}_{timestamp}/

    Examples:
        outputs/square_4x4_g0.5/ed_20260324_033000/
        outputs/square_4x4_g0.5/rbm_20260324_033015/
        outputs/chain_20_g0.3/qsann_run2_20260324_033030/

    Args:
        base_dir: Root outputs directory (e.g., "outputs").
        hamiltonian: Hamiltonian identifier (e.g., "square_4x4_g0.5").
        method: Solver or ansatz name (e.g., "ed", "rbm", "qsann").
        tag: Optional extra tag (e.g., "run2", "debug").

    Returns:
        Path to the new experiment directory (already created).
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    parts = [method]
    if tag:
        parts.append(tag)
    parts.append(timestamp)
    folder_name = "_".join(parts)

    path = Path(base_dir) / hamiltonian / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path