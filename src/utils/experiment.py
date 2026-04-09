"""
Experiment directory utilities.
"""
import os
import time
import uuid
from pathlib import Path


def create_experiment_dir(
    base_dir: str | Path,
    method: str,
    tag: str | None = None,
) -> Path:
    """
    Create a structured experiment output directory.

    Layout: {base_dir}/{method}/{timestamp}_{random}_{method}/
    With optional tag: {base_dir}/{method}/{timestamp}_{random}_{tag}_{method}/

    Examples:
        outputs/ed/20260324_033000_a3f2_ed/
        outputs/rbm/20260324_033015_b7c1_rbm/
        outputs/qmsan/20260324_033030_c5d3_qmsan/
        outputs/qmsan/20260324_033030_d8e4_debug_qmsan/

    Args:
        base_dir: Root outputs directory (e.g., "outputs").
        method: Solver or ansatz name (e.g., "ed", "rbm", "qsann").
        tag: Optional extra tag (e.g., "debug").

    Returns:
        Path to the new experiment directory (already created).
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:4]

    parts = [timestamp, short_id]
    if tag:
        parts.append(tag)
    parts.append(method)
    folder_name = "_".join(parts)

    path = Path(base_dir) / method / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path