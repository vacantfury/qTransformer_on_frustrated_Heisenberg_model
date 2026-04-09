"""
Stochastic Reconfiguration (SR) optimizer configuration.

SR preconditions gradient updates with the quantum geometric tensor
(Fisher information matrix), dramatically improving VMC convergence:

    δθ = -η S⁻¹ ∇E

where S_ij = Cov[O_i*, O_j] and O_k = ∂_{θ_k} log ψ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SRConfig:
    """
    Configuration for Stochastic Reconfiguration.

    Args:
        diag_shift: Regularisation added to S diagonal (prevents singularity).
        diag_scale: Multiplicative regularisation of S diagonal.
        holomorphic: Whether the model is holomorphic (complex params, no conjugates).
                     True for RBM with complex dtype.
                     False for real-param models (quantum circuits, ViTs, CNN).
        solver: Linear solver for S⁻¹ g. Options: "svd", "cholesky", "gmres".
    """
    diag_shift: float = 0.01
    diag_scale: float | None = None
    holomorphic: bool = False
    solver: str = "svd"


def build_sr_preconditioner(config: SRConfig) -> Any:
    """
    Build a NetKet SR preconditioner from configuration.

    Returns:
        NetKet nk.optimizer.SR object.
    """
    import netket as nk

    kwargs = {
        "diag_shift": config.diag_shift,
        "holomorphic": config.holomorphic,
    }
    if config.diag_scale is not None:
        kwargs["diag_scale"] = config.diag_scale

    return nk.optimizer.SR(**kwargs)
