"""
Restricted Boltzmann Machine (RBM) NQS.

The simplest NQS baseline: log ψ(x) = Σ_j log cosh(b_j + Σ_i W_ji x_i) + Σ_i a_i x_i

Uses NetKet's built-in RBM as the underlying implementation but
wraps it in our BaseModel interface.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import netket as nk

from src.models.base_model import BaseModel


class RBM(BaseModel):
    """
    RBM wave function ansatz.

    Args:
        alpha: Hidden unit density (n_hidden = alpha * n_visible).
        use_visible_bias: Whether to include visible bias terms.
        dtype: Parameter dtype (complex for complex-valued RBM).
    """
    alpha: int = 1
    use_visible_bias: bool = True
    dtype: type = complex

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x ∈ {+1,-1}^N → log ψ(x) ∈ ℂ"""
        return nk.models.RBM(
            alpha=self.alpha,
            use_visible_bias=self.use_visible_bias,
            dtype=self.dtype,
        )(x)
