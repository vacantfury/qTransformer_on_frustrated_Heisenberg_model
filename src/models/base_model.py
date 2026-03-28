"""
Base model class for all NQS wave function ansätze.

Every model (RBM, CNN, ViT, QSANN, QMSAN) subclasses BaseModel
and implements __call__ to map spin configurations to log ψ.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class BaseModel(nn.Module):
    """
    Abstract base class for NQS wave function ansätze.

    Subclasses must implement __call__ to define the mapping:
        x ∈ {+1, -1}^N  →  log ψ(x) ∈ ℂ

    This interface is compatible with NetKet's VMC driver, which
    expects a Flax nn.Module with this calling convention.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Map a spin configuration to log-amplitude.

        Args:
            x: Spin configuration, shape (N,) with values in {+1, -1}.

        Returns:
            log ψ(x), a complex scalar.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__"
        )
