"""
Base model class for all NQS wave function ansätze.

Every model (RBM, CNN, ViT, QSANN, QMSAN) subclasses BaseModel
and implements forward() to map spin configurations to log ψ.

BaseModel.__call__ handles the batch-dimension contract with NetKet:
  - NetKet may call with (N,) or (batch, N).
  - Subclasses always receive (batch, N) in forward().
  - Output is normalised to scalar / (batch,) automatically.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class BaseModel(nn.Module):
    """
    Abstract base class for NQS wave function ansätze.

    Subclasses must implement forward() to define the mapping:
        x ∈ {+1, -1}^(batch, N)  →  log ψ(x) ∈ ℂ^(batch,) or ℂ^(batch, 1)

    __call__ normalises the input/output dimensions so that:
      - (N,)       → forward gets (1, N)  → returns scalar
      - (batch, N) → forward gets (batch, N) → returns (batch,)

    This interface is compatible with NetKet's VMC and VMC_SR drivers.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Public API called by NetKet. Handles batch dims automatically.

        Args:
            x: Spin configuration(s), shape (N,) or (batch, N).

        Returns:
            log ψ(x): scalar for single sample, (batch,) for batched.
        """
        batched = x.ndim > 1

        if not batched:
            x = x[jnp.newaxis, :]  # (1, N)

        out = self.forward(x)  # subclass: (batch, N) → (batch, 1) or (batch,)

        # Normalise output shape
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)  # (batch, 1) → (batch,)

        if not batched:
            out = out.squeeze(0)  # (batch,) → scalar

        return out

    @nn.compact
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Core forward pass. Subclasses must implement this.

        Args:
            x: Spin configurations, shape (batch, N), values in {+1, -1}.

        Returns:
            log ψ(x), shape (batch, 1) or (batch,).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

