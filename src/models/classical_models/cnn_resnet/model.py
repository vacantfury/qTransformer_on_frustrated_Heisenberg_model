"""
CNN with residual blocks for NQS.

Convolutional neural network with skip connections, following
Liu et al. (2024) for the frustrated Heisenberg J1-J2 model.
Operates on 1D spin chains or flattened 2D lattices.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.models.base_model import BaseModel


class ResBlock(nn.Module):
    """Residual block: two convolutions with skip connection."""
    features: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.features, (self.kernel_size,), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Conv(self.features, (self.kernel_size,), padding="SAME")(x)
        # Match channel dim if needed
        if residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1,))(residual)
        return nn.gelu(x + residual)


class CNNResNet(BaseModel):
    """
    CNN + ResNet wave function ansatz.

    For 2D lattices, the input is kept as a 1D sequence (sites in row-major
    order); the CNN learns 1D correlations along the snake path. This is
    standard practice for CNN-NQS.

    Args:
        features: Number of channels in each conv layer.
        n_res_blocks: Number of residual blocks.
        kernel_size: Convolution kernel size.
        dtype: Parameter dtype.
    """
    features: int = 32
    n_res_blocks: int = 4
    kernel_size: int = 3
    dtype: type = complex

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x ∈ {+1,-1}^N → log ψ(x) ∈ ℂ"""
        # Reshape: (N,) → (N, 1) for 1D conv
        x = x.reshape(-1, 1).astype(self.dtype)

        # Initial projection
        x = nn.Conv(self.features, (self.kernel_size,), padding="SAME",
                     dtype=self.dtype)(x)
        x = nn.gelu(x)

        # Residual blocks
        for _ in range(self.n_res_blocks):
            x = ResBlock(self.features, self.kernel_size)(x)

        # Global pooling + output
        x = jnp.mean(x, axis=0)  # (features,)
        x = nn.Dense(1, dtype=self.dtype)(x)
        return x.squeeze(-1)  # scalar
