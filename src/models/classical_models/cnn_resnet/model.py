"""
CNN with residual blocks for NQS.

Convolutional neural network with skip connections, following
Liu et al. (2024) for the frustrated Heisenberg J1-J2 model.
Operates on 1D spin chains or flattened 2D lattices.

Uses purely array-based 1D convolution (slice + matmul) to avoid
cuDNN, which is incompatible on certain GPU architectures.
All computation stays on GPU via cuBLAS.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.models.base_model import BaseModel


class Conv1D(nn.Module):
    """
    1D convolution via slice-and-matmul.

    Avoids all jax.lax.conv* ops (which XLA routes through cuDNN).
    Uses pure array indexing + matmul (cuBLAS), which works on all GPUs.
    """
    features: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        """x: (batch, L, C_in) → (batch, L, features)"""
        C_in = x.shape[-1]
        L = x.shape[-2]
        ks = self.kernel_size

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (ks, C_in, self.features),
        )
        bias = self.param("bias", nn.initializers.zeros_init(), (self.features,))

        # SAME padding
        pad_total = ks - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_padded = jnp.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)))

        # Extract patches via concatenation of sliced windows
        # Each slice: (batch, L, C_in) → stack → (batch, L, ks * C_in)
        patches = jnp.concatenate(
            [x_padded[:, i:i + L, :] for i in range(ks)],
            axis=-1,
        )  # (batch, L, ks * C_in)

        # Matmul with kernel (cuBLAS, not cuDNN)
        kernel_flat = kernel.reshape(-1, self.features)  # (ks * C_in, features)
        return patches @ kernel_flat + bias  # (batch, L, features)


class ResBlock(nn.Module):
    """Residual block: two convolutions with skip connection."""
    features: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        residual = x
        x = Conv1D(self.features, self.kernel_size)(x)
        x = nn.gelu(x)
        x = Conv1D(self.features, self.kernel_size)(x)
        # Match channel dim if needed
        if residual.shape[-1] != self.features:
            residual = Conv1D(self.features, 1)(residual)
        return nn.gelu(x + residual)


class CNNResNet(BaseModel):
    """
    CNN + ResNet wave function ansatz.

    For 2D lattices, the input is kept as a 1D sequence (sites in row-major
    order); the CNN learns 1D correlations along the snake path. This is
    standard practice for CNN-NQS.

    Uses slice-and-matmul Conv1D (cuBLAS) to ensure compatibility
    across all GPU architectures without cuDNN dependency.

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
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (batch, N) → (batch, 1)"""
        # Use float32 for convolutions
        x = x[..., jnp.newaxis].astype(jnp.float32)  # (batch, N, 1)

        # Initial projection
        x = Conv1D(self.features, self.kernel_size)(x)
        x = nn.gelu(x)

        # Residual blocks
        for _ in range(self.n_res_blocks):
            x = ResBlock(self.features, self.kernel_size)(x)

        # Global pooling over sites → (batch, features)
        x = jnp.mean(x, axis=-2)
        # Cast to complex for output Dense layer
        x = x.astype(self.dtype)
        # Output → (batch, 1)
        return nn.Dense(1, dtype=self.dtype)(x)
