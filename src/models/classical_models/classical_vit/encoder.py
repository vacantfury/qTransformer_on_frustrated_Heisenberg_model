"""
Transformer encoder block for ViT NQS.

Each block: LayerNorm → Attention → residual → LayerNorm → FFN → residual
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.models.classical_models.classical_vit.attention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    """
    Single transformer encoder block.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout_rate: Dropout rate (0 for no dropout).
        dtype: Parameter dtype.
    """
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    dropout_rate: float = 0.0
    dtype: type = complex

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: shape (n_tokens, d_model)
        Returns:
            shape (n_tokens, d_model)
        """
        # Self-attention with residual
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dtype=self.dtype,
        )(x)
        x = residual + x

        # Feed-forward with residual
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(self.d_ff, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, dtype=self.dtype)(x)
        x = residual + x

        return x
