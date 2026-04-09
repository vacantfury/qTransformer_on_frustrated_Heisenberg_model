"""
Multi-head dot-product self-attention for ViT NQS.

Standard transformer attention:
    Attention(Q, K, V) = softmax(Q K^T / √d_k) V
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class MultiHeadAttention(nn.Module):
    """
    Multi-head dot-product self-attention.

    Args:
        d_model: Model dimension (input/output size per token).
        n_heads: Number of attention heads.
        dtype: Parameter dtype.
    """
    d_model: int = 64
    n_heads: int = 4
    dtype: type = complex

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (..., n_tokens, d_model).

        Returns:
            Output tensor, same shape (..., n_tokens, d_model).
        """
        n_tokens = x.shape[-2]
        d_k = self.d_model // self.n_heads

        # Project to Q, K, V — Dense broadcasts over leading dims
        Q = nn.Dense(self.d_model, dtype=self.dtype, name="query")(x)
        K = nn.Dense(self.d_model, dtype=self.dtype, name="key")(x)
        V = nn.Dense(self.d_model, dtype=self.dtype, name="value")(x)

        # Reshape: (..., n_tokens, d_model) → (..., n_tokens, n_heads, d_k)
        new_shape = x.shape[:-1] + (self.n_heads, d_k)
        Q = Q.reshape(new_shape)
        K = K.reshape(new_shape)
        V = V.reshape(new_shape)

        # Move heads before tokens: (..., n_heads, n_tokens, d_k)
        head_axis = len(x.shape) - 1  # the n_heads axis after reshape
        perm = list(range(len(new_shape)))
        perm[head_axis - 1], perm[head_axis] = perm[head_axis], perm[head_axis - 1]
        Q = Q.transpose(perm)
        K = K.transpose(perm)
        V = V.transpose(perm)

        # Scaled dot-product attention
        scale = jnp.sqrt(jnp.array(d_k, dtype=self.dtype))
        attn_weights = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / scale
        attn_weights = nn.softmax(attn_weights.real, axis=-1).astype(self.dtype)

        # Apply attention
        out = jnp.matmul(attn_weights, V)  # (..., n_heads, n_tokens, d_k)

        # Swap heads and tokens back, then concat heads
        out = out.transpose(perm)  # (..., n_tokens, n_heads, d_k)
        out = out.reshape(x.shape[:-1] + (self.d_model,))

        # Output projection
        out = nn.Dense(self.d_model, dtype=self.dtype, name="output")(out)

        return out

