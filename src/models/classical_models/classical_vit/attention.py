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
            x: Input tensor, shape (n_tokens, d_model).

        Returns:
            Output tensor, same shape (n_tokens, d_model).
        """
        n_tokens, d = x.shape
        d_k = self.d_model // self.n_heads

        # Project to Q, K, V
        Q = nn.Dense(self.d_model, dtype=self.dtype, name="query")(x)
        K = nn.Dense(self.d_model, dtype=self.dtype, name="key")(x)
        V = nn.Dense(self.d_model, dtype=self.dtype, name="value")(x)

        # Reshape for multi-head: (n_tokens, d_model) → (n_heads, n_tokens, d_k)
        Q = Q.reshape(n_tokens, self.n_heads, d_k).transpose(1, 0, 2)
        K = K.reshape(n_tokens, self.n_heads, d_k).transpose(1, 0, 2)
        V = V.reshape(n_tokens, self.n_heads, d_k).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = jnp.sqrt(jnp.array(d_k, dtype=self.dtype))
        attn_weights = jnp.matmul(Q, K.transpose(0, 2, 1)) / scale  # (n_heads, n_tokens, n_tokens)
        attn_weights = nn.softmax(attn_weights.real, axis=-1).astype(self.dtype)

        # Apply attention to values
        out = jnp.matmul(attn_weights, V)  # (n_heads, n_tokens, d_k)

        # Concatenate heads
        out = out.transpose(1, 0, 2).reshape(n_tokens, self.d_model)

        # Output projection
        out = nn.Dense(self.d_model, dtype=self.dtype, name="output")(out)

        return out
