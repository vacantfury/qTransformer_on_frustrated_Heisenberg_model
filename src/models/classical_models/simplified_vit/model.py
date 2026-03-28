"""
Simplified ViT (position-only attention) NQS.

Following Rende, Viteritti & Becca (2025): removes Q/K computation
entirely, using only learnable positional attention weights.
This is the Level 0 baseline in our quantum spectrum.

Key insight: for frustrated spin systems, standard dot-product attention
adds minimal expressivity beyond position-only encoding.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.models.base_model import BaseModel


class PositionOnlyAttention(nn.Module):
    """
    Position-only attention: no queries, no keys.

    Attention weights are learned as a (n_tokens × n_tokens) matrix
    that depends only on relative positions, not on input data.
    """
    n_heads: int = 4
    d_model: int = 64
    dtype: type = complex

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (n_tokens, d_model) → (n_tokens, d_model)"""
        n_tokens = x.shape[0]
        d_k = self.d_model // self.n_heads

        # Only V projection — no Q, K
        V = nn.Dense(self.d_model, dtype=self.dtype, name="value")(x)
        V = V.reshape(n_tokens, self.n_heads, d_k).transpose(1, 0, 2)

        # Learnable positional attention weights (input-independent)
        attn_logits = self.param(
            "pos_attn_logits",
            nn.initializers.normal(stddev=0.02),
            (self.n_heads, n_tokens, n_tokens),
        )
        attn_weights = nn.softmax(attn_logits.real, axis=-1).astype(self.dtype)

        # Apply fixed attention to values
        out = jnp.matmul(attn_weights, V)  # (n_heads, n_tokens, d_k)
        out = out.transpose(1, 0, 2).reshape(n_tokens, self.d_model)

        out = nn.Dense(self.d_model, dtype=self.dtype, name="output")(out)
        return out


class SimplifiedViT(BaseModel):
    """
    Simplified ViT with position-only attention (no Q/K).

    Same architecture as ClassicalViT but with PositionOnlyAttention
    replacing MultiHeadAttention.

    Args:
        d_model: Embedding dimension per site.
        n_heads: Number of attention heads.
        n_layers: Number of encoder blocks.
        d_ff: Feed-forward hidden dimension.
        dtype: Parameter dtype.
    """
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dtype: type = complex

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x ∈ {+1,-1}^N → log ψ(x) ∈ ℂ"""
        N = x.shape[0]

        # Per-site embedding
        x = x.reshape(N, 1).astype(self.dtype)
        x = nn.Dense(self.d_model, dtype=self.dtype, name="site_embed")(x)

        # Positional encoding
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (N, self.d_model),
        )
        x = x + pos_embed.astype(self.dtype)

        # Simplified encoder blocks
        for i in range(self.n_layers):
            # Attention with residual
            residual = x
            x = nn.LayerNorm(dtype=self.dtype)(x)
            x = PositionOnlyAttention(
                n_heads=self.n_heads,
                d_model=self.d_model,
                dtype=self.dtype,
                name=f"pos_attn_{i}",
            )(x)
            x = residual + x

            # FFN with residual
            residual = x
            x = nn.LayerNorm(dtype=self.dtype)(x)
            x = nn.Dense(self.d_ff, dtype=self.dtype)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.d_model, dtype=self.dtype)(x)
            x = residual + x

        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = jnp.mean(x, axis=0)
        log_psi = nn.Dense(1, dtype=self.dtype, name="output")(x)

        return log_psi.squeeze(-1)
