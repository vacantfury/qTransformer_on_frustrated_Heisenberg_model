"""
Vision Transformer (ViT) NQS model.

Full classical ViT wave function following Viteritti et al. (2023, PRL):
    spin config → patch embedding + positional encoding → transformer encoder → log ψ

Each "patch" is a single site (patch_size=1), making this equivalent to
a standard transformer operating on per-site embeddings.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.models.base_model import BaseModel
from src.models.classical_models.classical_vit.encoder import TransformerEncoder


class ClassicalViT(BaseModel):
    """
    Vision Transformer NQS.

    Args:
        d_model: Embedding dimension per site.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder blocks.
        d_ff: Feed-forward hidden dimension.
        dtype: Parameter dtype.
    """
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dtype: type = complex

    @nn.compact
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (batch, N) → (batch, 1)"""
        N = x.shape[-1]

        # Per-site embedding: (batch, N) → (batch, N, d_model)
        x = x[..., jnp.newaxis].astype(self.dtype)  # (batch, N, 1)
        x = nn.Dense(self.d_model, dtype=self.dtype, name="site_embed")(x)

        # Learnable positional encoding
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (N, self.d_model),
        )
        x = x + pos_embed.astype(self.dtype)  # broadcasts over batch

        # Transformer encoder stack
        for i in range(self.n_layers):
            x = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dtype=self.dtype,
                name=f"encoder_{i}",
            )(x)

        # Final LayerNorm
        x = nn.LayerNorm(dtype=self.dtype)(x)

        # Global average pooling over sites → (batch, d_model)
        x = jnp.mean(x, axis=-2)
        return nn.Dense(1, dtype=self.dtype, name="output")(x)  # (batch, 1)
