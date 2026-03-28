"""
QSANN NQS model.

Full QSANN wave function: spin config → tokenise → quantum attention → FFN → log ψ.

Tier 3, Level 2 (partial quantum): Q/K encoded in quantum space but projected
to classical before similarity computation (Gaussian kernel).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.models.quantum_models.base_quantum_model import BaseQuantumModel
from src.models.quantum_models.qsann.attention import GaussianProjectedAttention


class QSANN(BaseQuantumModel):
    """
    Quantum Self-Attention Neural Network for NQS.

    Inherits tokenisation, encoding, PQC layers, and FFN from BaseQuantumModel.
    Implements Gaussian-projected quantum attention (GPQSA).

    Args:
        n_qubits_per_token: Qubits per attention token.
        n_pqc_layers: Depth of Q/K/V PQC circuits.
        sigma: Gaussian kernel width.
        d_ff: Feed-forward hidden dimension.
        dtype: Output dtype.
    """
    sigma: float = 1.0

    def _quantum_attention(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Gaussian-projected quantum self-attention."""
        return GaussianProjectedAttention(
            n_qubits_per_token=self.n_qubits_per_token,
            n_pqc_layers=self.n_pqc_layers,
            sigma=self.sigma,
        )(tokens)
