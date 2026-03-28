"""
QMSAN attention mechanism.

Mixed-State Quantum Self-Attention via swap test:
1. Encode each token via PQC → quantum Q/K states
2. Compute pairwise similarity via swap test: |⟨q_i|k_j⟩|²
3. Normalise to attention weights
4. Apply to quantum-computed values

Level 3 (fully quantum): similarity computed entirely in Hilbert space.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from src.models.quantum_models.qmsan.circuits import (
    swap_test_circuit,
    mixed_state_value_circuit,
)


class SwapTestAttention(nn.Module):
    """
    Swap-test based quantum attention.

    Attention coefficient α_{ij} = softmax_j(|⟨q_i|k_j⟩|²)
    where |⟨q_i|k_j⟩|² is estimated via the swap test circuit.

    Args:
        n_qubits_per_token: Qubits per attention token.
        n_pqc_layers: Depth of Q/K/V circuits.
    """
    n_qubits_per_token: int = 4
    n_pqc_layers: int = 2

    @nn.compact
    def __call__(self, x_tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x_tokens: (n_tokens, n_qubits_per_token).

        Returns:
            (n_tokens, n_qubits_per_token) — attention output.
        """
        n_tokens, nq = x_tokens.shape
        nl = self.n_pqc_layers

        # Learnable PQC parameters
        params_q = self.param("params_q", nn.initializers.normal(0.1),
                              (nl, nq, 2))
        params_k = self.param("params_k", nn.initializers.normal(0.1),
                              (nl, nq, 2))
        params_v = self.param("params_v", nn.initializers.normal(0.1),
                              (nl, nq, 2))

        # Build circuits
        swap_circuit = swap_test_circuit(nq)
        val_circuit = mixed_state_value_circuit(nq)

        # Compute swap-test overlap for all (i, j) pairs
        def compute_overlap(x_q, x_k):
            """⟨ancilla Z⟩ → |⟨ψ|φ⟩|² = 2*P(0) - 1"""
            z_expval = swap_circuit(x_q, x_k, params_q, params_k, nl)
            overlap = z_expval  # Already in [-1, 1], proportional to |⟨ψ|φ⟩|²
            return overlap

        # Build attention matrix: (n_tokens, n_tokens)
        def row_overlaps(x_q):
            return jax.vmap(lambda x_k: compute_overlap(x_q, x_k))(x_tokens)

        attn_logits = jax.vmap(row_overlaps)(x_tokens)

        # Softmax to get attention weights
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        # Compute values
        V = jax.vmap(lambda xi: jnp.array(val_circuit(xi, params_v, nl)))(x_tokens)

        # Apply attention
        out = attn_weights @ V  # (n_tokens, nq)

        return out
