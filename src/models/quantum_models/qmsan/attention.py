"""
QMSAN attention mechanism.

Two modes controlled by `is_mixed_state`:
  - False (default): Pure-state swap test → |⟨ψ_q|ψ_k⟩|²
    Similarity computed on full pure states in 2^n-dim Hilbert space.
  - True: Mixed-state swap test → Tr(ρ_q ρ_k)
    Partial trace reduces states to density matrices on n_keep qubits.
    This is the original QMSAN approach (Chen et al. 2025).

Level 3 (fully quantum): similarity computed entirely in Hilbert space.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from src.models.quantum_models.qmsan.circuits import (
    swap_test_circuit,
    mixed_state_swap_test_circuit,
    mixed_state_value_circuit,
)


class SwapTestAttention(nn.Module):
    """
    Swap-test based quantum attention.

    Pure-state mode (is_mixed_state=False):
        α_{ij} = softmax_j(|⟨ψ_qi|ψ_kj⟩|²)
        Similarity is the fidelity of full pure states.

    Mixed-state mode (is_mixed_state=True):
        α_{ij} = softmax_j(Tr(ρ_qi ρ_kj))
        States are reduced via partial trace to n_keep qubits.
        Follows the original QMSAN paper (Chen et al. 2025).

    Args:
        n_qubits_per_token: Qubits per attention token.
        n_pqc_layers: Depth of Q/K/V circuits.
        is_mixed_state: If True, use mixed-state swap test with partial trace.
        n_keep_qubits: Number of qubits to keep (rest traced out).
                       Only used when is_mixed_state=True.
                       Defaults to n_qubits_per_token // 2.
    """
    n_qubits_per_token: int = 4
    n_pqc_layers: int = 2
    is_mixed_state: bool = False
    n_keep_qubits: int | None = None

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

        # Build swap test circuit based on mode
        if self.is_mixed_state:
            n_keep = self.n_keep_qubits if self.n_keep_qubits is not None else nq // 2
            swap_circuit = mixed_state_swap_test_circuit(nq, n_keep)
        else:
            swap_circuit = swap_test_circuit(nq)

        val_circuit = mixed_state_value_circuit(nq)

        # Compute swap-test overlap for all (i, j) pairs
        def compute_overlap(x_q, x_k):
            """Compute similarity via swap test."""
            return swap_circuit(x_q, x_k, params_q, params_k, nl)

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
