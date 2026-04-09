"""
QSANN attention mechanism.

Gaussian-Projected Quantum Self-Attention (GPQSA):
1. Encode each token (site) into quantum state via PQC
2. Measure expectation values → classical Q, K vectors
3. Compute attention weights classically via Gaussian kernel:
   α_{ij} = exp(-||q_i - k_j||² / 2σ²) / Z
4. Apply attention to quantum-computed values

Level 2 (partial quantum): quantum encoding, classical similarity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import pennylane as qml
import numpy as np

from src.models.quantum_models.device import get_device, get_diff_method


class GaussianProjectedAttention(nn.Module):
    """
    Gaussian-Projected Quantum Self-Attention (GPQSA).

    For each pair (i, j), the attention coefficient is:
        α_{ij} = softmax_j( -||q_i - k_j||² / (2σ²) )

    where q_i and k_j are PauliZ expectations from PQC circuits.

    Args:
        n_qubits_per_token: Qubits per attention token.
        n_pqc_layers: Depth of Q/K/V circuits.
        sigma: Width of Gaussian kernel.
    """
    n_qubits_per_token: int = 4
    n_pqc_layers: int = 2
    sigma: float = 1.0

    @nn.compact
    def __call__(self, x_tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x_tokens: (n_tokens, n_qubits_per_token) — spin values per token.

        Returns:
            (n_tokens, n_qubits_per_token) — attention output.
        """
        n_tokens, d = x_tokens.shape
        nq = self.n_qubits_per_token
        nl = self.n_pqc_layers

        # Learnable PQC parameters for Q, K, V circuits
        params_q = self.param("params_q", nn.initializers.normal(0.1),
                              (nl, nq, 2))
        params_k = self.param("params_k", nn.initializers.normal(0.1),
                              (nl, nq, 2))
        params_v = self.param("params_v", nn.initializers.normal(0.1),
                              (nl, nq, 2))

        dev = get_device(wires=nq)
        diff_method = get_diff_method()
        wires = list(range(nq))

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def qk_circuit(x_in, params):
            from src.models.quantum_models.base_quantum_model import BaseQuantumModel
            BaseQuantumModel.angle_encoding(x_in, wires)
            for l in range(nl):
                BaseQuantumModel.variational_layer(params[l], wires)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        # Compute Q, K, V for all tokens
        Q = jax.vmap(lambda xi: jnp.array(qk_circuit(xi, params_q)))(x_tokens)
        K = jax.vmap(lambda xi: jnp.array(qk_circuit(xi, params_k)))(x_tokens)
        V = jax.vmap(lambda xi: jnp.array(qk_circuit(xi, params_v)))(x_tokens)

        # Gaussian kernel attention (classical)
        # dist²_{ij} = ||q_i - k_j||²
        Q_sq = jnp.sum(Q**2, axis=-1, keepdims=True)
        K_sq = jnp.sum(K**2, axis=-1, keepdims=True)
        dist_sq = Q_sq - 2 * Q @ K.T + K_sq.T  # (n_tokens, n_tokens)

        attn_logits = -dist_sq / (2 * self.sigma**2)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        # Apply attention to values
        out = attn_weights @ V  # (n_tokens, n_qubits_per_token)

        return out
