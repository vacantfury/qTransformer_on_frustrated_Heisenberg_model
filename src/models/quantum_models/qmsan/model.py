"""
QMSAN NQS model.

Full QMSAN wave function: spin config → tokenise → swap-test attention → FFN → log ψ.

Tier 3, Level 3 (fully quantum): Q/K similarity computed entirely in Hilbert space
via swap test circuits. No classical projection of quantum states.

Two modes:
  - is_mixed_state=False (default): Pure-state swap test  → |⟨ψ_q|ψ_k⟩|²
  - is_mixed_state=True:            Mixed-state swap test → Tr(ρ_q ρ_k)
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.models.quantum_models.base_quantum_model import BaseQuantumModel
from src.models.quantum_models.qmsan.attention import SwapTestAttention


class QMSAN(BaseQuantumModel):
    """
    Quantum Mixed-State Self-Attention Network for NQS.

    Inherits tokenisation, encoding, PQC layers, and FFN from BaseQuantumModel.
    Implements swap-test quantum attention (fully quantum similarity).

    Args:
        n_qubits_per_token: Qubits per attention token.
        n_pqc_layers: Depth of Q/K/V PQC circuits.
        d_ff: Feed-forward hidden dimension.
        dtype: Output dtype.
        is_mixed_state: If True, use mixed-state swap test (partial trace).
                        If False (default), use pure-state swap test.
        n_keep_qubits: Qubits to keep after partial trace (rest traced out).
                       Only used when is_mixed_state=True.
                       Defaults to n_qubits_per_token // 2.
    """
    is_mixed_state: bool = False
    n_keep_qubits: int | None = None

    def _quantum_attention(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Swap-test quantum self-attention."""
        return SwapTestAttention(
            n_qubits_per_token=self.n_qubits_per_token,
            n_pqc_layers=self.n_pqc_layers,
            is_mixed_state=self.is_mixed_state,
            n_keep_qubits=self.n_keep_qubits,
        )(tokens)
