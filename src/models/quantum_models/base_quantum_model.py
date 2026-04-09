"""
Base class for quantum attention NQS models (Tier 3).

Carries the common parts shared by QSANN and QMSAN:
  - Spin → token encoding (tokenise + pad)
  - Angle encoding: spin value → qubit rotation
  - PQC variational layers: RY+RZ+CNOT hardware-efficient ansatz
  - HVA layers: Heisenberg-inspired RXX+RYY+RZZ
  - Classical FFN head: pooled attention → log ψ

Subclasses only need to implement the attention mechanism.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np

from src.models.base_model import BaseModel


class BaseQuantumModel(BaseModel):
    """
    Abstract base for quantum attention NQS.

    Subclasses must implement `_quantum_attention(tokens)` which takes
    tokenised spin patches and returns attention output of the same shape.

    Common hyperparameters:
        n_qubits_per_token: Qubits per attention token.
        n_pqc_layers: Depth of variational circuits.
        d_ff: Feed-forward hidden dimension.
        dtype: Output dtype.
    """
    n_qubits_per_token: int = 4
    n_pqc_layers: int = 2
    d_ff: int = 64
    dtype: type = complex

    # ──────────────── Tokenisation ────────────────

    @staticmethod
    def tokenise(x: jnp.ndarray, n_qubits_per_token: int) -> jnp.ndarray:
        """
        Split a flat spin config into tokens of size n_qubits_per_token.

        Pads with +1 if N is not divisible by n_qubits_per_token.

        Args:
            x: Spin configuration, shape (N,).
            n_qubits_per_token: Token size.

        Returns:
            Tokens, shape (n_tokens, n_qubits_per_token).
        """
        N = x.shape[0]
        nq = n_qubits_per_token
        n_tokens = (N + nq - 1) // nq
        x_padded = jnp.pad(x, (0, n_tokens * nq - N), constant_values=1.0)
        return x_padded.reshape(n_tokens, nq)

    # ──────────────── Quantum Encoding ────────────────

    @staticmethod
    def angle_encoding(x, wires: list[int]) -> None:
        """
        Encode spin config into qubit states via RY rotations.

        Maps s_i ∈ {+1, -1} → RY(arccos(s_i)) on wire i.
        +1 → |0⟩, -1 → |1⟩ (up to global phase).
        """
        for i, w in enumerate(wires):
            angle = jnp.arccos(jnp.clip(x[i], -1.0, 1.0))
            qml.RY(angle, wires=w)

    @staticmethod
    def dense_angle_encoding(x, wires: list[int]) -> None:
        """
        Denser encoding using both RY and RZ per qubit.

        Each qubit encodes one spin via RY(arccos(s)) · RZ(arcsin(s)).
        """
        for i, w in enumerate(wires):
            qml.RY(jnp.arccos(jnp.clip(x[i], -1.0, 1.0)), wires=w)
            qml.RZ(jnp.arcsin(jnp.clip(x[i], -1.0, 1.0)), wires=w)

    # ──────────────── PQC Layers ────────────────

    @staticmethod
    def variational_layer(params: np.ndarray, wires: list[int]) -> None:
        """
        Single variational layer: RY+RZ rotations + circular CNOT entangling.

        Args:
            params: Shape (n_qubits, 2) — two rotation angles per qubit.
            wires: Qubit wire indices.
        """
        n_qubits = len(wires)
        for i, w in enumerate(wires):
            qml.RY(params[i, 0], wires=w)
            qml.RZ(params[i, 1], wires=w)
        for i in range(n_qubits):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])

    @staticmethod
    def hva_layer(params: np.ndarray, wires: list[int]) -> None:
        """
        Hamiltonian Variational Ansatz (HVA) layer.

        Uses RXX, RYY, RZZ gates mirroring Heisenberg interaction structure.
        Better inductive bias for spin Hamiltonians.

        Args:
            params: Shape (n_qubits, 3) — three angles per qubit.
            wires: Qubit wire indices.
        """
        n_qubits = len(wires)
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.IsingXX(params[i, 0], wires=[wires[i], wires[j]])
            qml.IsingYY(params[i, 1], wires=[wires[i], wires[j]])
            qml.IsingZZ(params[i, 2], wires=[wires[i], wires[j]])

    @staticmethod
    def multi_layer_pqc(
        params: np.ndarray, wires: list[int], n_layers: int,
        layer_type: str = "variational",
    ) -> None:
        """
        Stack multiple PQC layers.

        Args:
            params: (n_layers, n_qubits, 2) for variational, (n_layers, n_qubits, 3) for hva.
            wires: Qubit wires.
            n_layers: Number of layers.
            layer_type: "variational" or "hva".
        """
        layer_fn = BaseQuantumModel.variational_layer if layer_type == "variational" else BaseQuantumModel.hva_layer
        for l in range(n_layers):
            layer_fn(params[l], wires)

    # ──────────────── Forward Pass ────────────────

    def _quantum_attention(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Subclasses implement their attention mechanism here.

        Args:
            tokens: (n_tokens, n_qubits_per_token).

        Returns:
            Attention output, same shape.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _quantum_attention"
        )

    @nn.compact
    def _call_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Process a single spin configuration.

        Args:
            x: (N,) spin config.

        Returns:
            Scalar log ψ(x).
        """
        # Tokenise
        tokens = self.tokenise(x, self.n_qubits_per_token)

        # Quantum attention (subclass-specific)
        attn_out = self._quantum_attention(tokens)

        # Global pooling
        pooled = jnp.mean(attn_out, axis=0)

        # Classical FFN → log ψ
        h = nn.Dense(self.d_ff, dtype=self.dtype)(pooled.astype(self.dtype))
        h = nn.gelu(h)
        log_psi = nn.Dense(1, dtype=self.dtype)(h)

        return log_psi.squeeze(-1)  # scalar

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Override BaseModel.__call__ for quantum models.

        Quantum circuits process single samples only.
        For batched input (batch, N), we use jax.vmap to map
        _call_single over the batch dimension.

        This works because default.qubit + backprop uses native JAX
        operations — all JAX transforms (vmap, jit, jacobian) are
        fully compatible. No custom JVP issues.

        Args:
            x: (N,) or (batch, N).

        Returns:
            scalar for (N,), (batch,) for (batch, N).
        """
        if x.ndim == 1:
            return self._call_single(x)
        else:
            return jax.vmap(self._call_single)(x)

