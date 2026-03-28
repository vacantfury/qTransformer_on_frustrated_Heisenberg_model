"""
QSANN quantum circuits.

PQC circuits for the Gaussian-Projected Quantum Self-Attention (QSANN).
Q and K are encoded in quantum feature space, but similarity is computed
classically after measurement (partial quantum — Level 2).
"""
from __future__ import annotations

import pennylane as qml
import numpy as np

from src.models.quantum_models.base_quantum_model import BaseQuantumModel

# Use static methods from BaseQuantumModel
angle_encoding = BaseQuantumModel.angle_encoding
variational_layer = BaseQuantumModel.variational_layer


def qsann_qk_circuit(
    x: np.ndarray,
    params_q: np.ndarray,
    params_k: np.ndarray,
    n_layers: int,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute quantum Q and K representations for QSANN.

    Encodes input, applies parameterised Q and K circuits,
    returns expectation values (classical vectors) for each.

    Args:
        x: Spin configuration for this "token", shape (n_qubits,).
        params_q: Q-circuit params, shape (n_layers, n_qubits, 2).
        params_k: K-circuit params, shape (n_layers, n_qubits, 2).
        n_layers: Number of variational layers.
        n_qubits: Number of qubits per token.

    Returns:
        (q_vec, k_vec) — classical vectors of PauliZ expectations,
        each shape (n_qubits,).
    """
    wires = list(range(n_qubits))
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def q_circuit(x_in, params):
        angle_encoding(x_in, wires)
        for l in range(n_layers):
            variational_layer(params[l], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    @qml.qnode(dev, interface="jax")
    def k_circuit(x_in, params):
        angle_encoding(x_in, wires)
        for l in range(n_layers):
            variational_layer(params[l], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    q_vec = np.array(q_circuit(x, params_q))
    k_vec = np.array(k_circuit(x, params_k))

    return q_vec, k_vec


def qsann_value_circuit(
    x: np.ndarray,
    params_v: np.ndarray,
    n_layers: int,
    n_qubits: int,
) -> np.ndarray:
    """
    Compute quantum V (value) representation for QSANN.

    Args:
        x: Spin configuration for this token, shape (n_qubits,).
        params_v: V-circuit params, shape (n_layers, n_qubits, 2).
        n_layers: Number of variational layers.
        n_qubits: Number of qubits per token.

    Returns:
        v_vec — classical vector of expectations, shape (n_qubits,).
    """
    wires = list(range(n_qubits))
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def v_circuit(x_in, params):
        angle_encoding(x_in, wires)
        for l in range(n_layers):
            variational_layer(params[l], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return np.array(v_circuit(x, params_v))
