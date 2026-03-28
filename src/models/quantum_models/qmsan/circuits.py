"""
QMSAN quantum circuits.

PQC circuits for the Quantum Mixed-State Self-Attention Network.
Similarity is computed entirely in Hilbert space via swap test —
no projection to classical before similarity (fully quantum — Level 3).
"""
from __future__ import annotations

import pennylane as qml
import numpy as np

from src.models.quantum_models.base_quantum_model import BaseQuantumModel

# Use static methods from BaseQuantumModel
angle_encoding = BaseQuantumModel.angle_encoding
variational_layer = BaseQuantumModel.variational_layer


def swap_test_circuit(n_qubits: int):
    """
    Build a swap test circuit for computing |⟨ψ|φ⟩|².

    Uses 2*n_qubits data qubits + 1 ancilla qubit.
    Wires layout:
        0: ancilla
        1..n_qubits: register A (query state)
        n_qubits+1..2*n_qubits: register B (key state)

    Returns:
        PennyLane QNode that computes swap test overlap.
    """
    total_wires = 2 * n_qubits + 1
    dev = qml.device("lightning.qubit", wires=total_wires)

    wires_a = list(range(1, n_qubits + 1))
    wires_b = list(range(n_qubits + 1, 2 * n_qubits + 1))
    ancilla = 0

    @qml.qnode(dev, interface="jax")
    def circuit(x_q, x_k, params_q, params_k, n_layers):
        # Prepare query state on register A
        angle_encoding(x_q, wires_a)
        for l in range(n_layers):
            variational_layer(params_q[l], wires_a)

        # Prepare key state on register B
        angle_encoding(x_k, wires_b)
        for l in range(n_layers):
            variational_layer(params_k[l], wires_b)

        # Swap test
        qml.Hadamard(wires=ancilla)
        for i in range(n_qubits):
            qml.CSWAP(wires=[ancilla, wires_a[i], wires_b[i]])
        qml.Hadamard(wires=ancilla)

        # P(ancilla=0) = (1 + |⟨ψ|φ⟩|²) / 2
        return qml.expval(qml.PauliZ(ancilla))

    return circuit


def mixed_state_value_circuit(n_qubits: int):
    """
    Build a value circuit that returns expectation values.

    The value representation uses a partial trace approach:
    encode input, apply V-circuit, measure subset of qubits.
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="jax")
    def circuit(x, params_v, n_layers):
        angle_encoding(x, wires)
        for l in range(n_layers):
            variational_layer(params_v[l], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return circuit
