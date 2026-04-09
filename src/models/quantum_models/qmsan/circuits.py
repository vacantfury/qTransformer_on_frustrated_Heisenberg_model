"""
QMSAN quantum circuits.

PQC circuits for the Quantum Mixed-State Self-Attention Network.
Two modes:
  - Pure-state swap test: |⟨ψ_q|ψ_k⟩|² (all qubits swapped)
  - Mixed-state swap test: Tr(ρ_q ρ_k) (partial trace, only subset swapped)

Level 3 (fully quantum): similarity computed entirely in Hilbert space.
"""
from __future__ import annotations

import pennylane as qml
import numpy as np

from src.models.quantum_models.base_quantum_model import BaseQuantumModel
from src.models.quantum_models.device import get_device, get_diff_method

# Use static methods from BaseQuantumModel
angle_encoding = BaseQuantumModel.angle_encoding
variational_layer = BaseQuantumModel.variational_layer


def swap_test_circuit(n_qubits: int):
    """
    Build a PURE-STATE swap test circuit for computing |⟨ψ_q|ψ_k⟩|².

    Uses 2*n_qubits data qubits + 1 ancilla qubit.
    All qubits are swapped → computes overlap of the full pure states.

    Wires layout:
        0: ancilla
        1..n_qubits: register A (query state)
        n_qubits+1..2*n_qubits: register B (key state)

    Returns:
        PennyLane QNode computing ⟨Z_ancilla⟩ = |⟨ψ_q|ψ_k⟩|².
    """
    total_wires = 2 * n_qubits + 1
    dev = get_device(wires=total_wires)
    diff_method = get_diff_method()

    wires_a = list(range(1, n_qubits + 1))
    wires_b = list(range(n_qubits + 1, 2 * n_qubits + 1))
    ancilla = 0

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(x_q, x_k, params_q, params_k, n_layers):
        # Prepare query state on register A
        angle_encoding(x_q, wires_a)
        for l in range(n_layers):
            variational_layer(params_q[l], wires_a)

        # Prepare key state on register B
        angle_encoding(x_k, wires_b)
        for l in range(n_layers):
            variational_layer(params_k[l], wires_b)

        # Swap test on ALL qubits
        qml.Hadamard(wires=ancilla)
        for i in range(n_qubits):
            qml.CSWAP(wires=[ancilla, wires_a[i], wires_b[i]])
        qml.Hadamard(wires=ancilla)

        # ⟨Z⟩ = |⟨ψ_q|ψ_k⟩|²
        return qml.expval(qml.PauliZ(ancilla))

    return circuit


def mixed_state_swap_test_circuit(n_qubits: int, n_keep: int):
    """
    Build a MIXED-STATE swap test circuit for computing Tr(ρ_q ρ_k).

    Each register has n_qubits, but only n_keep qubits participate in
    the swap test. The remaining (n_qubits - n_keep) qubits are
    effectively traced out, yielding reduced density matrices:

        ρ_q = Tr_{traced}(|ψ_q⟩⟨ψ_q|)
        ρ_k = Tr_{traced}(|ψ_k⟩⟨ψ_k|)

    The swap test then computes ⟨Z_ancilla⟩ = Tr(ρ_q ρ_k).

    This is the approach from the original QMSAN paper (Chen et al. 2025).

    Wires layout:
        0: ancilla
        1..n_qubits: register A (query)
            - wires 1..n_keep: kept (swapped)
            - wires n_keep+1..n_qubits: traced out (not swapped)
        n_qubits+1..2*n_qubits: register B (key)
            - wires n_qubits+1..n_qubits+n_keep: kept (swapped)
            - wires n_qubits+n_keep+1..2*n_qubits: traced out (not swapped)

    Args:
        n_qubits: Total qubits per register.
        n_keep: Number of qubits to keep (swap). Must satisfy 1 <= n_keep < n_qubits.

    Returns:
        PennyLane QNode computing ⟨Z_ancilla⟩ = Tr(ρ_q ρ_k).
    """
    assert 1 <= n_keep < n_qubits, (
        f"n_keep must satisfy 1 <= n_keep < n_qubits, got n_keep={n_keep}, n_qubits={n_qubits}"
    )

    total_wires = 2 * n_qubits + 1
    dev = get_device(wires=total_wires)
    diff_method = get_diff_method()

    wires_a = list(range(1, n_qubits + 1))
    wires_b = list(range(n_qubits + 1, 2 * n_qubits + 1))
    ancilla = 0

    # Only the first n_keep qubits of each register are swapped
    swap_a = wires_a[:n_keep]
    swap_b = wires_b[:n_keep]

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(x_q, x_k, params_q, params_k, n_layers):
        # Prepare query state on register A (all n_qubits)
        angle_encoding(x_q, wires_a)
        for l in range(n_layers):
            variational_layer(params_q[l], wires_a)

        # Prepare key state on register B (all n_qubits)
        angle_encoding(x_k, wires_b)
        for l in range(n_layers):
            variational_layer(params_k[l], wires_b)

        # Swap test on ONLY the kept qubits
        # The un-swapped qubits are traced out by not participating
        qml.Hadamard(wires=ancilla)
        for wa, wb in zip(swap_a, swap_b):
            qml.CSWAP(wires=[ancilla, wa, wb])
        qml.Hadamard(wires=ancilla)

        # ⟨Z⟩ = Tr(ρ_q ρ_k)
        return qml.expval(qml.PauliZ(ancilla))

    return circuit


def mixed_state_value_circuit(n_qubits: int):
    """
    Build a value circuit that returns expectation values.

    The value representation: encode input, apply V-circuit, measure
    PauliZ on all qubits. Same for both pure-state and mixed-state modes.
    """
    dev = get_device(wires=n_qubits)
    diff_method = get_diff_method()
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(x, params_v, n_layers):
        angle_encoding(x, wires)
        for l in range(n_layers):
            variational_layer(params_v[l], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return circuit
