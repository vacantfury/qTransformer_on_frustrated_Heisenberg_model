"""
PennyLane device utilities — config-driven backend selection.

Backend priority for this project (n_qubits_per_token ≤ 8):

    default.qubit + backprop (primary)
    ├─ Uses JAX for statevector simulation
    ├─ Runs on GPU via JAX XLA — no cuQuantum needed
    ├─ Full JAX transform support (vmap, jit, jacobian)
    └─ Ideal for circuits with ≤ ~20 qubits

For future large-qubit experiments (n_qubits_per_token > 20):
    lightning.gpu + adjoint  → cuQuantum GPU, memory-efficient
    lightning.qubit + adjoint → CPU fallback
"""
from __future__ import annotations

from typing import Literal

import pennylane as qml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported backends and their differentiation methods
BackendName = Literal["default.qubit", "lightning.gpu", "lightning.qubit"]

_DIFF_METHODS: dict[str, str] = {
    "default.qubit": "backprop",
    "lightning.gpu": "adjoint",
    "lightning.qubit": "adjoint",
}

# Cached auto-detected backend
_AUTO_BACKEND: BackendName | None = None


def _auto_detect_backend() -> BackendName:
    """Auto-detect the best backend for JAX GPU simulation."""
    global _AUTO_BACKEND
    if _AUTO_BACKEND is not None:
        return _AUTO_BACKEND

    # For this project, default.qubit is preferred because:
    # 1. Circuits are small (4-9 qubits) — cuQuantum overhead dominates
    # 2. JAX backprop supports vmap/jit/jacobian natively
    # 3. Runs on GPU through JAX XLA — no extra dependency
    try:
        import jax
        backend_name = jax.default_backend()
        if backend_name == "gpu":
            logger.info("Quantum backend: default.qubit (JAX GPU via XLA)")
        else:
            logger.info(f"Quantum backend: default.qubit (JAX {backend_name})")
    except Exception:
        logger.info("Quantum backend: default.qubit")

    _AUTO_BACKEND = "default.qubit"
    return _AUTO_BACKEND


def get_device(wires: int | list[int], *, backend: BackendName | None = None,
               **kwargs) -> qml.Device:
    """
    Create a PennyLane device.

    Args:
        wires: Number of qubits or list of wire indices.
        backend: Explicit backend choice. If None, auto-detects.
        **kwargs: Additional args passed to qml.device().

    Returns:
        PennyLane device.
    """
    if backend is None:
        backend = _auto_detect_backend()

    return qml.device(backend, wires=wires, **kwargs)


def get_diff_method(backend: BackendName | None = None) -> str:
    """
    Get the differentiation method for a given backend.

    Args:
        backend: Backend name. If None, uses auto-detected backend.

    Returns:
        Differentiation method string for use in @qml.qnode().
    """
    if backend is None:
        backend = _auto_detect_backend()
    return _DIFF_METHODS[backend]
