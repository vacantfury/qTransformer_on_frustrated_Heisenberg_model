"""
Model factory for NQS wave function ansätze.

Single entry point to create any model from a solution config dict.
Keeps model construction logic out of the experiment runner.
"""
from __future__ import annotations

from typing import Any

from src.models.base_model import BaseModel


# Registry: solution type → (module path, class name, default kwargs)
_REGISTRY: dict[str, tuple[str, str, dict[str, Any]]] = {
    "rbm": (
        "src.models.classical_models.rbm.model",
        "RBM",
        {"alpha": 1},
    ),
    "cnn_resnet": (
        "src.models.classical_models.cnn_resnet.model",
        "CNNResNet",
        {"features": 32, "n_res_blocks": 4},
    ),
    "classical_vit": (
        "src.models.classical_models.classical_vit.model",
        "ClassicalViT",
        {"d_model": 64, "n_heads": 4, "n_layers": 2},
    ),
    "simplified_vit": (
        "src.models.classical_models.simplified_vit.model",
        "SimplifiedViT",
        {"d_model": 64, "n_heads": 4, "n_layers": 2},
    ),
    "qsann": (
        "src.models.quantum_models.qsann.model",
        "QSANN",
        {"n_qubits_per_token": 4, "n_pqc_layers": 2},
    ),
    "qmsan": (
        "src.models.quantum_models.qmsan.model",
        "QMSAN",
        {"n_qubits_per_token": 4, "n_pqc_layers": 2, "is_mixed_state": False},
    ),
}


def available_models() -> list[str]:
    """Return list of registered model type names."""
    return list(_REGISTRY.keys())


def create_model(sol_cfg: dict[str, Any]) -> BaseModel:
    """
    Create an NQS model from a solution config dict.

    Args:
        sol_cfg: Solution config with at least a "type" key.
                 Other keys are passed as kwargs to the model constructor,
                 falling back to defaults from the registry.

    Returns:
        Instantiated BaseModel subclass.

    Raises:
        ValueError: If the solution type is not registered.
    """
    sol_type = sol_cfg["type"]

    if sol_type not in _REGISTRY:
        raise ValueError(
            f"Unknown solution type: '{sol_type}'. "
            f"Available: {available_models()}"
        )

    module_path, class_name, defaults = _REGISTRY[sol_type]

    # Lazy import to avoid loading all model dependencies at startup
    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    # Merge: config values override defaults; exclude meta keys
    meta_keys = {"type", "name"}
    kwargs = {**defaults}
    for k, v in sol_cfg.items():
        if k not in meta_keys and v is not None:
            kwargs[k] = v

    return model_cls(**kwargs)
