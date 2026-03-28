"""
MLflow experiment tracking wrapper for PTP.

Thin wrapper around MLflow that handles run lifecycle, param/metric logging,
and artifact storage. All data is stored locally in mlruns/ (no server needed).

Usage (called by BaseOptimizer — not directly):
    tracker = MLflowTracker()
    run_id = tracker.start_run(mode="baseline", dataset="M3CoT", model="Pixtral-12B")
    tracker.log_params(llm_config, data_loader_config)
    tracker.log_metrics({"accuracy": 0.85, "f1_macro": 0.67})
    tracker.log_artifact("/path/to/results.json")
    tracker.end_run()
"""
import os
from typing import Any, Optional

import mlflow

from src.paths import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MLflow experiment name
EXPERIMENT_NAME = "PTP"

# Local tracking directory (relative to project root)
TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"


class MLflowTracker:
    """Thin wrapper for MLflow experiment tracking.
    
    Manages run lifecycle and provides simple methods for logging
    params, metrics, and artifacts. All data stored locally in mlruns/.
    """
    
    def __init__(self):
        self._run_id: Optional[str] = None
        self._active = False
        
        # Point MLflow to local storage
        mlflow.set_tracking_uri(TRACKING_URI)
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID (None if no active run)."""
        return self._run_id
    
    def start_run(self, mode: str, dataset: str, model: str) -> str:
        """Start an MLflow run.
        
        Args:
            mode: Experiment mode (e.g., "baseline", "naive")
            dataset: Dataset name (e.g., "M3CoT")
            model: Model identifier (e.g., "Pixtral-12B-2409")
        
        Returns:
            The MLflow run ID (UUID string)
        """
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        run_name = f"{mode}_{dataset}_{model}"
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        self._active = True
        
        # Set tags for easy filtering
        mlflow.set_tags({
            "mode": mode,
            "dataset": dataset,
            "model": model,
        })
        
        logger.info(f"MLflow run started: {run_name} (ID: {self._run_id})")
        return self._run_id
    
    def log_params(self, llm_config, data_loader_config) -> None:
        """Log experiment parameters from configs.
        
        Flattens nested config into dot-notation keys:
            llm_config.temperature -> "llm.temperature"
            data_loader_config['name'] -> "data.name"
        
        Args:
            llm_config: DictConfig or dict with LLM settings
            data_loader_config: DictConfig or dict with data loader settings
        """
        if not self._active:
            return
        
        try:
            # DictConfig supports .items() natively — no conversion needed
            llm_dict = dict(llm_config)
            if "model" in llm_dict:
                llm_dict["model"] = str(llm_dict["model"])
            llm_params = {f"llm.{k}": str(v) for k, v in llm_dict.items()}
            
            data_params = {f"data.{k}": str(v) for k, v in data_loader_config.items()}
            
            # MLflow has a 500-param limit, but we're well under it
            mlflow.log_params({**llm_params, **data_params})
            logger.debug(f"Logged {len(llm_params) + len(data_params)} params to MLflow")
        except Exception as e:
            # Don't let MLflow errors break the experiment
            logger.warning(f"MLflow param logging failed (non-fatal): {e}")
    
    def log_metrics(self, eval_results: dict[str, Any]) -> None:
        """Log evaluation metrics.
        
        Only logs numeric values (int, float). Skips non-numeric fields
        like question_type.
        
        Args:
            eval_results: Dict with evaluation metrics from Evaluator
        """
        if not self._active:
            return
        
        try:
            metrics = {}
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
            
            if metrics:
                mlflow.log_metrics(metrics)
                logger.debug(f"Logged {len(metrics)} metrics to MLflow: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"MLflow metric logging failed (non-fatal): {e}")
    
    def log_artifact(self, file_path: str) -> None:
        """Log a file as an MLflow artifact.
        
        Args:
            file_path: Absolute path to the file to log
        """
        if not self._active:
            return
        
        try:
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)
                logger.debug(f"Logged artifact to MLflow: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"MLflow artifact logging failed (non-fatal): {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._active:
            return
        
        try:
            mlflow.end_run()
            logger.info(f"MLflow run ended: {self._run_id}")
        except Exception as e:
            logger.warning(f"MLflow end_run failed (non-fatal): {e}")
        finally:
            self._active = False
