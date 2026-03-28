"""
Experiment orchestrator for qTransformer.

Manages sequential execution of multiple tasks.
Each task is a (solution, hamiltonian, training, evaluation) combination
defined in conf/experiment/*.yaml.

Usage:
    exp = Experiment(conf_dir)
    exp.add_tasks(tasks_from_yaml)
    results = exp.run()
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .task import run_task

logger = logging.getLogger(__name__)


# ==================== Config Loading ====================

def load_preset(name: str, conf_dir: Path = None) -> dict:
    """
    Load an experiment preset from conf/experiment/<name>.yaml.

    Args:
        name: Preset name (without .yaml extension).
        conf_dir: Path to conf/ directory. Defaults to project conf/.

    Returns:
        Raw dict from YAML.
    """
    if conf_dir is None:
        conf_dir = Path(__file__).resolve().parent.parent.parent / "conf"
    path = conf_dir / "experiment" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (conf_dir / "experiment").glob("*.yaml"))
        raise FileNotFoundError(
            f"Preset '{name}' not found at {path}\n"
            f"Available presets: {', '.join(available)}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def load_component(component_type: str, name: str, conf_dir: Path) -> dict:
    """
    Load a component config from conf/<component_type>/<name>.yaml.

    Args:
        component_type: e.g., "solution", "hamiltonian", "training", "evaluation"
        name: Component name (without .yaml).
        conf_dir: Path to conf/ directory.
    """
    path = conf_dir / component_type / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (conf_dir / component_type).glob("*.yaml"))
        raise FileNotFoundError(
            f"{component_type}/{name} not found at {path}\n"
            f"Available: {', '.join(available)}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_task(task_def: dict, conf_dir: Path) -> dict:
    """
    Resolve a task definition by loading referenced component configs.

    A task_def like:
        {solution: rbm, hamiltonian: square_4x4, g: 0.5}

    becomes a dict with solution, hamiltonian, training, evaluation
    sub-dicts, plus per-task overrides applied.
    """
    resolved = {}

    # Load component configs
    resolved["solution"] = load_component("solution", task_def["solution"], conf_dir)
    resolved["hamiltonian"] = load_component("hamiltonian", task_def["hamiltonian"], conf_dir)
    resolved["training"] = load_component("training", task_def.get("training", "default"), conf_dir)
    resolved["evaluation"] = load_component("evaluation", task_def.get("evaluation", "default"), conf_dir)

    # Per-task overrides
    for key, val in task_def.items():
        if key in ("solution", "hamiltonian", "training", "evaluation"):
            continue
        if key == "g":
            resolved["hamiltonian"]["g"] = val
        else:
            resolved[key] = val

    return resolved


# ==================== Task Scheduling ====================

@dataclass
class TaskInfo:
    """Task metadata for scheduling."""
    index: int
    config: dict
    solution: str
    hamiltonian: str
    name: str


def _get_task_name(task_def: dict, index: int) -> str:
    """Generate a descriptive name for a task."""
    solution = task_def.get("solution", "unknown")
    hamiltonian = task_def.get("hamiltonian", "unknown")
    g = task_def.get("g", 0.0)
    return f"{solution}_{hamiltonian}_g{g}_{index}"


# ==================== Orchestrator ====================

class Experiment:
    """
    Experiment orchestrator with sequential task execution.

    Tasks are loaded from conf/experiment/*.yaml and resolved
    against component configs in conf/solution/, conf/hamiltonian/, etc.

    Usage:
        exp = Experiment(conf_dir)
        exp.add_task({"solution": "rbm", "hamiltonian": "square_4x4", "g": 0.0})
        results = exp.run()
    """

    def __init__(self, conf_dir: Path):
        self.conf_dir = conf_dir
        self.tasks: list[TaskInfo] = []
        self.results: list[dict[str, Any]] = []
        self._task_counter = 0

    def add_task(self, task_def: dict):
        """Add a task definition (from experiment YAML) to the queue."""
        task_info = TaskInfo(
            index=self._task_counter,
            config=task_def,
            solution=task_def.get("solution", "unknown"),
            hamiltonian=task_def.get("hamiltonian", "unknown"),
            name=_get_task_name(task_def, self._task_counter),
        )
        self.tasks.append(task_info)
        self._task_counter += 1
        logger.debug(f"Added task '{task_info.name}' (total: {len(self.tasks)})")

    def add_tasks(self, task_defs: list[dict]):
        """Add multiple task definitions to the queue."""
        for task_def in task_defs:
            self.add_task(task_def)

    def run(self, num_of_tasks: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Execute tasks sequentially.

        Args:
            num_of_tasks: Max tasks to run. None = all.

        Returns:
            List of task result dicts.
        """
        if not self.tasks:
            logger.warning("No tasks to execute")
            return []

        tasks_to_run = list(self.tasks)
        if num_of_tasks is not None:
            tasks_to_run = tasks_to_run[:num_of_tasks]
        self.tasks.clear()
        total = len(tasks_to_run)

        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: {total} tasks")
        logger.info(f"{'='*70}\n")

        self.results = []
        for task in tasks_to_run:
            logger.info(f"\n{'~'*70}")
            logger.info(f"Task {task.index+1}/{total}: {task.name}")
            logger.info(f"{'~'*70}\n")

            try:
                resolved = resolve_task(task.config, self.conf_dir)
                result = run_task(resolved)
                result["task_name"] = task.name
                result["original_index"] = task.index
                result["status"] = "success"

                logger.info(f"Completed: {task.name}")
                self.results.append(result)
            except Exception as e:
                logger.error(f"Failed: {task.name} — {e}")
                logger.error(traceback.format_exc())
                self.results.append({
                    "task_name": task.name,
                    "original_index": task.index,
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        self._print_summary()
        return self.results

    def _print_summary(self):
        """Print experiment summary."""
        logger.info(f"\n{'='*70}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total tasks executed: {len(self.results)}")

        successful = sum(1 for r in self.results if r.get("status") == "success")
        failed = len(self.results) - successful

        logger.info(f"Successful: {successful}")
        if failed > 0:
            logger.info(f"Failed: {failed}")
            for r in self.results:
                if r.get("status") == "failed":
                    logger.info(f"  ✗ {r.get('task_name', 'unknown')}: {r.get('error', 'unknown')}")

        logger.info(f"{'='*70}\n")

    def __repr__(self) -> str:
        return f"Experiment(tasks={len(self.tasks)})"


# ==================== Convenience ====================

def run_experiment_from_cfg(experiment_data: dict, conf_dir: Path) -> list[dict[str, Any]]:
    """Run an experiment from parsed YAML data."""
    tasks = experiment_data.get("tasks", [])
    exp = Experiment(conf_dir)
    exp.add_tasks(tasks)
    return exp.run()
