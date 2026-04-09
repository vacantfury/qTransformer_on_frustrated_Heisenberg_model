"""
Experiment orchestrator for qTransformer.

Manages task execution: either sequentially in one process, or by
submitting parallel SLURM jobs. The parallelism is controlled by
the ``parallel_jobs`` field in the experiment YAML.

Config resolution uses Hydra's compose API:
- Experiment YAML (task list) is read as plain YAML → orchestration layer
- Individual task configs are resolved via Hydra compose → typed DictConfig

Usage:
    exp = Experiment(conf_dir)
    exp.add_tasks(tasks_from_yaml)
    results = exp.run(parallel_jobs=4, cluster_config={...})
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

from .task import run_task
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Hydra Initialization ====================

_hydra_initialized = False


def _init_hydra(conf_dir: Path):
    """Initialize Hydra config resolution (idempotent)."""
    global _hydra_initialized
    if _hydra_initialized:
        return
    # Clear any previous Hydra state (e.g., from tests or re-runs)
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(conf_dir.resolve()),
        version_base=None,
    )
    _hydra_initialized = True


# ==================== Config Loading ====================

def load_preset(name: str, conf_dir: Path = None) -> dict:
    """
    Load an experiment preset from conf/experiment/<name>.yaml.

    Merges with conf/experiment/default.yaml so that default parameters
    (like parallel_jobs) are always available.

    NOTE: Experiment files are plain YAML, NOT Hydra configs.
    They define task lists which are resolved individually via Hydra.

    Args:
        name: Preset name (without .yaml extension).
        conf_dir: Path to conf/ directory. Defaults to project conf/.

    Returns:
        Merged dict (defaults + preset, preset wins).
    """
    if conf_dir is None:
        conf_dir = Path(__file__).resolve().parent.parent.parent / "conf"

    # Load defaults
    defaults_path = conf_dir / "experiment" / "default.yaml"
    defaults = {}
    if defaults_path.exists():
        with open(defaults_path) as f:
            defaults = yaml.safe_load(f) or {}

    # Load preset
    path = conf_dir / "experiment" / f"{name}.yaml"
    if not path.exists():
        available = sorted(
            p.stem for p in (conf_dir / "experiment").glob("*.yaml")
            if p.stem != "default"
        )
        raise FileNotFoundError(
            f"Preset '{name}' not found at {path}\n"
            f"Available presets: {', '.join(available)}"
        )
    with open(path) as f:
        preset = yaml.safe_load(f) or {}

    # Merge: defaults first, preset overrides
    merged = {**defaults, **preset}
    return merged


def resolve_task(task_def: dict, conf_dir: Path) -> DictConfig:
    """
    Resolve a task definition via Hydra config composition.

    A task_def like:
        {solution: rbm, hamiltonian: chain_10, g: 0.5, training: medium}

    is resolved by composing Hydra config groups:
        conf/solution/rbm.yaml + conf/hamiltonian/chain_10.yaml + ...

    Returns a typed OmegaConf DictConfig with dot access:
        cfg.solution.type, cfg.training.n_steps, cfg.hamiltonian.g
    """
    _init_hydra(conf_dir)

    # Build Hydra overrides from task definition
    overrides = [
        f"solution={task_def['solution']}",
        f"hamiltonian={task_def['hamiltonian']}",
        f"training={task_def.get('training', 'default')}",
        f"evaluation={task_def.get('evaluation', 'default')}",
    ]

    # Per-task scalar overrides (e.g., g=0.5)
    skip_keys = {"solution", "hamiltonian", "training", "evaluation"}
    for key, val in task_def.items():
        if key in skip_keys:
            continue
        if key == "g":
            overrides.append(f"hamiltonian.g={val}")
        else:
            overrides.append(f"+{key}={val}")

    cfg = compose(config_name="config", overrides=overrides)
    return cfg


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
    Experiment orchestrator.

    Supports two execution modes:
    - Sequential: run all tasks in-process (parallel_jobs=1 or no SLURM)
    - Parallel: submit SLURM job array, each job runs a round-robin subset

    Usage:
        exp = Experiment(conf_dir)
        exp.add_tasks(tasks)
        exp.run(parallel_jobs=4, cluster_config={...}, experiment_name="test")
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

    def run(
        self,
        parallel_jobs: int = 1,
        cluster_config: Optional[dict] = None,
        experiment_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute tasks.

        If parallel_jobs > 1 and not already inside a SLURM array job,
        generates and submits a SLURM job array. Each array element
        runs a round-robin subset of tasks.

        If parallel_jobs == 1 or SLURM_ARRAY_TASK_ID is set (meaning
        we ARE a worker), execute tasks in-process.

        Args:
            parallel_jobs: Number of SLURM jobs to split tasks across.
            cluster_config: SLURM resource config from default.yaml.
            experiment_name: Name of the experiment preset.

        Returns:
            List of task result dicts (empty if jobs were submitted).
        """
        slurm_array_id = os.environ.get("SLURM_ARRAY_TASK_ID")

        if parallel_jobs > 1 and slurm_array_id is None:
            # Orchestrator mode: submit worker jobs
            self._submit_parallel_jobs(
                parallel_jobs, cluster_config or {}, experiment_name or "experiment"
            )
            return []
        elif slurm_array_id is not None and parallel_jobs > 1:
            # Worker mode: run round-robin subset
            job_index = int(slurm_array_id)
            subset = [t for t in self.tasks if t.index % parallel_jobs == job_index]
            logger.info(
                f"Worker job {job_index}/{parallel_jobs}: "
                f"{len(subset)}/{len(self.tasks)} tasks "
                f"(indices: {[t.index for t in subset]})"
            )
            return self._execute(subset)
        else:
            # Sequential mode: run all tasks in-process
            return self._execute(list(self.tasks))

    # -------------------- SLURM Submission --------------------

    def _submit_parallel_jobs(
        self, num_jobs: int, cluster_config: dict, experiment_name: str
    ):
        """
        Generate a worker sbatch script and submit it as a SLURM job array.
        """
        sbatch_content = self._generate_worker_sbatch(
            cluster_config, experiment_name
        )

        # Write to a temporary file (SLURM reads it at submission time)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sbatch", prefix=f"qt_{experiment_name}_",
            delete=False,
        ) as f:
            f.write(sbatch_content)
            sbatch_path = f.name
        os.chmod(sbatch_path, 0o755)

        array_spec = f"0-{num_jobs - 1}"
        cmd = [
            "sbatch",
            f"--array={array_spec}",
            f"--job-name=qt_{experiment_name}",
            str(sbatch_path),
            experiment_name,
        ]

        logger.info(f"Submitting {num_jobs} parallel jobs (array {array_spec})")
        logger.info(f"Worker script: {sbatch_path}")
        for i in range(num_jobs):
            subset = [t for t in self.tasks if t.index % num_jobs == i]
            logger.info(f"  Job {i}: {[t.name for t in subset]}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"SLURM: {result.stdout.strip()}")
        else:
            logger.error(f"sbatch failed: {result.stderr.strip()}")
            raise RuntimeError(f"Failed to submit SLURM jobs: {result.stderr}")

    def _generate_worker_sbatch(self, cluster: dict, experiment_name: str) -> str:
        """
        Generate a worker sbatch script from cluster config.

        The project directory is derived from conf_dir (not hardcoded).
        """
        partition = cluster.get("partition", "gpu")
        gres = cluster.get("gres", "gpu:1")
        cpus = cluster.get("cpus_per_task", 4)
        mem = cluster.get("mem", "32G")
        time_limit = cluster.get("time", "08:00:00")
        modules = cluster.get("modules", ["anaconda3/2024.06"])
        conda_env = cluster.get("conda_env", "qml")
        project_dir = str(self.conf_dir.parent.resolve())

        module_lines = "\n".join(f"module load {m}" for m in modules)

        return f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output=logs/qt_{experiment_name}_%A_%a.out
#SBATCH --error=logs/qt_{experiment_name}_%A_%a.err

cd {project_dir}
mkdir -p logs

{module_lines}
source activate {conda_env}

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:   $(date)"
echo "Experiment:   $1"
echo "============================================"

python main.py "$1"

echo "Finished at: $(date)"
"""

    # -------------------- Task Execution --------------------

    def _execute(self, tasks_to_run: list, num_of_tasks: Optional[int] = None) -> list[dict[str, Any]]:
        """Execute a list of tasks sequentially in-process."""
        if not tasks_to_run:
            logger.warning("No tasks to execute")
            return []

        if num_of_tasks is not None:
            tasks_to_run = tasks_to_run[:num_of_tasks]
        self.tasks.clear()
        total = len(tasks_to_run)

        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: {total} tasks")
        logger.info(f"{'='*70}\n")

        self.results = []
        for i, task in enumerate(tasks_to_run):
            logger.info(f"\n{'~'*70}")
            logger.info(f"Task {i+1}/{total}: {task.name}")
            logger.info(f"{'~'*70}\n")

            try:
                cfg = resolve_task(task.config, self.conf_dir)
                result = run_task(cfg)
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
