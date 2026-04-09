"""
Main entry point for qTransformer experiments.

CLI:
    python main.py <experiment>           # run experiment preset
    python main.py test                   # quick test
    python main.py baseline --dry-run     # print resolved tasks and exit

For SLURM: sbatch scripts/run_experiment.sbatch <preset>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from src.experiment import Experiment, load_preset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qTransformer")

CONF_DIR = Path(__file__).parent / "conf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="qTransformer: Quantum Attention NQS for J1-J2 Heisenberg Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test
  python main.py baseline
  python main.py test --dry-run

Available experiments: see conf/experiment/*.yaml""",
    )

    parser.add_argument("experiment",
                        help="Experiment preset name (from conf/experiment/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved tasks and exit")

    return parser.parse_args()


def main():
    args = parse_args()

    experiment_data = load_preset(args.experiment, CONF_DIR)
    tasks = experiment_data.get("tasks", [])
    parallel_jobs = experiment_data.get("parallel_jobs", 1)

    logger.info(f"Experiment: {args.experiment} ({len(tasks)} tasks)")

    if args.dry_run:
        print(json.dumps(experiment_data, indent=2, default=str))
        return

    exp = Experiment(CONF_DIR)
    exp.add_tasks(tasks)

    # Pass parallel config to Experiment — it decides whether
    # to submit SLURM jobs or run sequentially
    exp.run(
        parallel_jobs=parallel_jobs,
        cluster_config=experiment_data.get("cluster"),
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
