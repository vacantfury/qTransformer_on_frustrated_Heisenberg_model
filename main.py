"""
Main entry point for qTransformer experiments.

CLI follows the PTP pattern:
    python main.py --batch --experiment <preset>    # run all tasks in preset
    python main.py --batch --experiment test         # quick test
    python main.py --batch --experiment benchmark    # full sweep

For SLURM: sbatch scripts/run_experiment.sbatch <preset>
"""
from __future__ import annotations

import argparse
import json
import logging
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
  python main.py --batch --experiment test
  python main.py --batch --experiment baseline
  python main.py --batch --experiment benchmark
  python main.py --batch --experiment test --dry-run

Available experiments: see conf/experiment/*.yaml""",
    )

    parser.add_argument("--experiment", "-e", required=True,
                        help="Experiment preset name (from conf/experiment/)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode (no interactive prompts)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved tasks and exit")

    return parser.parse_args()


def main():
    args = parse_args()

    experiment_data = load_preset(args.experiment, CONF_DIR)
    tasks = experiment_data.get("tasks", [])

    logger.info(f"Experiment: {args.experiment} ({len(tasks)} tasks)")

    if args.dry_run:
        print(json.dumps(experiment_data, indent=2, default=str))
        return

    exp = Experiment(CONF_DIR)
    exp.add_tasks(tasks)
    exp.run()


if __name__ == "__main__":
    main()
