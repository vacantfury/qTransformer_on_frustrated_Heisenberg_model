"""Project path constants."""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Source code
SRC_DIR = ROOT_DIR / "src"

# Configuration
CONF_DIR = ROOT_DIR / "conf"

# Outputs
OUTPUTS_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = ROOT_DIR / "results"
MLRUNS_DIR = ROOT_DIR / "mlruns"
