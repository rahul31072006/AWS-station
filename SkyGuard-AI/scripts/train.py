"""Command-line entry point for model training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import main


if __name__ == "__main__":
    main()