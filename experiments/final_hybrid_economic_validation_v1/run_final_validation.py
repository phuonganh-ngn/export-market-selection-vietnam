from __future__ import annotations

import argparse
from pathlib import Path

from src.final_validation import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final hybrid and decision reliability validation.")
    parser.add_argument("--mode", choices=["all", "audit", "blend", "risk", "reports"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run(root, args.mode)


if __name__ == "__main__":
    main()
