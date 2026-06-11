"""Compatibility module for the legacy AI trainer."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    legacy_train = Path(__file__).resolve().parents[2] / "AI" / "train.py"
    runpy.run_path(str(legacy_train), run_name="__main__")


if __name__ == "__main__":
    main()
