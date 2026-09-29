#!/usr/bin/env python3
"""Add a common batch-size configuration while preserving nnU-Net's plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    for path in args.plans:
        plans = json.loads(path.read_text(encoding="utf-8"))
        plans["configurations"]["3d_fullres_common"] = {
            "inherits_from": "3d_fullres",
            "batch_size": args.batch_size,
        }
        path.write_text(json.dumps(plans, indent=4), encoding="utf-8")
        print(f"Updated {path} with batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
