#!/usr/bin/env python3
"""After folds 3-4 finish, run folds 1-2 and evaluate all 20 models."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LOGS = PROJECT / "logs"
QUEUE_STATUS = LOGS / "cv_queue_status.json"
STATUS = LOGS / "cv_followon_status.json"


def update(state: str, detail: str = "") -> None:
    STATUS.write_text(
        json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), "state": state, "detail": detail}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{state}: {detail}", flush=True)


def main() -> None:
    update("waiting_for_folds_3_4")
    while True:
        if QUEUE_STATUS.is_file():
            queue = json.loads(QUEUE_STATUS.read_text(encoding="utf-8"))
            state = queue.get("state")
            if state == "complete":
                break
            if state in {"failed", "failed_overheat"}:
                update("failed", f"First desktop queue: {state}")
                raise RuntimeError(f"First desktop queue failed: {queue}")
        time.sleep(120)
    update("running_folds_1_2")
    with (LOGS / "cv_queue_12_console.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(PROJECT / "scripts/run_fivefold_queue.py"),
             "--folds", "1", "2", "--max-temperature", "80", "--resume-temperature", "70"],
            cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    update("evaluating")
    with (LOGS / "cv_final_evaluation.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(PROJECT / "scripts/finalize_crossval.py")],
            cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    update("complete", "All 20 fold-model results verified and cross-validation evaluated")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        update("failed", str(exc))
        raise
