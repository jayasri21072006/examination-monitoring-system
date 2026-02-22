import os

REPORT_DIR = "reports"
SNAPSHOT_DIR = "snapshots"
MAX_SCAN_INDEX = 8


def ensure_dirs() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
