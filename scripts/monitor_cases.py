"""Create an XRayMind workflow monitoring snapshot.

Example:
    python scripts/monitor_cases.py --db outputs/xraymind_cases.sqlite3 --out outputs/monitoring/snapshot.json
"""

from __future__ import annotations

import argparse
import json

from xraymind.monitoring import save_monitoring_snapshot
from xraymind.store import DEFAULT_DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a monitoring snapshot for XRayMind case workflows.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite workflow database path")
    parser.add_argument("--out", default="outputs/monitoring/snapshot.json", help="Snapshot JSON path")
    parser.add_argument("--baseline", default=None, help="Optional previous snapshot for drift comparison")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--drift-threshold", type=float, default=0.15)
    args = parser.parse_args()

    snapshot = save_monitoring_snapshot(
        args.out,
        baseline=args.baseline,
        db_path=args.db,
        limit=args.limit,
        drift_threshold=args.drift_threshold,
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
