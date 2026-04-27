"""Export XRayMind case workflow data.

Example:
    python scripts/export_cases.py --db outputs/xraymind_cases.sqlite3 --out-dir outputs/exports
"""

from __future__ import annotations

import argparse
import json

from xraymind.export import export_cases
from xraymind.store import DEFAULT_DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Export XRayMind case data to CSV and JSONL.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite workflow database path")
    parser.add_argument("--out-dir", default="outputs/exports", help="Export directory")
    parser.add_argument("--status", default=None, help="Optional case status filter")
    parser.add_argument("--priority", default=None, help="Optional case priority filter")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    manifest = export_cases(
        args.out_dir,
        status=args.status,
        priority=args.priority,
        limit=args.limit,
        offset=args.offset,
        db_path=args.db,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
