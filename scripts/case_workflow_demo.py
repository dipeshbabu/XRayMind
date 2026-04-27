"""Create a local XRayMind case, add a review, and print the dashboard summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xraymind.cases import add_review, create_case_with_prediction, get_case_detail
from xraymind.config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from xraymind.dashboard import dashboard_summary
from xraymind.store import DEFAULT_DB_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small local case workflow demo")
    parser.add_argument("--image", required=True, help="Chest X-ray image or DICOM path")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite workflow DB path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reviewer", default="demo_reviewer")
    parser.add_argument("--decision", default="uncertain", choices=["agree", "disagree", "uncertain", "defer", "flag"])
    parser.add_argument("--notes", default="Demo review added from scripts/case_workflow_demo.py")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    created = create_case_with_prediction(
        args.image,
        model_name=args.model,
        top_k=args.top_k,
        threshold=args.threshold,
        tags=["demo"],
        db_path=args.db,
    )
    case_id = created["case"]["id"]
    review = add_review(
        case_id,
        decision=args.decision,
        reviewer=args.reviewer,
        notes=args.notes,
        db_path=args.db,
    )
    payload = {
        "created": created,
        "review": review,
        "case_detail": get_case_detail(case_id, db_path=args.db),
        "dashboard": dashboard_summary(db_path=args.db),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
