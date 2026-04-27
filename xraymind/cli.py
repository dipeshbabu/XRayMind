"""Command-line interface for XRayMind."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .audit import audit_prediction
from .cases import (
    add_review,
    create_case_with_prediction,
    get_case_detail,
    list_cases,
    update_case_status,
)
from .config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from .dashboard import cases_requiring_attention, dashboard_summary
from .dicom import dicom_to_png, redact_dicom, write_safe_metadata_json
from .explainability import explain_to_file
from .inference import predict_image, save_prediction
from .packet import create_study_packet
from .pdf import maybe_html_to_pdf
from .report import save_html_report
from .store import DEFAULT_DB_PATH
from .tta import predict_with_tta
from .visualization import save_original_preview


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="TorchXRayVision model name")


def _add_audit_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit-log", default=None, help="Optional JSONL audit log path")


def _print_json(payload: dict | list) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xraymind", description="Explainable chest X-ray inference utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run prediction on one image")
    predict.add_argument("--image", required=True, help="Path to chest X-ray image")
    predict.add_argument("--out", default="outputs/prediction.json", help="Output JSON path")
    predict.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    predict.add_argument("--threshold", type=float, default=0.5)
    predict.add_argument("--html", default=None, help="Optional HTML report path")
    _add_common_model_args(predict)
    _add_audit_arg(predict)

    tta = subparsers.add_parser("tta", help="Run test-time augmentation uncertainty prediction")
    tta.add_argument("--image", required=True, help="Path to chest X-ray image")
    tta.add_argument("--out", default="outputs/tta_prediction.json", help="Output JSON path")
    tta.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    tta.add_argument("--threshold", type=float, default=0.5)
    _add_common_model_args(tta)
    _add_audit_arg(tta)

    explain = subparsers.add_parser("explain", help="Generate an attribution heatmap")
    explain.add_argument("--image", required=True, help="Path to chest X-ray image")
    explain.add_argument("--label", required=True, help="Target pathology label")
    explain.add_argument("--out", default="outputs/heatmap.png", help="Output PNG path")
    explain.add_argument(
        "--method",
        default="integrated_gradients",
        choices=["saliency", "input_x_gradient", "integrated_gradients"],
    )
    _add_common_model_args(explain)

    report = subparsers.add_parser("report", help="Prediction + optional heatmap + HTML report")
    report.add_argument("--image", required=True, help="Path to chest X-ray image")
    report.add_argument("--label", default=None, help="Optional target label for heatmap")
    report.add_argument("--json", default="outputs/prediction.json", help="Output JSON path")
    report.add_argument("--html", default="outputs/report.html", help="Output HTML path")
    report.add_argument("--pdf", default=None, help="Optional PDF report path")
    report.add_argument("--heatmap", default="outputs/heatmap.png", help="Output heatmap path")
    report.add_argument("--original", default="outputs/original_preview.png", help="Output original preview path")
    report.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    report.add_argument("--threshold", type=float, default=0.5)
    report.add_argument("--image-id", default=None)
    report.add_argument(
        "--method",
        default="integrated_gradients",
        choices=["saliency", "input_x_gradient", "integrated_gradients"],
    )
    _add_common_model_args(report)
    _add_audit_arg(report)

    packet = subparsers.add_parser("packet", help="Create a complete single-image study packet")
    packet.add_argument("--image", required=True, help="Path to chest X-ray image")
    packet.add_argument("--out-dir", default="outputs/study_packet")
    packet.add_argument("--label", default=None, help="Optional target label for heatmap")
    packet.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    packet.add_argument("--threshold", type=float, default=0.5)
    packet.add_argument("--image-id", default=None)
    packet.add_argument("--pdf", action="store_true", help="Also export report.pdf")
    packet.add_argument(
        "--method",
        default="integrated_gradients",
        choices=["saliency", "input_x_gradient", "integrated_gradients"],
    )
    _add_common_model_args(packet)
    _add_audit_arg(packet)

    dicom = subparsers.add_parser("dicom", help="Convert/redact DICOM files")
    dicom.add_argument("--dicom", required=True, help="Input DICOM path")
    dicom.add_argument("--png", default="outputs/dicom/preview.png", help="Output PNG path")
    dicom.add_argument("--metadata", default=None, help="Optional safe metadata JSON path")
    dicom.add_argument("--redacted", default=None, help="Optional redacted DICOM output path")

    case = subparsers.add_parser("case", help="Local human-in-the-loop case workflow")
    case_subparsers = case.add_subparsers(dest="case_command", required=True)

    case_create = case_subparsers.add_parser("create", help="Create a case and run prediction")
    case_create.add_argument("--image", required=True, help="Path to chest X-ray image")
    case_create.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite workflow database path")
    case_create.add_argument("--image-id", default=None)
    case_create.add_argument("--priority", default="routine", choices=["routine", "elevated", "urgent"])
    case_create.add_argument("--tags", default="", help="Comma-separated tags")
    case_create.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    case_create.add_argument("--threshold", type=float, default=0.5)
    _add_common_model_args(case_create)

    case_list = case_subparsers.add_parser("list", help="List cases")
    case_list.add_argument("--db", default=DEFAULT_DB_PATH)
    case_list.add_argument("--status", default=None)
    case_list.add_argument("--priority", default=None)
    case_list.add_argument("--limit", type=int, default=25)
    case_list.add_argument("--offset", type=int, default=0)

    case_show = case_subparsers.add_parser("show", help="Show case detail")
    case_show.add_argument("case_id", type=int)
    case_show.add_argument("--db", default=DEFAULT_DB_PATH)

    case_status = case_subparsers.add_parser("status", help="Update case status")
    case_status.add_argument("case_id", type=int)
    case_status.add_argument("status", choices=["pending", "reviewed", "deferred", "flagged", "archived"])
    case_status.add_argument("--db", default=DEFAULT_DB_PATH)

    case_review = case_subparsers.add_parser("review", help="Add a human review")
    case_review.add_argument("case_id", type=int)
    case_review.add_argument("--decision", required=True, choices=["agree", "disagree", "uncertain", "defer", "flag"])
    case_review.add_argument("--reviewer", default=None)
    case_review.add_argument("--notes", default=None)
    case_review.add_argument("--final-labels-json", default="{}", help="JSON object with final labels")
    case_review.add_argument("--next-status", default=None)
    case_review.add_argument("--db", default=DEFAULT_DB_PATH)

    dashboard = subparsers.add_parser("dashboard", help="Show local case workflow dashboard")
    dashboard.add_argument("--db", default=DEFAULT_DB_PATH)
    dashboard.add_argument("--attention", action="store_true", help="Show cases requiring attention instead of summary")
    dashboard.add_argument("--limit", type=int, default=25)

    subparsers.add_parser("demo", help="Launch the Gradio demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "predict":
        result = predict_image(
            args.image, model_name=args.model, top_k=args.top_k, threshold=args.threshold
        )
        save_prediction(result, args.out)
        if args.html:
            save_html_report(result, args.html)
        if args.audit_log:
            audit_prediction(args.image, args.model, args.audit_log, extra={"command": "predict"})
        print(f"Prediction saved to {args.out}")
        return 0

    if args.command == "tta":
        result = predict_with_tta(
            args.image, model_name=args.model, top_k=args.top_k, threshold=args.threshold
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if args.audit_log:
            audit_prediction(args.image, args.model, args.audit_log, extra={"command": "tta"})
        print(f"TTA prediction saved to {args.out}")
        return 0

    if args.command == "explain":
        explain_to_file(
            image=args.image,
            label=args.label,
            output_path=args.out,
            model_name=args.model,
            method=args.method,
        )
        print(f"Heatmap saved to {args.out}")
        return 0

    if args.command == "report":
        result = predict_image(
            args.image, model_name=args.model, top_k=args.top_k, threshold=args.threshold
        )
        save_prediction(result, args.json)
        original_path = save_original_preview(args.image, args.original)
        heatmap_path = None
        label = args.label or (result["top_findings"][0]["label"] if result["top_findings"] else None)
        if label:
            heatmap_path = explain_to_file(
                image=args.image,
                label=label,
                output_path=args.heatmap,
                model_name=args.model,
                method=args.method,
            )
        save_html_report(
            result,
            args.html,
            heatmap_path=heatmap_path,
            original_path=original_path,
            image_id=args.image_id,
        )
        maybe_html_to_pdf(args.html, args.pdf)
        if args.audit_log:
            audit_prediction(args.image, args.model, args.audit_log, extra={"command": "report", "label": label})
        print(f"Prediction saved to {args.json}")
        print(f"Report saved to {args.html}")
        if args.pdf:
            print(f"PDF saved to {args.pdf}")
        return 0

    if args.command == "packet":
        manifest = create_study_packet(
            image=args.image,
            output_dir=args.out_dir,
            model_name=args.model,
            label=args.label,
            top_k=args.top_k,
            threshold=args.threshold,
            method=args.method,
            make_pdf=args.pdf,
            image_id=args.image_id,
        )
        if args.audit_log:
            audit_prediction(args.image, args.model, args.audit_log, extra={"command": "packet", "out_dir": args.out_dir})
        print(f"Study packet saved to {args.out_dir}")
        print(f"Manifest: {manifest['files']['manifest']}")
        print(f"ZIP: {manifest['files']['zip']}")
        return 0

    if args.command == "dicom":
        png_path = dicom_to_png(args.dicom, args.png)
        print(f"Saved PNG preview to {png_path}")
        if args.metadata:
            metadata_path = write_safe_metadata_json(args.dicom, args.metadata)
            print(f"Saved safe metadata to {metadata_path}")
        if args.redacted:
            redacted_path = redact_dicom(args.dicom, args.redacted)
            print(f"Saved redacted DICOM copy to {redacted_path}")
        return 0

    if args.command == "case":
        if args.case_command == "create":
            tag_list = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
            result = create_case_with_prediction(
                args.image,
                image_id=args.image_id,
                model_name=args.model,
                top_k=args.top_k,
                threshold=args.threshold,
                priority=args.priority,
                tags=tag_list,
                db_path=args.db,
            )
            _print_json(result)
            return 0
        if args.case_command == "list":
            _print_json(
                {
                    "cases": list_cases(
                        status=args.status,
                        priority=args.priority,
                        limit=args.limit,
                        offset=args.offset,
                        db_path=args.db,
                    )
                }
            )
            return 0
        if args.case_command == "show":
            _print_json(get_case_detail(args.case_id, db_path=args.db))
            return 0
        if args.case_command == "status":
            _print_json({"case": update_case_status(args.case_id, args.status, db_path=args.db)})
            return 0
        if args.case_command == "review":
            final_labels = json.loads(args.final_labels_json or "{}")
            review = add_review(
                args.case_id,
                decision=args.decision,
                reviewer=args.reviewer,
                notes=args.notes,
                final_labels=final_labels,
                next_status=args.next_status,
                db_path=args.db,
            )
            _print_json({"review": review, "case_detail": get_case_detail(args.case_id, db_path=args.db)})
            return 0

    if args.command == "dashboard":
        if args.attention:
            _print_json({"cases": cases_requiring_attention(db_path=args.db, limit=args.limit)})
        else:
            _print_json(dashboard_summary(db_path=args.db))
        return 0

    if args.command == "demo":
        subprocess.run([sys.executable, "app.py"], check=True)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
