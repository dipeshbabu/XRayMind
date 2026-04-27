"""Convert a DICOM image to PNG and optionally save safe metadata."""

from __future__ import annotations

import argparse

from xraymind.dicom import dicom_to_png, redact_dicom, write_safe_metadata_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert/redact DICOM files for XRayMind research workflows")
    parser.add_argument("--dicom", required=True, help="Input DICOM path")
    parser.add_argument("--png", default="outputs/dicom/preview.png", help="Output PNG path")
    parser.add_argument("--metadata", default=None, help="Optional safe metadata JSON path")
    parser.add_argument("--redacted", default=None, help="Optional redacted DICOM copy path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    png_path = dicom_to_png(args.dicom, args.png)
    print(f"Saved PNG preview to {png_path}")
    if args.metadata:
        metadata_path = write_safe_metadata_json(args.dicom, args.metadata)
        print(f"Saved safe metadata to {metadata_path}")
    if args.redacted:
        redacted_path = redact_dicom(args.dicom, args.redacted)
        print(f"Saved redacted DICOM copy to {redacted_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
