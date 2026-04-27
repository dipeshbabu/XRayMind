"""Tiny API smoke test for a running XRayMind FastAPI server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a running XRayMind API server")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--image", required=True, help="Image or DICOM path to upload")
    parser.add_argument("--api-key", default=os.getenv("XRAYMIND_API_KEY"), help="Optional API key")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    health = requests.get(f"{args.url.rstrip('/')}/health", timeout=20)
    health.raise_for_status()
    print("Health:", health.json())

    image_path = Path(args.image)
    with image_path.open("rb") as handle:
        response = requests.post(
            f"{args.url.rstrip('/')}/predict",
            headers=headers,
            files={"file": (image_path.name, handle)},
            timeout=120,
        )
    response.raise_for_status()
    print("Prediction:", response.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
