"""Create a full XRayMind single-image study packet."""

from __future__ import annotations

import argparse

from xraymind.config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from xraymind.packet import create_study_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an XRayMind study packet")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default="outputs/study_packet")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--label", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-id", default=None)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument(
        "--method",
        default="integrated_gradients",
        choices=["saliency", "input_x_gradient", "integrated_gradients"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    print(f"Study packet saved to {args.out_dir}")
    print(f"Manifest: {manifest['files']['manifest']}")
    print(f"ZIP: {manifest['files']['zip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
