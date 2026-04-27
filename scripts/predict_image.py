"""Run XRayMind prediction from a script entry point."""

from xraymind.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["predict"] + __import__("sys").argv[1:]))
