from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the existing 7-arm script with local paths without editing repository code")
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=5)
    args = parser.parse_args()

    source = Path(__file__).with_name("run_intraday_7arm_5m.py")
    text = source.read_text()
    replacements = {
        "ROOT = '/mnt/data/phase1_input'": f"ROOT = {str(Path(args.phase1_root).expanduser().resolve())!r}",
        "OUT = '/mnt/data/intraday_7arm_5m'": f"OUT = {str(Path(args.output).expanduser().resolve())!r}",
        "MIN_SIZE = 20": f"MIN_SIZE = {args.min_size}",
        "SAMPLE_EVERY = 5": f"SAMPLE_EVERY = {args.sample_every}",
    }
    for old, new in replacements.items():
        if old not in text:
            raise SystemExit(f"Expected source line not found: {old}")
        text = text.replace(old, new, 1)

    Path(args.output).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gff_7arm_") as temp_dir:
        temp_script = Path(temp_dir) / "run_intraday_7arm_5m.py"
        temp_script.write_text(text)
        subprocess.run([sys.executable, str(temp_script)], check=True)


if __name__ == "__main__":
    main()
