import argparse
import logging
import subprocess
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_day(edge_dir: Path, out_dir: Path) -> bool:
    try:
        cmd = [
            "python", "scripts/build_b50_b35_theme_forest.py",
            "--p0-edges", str(edge_dir),
            "--out-dir", str(out_dir),
            "--output-format", "parquet"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed for {edge_dir.name}:\n{e.stderr}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    p0_root = Path(args.p0_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    date_dirs = sorted([d for d in p0_root.iterdir() if d.is_dir() and d.name.startswith("date=")])
    logging.info(f"Found {len(date_dirs)} dates to process.")

    completed = 0
    start_t = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for d in date_dirs:
            out_d = out_root / d.name
            if out_d.exists() and (out_d / "manifest.json").exists():
                logging.info(f"Skipping {d.name}, already done.")
                completed += 1
                continue
            futures[pool.submit(run_day, d, out_d)] = d.name

        for f in as_completed(futures):
            name = futures[f]
            try:
                if f.result():
                    completed += 1
                    logging.info(f"[{completed}/{len(date_dirs)}] Completed {name}")
                else:
                    logging.error(f"Failed {name}")
            except Exception as e:
                logging.exception(f"Exception processing {name}")

    logging.info(f"All done in {time.time() - start_t:.1f} seconds.")

if __name__ == "__main__":
    main()
