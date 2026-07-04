import logging
import time
from pathlib import Path
from graphfactorfactory.themes.pipeline import ThemeDiscoveryPipeline, ThemeDiscoveryConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_benchmark(workers):
    output_root = Path("data/graph_store").resolve()
    # Reset themes output so it's a clean run each time
    import shutil
    themes_dir = output_root / "themes"
    if themes_dir.exists():
        shutil.rmtree(themes_dir)

    theme_config = ThemeDiscoveryConfig(run_id=f"benchmark_{workers}w")
    theme_pipeline = ThemeDiscoveryPipeline(output_root, themes_dir, theme_config)
    
    start_time = time.time()
    theme_pipeline.run(max_workers=workers)
    elapsed = time.time() - start_time
    return elapsed

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Benchmark: 13 Workers (5-min frequency, 2 days)")
    print("=" * 50)
    time_13 = run_benchmark(13)
    
    print("=" * 50)
    print("Starting Benchmark: 26 Workers (5-min frequency, 2 days)")
    print("=" * 50)
    time_26 = run_benchmark(26)

    print("=" * 50)
    print("BENCHMARK RESULTS")
    print(f"13 Workers: {time_13:.2f} seconds")
    print(f"26 Workers: {time_26:.2f} seconds")
    print("=" * 50)
