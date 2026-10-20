"""Run the full training pipeline.

    python run_pipeline.py                 # full run using config/config.yaml
    python run_pipeline.py --quick         # small, fast run for smoke testing
    python run_pipeline.py --n-samples 5000 --n-iter 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from relcompat.config import load_config  # noqa: E402
from relcompat.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to a YAML config (default: config/config.yaml)")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--n-iter", type=int, help="RandomizedSearchCV iterations per model")
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quick", action="store_true", help="3000 samples, 3 folds, 2 iterations")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    quick = {"n_samples": 3000, "cv_folds": 3, "n_iter": 2} if args.quick else {}
    cfg = load_config(args.config, **quick).with_overrides(
        n_samples=args.n_samples, n_iter=args.n_iter, cv_folds=args.cv_folds, seed=args.seed
    )

    metrics = run_pipeline(cfg, make_plots=not args.no_plots)
    print("\n=== Selected models (hold-out test set) ===")
    for task, info in metrics["tasks"].items():
        print(f"{task:15s} {info['selected']:20s} {json.dumps(info['test_metrics'])}")
    print(f"\nArtifacts: {cfg.models_dir}  |  Reports: {cfg.reports_dir}  |  {metrics['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
