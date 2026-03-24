
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcr_m_stage_repo.analysis import AnalysisConfig, DEFAULT_METRICS, VALID_STAGES, infer_receptor_name, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis on all receptor CSV files in a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for result folders.")
    parser.add_argument(
        "--therapy-status",
        default="baseline_only",
        choices=["baseline_only", "all"],
        help="Use baseline only (default) or all samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config = AnalysisConfig(metrics=DEFAULT_METRICS, stage_order=VALID_STAGES)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    for csv_path in csv_files:
        receptor = infer_receptor_name(csv_path)
        target = output_dir / receptor
        run_pipeline(csv_path=csv_path, output_dir=target, config=config, therapy_status=args.therapy_status)
        print(f"Finished {csv_path.name} -> {target}")


if __name__ == "__main__":
    main()
