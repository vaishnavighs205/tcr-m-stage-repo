
from __future__ import annotations

import argparse
from pathlib import Path

from tcr_m_stage_repo.analysis import AnalysisConfig, DEFAULT_METRICS, VALID_STAGES, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TCR diversity by M stage analysis.")
    parser.add_argument("--input", required=True, help="Path to one receptor CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--therapy-status",
        default="baseline_only",
        choices=["baseline_only", "all"],
        help="Use baseline only (default) or all samples.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to analyze.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AnalysisConfig(metrics=args.metrics, stage_order=VALID_STAGES)
    run_pipeline(
        csv_path=args.input,
        output_dir=args.output_dir,
        config=config,
        therapy_status=args.therapy_status,
    )
    print(f"Done. Results saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
