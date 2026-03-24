
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcr_m_stage_repo.analysis import AnalysisConfig, DEFAULT_METRICS, VALID_STAGES, load_and_filter


def test_load_and_filter_keeps_only_baseline_and_m1_stages():
    csv_path = ROOT / "data" / "raw" / "resultsTRA_CDR3_Div+Clin_2025-07-26.csv"
    config = AnalysisConfig(metrics=DEFAULT_METRICS, stage_order=VALID_STAGES)
    df = load_and_filter(csv_path, config=config, therapy_status="baseline_only")
    assert not df.empty
    assert set(df["Therapy_status_Pre_On"].unique()) == {"PRE"}
    assert set(df["M_Stage"].unique()).issubset({"M1A", "M1B", "M1C"})
    assert df["Subject"].nunique() == len(df)
