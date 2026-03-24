
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


DEFAULT_METRICS = ["Shan", "Simp", "lowQ", "highQ", "deltaqD", "IPq", "IPslope"]
VALID_STAGES = ["M1A", "M1B", "M1C"]


@dataclass
class AnalysisConfig:
    metrics: list[str]
    stage_order: list[str]
    baseline_label: str = "PRE"
    stage_column: str = "M_Stage"
    therapy_column: str = "Therapy_status_Pre_On"
    subject_column: str = "Subject"


def infer_receptor_name(path: str | Path) -> str:
    name = Path(path).stem.upper()
    for receptor in ["TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"]:
        if receptor in name:
            return receptor
    return Path(path).stem


def clean_stage_values(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .replace({"M1A ": "M1A", "M1B ": "M1B", "M1C ": "M1C"})
    )


def load_and_filter(
    csv_path: str | Path,
    config: AnalysisConfig,
    therapy_status: str = "baseline_only",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    needed = {config.subject_column, config.therapy_column, config.stage_column}
    missing_needed = needed - set(df.columns)
    if missing_needed:
        missing_str = ", ".join(sorted(missing_needed))
        raise ValueError(f"Missing required columns: {missing_str}")

    df[config.stage_column] = clean_stage_values(df[config.stage_column])
    df[config.therapy_column] = df[config.therapy_column].astype(str).str.upper().str.strip()

    available_metrics = [m for m in config.metrics if m in df.columns]
    if not available_metrics:
        raise ValueError("None of the configured metrics are present in the file.")

    keep_cols = list(dict.fromkeys(
        [config.subject_column, config.therapy_column, config.stage_column] + available_metrics
        + [c for c in ["Response", "Cohort", "Mutation_Load", "Neoantigen_Load", "Cytolytic_Score"] if c in df.columns]
    ))
    out = df[keep_cols].copy()

    if therapy_status == "baseline_only":
        out = out[out[config.therapy_column] == config.baseline_label].copy()
    elif therapy_status != "all":
        raise ValueError("therapy_status must be 'baseline_only' or 'all'.")

    out = out[out[config.stage_column].isin(config.stage_order)].copy()

    # If multiple rows remain per subject after filtering, keep the first and warn through column.
    dupes = out.duplicated(subset=[config.subject_column], keep=False)
    if dupes.any():
        out = (
            out.sort_values([config.subject_column, config.stage_column])
            .drop_duplicates(subset=[config.subject_column], keep="first")
            .copy()
        )

    return out.reset_index(drop=True)


def summarize_by_stage(df: pd.DataFrame, metrics: Iterable[str], stage_column: str = "M_Stage") -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for stage, group in df.groupby(stage_column):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) == 0:
                continue
            rows.append(
                {
                    "metric": metric,
                    "stage": stage,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["metric", "stage"]).reset_index(drop=True)
    return result


def run_kruskal_and_pairwise(
    df: pd.DataFrame,
    metrics: Iterable[str],
    stage_order: Iterable[str],
    stage_column: str = "M_Stage",
) -> pd.DataFrame:
    rows: list[dict] = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        groups = []
        nonempty_stages = []
        for stage in stage_order:
            values = pd.to_numeric(df.loc[df[stage_column] == stage, metric], errors="coerce").dropna()
            if len(values) > 0:
                groups.append(values)
                nonempty_stages.append(stage)

        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            kw_stat, kw_p = kruskal(*groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        rows.append(
            {
                "metric": metric,
                "comparison": "Kruskal-Wallis",
                "group1": ",".join(nonempty_stages),
                "group2": "",
                "n1": int(sum(len(g) for g in groups)),
                "n2": np.nan,
                "statistic": kw_stat,
                "p_value": kw_p,
                "p_adj_bh": np.nan,
            }
        )

        pairwise_results = []
        for i, stage1 in enumerate(stage_order):
            for stage2 in list(stage_order)[i + 1:]:
                x = pd.to_numeric(df.loc[df[stage_column] == stage1, metric], errors="coerce").dropna()
                y = pd.to_numeric(df.loc[df[stage_column] == stage2, metric], errors="coerce").dropna()
                if len(x) < 2 or len(y) < 2:
                    stat, p = np.nan, np.nan
                else:
                    stat, p = mannwhitneyu(x, y, alternative="two-sided")
                pairwise_results.append(
                    {
                        "metric": metric,
                        "comparison": "Mann-Whitney U",
                        "group1": stage1,
                        "group2": stage2,
                        "n1": int(len(x)),
                        "n2": int(len(y)),
                        "statistic": stat,
                        "p_value": p,
                    }
                )

        valid_ps = [r["p_value"] for r in pairwise_results if pd.notna(r["p_value"])]
        adjusted = []
        if valid_ps:
            adjusted = multipletests(valid_ps, method="fdr_bh")[1].tolist()

        j = 0
        for r in pairwise_results:
            if pd.notna(r["p_value"]):
                r["p_adj_bh"] = adjusted[j]
                j += 1
            else:
                r["p_adj_bh"] = np.nan
            rows.append(r)

    return pd.DataFrame(rows)


def run_collapsed_test(
    df: pd.DataFrame,
    metrics: Iterable[str],
    stage_column: str = "M_Stage",
) -> pd.DataFrame:
    rows = []
    tmp = df.copy()
    tmp["collapsed_stage"] = tmp[stage_column].replace({"M1A": "M1AB", "M1B": "M1AB", "M1C": "M1C"})

    for metric in metrics:
        if metric not in tmp.columns:
            continue
        x = pd.to_numeric(tmp.loc[tmp["collapsed_stage"] == "M1AB", metric], errors="coerce").dropna()
        y = pd.to_numeric(tmp.loc[tmp["collapsed_stage"] == "M1C", metric], errors="coerce").dropna()
        if len(x) < 2 or len(y) < 2:
            stat, p = np.nan, np.nan
        else:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")

        rows.append(
            {
                "metric": metric,
                "group1": "M1AB",
                "group2": "M1C",
                "n1": int(len(x)),
                "n2": int(len(y)),
                "median_group1": float(x.median()) if len(x) else np.nan,
                "median_group2": float(y.median()) if len(y) else np.nan,
                "statistic": stat,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def save_metric_boxplots(
    df: pd.DataFrame,
    metrics: Iterable[str],
    output_dir: str | Path,
    receptor_name: str,
    stage_order: Iterable[str],
    stage_column: str = "M_Stage",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        data = []
        labels = []
        for stage in stage_order:
            values = pd.to_numeric(df.loc[df[stage_column] == stage, metric], errors="coerce").dropna()
            if len(values) > 0:
                data.append(values)
                labels.append(stage)
        if not data:
            plt.close(fig)
            continue
        ax.boxplot(data, labels=labels)
        ax.set_title(f"{receptor_name}: {metric} by M stage (baseline)")
        ax.set_xlabel("M stage")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}_boxplot.png", dpi=200)
        plt.close(fig)


def run_pipeline(
    csv_path: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig | None = None,
    therapy_status: str = "baseline_only",
) -> dict[str, Path]:
    if config is None:
        config = AnalysisConfig(metrics=DEFAULT_METRICS, stage_order=VALID_STAGES)

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    receptor_name = infer_receptor_name(csv_path)
    filtered = load_and_filter(csv_path, config=config, therapy_status=therapy_status)
    available_metrics = [m for m in config.metrics if m in filtered.columns]

    filtered_path = output_dir / "filtered_baseline_data.csv"
    summary_path = output_dir / "summary_by_stage.csv"
    stats_path = output_dir / "kruskal_pairwise_results.csv"
    collapsed_path = output_dir / "collapsed_m1ab_vs_m1c.csv"

    filtered.to_csv(filtered_path, index=False)

    summary_df = summarize_by_stage(filtered, available_metrics, stage_column=config.stage_column)
    summary_df.to_csv(summary_path, index=False)

    stats_df = run_kruskal_and_pairwise(
        filtered, available_metrics, stage_order=config.stage_order, stage_column=config.stage_column
    )
    stats_df.to_csv(stats_path, index=False)

    collapsed_df = run_collapsed_test(filtered, available_metrics, stage_column=config.stage_column)
    collapsed_df.to_csv(collapsed_path, index=False)

    save_metric_boxplots(
        filtered,
        available_metrics,
        plots_dir,
        receptor_name=receptor_name,
        stage_order=config.stage_order,
        stage_column=config.stage_column,
    )

    return {
        "filtered": filtered_path,
        "summary": summary_path,
        "stats": stats_path,
        "collapsed": collapsed_path,
        "plots_dir": plots_dir,
    }
