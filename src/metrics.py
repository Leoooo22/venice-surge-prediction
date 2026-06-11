"""
Metrics utilities for the surge and PAW notebooks.

Functions:
    compute_metrics          -- MAE, RMSE, R² for a single model / split
    print_metrics_table      -- formatted console table from compute_metrics results
    print_comparative_table  -- 4-row baseline vs tuned table with a mid-separator
    print_delta              -- (tuned - baseline) deltas with improvement labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, label: str = "") -> dict:
    """Compute MAE, RMSE, and R² for a single model / split.

    Parameters
    ----------
    y_true:
        True target values.
    y_pred:
        Model predictions.
    label:
        Name for this row
        
    Returns
    -------
    dict with keys ``'Set'``, ``'MAE (m)'``, ``'RMSE (m)'``, ``'R²'``.
    """
    
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Set": label, "MAE (m)": mae, "RMSE (m)": rmse, "R²": r2}


def print_metrics_table(rows: list[dict], title: str = "METRICS") -> None:
    """Print a formatted metrics table.

    Parameters
    ----------
    rows:
        List of dicts returned by func:`compute_metrics`.
    title:
        Header text.
    """
    
    w = 54
    print("=" * w)
    print(f"  {title}")
    print("=" * w)
    print(f"  {'Set':<22} {'MAE (m)':>10} {'RMSE (m)':>10} {'R²':>8}")
    print("  " + "-" * (w - 2))
    for r in rows:
        print(
            f"  {r['Set']:<22} {r['MAE (m)']:>10.5f} "
            f"{r['RMSE (m)']:>10.5f} {r['R²']:>8.4f}"
        )
    print("=" * w)


def print_comparative_table(
    rows: list[dict], title: str = "COMPARATIVE METRICS"
) -> None:
    """Print a 4-row baseline vs tuned table with a separator after row 2.

    Parameters
    ----------
    rows:
        ``[baseline_train, baseline_test, tuned_train, tuned_test]`` dicts
        from :func:`compute_metrics`.
    title:
        Header text.
    """
    
    w = 54
    print("=" * w)
    print(f"  {title}")
    print("=" * w)
    print(f"  {'Model | Set':<18} {'MAE (m)':>8} {'RMSE (m)':>9} {'R²':>7}")
    print("  " + "-" * (w - 2))
    for i, r in enumerate(rows):
        if i == 2:
            print("  " + "-" * (w - 2))
        print(
            f"  {r['Set']:<18} {r['MAE (m)']:>8.5f} "
            f"{r['RMSE (m)']:>9.5f} {r['R²']:>7.4f}"
        )
    print("=" * w)

def print_delta(baseline_row: dict, tuned_row: dict) -> None:
    """Print MAE, RMSE, and R² deltas (tuned - baseline) with improvement labels.

    Parameters
    ----------
    baseline_row, tuned_row:
        Output of :func:`compute_metrics` for the baseline and tuned test rows.
    """
    
    delta_mae = tuned_row["MAE (m)"] - baseline_row["MAE (m)"]
    delta_rmse = tuned_row["RMSE (m)"] - baseline_row["RMSE (m)"]
    delta_r2 = tuned_row["R²"] - baseline_row["R²"]
    print("\n  Delta test (Tuned - Baseline):")
    print(
        f"  MAE:  {delta_mae:+.5f} m  "
        f"({'improvement' if delta_mae < 0 else 'degradation'})"
    )
    print(
        f"  RMSE: {delta_rmse:+.5f} m  "
        f"({'improvement' if delta_rmse < 0 else 'degradation'})"
    )
    print(
        f"  R²:   {delta_r2:+.5f}    "
        f"({'improvement' if delta_r2 > 0 else 'degradation'})"
    )


def print_final_summary(
    title: str,
    dataset_info: str,
    metrics: list[dict],
    top_shap_lines: list[str],
    width: int = 62,
) -> None:
    """Print the final narrative summary for a model notebook.

    Parameters
    ----------
    title:
        Model title, e.g. ``'XGBoost Surge Model'``.
    dataset_info:
        Pre-indented multi-line string describing the dataset
        (period, train/test split, features, target).
    metrics:
        Four dicts from :func:`compute_metrics` in order:
        ``[baseline_train, baseline_test, tuned_train, tuned_test]``.
    top_shap_lines:
        Pre-formatted strings for the top-N SHAP features section
        (one string per feature, without leading indentation — the
        function adds two spaces automatically).
    width:
        Total width of the separator lines.
    """
    sep  = "=" * width
    dash = "  " + "-" * (width - 2)

    print(sep)
    print(f"  FINAL SUMMARY — {title}")
    print(sep)

    print("\n  Dataset")
    print(dash)
    print(dataset_info)

    print("\n  Model performance")
    print(dash)
    print(f"  {'Model':<22} {'MAE (m)':>10} {'RMSE (m)':>10} {'R²':>8}")
    print(dash)
    for i, r in enumerate(metrics):
        if i == 2:
            print(dash)
        print(
            f"  {r['Set']:<22} {r['MAE (m)']:>10.5f} "
            f"{r['RMSE (m)']:>10.5f} {r['R²']:>8.4f}"
        )
    print(dash)

    print("\n  Top SHAP features")
    print(dash)
    for line in top_shap_lines:
        print(f"  {line}")
    print(sep)


