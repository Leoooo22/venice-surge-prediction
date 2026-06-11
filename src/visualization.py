"""
Visualization utilities for the surge and PAW notebooks.

Plotting functions are grouped into four sections:

  Data Analysis
  ----
  plot_spatial_grid_map          -- interactive Plotly geo scatter of the model grid
  plot_target_timeseries         -- full train/test time series with split marker
  plot_target_distribution       -- histogram + key-statistic overlay
  plot_monthly_boxplot           -- monthly seasonality boxplot (surge only)

  Model evaluation
  ----------------
  plot_predictions_vs_actual     -- predicted vs actual overlay, P95 outliers highlighted
  plot_residuals                 -- residual distribution histogram
  plot_baseline_vs_tuned_preds   -- single-panel baseline/tuned/actual overlay
  plot_metrics_comparison        -- grouped-bar or two-panel MAE/RMSE comparison

  SHAP
  ----
  plot_shap_summary_beeswarm     -- SHAP beeswarm summary (top-N features)
  plot_shap_importance_bar       -- horizontal bar chart of mean-abs SHAP by feature
  plot_shap_waterfall            -- SHAP waterfall for a single observation
  plot_shap_geo_importance       -- Plotly choropleth of aggregated SHAP per grid point

"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_mpl(path: Optional[str], dpi: int = 150) -> None:
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")


def _save_plotly(fig: go.Figure, path: Optional[str], scale: int = 2) -> None:
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.write_image(path, scale=scale)


# ---------------------------------------------------------------------------
# Data Analysis
# ---------------------------------------------------------------------------


def plot_spatial_grid_map(
    lats: np.ndarray,
    lons: np.ndarray,
    venice_lat: float = 45.44,
    venice_lon: float = 12.33,
    venice_grid_lat: float = 45.75,
    venice_grid_lon: float = 12.25,
    title: str = "Spatial grid — Europe and Mediterranean",
    save_path: Optional[str] = "plots/grid_map.png",
    lon_range: tuple[float, float] = (-22, 32),
    lat_range: tuple[float, float] = (28, 62),
    n_grid_points: Optional[int] = None,
) -> go.Figure:
    
    """Interactive Plotly geo scatter of the model grid with Venice highlighted.

    Parameters
    ----------
    lats, lons:
        Flat arrays of all grid-point coordinates.
    venice_lat, venice_lon:
        True geographic coordinates of Venice.
    venice_grid_lat, venice_grid_lon:
        Coordinates of the grid point closest to Venice.
    title:
        Figure title.
    save_path:
        PNG output path.  ``None`` skips saving.
    lon_range, lat_range:
        Geographic extent of the map view.
    n_grid_points:
        Optional total shown in the legend label (e.g. 1500).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    label = f"Grid points ({n_grid_points:,})" if n_grid_points else "Grid points"

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(size=3, color="#4C72B0", opacity=0.5),
        name=label,
        hovertemplate="lat=%{lat:.2f}<br>lon=%{lon:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[venice_grid_lat], lon=[venice_grid_lon],
        mode="markers+text",
        marker=dict(size=10, color="orange", symbol="diamond"),
        text=["Venice (grid)"],
        textposition="top right",
        name=f"Venice grid point ({venice_grid_lat}, {venice_grid_lon})",
        hovertemplate=(
            f"Nearest grid point to Venice<br>"
            f"lat={venice_grid_lat}, lon={venice_grid_lon}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scattergeo(
        lat=[venice_lat], lon=[venice_lon],
        mode="markers+text",
        marker=dict(size=12, color="red", symbol="star"),
        text=["Venice"],
        textposition="bottom right",
        name=f"Venice ({venice_lat}, {venice_lon})",
        hovertemplate=f"Venice<br>lat={venice_lat}, lon={venice_lon}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        geo=dict(
            scope="europe",
            showland=True, landcolor="#F0F0F0",
            showcoastlines=True, coastlinecolor="#AAAAAA",
            showocean=True, oceancolor="#D6EAF8",
            showcountries=True, countrycolor="#CCCCCC",
            projection_type="natural earth",
            lonaxis_range=list(lon_range),
            lataxis_range=list(lat_range),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        width=900, height=600,
    )
    _save_plotly(fig, save_path)
    return fig


def plot_target_timeseries(
    y_train: pd.Series,
    y_test: pd.Series,
    split_time: pd.Timestamp,
    title: str,
    ylabel: str = "Surge (m)",
    save_path: Optional[str] = None,
    split_date_fmt: str = "%Y-%m-%d %H:%M",
) -> None:
    """Full train/test time series with a vertical split marker.

    Parameters
    ----------
    y_train, y_test:
        Target series for the training and test periods.
    split_time:
        Timestamp of the train/test boundary.
    title:
        Axes title.
    ylabel:
        Y-axis label.
    save_path:
        PNG output path.
    split_date_fmt:
        ``strftime`` format used in the split-marker legend label.
    """
    
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(16, 4))
    ax.plot(y_train.index, y_train.values, color="#4C72B0", linewidth=0.6,
            label="Train", alpha=0.85)
    ax.plot(y_test.index, y_test.values, color="#DD8452", linewidth=0.6,
            label="Test", alpha=0.85)
    ax.axvline(x=split_time, color="crimson", linewidth=1.5, linestyle="--",
               label=f"Split: {split_time.strftime(split_date_fmt)}")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def plot_target_distribution(
    y: pd.Series,
    stat_lines: Sequence[tuple[float, str, str, str]],
    title: str,
    xlabel: str = "Surge (m)",
    save_path: Optional[str] = None,
    bins: int = 60,
) -> None:
    """Histogram of target values with key-statistic vertical lines and a stats box.

    Parameters
    ----------
    y:
        Full target series (train + test).
    stat_lines:
        Each element is (value, label, colour, linestyle) for one vertical
        reference line (mean, median, P95, P99).
    title:
        Axes title.
    xlabel:
        X-axis label.
    save_path:
        PNG output path.
    bins:
        Number of histogram bins.
    """
    
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y.values, bins=bins, color="#4C72B0", edgecolor="white",
            alpha=0.75, density=True, label="Histogram")
    for val, lbl, col, ls in stat_lines:
        ax.axvline(x=val, color=col, linestyle=ls, linewidth=1.6, label=lbl)

    mean_, std_, min_, max_ = y.mean(), y.std(), y.min(), y.max()
    p95_, p99_ = y.quantile(0.95), y.quantile(0.99)
    stats_text = (
        f"Mean:   {mean_:.4f} m\n"
        f"Std:    {std_:.4f} m\n"
        f"Min:    {min_:.4f} m\n"
        f"Max:    {max_:.4f} m\n"
        f"P95:    {p95_:.4f} m\n"
        f"P99:    {p99_:.4f} m"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def plot_monthly_boxplot(
    y_train: pd.Series,
    title: str,
    ylabel: str = "Surge (m)",
    save_path: Optional[str] = None,
) -> None:
    """Monthly seasonality boxplot of the training-set target variable.

    Parameters
    ----------
    y_train:
        Training target series with a DatetimeIndex.
    title:
        Axes title.
    ylabel:
        Y-axis label.
    save_path:
        PNG output path.
    """
    
    plt.style.use("seaborn-v0_8-whitegrid")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = y_train.index.month
    monthly_groups = [y_train.values[months == m] for m in range(1, 13)]

    _, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(
        monthly_groups,
        labels=month_names,
        patch_artist=True,
        medianprops=dict(color="crimson", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#4C72B0", alpha=0.4),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#4C72B0" if i % 2 == 0 else "#6CA6C1")
        patch.set_alpha(0.6)
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def plot_predictions_vs_actual(
    y_test: pd.Series,
    y_pred: np.ndarray,
    title: str,
    ylabel: str = "Surge (m)",
    save_path: Optional[str] = None,
) -> None:
    """Predicted vs actual time series with P95-error outliers highlighted.

    Parameters
    ----------
    y_test:
        Actual test-set values.
    y_pred:
        Model predictions, same length as ``y_test``.
    title:
        Axes title.
    ylabel:
        Y-axis label.
    save_path:
        PNG output path.
    """
    
    plt.style.use("seaborn-v0_8-whitegrid")
    abs_errors = np.abs(y_test.values - y_pred)
    err_p95 = np.percentile(abs_errors, 95)
    mask_outlier = abs_errors > err_p95

    _, ax = plt.subplots(figsize=(16, 4))
    ax.plot(y_test.index, y_test.values, color="#4C72B0", linewidth=0.8,
            label="Actual", zorder=2)
    ax.plot(y_test.index, y_pred, color="#DD8452", linewidth=0.8, linestyle="--",
            label="Predicted (XGBoost)", zorder=3)
    ax.scatter(
        y_test.index[mask_outlier], y_test.values[mask_outlier],
        color="crimson", s=18, zorder=4,
        label=f"Error > P95 ({err_p95:.4f} m)  [{mask_outlier.sum()} points]",
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Residual distribution histogram (predicted - actual).

    Parameters
    ----------
    y_true:
        Actual values.
    y_pred:
        Predicted values.
    title:
        Axes title.
    save_path:
        PNG output path.
    """
    
    plt.style.use("seaborn-v0_8-whitegrid")
    residuals = y_pred - np.asarray(y_true)
    res_mean = residuals.mean()
    res_std = residuals.std()

    _, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=60, color="#4C72B0", edgecolor="white",
            alpha=0.75, density=True, label="Residuals (density)")
    ax.axvline(x=0, color="black", linewidth=1.5, linestyle="-", label="Zero")
    ax.axvline(x=res_mean, color="#2ca02c", linewidth=1.4, linestyle="--",
               label=f"Mean residual ({res_mean:.4f} m)")
    ax.text(
        0.98, 0.97,
        f"Mean:   {res_mean:.4f} m\nStd:    {res_std:.4f} m",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Residual (m)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def  plot_baseline_vs_tuned_preds(
    y_test: pd.Series,
    y_pred_baseline: np.ndarray,
    y_pred_tuned: np.ndarray,
    title: str,
    ylabel: str = "Surge (m)",
    save_path: Optional[str] = None,
) -> None:
    """Actual, baseline, and tuned predictions on one axes, P95 errors highlighted.

    Parameters
    ----------
    y_test:
        Actual test-set values (time-indexed Series).
    y_pred_baseline:
        Baseline model predictions.
    y_pred_tuned:
        Tuned model predictions.
    title:
        Axes title.
    ylabel:
        Y-axis label.
    save_path:
        PNG output path.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    abs_errors_tuned = np.abs(y_test.values - y_pred_tuned)
    err_p95_tuned = np.percentile(abs_errors_tuned, 95)
    mask_outlier_tuned = abs_errors_tuned > err_p95_tuned

    _, ax = plt.subplots(figsize=(16, 4))
    ax.plot(y_test.index, y_test.values, color="#626161", linewidth=0.9,
            label="Actual", zorder=2)
    ax.plot(y_test.index, y_pred_baseline, color="#0062FF", linewidth=0.7,
            linestyle="--", label="Baseline (XGBoost)", alpha=0.75, zorder=3)
    ax.plot(y_test.index, y_pred_tuned, color="#FF9500", linewidth=0.7,
            linestyle="--", label="Tuned (XGBoost)", alpha=0.75, zorder=4)
    ax.scatter(
        y_test.index[mask_outlier_tuned], y_test.values[mask_outlier_tuned],
        color="crimson", s=18, zorder=5,
        label=(
            f"Tuned error > P95 ({err_p95_tuned:.4f} m)  "
            f"[{mask_outlier_tuned.sum()} points]"
        ),
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def plot_metrics_comparison(
    baseline_mae: float,
    baseline_rmse: float,
    tuned_mae: float,
    tuned_rmse: float,
    title: str,
    save_path: Optional[str] = None,
    layout: str = "grouped",
) -> None:
    """Bar chart comparing baseline vs tuned MAE and RMSE.

    Parameters
    ----------
    baseline_mae, baseline_rmse:
        Error metrics for the baseline model on the test set.
    tuned_mae, tuned_rmse:
        Error metrics for the tuned model on the test set.
    title:
        Figure title.
    save_path:
        PNG output path.
    layout:
        ``'grouped'`` — single axes with two metric groups (MAE, RMSE), each
        bar pair side-by-side (surge style).
        ``'subplots'`` — two side-by-side subplots, one per metric (PAW style).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    if layout == "grouped":
        labels = ["MAE", "RMSE"]
        baseline_vals = [baseline_mae, baseline_rmse]
        tuned_vals = [tuned_mae, tuned_rmse]
        x = np.arange(len(labels))
        width = 0.35

        _, ax = plt.subplots(figsize=(7, 5))
        bars_b = ax.bar(x - width / 2, baseline_vals, width,
                        color="#4C72B0", alpha=0.85, label="Baseline", edgecolor="white")
        bars_t = ax.bar(x + width / 2, tuned_vals, width,
                        color="#DD8452", alpha=0.85, label="Tuned", edgecolor="white")
        for bar in bars_b:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
        for bar in bars_t:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_ylabel("Error (m)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(baseline_vals + tuned_vals) * 1.18)

    elif layout == "subplots":
        metric_labels = ["MAE", "RMSE"]
        vals_base = [baseline_mae, baseline_rmse]
        vals_tuned = [tuned_mae, tuned_rmse]
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, metric, vb, vt in zip(axes, metric_labels, vals_base, vals_tuned):
            bars_b = ax.bar(-width / 2, vb, width, label="Baseline",
                            color="#4C72B0", alpha=0.85)
            bars_t = ax.bar(+width / 2, vt, width, label="Tuned",
                            color="#DD8452", alpha=0.85)
            ax.set_title(metric)
            ax.set_xticks([])
            ax.set_ylabel("m")
            ax.legend()
            for container, val in [(bars_b, vb), (bars_t, vt)]:
                rect = container[0]
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.0005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle(title, fontsize=13)

    else:
        raise ValueError(f"layout must be 'grouped' or 'subplots', got {layout!r}")

    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def plot_shap_summary_beeswarm(
    shap_values: np.ndarray,
    X_data: np.ndarray,
    feature_names: list[str],
    title: str,
    save_path: Optional[str] = None,
    max_display: int = 20,
) -> None:
    """SHAP beeswarm summary plot for the top-N most important features.

    Parameters
    ----------
    shap_values:
        2-D array of shape ``(n_samples, n_features)``.
    X_data:
        Feature matrix (same shape as shap_values).
    feature_names:
        Column names corresponding to the second axis.
    title:
        Axes title.
    save_path:
        PNG output path.
    max_display:
        Maximum number of features to display.
    """
    import shap as _shap  # lazy import — heavy dependency

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.subplots(figsize=(12, 8))
    _shap.summary_plot(
        shap_values, X_data,
        feature_names=feature_names,
        plot_type="dot",
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def plot_shap_importance_bar(
    mean_abs_shap: np.ndarray,
    feature_names: list[str],
    title: str,
    save_path: Optional[str] = None,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of mean absolute SHAP importance, top-N features.

    Parameters
    ----------
    mean_abs_shap:
        1-D array of mean absolute SHAP values, one per feature.
    feature_names:
        Names aligned with ``mean_abs_shap``.
    title:
        Axes title.
    save_path:
        PNG output path.
    top_n:
        Number of top features to show.
    """
    
    order = np.argsort(mean_abs_shap)[::-1]
    top_idx = order[:top_n]
    top_feat = [feature_names[i] for i in top_idx]
    top_imp = mean_abs_shap[top_idx]

    _, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), top_imp[::-1], color="#4C72B0",
                   edgecolor="white", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feat[::-1], fontsize=9)
    for bar, val in zip(bars, top_imp[::-1]):
        ax.text(
            bar.get_width() + max(top_imp) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f} m", va="center", fontsize=8,
        )
    ax.set_xlabel("Mean absolute SHAP importance (m)")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlim(0, max(top_imp) * 1.18)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def plot_shap_waterfall(
    shap_explanation,
    idx: int,
    title: str,
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> None:
    """SHAP waterfall plot for a single observation.

    Parameters
    ----------
    shap_explanation:
        A ``shap.Explanation`` object (full test set).
    idx:
        Row index of the observation to display.
    title:
        Axes title.
    save_path:
        PNG output path.
    max_display:
        Maximum number of features to show in the waterfall.
    """
    
    import shap as _shap  # lazy import

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.subplots(figsize=(12, 7))
    _shap.plots.waterfall(shap_explanation[idx], max_display=max_display, show=False)
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    _save_mpl(save_path)
    plt.show()


def plot_shap_geo_importance(
    lats: np.ndarray,
    lons: np.ndarray,
    importance_values: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
    venice_grid_lat: float = 45.75,
    venice_grid_lon: float = 12.25,
    lon_range: tuple[float, float] = (8, 24),
    lat_range: tuple[float, float] = (36, 49),
    colorbar_title: str = "SHAP imp.<br>(m)",
    projection: str = "natural earth",
) -> go.Figure:
    """Plotly choropleth of aggregated SHAP importance per grid point.

    The function accepts pre-aggregated ``(lats, lons, importance_values)``
    arrays — the aggregation logic (direct SHAP or PCA back-projection) is
    left to the notebook so that this function is reusable for both datasets.

    Parameters
    ----------
    lats, lons:
        Coordinates of each grid point, shape ``(n_points,)``.
    importance_values:
        Aggregated SHAP importance per grid point, same shape.
    title:
        Figure title.
    save_path:
        PNG output path.
    venice_grid_lat, venice_grid_lon:
        Coordinates of the Venice marker.
    lon_range, lat_range:
        Geographic extent of the map view.
    colorbar_title:
        Title of the colour scale bar.
    projection:
        Plotly geo projection type.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    
    imp_min, imp_max = importance_values.min(), importance_values.max()
    marker_size = 6 + 20 * (importance_values - imp_min) / (imp_max - imp_min + 1e-9)

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(
            size=marker_size,
            color=importance_values,
            colorscale="RdYlBu_r",
            showscale=True,
            colorbar=dict(title=colorbar_title, thickness=14, len=0.6),
            line=dict(width=0.3, color="white"),
        ),
        hovertemplate=(
            "lat=%{lat:.2f}  lon=%{lon:.2f}<br>"
            "SHAP importance: %{marker.color:.4f} m<extra></extra>"
        ),
        name="Grid",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[venice_grid_lat], lon=[venice_grid_lon],
        mode="markers+text",
        marker=dict(size=15, color="red", symbol="star",
                    line=dict(width=1, color="white")),
        text=["Venice"],
        textposition="top right",
        textfont=dict(size=11, color="red"),
        name=f"Venice ({venice_grid_lat}, {venice_grid_lon})",
        hovertemplate=(
            f"Venice<br>lat={venice_grid_lat}, lon={venice_grid_lon}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        geo=dict(
            scope="europe",
            showland=True, landcolor="#F0F0F0",
            showcoastlines=True, coastlinecolor="#AAAAAA",
            showocean=True, oceancolor="#D6EAF8",
            showcountries=True, countrycolor="#CCCCCC",
            projection_type=projection,
            lonaxis_range=list(lon_range),
            lataxis_range=list(lat_range),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        width=850, height=580,
    )
    _save_plotly(fig, save_path)
    return fig
