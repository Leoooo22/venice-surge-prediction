"""
Preprocessing utilities for the surge and PAW notebooks.

Shared fuctions:
    print_dataset_structure  -- print shape, index levels and column dtypes
    plot_missing             -- missing-value report and bar chart
    check_grid               -- spatial grid uniformity check
    temporal_split           -- chronological 80/20 train/test split

Surge-specific:
    build_surge_features     -- Adriatic filtering, NaN removal, wide pivot

PAW-specific:
    build_paw_features       -- lag filtering, NaN removal, wide pivot
    explore_pca_variance     -- plot cumulative PCA variance and return component counts
    fit_standard_scaler      -- fit StandardScaler on train, transform both splits, persist
    apply_pca                -- fit PCA on scaled train, transform both splits, persist
"""

from __future__ import annotations
import os
from typing import Optional
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared functions exploratory
# ---------------------------------------------------------------------------


def print_dataset_structure(df: pd.DataFrame, name: str) -> None:
    """Print shape, index levels, and column dtypes of a MultiIndex DataFrame.

    Parameters
    ----------
    df:
        DataFrame with a multi-level index.
    name:
        Label printed in the header.
    """
    
    print(f"{'=' * 40}")
    print(f"  Dataset: {name}")
    print(f"{'=' * 40}")
    print(f"  Shape (rows, columns): {df.shape}")
    print(f"  N. rows: {len(df):,}")
    print(f"  N. columns: {len(df.columns)}")
    print()
    print("  --- Indexes per row ---")
    for i, (idx_name, idx_dtype) in enumerate(zip(df.index.names, df.index.dtypes)):
        vals = df.index.get_level_values(i)
        print(
            f"  [{i}] {idx_name:<12} dtype={idx_dtype}  "
            f"range=[{vals.min()}, {vals.max()}]  unique={vals.nunique()}"
        )
    print()
    print("  --- Columns ---")
    for col in df.columns:
        print(f"  {col:<14} dtype={df[col].dtype}")
    print()


def plot_missing(df: pd.DataFrame, name: str, filepath: str) -> None:
    """Print missing-value statistics and save a bar chart.

    Parameters
    ----------
    df:
        Source DataFrame.
    name:
        Label used in the title and console output.
    filepath:
        Full path (including filename) where the PNG is saved.
        Parent directory is created automatically.
    """
    
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100

    print(f"Missing values — {name}")
    print(f"{'Column':<14} {'Count':>12} {'%':>8}")
    print("-" * 38)
    for col in df.columns:
        print(f"{col:<14} {missing[col]:>12,} {missing_pct[col]:>7.2f}%")

    total_cells = len(df) * len(df.columns)
    total_missing = missing.sum()
    print(f"\nTotal cells: {total_cells:,}")
    print(f"Total NaN: {total_missing:,}  ({total_missing / total_cells * 100:.2f}%)\n")

    cols_with_missing = missing[missing > 0]
    if cols_with_missing.empty:
        print(f"No missing values in {name}.\n")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(cols_with_missing) * 0.6), 5))
    bars = ax.bar(
        cols_with_missing.index,
        cols_with_missing.values / 1e6,
        color="#4C72B0",
        edgecolor="white",
    )
    ax.set_title(f"Missing values per column — {name}", fontsize=13, pad=12)
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing values (millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    plt.xticks(rotation=45, ha="right", fontsize=9)
    for bar, pct in zip(bars, cols_with_missing / len(df) * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()


def check_grid(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Print spatial grid uniformity info and return the unique coordinate arrays.

    Parameters
    ----------
    df:
        DataFrame with a MultiIndex containing ``latitude`` and ``longitude``
        levels.
    name:
        Label printed in the console output.

    Returns
    -------
    lats, lons:
        Sorted arrays of unique latitude and longitude values.
    """
    
    lats = np.sort(df.index.get_level_values("latitude").unique())
    lons = np.sort(df.index.get_level_values("longitude").unique())
    lat_diffs = np.diff(lats)
    lon_diffs = np.diff(lons)

    print(f"Spatial grid — {name}")
    print(f"  Unique latitudes:   {len(lats)}  range [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"  Unique longitudes:  {len(lons)}  range [{lons.min():.2f}, {lons.max():.2f}]")
    print(f"  Total grid points: {len(lats) * len(lons)}")
    print()
    print(
        f"  Spacing lat — min={lat_diffs.min():.4f}  max={lat_diffs.max():.4f}  "
        f"std={lat_diffs.std():.6f}  uniform={np.allclose(lat_diffs, lat_diffs[0])}"
    )
    print(
        f"  Spacing lon — min={lon_diffs.min():.4f}  max={lon_diffs.max():.4f}  "
        f"std={lon_diffs.std():.6f}  uniform={np.allclose(lon_diffs, lon_diffs[0])}"
    )
    print()
    return lats, lons



def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    """Chronological split without shuffling, preserving temporal order.

    Parameters
    ----------
    X:
        Feature matrix indexed by time.
    y:
        Target series indexed by time.
    ratio:
        Fraction of time steps assigned to the training set.

    Returns
    -------
    X_train, X_test, y_train, y_test, split_time
    """
    
    times = X.index.sort_values().unique()
    n_train = int(len(times) * ratio)
    split_time = times[n_train]

    X_train = X[X.index < split_time]
    X_test = X[X.index >= split_time]
    y_train = y[y.index < split_time]
    y_test = y[y.index >= split_time]
    
    print("=" * 65)
    print(" TEMPORAL SPLIT — 80% train | 20% test (chronological order)")
    print("=" * 65)
    print()
    print(f"  Split point:  {split_time}")
    print(f"  Train: X={X_train.shape}  y = {len(y_train):,}")
    print(f"  period: {X_train.index.min()}  -> {X_train.index.max()}")
    print()
    print(f"  Test: X={X_test.shape}  y = {len(y_test):,}")
    print(f"  period: {X_test.index.min()}  -> {X_test.index.max()}")
    print()
    print("=" * 65)
    
    return X_train, X_test, y_train, y_test, split_time


# ---------------------------------------------------------------------------
# Surge: feature matrix construction
# ---------------------------------------------------------------------------


def build_surge_features(
    surge_pressures: pd.DataFrame,
    lat_min: float = 38.0,
    lat_max: float = 47.0,
    lon_min: float = 10.0,
    lon_max: float = 22.0,
    target_lat: float = 45.75,
    target_lon: float = 12.25,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    
    """Filter, clean and pivot ``surge_pressures`` into a wide feature matrix.

    Steps
    -----
    1. Restrict to the Adriatic bounding box [lat_min, lat_max] * [lon_min, lon_max].
    2. Drop rows with any NaN.
    3. Remove grid points whose time-series length is shorter than the maximum
       (ensures a perfectly rectangular panel before pivoting).
    4. Extract the target variable (surge) at the grid point closest to Venice.
    5. Pivot all pressure features to wide format:
       one column per (feature, lat, lon) combination.

    Parameters
    ----------
    surge_pressures:
    Raw DataFrame loaded from ``data/surge_pressures.parquet``.
    Expected MultiIndex: (latitude, longitude, time).
    lat_min, lat_max, lon_min, lon_max:
        Bounding box for the Adriatic sub-grid.
    target_lat, target_lon:
        Grid point used to extract the target (Venice proxy).
    verbose:
        Print step-by-step row counts when True.

    Returns
    -------
    X_wide:
        Feature matrix indexed by ``time``, shape
        (n_timestamps, n_features * n_grid_points).
    y_surge:
        Target series indexed by ``time``.
    """
    
    if verbose:
        print("=== surge_pressures — Preprocessing ===\n")
        print(f"[0] Original:  {len(surge_pressures):,} rows")

    # Step 1 — Adriatic bounding box
    lats = surge_pressures.index.get_level_values("latitude")
    lons = surge_pressures.index.get_level_values("longitude")
    mask_adria = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
    sp = surge_pressures[mask_adria]
    n_lats_a = sp.index.get_level_values("latitude").nunique()
    n_lons_a = sp.index.get_level_values("longitude").nunique()
    if verbose:
        print(
            f"[1] Adriatic area:  {len(sp):,} rows  "
            f"({n_lats_a} lat * {n_lons_a} lon = {n_lats_a * n_lons_a} theoretical points)"
        )

    # Step 2 — Drop NaN rows
    n_before_dropna = len(sp)
    sp = sp.dropna()
    if verbose:
        print(f"[2] Drop NaN:  {len(sp):,} rows  (removed {n_before_dropna - len(sp):,})")

    # Step 3 — Remove grid points with an incomplete time series
    n_timestamps = sp.index.get_level_values("time").nunique()
    points_coverage = sp.groupby(level=["latitude", "longitude"]).size()
    complete_points = set(points_coverage[points_coverage == n_timestamps].index.tolist())

    theoretical_points = set(
        (lat, lon)
        for lat in surge_pressures[mask_adria].index.get_level_values("latitude").unique()
        for lon in surge_pressures[mask_adria].index.get_level_values("longitude").unique()
    )
    
    excluded_points = theoretical_points - complete_points

    mask_complete = [
        (lat, lon) in complete_points
        for lat, lon in zip(
            sp.index.get_level_values("latitude"),
            sp.index.get_level_values("longitude"),
        )
    ]
    sp = sp[mask_complete].copy()
    sp.index = sp.index.remove_unused_levels()
    if verbose:
        print(
            f"[3] Complete grid points:  {len(sp):,} rows  "
            f"({len(complete_points)} points — excluded {len(excluded_points)}: {excluded_points})"
        )

    # Step 4 — Target at the Venice grid point
    y_surge = sp.xs((target_lat, target_lon), level=["latitude", "longitude"])["surge"]
    y_surge = y_surge.sort_index()
    if verbose:
        print(f"[4] Target y_surge:  {len(y_surge):,} time steps")

    # Step 5 — Pivot to wide format
    feature_cols = [c for c in sp.columns if c not in ("valid_time", "surge")]
    sp_wide = sp[feature_cols].unstack(level=["latitude", "longitude"])
    sp_wide.columns = [
        f"{feat}_lat{lat}_lon{lon}" for feat, lat, lon in sp_wide.columns
    ]
    sp_wide = sp_wide.sort_index()
    if verbose:
        print(
            f"[5] Pivot (wide):  {sp_wide.shape[0]:,} rows * {sp_wide.shape[1]:,} columns  "
            f"({len(feature_cols)} features * {sp_wide.shape[1] // len(feature_cols)} points)"
        )
        print(f"    NaN after pivot:  {sp_wide.isnull().sum().sum():,}")
        

    return sp_wide, y_surge


# ---------------------------------------------------------------------------
# PAW: feature matrix construction
# ---------------------------------------------------------------------------


def build_paw_features(
    paw_pressures: pd.DataFrame,
    target_lat: float = 45.75,
    target_lon: float = 12.25,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and pivot ``paw_pressures`` into a wide feature matrix.
    Steps
    -----
    1. Keep only rows where ``msl_lag6d`` is not NaN (drops timestamps that
       lack a complete 6-day lag window).
    2. Drop any remaining NaN rows.
    3. Extract the target variable (paw_surge) at the Venice grid point.
    4. Pivot all pressure features to wide format.

    Parameters
    ----------
    paw_pressures:
        Raw DataFrame loaded from ``data/paw_pressures.parquet``.
        Expected MultiIndex: (latitude, longitude, time).
    target_lat, target_lon:
        Grid point used to extract the target.
    verbose:
        Print step-by-step row counts when True.

    Returns
    -------
    X_wide:
        Feature matrix indexed by ``time``.
    y_paw:
        Target series indexed by ``time``.
    """
    
    if verbose:
        print("=== paw_pressures — Preprocessing ===\n")
        n_times = paw_pressures.index.get_level_values("time").nunique()
        n_pts = paw_pressures.index.droplevel("time").nunique()
        print(f"[0] Original:          {len(paw_pressures):>12,} rows")
        print(f"    Timestamps: {n_times}  —  Grid points: {n_pts}")

    # Step 1 — Keep rows with a complete 6-day lag window
    pp = paw_pressures[paw_pressures["msl_lag6d"].notna()]
    if verbose:
        n_times_1 = pp.index.get_level_values("time").nunique()
        n_pts_1 = pp.index.droplevel("time").nunique()
        print(f"[1] Filter lag6d ≠ NaN:  {len(pp):>12,} rows")
        print(f"    Timestamps: {n_times_1}  —  Grid points: {n_pts_1}")

    # Step 2 — Drop remaining NaN
    pp = pp.dropna()
    if verbose:
        n_times_2 = pp.index.get_level_values("time").nunique()
        n_pts_2 = pp.index.droplevel("time").nunique()
        print(f"[2] Drop residual NaN: {len(pp):>12,} rows")
        print(f"    Timestamps: {n_times_2}  —  Grid points: {n_pts_2}")

    # Step 3 — Remove grid points with an incomplete time series
    n_timestamps = pp.index.get_level_values("time").nunique()
    points_coverage = pp.groupby(level=["latitude", "longitude"]).size()
    complete_points = set(points_coverage[points_coverage == n_timestamps].index.tolist())
    theoretical_points = set(
        (lat, lon)
        for lat in pp.index.get_level_values("latitude").unique()
        for lon in pp.index.get_level_values("longitude").unique()
    )
    excluded_points = theoretical_points - complete_points
    mask_complete = [
        (lat, lon) in complete_points
        for lat, lon in zip(
            pp.index.get_level_values("latitude"),
            pp.index.get_level_values("longitude"),
        )
    ]
    pp = pp[mask_complete].copy()
    pp.index = pp.index.remove_unused_levels()
    if verbose:
        n_lats = pp.index.get_level_values("latitude").nunique()
        n_lons = pp.index.get_level_values("longitude").nunique()
        print(
            f"[3] Complete grid points:  {len(pp):>12,} rows  "
            f"({len(complete_points)} points — excluded {len(excluded_points)}: {excluded_points})"
        )

    # Step 4 — Target at the Venice grid point
    y_paw = pp.xs((target_lat, target_lon), level=["latitude", "longitude"])["paw_surge"]
    y_paw = y_paw.sort_index()
    if verbose:
        print(f"[4] Target y_paw:      {len(y_paw):>12,} time steps")

    # Step 5 — Pivot to wide format
    feature_cols = [c for c in pp.columns if c not in ("valid_time", "paw_surge")]
    pp_wide = pp[feature_cols].unstack(level=["latitude", "longitude"])
    pp_wide.columns = [
        f"{feat}_lat{lat}_lon{lon}" for feat, lat, lon in pp_wide.columns
    ]
    pp_wide = pp_wide.sort_index()
    if verbose:
        print(
            f"[5] Pivot (wide):      {pp_wide.shape[0]:>12,} rows × {pp_wide.shape[1]:,} columns"
        )
        print(f"    NaN after pivot:   {pp_wide.isnull().sum().sum()}")

    return pp_wide, y_paw


# ---------------------------------------------------------------------------
# PAW: dimensionality reduction (PCA)
# ---------------------------------------------------------------------------


def fit_standard_scaler(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: Optional[str] = "models/paw/paw_scaler.pkl",
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a ``StandardScaler`` on training data and transform both splits.

    The scaler is fitted on ``X_train`` only to prevent data leakage.

    Parameters
    ----------
    X_train, X_test:
        Raw (un-scaled) feature matrices.
    save_path:
        File path for persisting the fitted scaler with ``joblib``.
        Pass ``None`` to skip saving.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled  shape: {X_test_scaled.shape}")
    print("Standardisation complete (fit on train only).")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    return X_train_scaled, X_test_scaled, scaler


def explore_pca_variance(
    X_train_scaled: np.ndarray,
    plot_path: str = "plots/paw/pca_variance_explained.png",
    thresholds: Optional[dict[str, float]] = None,
    random_state: int = 42,
) -> dict[str, int]:
    """Fit a full exploratory PCA and plot cumulative explained variance.

    Parameters
    ----------
    X_train_scaled:
        Standardised training data (output of ``fit_standard_scaler``).
    plot_path:
        File path for saving the variance plot.  Parent directory is
        created automatically.
    thresholds:
        Mapping of label → fraction, e.g. ``{'95%': 0.95}``.
        Defaults to ``{'90%': 0.90, '95%': 0.95, '99%': 0.99}``.

    Returns
    -------
    n_comp_per_threshold:
        Dict mapping each threshold label to the minimum number of
        components required to reach that variance fraction.
    """
    if thresholds is None:
        thresholds = {"90%": 0.90, "95%": 0.95, "99%": 0.99}

    n_components_max = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
    print(f"Maximum computable components: {n_components_max}")

    pca_expl = PCA(n_components=n_components_max, random_state=random_state)
    pca_expl.fit(X_train_scaled)
    cum_var = np.cumsum(pca_expl.explained_variance_ratio_)

    n_comp_per_threshold: dict[str, int] = {}
    for label, thr in thresholds.items():
        n = int(np.argmax(cum_var >= thr)) + 1
        n_comp_per_threshold[label] = n
        print(f"  Components for {label} of variance: {n}")

    colors = {"90%": "green", "95%": "orange", "99%": "red"}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, linewidth=1.5, color="steelblue")
    for label, thr in thresholds.items():
        n = n_comp_per_threshold[label]
        color = colors.get(label, "grey")
        ax.axhline(thr, linestyle="--", color=color, alpha=0.7, label=f"{label} ({n} components)")
        ax.axvline(n, linestyle=":", color=color, alpha=0.5)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA — Cumulative explained variance (paw_pressures)")
    ax.legend()
    ax.set_xlim(0, n_components_max)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.show()

    return n_comp_per_threshold


def apply_pca(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    n_components: int,
    save_path: Optional[str] = "models/paw/paw_pca.pkl",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on scaled training data and transform both splits.

    Parameters
    ----------
    X_train_scaled, X_test_scaled:
        Standardised arrays (output of ``fit_standard_scaler``).
    n_components:
        Number of principal components to retain.
    save_path:
        File path for persisting the fitted PCA with ``joblib``.
        Pass ``None`` to skip saving.

    Returns
    -------
    X_train_pca, X_test_pca, pca
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Selected components: {n_components}")
    print(f"X_train_pca shape: {X_train_pca.shape}")
    print(f"X_test_pca  shape: {X_test_pca.shape}")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pca, save_path)
    return X_train_pca, X_test_pca, pca
