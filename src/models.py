"""
XGBoost model training utilities for the surge and PAW notebooks.

Functions:
    train_xgb_baseline                -- fit a default XGBRegressor
    run_or_load_hyperparameter_search -- RandomizedSearchCV with TimeSeriesSplit
                                         and optional JSON result caching
    load_or_train_tuned               -- load a saved tuned model or train from
                                         scratch and persist
    save_model                        -- joblib.dump wrapper
    load_model                        -- joblib.load wrapper
"""

from __future__ import annotations
import json
import os
import time
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor


def train_xgb_baseline(
    X_train,
    y_train,
    save_path: Optional[str] = None,
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBRegressor:
    """Fit a default XGBRegressor on training data and optionally persist it.

    Parameters
    ----------
    X_train, y_train:
        Training features and target.
    save_path:
        If given, the trained model is saved with ``joblib``.
    random_state, n_jobs:
        Passed to class:`XGBRegressor`.

    Returns
    -------
    model : XGBRegressor
    """
    
    model = XGBRegressor(random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    print("Training complete")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
    return model


def run_or_load_hyperparameter_search(
    X_train,
    y_train,
    param_dist: dict,
    results_path: Optional[str] = None,
    n_iter: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: int = 1,
) -> tuple[dict, float]:
    """Load cached RandomizedSearchCV results or run a fresh search.

    If ``results_path`` is given and the file already exists the search is
    skipped entirely and the saved best parameters / CV-MAE are returned.
    Otherwise the search runs, and (if ``results_path`` is given) the results
    are written to a JSON file for future reuse.

    Parameters
    ----------
    X_train, y_train:
        Training data.
    param_dist:
        Hyperparameter search space passed to
        class:`~sklearn.model_selection.RandomizedSearchCV`.
    results_path:
        JSON file for caching.  ``None`` means always run and never save.
    n_iter:
        Number of random candidates to evaluate.
    n_splits:
        Number of :class:`~sklearn.model_selection.TimeSeriesSplit` folds.
    random_state, n_jobs, verbose:
        Forwarded to :class:`~sklearn.model_selection.RandomizedSearchCV`.

    Returns
    -------
    best_params : dict
    best_cv_mae : float
    """
    
    if results_path is not None and os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        best_params = results["best_params"]
        best_cv_mae = results["best_cv_mae"]
        print(f"Search results loaded from {results_path}")
        return best_params, best_cv_mae

    tscv = TimeSeriesSplit(n_splits=n_splits)
    base = XGBRegressor(random_state=random_state, n_jobs=n_jobs)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=False,
    )

    print(
        f"Starting RandomizedSearchCV "
        f"({n_iter} iter * {n_splits} folds = {n_iter * n_splits} fits)"
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Search completed in {elapsed:.1f}s")

    best_params = search.best_params_
    best_cv_mae = -search.best_score_

    if results_path is not None:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({"best_params": best_params, "best_cv_mae": best_cv_mae}, f, indent=2)
        print(f"Search results saved to {results_path}")

    return best_params, best_cv_mae


def load_or_train_tuned(
    X_train,
    y_train,
    best_params: dict,
    model_path: Optional[str] = None,
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBRegressor:
    """Load a saved tuned XGBRegressor or train from scratch and persist.

    If ``model_path`` points to an existing file the model is loaded from
    disk.  Otherwise it is trained with ``best_params`` and, when
    ``model_path`` is given, saved for future reuse.

    Parameters
    ----------
    X_train, y_train:
        Training data used when training from scratch.
    best_params:
        Hyperparameters — output of func:`run_or_load_hyperparameter_search`.
    model_path:
        Path for loading / saving.  ``None`` means train without saving.
    random_state, n_jobs:
        Forwarded to class:`XGBRegressor`.

    Returns
    -------
    model : XGBRegressor
    """
    
    if model_path is not None and os.path.exists(model_path):
        model = joblib.load(model_path)
        expected = X_train.shape[1]
        cached   = getattr(model, "n_features_in_", None)
        if cached is not None and cached != expected:
            print(
                f"Cached model has {cached} features but X_train has {expected} — "
                f"retraining (PCA variant changed)."
            )
        else:
            print(f"model_tuned loaded from {model_path}")
            return model

    model = XGBRegressor(**best_params, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)

    if model_path is not None:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(model, model_path)
        print(f"model_tuned trained and saved to {model_path}")

    return model


def select_pca_variant(
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    y_train,
    y_test,
    n_components_90: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, str, XGBRegressor, XGBRegressor]:
    """Compare a PCA-95 and a PCA-90 variant with default XGBoost; return the better one.

    The PCA-90 variant is obtained by keeping only the first
    ``n_components_90`` columns of the full PCA matrix.  Both variants are
    evaluated with default XGBoost parameters on the test set; the one with
    the higher R² is selected for downstream tuning.

    ``n_jobs=1`` is used internally so that results are fully reproducible
    across runs regardless of the execution environment.

    Parameters
    ----------
    X_train_pca, X_test_pca:
        Full PCA-95 feature matrices ``(n_samples, n_components_95)``.
    y_train, y_test:
        Target arrays for the two splits.
    n_components_90:
        Number of leading components for the PCA-90 variant.
    random_state:
        Seed forwarded to :class:`XGBRegressor`.

    Returns
    -------
    X_tune_train, X_tune_test : np.ndarray
        Feature matrices of the selected variant.
    selected : str
        Human-readable name (e.g. ``'PCA-95 (115 components)'``).
    xgb_95, xgb_90 : XGBRegressor
        Trained default models — needed later for baseline-vs-tuned predictions.
    """
    n95 = X_train_pca.shape[1]
    n90 = n_components_90

    X_train_90 = X_train_pca[:, :n90]
    X_test_90  = X_test_pca[:, :n90]

    print(f"PCA-95 shape train/test: {X_train_pca.shape} | {X_test_pca.shape}")
    print(f"PCA-90 shape train/test: {X_train_90.shape} | {X_test_90.shape}")

    xgb_95 = XGBRegressor(random_state=random_state, n_jobs=1)
    xgb_95.fit(X_train_pca, y_train)
    xgb_90 = XGBRegressor(random_state=random_state, n_jobs=1)
    xgb_90.fit(X_train_90, y_train)
    print("Training complete")

    pred_95 = xgb_95.predict(X_test_pca)
    pred_90 = xgb_90.predict(X_test_90)

    def _metrics(y_true, y_pred):
        return {
            "MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²":   r2_score(y_true, y_pred),
        }

    name_95 = f"PCA-95 ({n95} comp.)"
    name_90 = f"PCA-90 ({n90} comp.)"
    df_cmp = pd.DataFrame({name_95: _metrics(y_test, pred_95),
                           name_90: _metrics(y_test, pred_90)}).T
    print(f"\n=== PCA-95 vs PCA-90 comparison (test set, default XGBoost) ===")
    print(df_cmp.to_string(float_format="{:.5f}".format))

    if df_cmp.loc[name_95, "R²"] >= df_cmp.loc[name_90, "R²"]:
        X_tune_train = X_train_pca
        X_tune_test  = X_test_pca
        selected     = f"PCA-95 ({n95} components)"
        r2_sel       = df_cmp.loc[name_95, "R²"]
    else:
        X_tune_train = X_train_90
        X_tune_test  = X_test_90
        selected     = f"PCA-90 ({n90} components)"
        r2_sel       = df_cmp.loc[name_90, "R²"]

    print(f"\nVariant selected for tuning: {selected}")
    print(f"  → Test R²: {r2_sel:.5f} (highest among the two variants)")

    return X_tune_train, X_tune_test, selected, xgb_95, xgb_90


def save_model(model, path: str) -> None:
    """Persist a model to disk using joblib.

    Parameters
    ----------
    model:
        Any fitted estimator.
    path:
        Destination file path.  Parent directory is created automatically.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    """Load a joblib-persisted model.

    Parameters
    ----------
    path:
        File path of the saved model.

    Returns
    -------
    The loaded object.
    """
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
