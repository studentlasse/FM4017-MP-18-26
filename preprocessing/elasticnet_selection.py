"""ElasticNet based feature selection.

ElasticNet combines the L1 penalty of Lasso with the L2 penalty of Ridge.
The L1 part shrinks the coefficients of uninformative features to exactly
zero, which makes ElasticNet a natural embedded feature selector while the
L2 part keeps groups of correlated features from being dropped arbitrarily.

The regularization strength ``alpha`` and mixing parameter ``rho`` are
selected jointly through expanding-window ``TimeSeriesSplit``
cross-validation. Imputation and standardization are fitted inside each fold,
on that fold's training rows only, so the validation rows of a fold never
influence how that fold is preprocessed. The final coefficients are then
estimated on the whole training window, which keeps the time ordering of the
energy price data intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, enet_path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

DEFAULT_L1_RATIOS = tuple(
    float(round(value, 2)) for value in np.linspace(0.05, 1.0, 20)
)


@dataclass
class ElasticNetSelection:
    """Result of an ElasticNet feature selection run."""

    selected_features: list[str]
    all_features: list[str]
    coefficients: pd.DataFrame
    alpha: float
    l1_ratio: float
    intercept: float
    cv_alpha: float | None = None
    alpha_scale: float = 1.0
    cv_rmse: float | None = None
    cv_rmse_std: float | None = None
    cv_fold_rmse: list[float] | None = None

    @property
    def n_features(self):
        return len(self.all_features)

    @property
    def n_selected(self):
        return len(self.selected_features)


def _to_feature_frame(X, feature_names=None):
    """Return X as a DataFrame with stable column names."""
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        if feature_names is not None and list(feature_names) != list(frame.columns):
            raise ValueError(
                "feature_names must match the columns of the input DataFrame."
            )
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a 2D array or a pandas DataFrame.")

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(array.shape[1])]

        if len(feature_names) != array.shape[1]:
            raise ValueError("feature_names must have one entry per column in X.")

        frame = pd.DataFrame(array, columns=list(feature_names))

    return frame.replace([np.inf, -np.inf], np.nan)


def _default_cv(n_samples, n_splits):
    if n_samples < 10:
        raise ValueError(
            "At least 10 samples are required for time series cross-validation."
        )

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    return TimeSeriesSplit(n_splits=min(n_splits, max(2, n_samples // 10)))


def _alpha_grid(X, y, l1_ratio, n_alphas, eps):
    """Return a descending, log-spaced alpha path for one rho value.

    The largest value is the smallest penalty that drives every coefficient
    to zero; the path then decreases geometrically to ``eps`` times that
    value. Features are standardized first so the path matches the scale the
    ElasticNet is fitted on. The grid only ever sees training rows.
    """
    if n_alphas < 1:
        raise ValueError("n_alphas must be at least 1.")

    if not 0 < eps < 1:
        raise ValueError("eps must lie strictly between 0 and 1.")

    X_centered = np.asarray(X, dtype=float) - np.mean(X, axis=0)
    scale = X_centered.std(axis=0)
    scale[scale == 0] = 1.0
    X_centered = X_centered / scale
    y_centered = np.asarray(y, dtype=float) - np.mean(y)
    Xy = X_centered.T @ y_centered
    n_samples = X_centered.shape[0]

    if l1_ratio == 1.0:
        alpha_max = np.max(np.abs(Xy)) / (n_samples * l1_ratio)
    else:
        alpha_max = np.sqrt(np.sum(Xy**2)) / (n_samples * l1_ratio)

    alpha_max = max(float(alpha_max), np.finfo(float).eps)
    return np.logspace(np.log10(alpha_max), np.log10(alpha_max * eps), n_alphas)


def _fold_mse(
    X_train,
    y_train,
    X_valid,
    y_valid,
    l1_ratio_values,
    alpha_grids,
    max_iter,
    tol,
):
    """Return the validation MSE for every (rho, alpha) pair in one fold.

    Imputation and scaling are fitted on this fold's training rows only, so
    later observations never influence how earlier rows are preprocessed.
    """
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_valid_scaled = scaler.transform(imputer.transform(X_valid))

    # The scaler centres the training columns, so enet_path (which fits no
    # intercept) recovers the same coefficients as ElasticNet. The intercept
    # follows from the target mean.
    train_column_means = X_train_scaled.mean(axis=0)
    target_mean = float(np.mean(y_train))

    mse = np.empty((len(l1_ratio_values), len(alpha_grids[0])))
    for index, (l1_ratio, alphas) in enumerate(zip(l1_ratio_values, alpha_grids)):
        _, coefficients, _ = enet_path(
            X_train_scaled,
            y_train,
            l1_ratio=l1_ratio,
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
        )
        intercepts = target_mean - train_column_means @ coefficients
        predictions = X_valid_scaled @ coefficients + intercepts
        mse[index] = np.mean((predictions - y_valid[:, None]) ** 2, axis=0)

    return mse


def _fit_final_selection(
    X_frame,
    y_array,
    *,
    alpha,
    l1_ratio,
    min_abs_coef,
    max_features,
    max_iter,
    tol,
):
    """Fit the final ElasticNet at a fixed penalty and build the result."""
    final_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(final_imputer.fit_transform(X_frame))
    final_model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
    )
    final_model.fit(X_final, y_array)

    coefficient_values = final_model.coef_
    intercept = float(final_model.intercept_)

    coefficients = pd.DataFrame(
        {
            "feature": X_frame.columns,
            "coefficient": coefficient_values,
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()

    # Rank by importance, then optionally keep only the strongest features.
    coefficients = coefficients.sort_values(
        "abs_coefficient", ascending=False, kind="mergesort"
    ).reset_index(drop=True)

    selected_mask = coefficients["abs_coefficient"] > min_abs_coef

    if max_features is not None:
        selected_mask &= coefficients.index < max_features

    if not selected_mask.any():
        raise ValueError(
            "ElasticNet did not select any features. Lower min_abs_coef, "
            "increase max_features, or provide more informative features."
        )

    coefficients["selected"] = selected_mask

    selected_set = set(coefficients.loc[selected_mask, "feature"])
    selected_features = [
        feature for feature in X_frame.columns if feature in selected_set
    ]

    return ElasticNetSelection(
        selected_features=selected_features,
        all_features=list(X_frame.columns),
        coefficients=coefficients,
        alpha=alpha,
        l1_ratio=l1_ratio,
        intercept=intercept,
    )


def select_features_at_alpha(
    X,
    y,
    *,
    alpha,
    l1_ratio,
    feature_names=None,
    min_abs_coef=1e-8,
    max_features=None,
    max_iter=50_000,
    tol=1e-4,
):
    """Fit ElasticNet at a fixed ``(alpha, l1_ratio)`` and return its features.

    This skips cross-validation, so a single CV run can be reused to derive
    several feature sets (for example an aggressive and a moderate one) by
    scaling the CV-selected alpha.
    """
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    if not 0 < l1_ratio <= 1:
        raise ValueError("l1_ratio must lie in (0, 1].")

    X_frame = _to_feature_frame(X, feature_names)
    y_array = np.asarray(y).ravel()

    if len(X_frame) != len(y_array):
        raise ValueError("X and y must contain the same number of samples.")

    if X_frame.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    if np.isnan(y_array).any():
        raise ValueError("y contains missing values; drop or impute them first.")

    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be a positive integer or None.")

    return _fit_final_selection(
        X_frame,
        y_array,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        min_abs_coef=min_abs_coef,
        max_features=max_features,
        max_iter=max_iter,
        tol=tol,
    )


def select_features_elasticnet(
    X,
    y,
    *,
    feature_names=None,
    l1_ratios=DEFAULT_L1_RATIOS,
    n_alphas=100,
    eps=1e-3,
    cv=None,
    n_splits=5,
    min_abs_coef=1e-8,
    max_features=None,
    alpha_scale=1.0,
    max_iter=50_000,
    tol=1e-4,
    n_jobs=-1,
):
    """Fit an ElasticNetCV model and return the features it keeps.

    Parameters
    ----------
    X : pandas.DataFrame or array-like of shape (n_samples, n_features)
        Candidate features. Infinite values are treated as missing.
    y : array-like of shape (n_samples,)
        Regression target.
    feature_names : sequence of str, optional
        Column names used when ``X`` is not a DataFrame.
    l1_ratios : sequence of float or float, default ``DEFAULT_L1_RATIOS``
        Mix ratios (rho) tried by ``ElasticNetCV``. The default grid spans
        ``0.05`` to ``1.0``. ``1.0`` is pure Lasso; smaller values behave
        more like Ridge.
    n_alphas : int, default 100
        Number of alpha values evaluated along the regularization path.
    eps : float, default 1e-3
        Ratio between the smallest and largest alpha on the logarithmically
        spaced path.
    cv : cross-validation splitter, optional
        Defaults to an expanding-window ``TimeSeriesSplit(n_splits=n_splits)``.
        Any splitter exposing ``split`` is accepted; imputation and scaling
        are fitted separately on each fold's training rows.
    n_splits : int, default 5
        Number of folds used by the default time series splitter.
    min_abs_coef : float, default 1e-8
        Minimum absolute standardized coefficient for a feature to be kept.
    max_features : int, optional
        Keep at most this many features, ranked by absolute coefficient.
    alpha_scale : float, default 1.0
        Multiplier applied to the alpha chosen by cross-validation. ``1.0``
        keeps the CV-optimal, most aggressive selection. Values below ``1.0``
        weaken the penalty and keep more (possibly correlated) features.
    max_iter : int, default 50000
        Maximum number of iterations for the ElasticNet solver.
    tol : float, default 1e-4
        Solver tolerance.
    n_jobs : int, default -1
        Kept for backwards compatibility. The fold loop is sequential because
        the regularization paths are evaluated with warm starts.

    Returns
    -------
    ElasticNetSelection
        Selected feature names, the full coefficient table, and the
        hyperparameters chosen by cross-validation.
    """
    X_frame = _to_feature_frame(X, feature_names)
    y_array = np.asarray(y).ravel()

    if len(X_frame) != len(y_array):
        raise ValueError("X and y must contain the same number of samples.")

    if X_frame.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    if np.isnan(y_array).any():
        raise ValueError("y contains missing values; drop or impute them first.")

    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be a positive integer or None.")

    if alpha_scale <= 0:
        raise ValueError("alpha_scale must be greater than 0.")

    if cv is None:
        cv = _default_cv(len(X_frame), n_splits)

    if np.isscalar(l1_ratios):
        l1_ratio_values = [float(l1_ratios)]
    else:
        l1_ratio_values = [float(value) for value in l1_ratios]

    if not l1_ratio_values:
        raise ValueError("l1_ratios must contain at least one value.")

    folds = list(cv.split(X_frame))
    if len(folds) < 2:
        raise ValueError("Cross-validation must produce at least two folds.")

    X_values = X_frame.to_numpy(dtype=float)

    # The alpha grids only set the candidate range shared by every fold, and
    # are built from the training window (never the test period).
    grid_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_for_grid = grid_imputer.fit_transform(X_frame)
    alpha_grids = [
        _alpha_grid(X_for_grid, y_array, l1_ratio, n_alphas, eps)
        for l1_ratio in l1_ratio_values
    ]

    fold_mse = np.stack(
        [
            _fold_mse(
                X_values[train_index],
                y_array[train_index],
                X_values[valid_index],
                y_array[valid_index],
                l1_ratio_values,
                alpha_grids,
                max_iter=max_iter,
                tol=tol,
            )
            for train_index, valid_index in folds
        ]
    )

    mean_mse = fold_mse.mean(axis=0)
    best_l1_index, best_alpha_index = np.unravel_index(
        np.argmin(mean_mse), mean_mse.shape
    )

    cv_alpha = float(alpha_grids[best_l1_index][best_alpha_index])
    l1_ratio = float(l1_ratio_values[best_l1_index])
    used_alpha = cv_alpha * alpha_scale

    fold_rmse = np.sqrt(fold_mse[:, best_l1_index, best_alpha_index])
    cv_rmse = float(fold_rmse.mean())
    cv_rmse_std = float(fold_rmse.std())

    # Final coefficients: preprocessing fitted on the full training window,
    # then ElasticNet at the selected (alpha, rho).
    selection = _fit_final_selection(
        X_frame,
        y_array,
        alpha=used_alpha,
        l1_ratio=l1_ratio,
        min_abs_coef=min_abs_coef,
        max_features=max_features,
        max_iter=max_iter,
        tol=tol,
    )
    selection.cv_alpha = cv_alpha
    selection.alpha_scale = alpha_scale
    selection.cv_rmse = cv_rmse
    selection.cv_rmse_std = cv_rmse_std
    selection.cv_fold_rmse = fold_rmse.tolist()

    return selection


class ElasticNetFeatureSelector(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer that keeps the features selected by ElasticNet.

    The selector must be fitted on training data only. It can be used on its
    own or as a step of a pipeline::

        selector = ElasticNetFeatureSelector(max_features=20)
        selected = selector.fit_transform(X_train, y_train)

    Missing values are imputed while fitting, but ``transform`` only selects
    columns and leaves their values unchanged. Add an imputer before the
    selector in a pipeline if the downstream estimator cannot handle NaNs.
    ``transform`` accepts both DataFrames and NumPy arrays.
    """

    def __init__(
        self,
        l1_ratios=DEFAULT_L1_RATIOS,
        n_alphas=100,
        eps=1e-3,
        cv=None,
        n_splits=5,
        min_abs_coef=1e-8,
        max_features=None,
        alpha_scale=1.0,
        max_iter=50_000,
        tol=1e-4,
        n_jobs=-1,
    ):
        self.l1_ratios = l1_ratios
        self.n_alphas = n_alphas
        self.eps = eps
        self.cv = cv
        self.n_splits = n_splits
        self.min_abs_coef = min_abs_coef
        self.max_features = max_features
        self.alpha_scale = alpha_scale
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.selection_ = select_features_elasticnet(
            X,
            y,
            l1_ratios=self.l1_ratios,
            n_alphas=self.n_alphas,
            eps=self.eps,
            cv=self.cv,
            n_splits=self.n_splits,
            min_abs_coef=self.min_abs_coef,
            max_features=self.max_features,
            alpha_scale=self.alpha_scale,
            max_iter=self.max_iter,
            tol=self.tol,
            n_jobs=self.n_jobs,
        )

        self.n_features_in_ = self.selection_.n_features
        self.feature_names_in_ = np.asarray(self.selection_.all_features, dtype=object)
        selected = set(self.selection_.selected_features)
        self.support_ = np.asarray(
            [feature in selected for feature in self.selection_.all_features],
            dtype=bool,
        )

        return self

    def transform(self, X):
        check_is_fitted(self, "selection_")

        if isinstance(X, pd.DataFrame):
            missing = [
                feature
                for feature in self.selection_.selected_features
                if feature not in X.columns
            ]
            if missing:
                raise ValueError(f"X is missing selected features: {missing}")

            return X.loc[:, self.selection_.selected_features]

        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a 2D array or a pandas DataFrame.")

        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {array.shape[1]} features, but the selector was "
                f"fitted with {self.n_features_in_}."
            )

        return array[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selection_")

        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if input_features.shape[0] != self.n_features_in_:
                raise ValueError(
                    "input_features must match the number of fitted features."
                )

        return np.asarray(self.selection_.selected_features, dtype=object)
