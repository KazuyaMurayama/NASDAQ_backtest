"""
LightGBM model utilities for NASDAQ strategy ML improvement.
Core functions: feature preparation, training, prediction, evaluation.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTINEL = -9999.0      # Fill value for structurally-missing macro data
TARGET_COLS = ['target_ret_21d', 'target_ret_5d', 'target_sharpe21']
DROP_COLS = TARGET_COLS + ['bond_source', 'vix_source']   # non-numeric / leakage

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 200,
    "learning_rate": 0.02,
    "n_estimators": 2000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.01,
    "verbose": -1,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame,
                     target_col: str = 'target_ret_21d',
                     winsorize_pct: float = 0.01) -> tuple:
    """
    Returns (X, y) with:
    - target winsorized at [winsorize_pct, 1-winsorize_pct]
    - NaN in features replaced by SENTINEL (-9999)
    - target/drop columns removed from X
    - rows where target is NaN dropped
    """
    df = df.copy()

    # Build feature matrix
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()

    # Replace NaN with sentinel (LightGBM handles -9999 as structural missing)
    X = X.fillna(SENTINEL)

    # Target
    y_raw = df[target_col].copy()
    lo = y_raw.quantile(winsorize_pct)
    hi = y_raw.quantile(1 - winsorize_pct)
    y = y_raw.clip(lo, hi)

    # Drop rows where target is NaN (first/last 21 rows due to forward shift)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
               params: dict = None,
               valid_frac: float = 0.10) -> lgb.Booster:
    """
    Train LightGBM with early stopping on held-out tail fraction.
    valid_frac: last fraction of training data used as validation set.
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    n = len(X_train)
    split = int(n * (1 - valid_frac))
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    p = params.copy()
    n_est = p.pop('n_estimators', 2000)
    early = p.pop('early_stopping_rounds', 100)

    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early, verbose=False),
            lgb.log_evaluation(period=-1),   # silent
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predict OOS scores."""
    X_filled = X.fillna(SENTINEL)
    return model.predict(X_filled)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_ic(pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman Information Coefficient (IC)."""
    mask = np.isfinite(pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return np.nan
    rho, _ = stats.spearmanr(pred[mask], y_true[mask])
    return float(rho)


def compute_hit_rate(pred: np.ndarray, y_true: np.ndarray) -> float:
    """Sign agreement rate (hit rate)."""
    mask = np.isfinite(pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return np.nan
    return float((np.sign(pred[mask]) == np.sign(y_true[mask])).mean())


def evaluate_fold(pred: np.ndarray, y_true: np.ndarray) -> dict:
    return {
        'ic':       compute_ic(pred, y_true),
        'hit_rate': compute_hit_rate(pred, y_true),
        'rmse':     float(np.sqrt(np.nanmean((pred - y_true) ** 2))),
        'n':        int(np.isfinite(pred).sum()),
    }


# ---------------------------------------------------------------------------
# Leverage adjustment (prediction → position size)
# ---------------------------------------------------------------------------

def pred_to_leverage_multiplier(pred_oos: np.ndarray,
                                pred_train: np.ndarray) -> np.ndarray:
    """
    Convert raw prediction to leverage multiplier in [0.5, 1.5].
    Uses sigmoid with z-score normalization against training predictions.
    """
    mu    = np.nanmean(pred_train)
    sigma = np.nanstd(pred_train)
    if sigma < 1e-10:
        return np.ones(len(pred_oos))
    z = (pred_oos - mu) / sigma
    # Sigmoid: 0.5 + 1/(1 + exp(-z))  → range (0.5, 1.5)
    mult = 0.5 + 1.0 / (1.0 + np.exp(-z))
    return mult


def apply_ml_to_leverage(raw_leverage: np.ndarray,
                         ml_mult: np.ndarray,
                         lo: float = 0.0,
                         hi: float = 1.2) -> np.ndarray:
    """
    Multiply existing raw_leverage by ML multiplier, then clip to [lo, hi].
    """
    return np.clip(raw_leverage * ml_mult, lo, hi)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model: lgb.Booster,
                           feature_names: list,
                           top_n: int = 20) -> pd.DataFrame:
    imp = model.feature_importance(importance_type='gain')
    df = pd.DataFrame({'feature': feature_names, 'gain': imp})
    return df.sort_values('gain', ascending=False).head(top_n).reset_index(drop=True)
