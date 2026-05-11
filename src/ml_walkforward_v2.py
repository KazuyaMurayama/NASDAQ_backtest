"""
Walk-Forward v2: Meta-labeling + Rolling 10-year window + Top-10 features.

Changes vs v1:
  - Target: target_meta_21d (binary: DH signal correct or not) → AUC/Accuracy
  - Training window: 10-year rolling (not expanding)
  - Features: Top 10 from v1 importance analysis
  - Model: LightGBM binary classification

Fold schedule (10-year rolling train, 3-year OOS, 22-day embargo):
  Fold 1: train 1990-01 to 1999-12  |  OOS 2000-02 to 2002-12
  Fold 2: train 1993-01 to 2002-12  |  OOS 2003-02 to 2005-12
  ...
  Fold 9: train 2014-01 to 2023-12  |  OOS 2024-02 to 2026-03
"""
import os
import sys
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import SENTINEL, get_feature_importance

# ---- Configuration --------------------------------------------------------
TRAIN_WINDOW_YEARS = 10
FIRST_OOS_START    = '2000-01-01'
OOS_YEARS          = 3
EMBARGO_DAYS       = 22
TARGET_COL         = 'target_meta_21d'

# Top 10 features from v1 importance (macro + correlation + existing DH signal)
TOP10_FEATURES = [
    'nas_skew126',
    'yc_3m10y',
    'credit_spread_z252',
    'dh_mom_decel',
    'corr_nas_bond_63',
    'nfci_z52w',
    'nas_kurt126',
    'dgs10_z252',
    'vix',
    'month_cos',
]

LGBM_PARAMS_V2 = {
    "objective":          "binary",
    "metric":             "auc",
    "num_leaves":         15,
    "max_depth":          4,
    "min_data_in_leaf":   300,
    "learning_rate":      0.02,
    "n_estimators":       1500,
    "feature_fraction":   0.8,
    "bagging_fraction":   0.8,
    "bagging_freq":       5,
    "lambda_l1":          0.5,
    "lambda_l2":          1.0,
    "min_gain_to_split":  0.01,
    "verbose":            -1,
    "random_state":       42,
    "is_unbalance":       True,   # handle class imbalance
}

# ---- Fold generation -------------------------------------------------------

def make_folds_rolling(dates: pd.DatetimeIndex) -> list:
    folds = []
    fold_num = 1
    oos_start = pd.Timestamp(FIRST_OOS_START)
    max_date = dates.max()

    while True:
        oos_end = oos_start + relativedelta(years=OOS_YEARS) - pd.Timedelta(days=1)
        if oos_start >= max_date:
            break

        # train window: [oos_start - 10y - embargo, oos_start - embargo)
        oos_idx = dates.searchsorted(oos_start)
        emb_idx = max(0, oos_idx - EMBARGO_DAYS)
        train_end = dates[emb_idx] if emb_idx < len(dates) else max_date

        train_start = oos_start - relativedelta(years=TRAIN_WINDOW_YEARS)

        if train_end <= train_start or train_end <= dates[0]:
            oos_start += relativedelta(years=OOS_YEARS)
            fold_num += 1
            continue

        folds.append({
            'fold':        fold_num,
            'train_start': train_start,
            'train_end':   train_end,
            'oos_start':   oos_start,
            'oos_end':     min(oos_end, max_date),
        })
        oos_start += relativedelta(years=OOS_YEARS)
        fold_num += 1

    return folds

# ---- Training helpers -------------------------------------------------------

def train_binary_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
                      valid_frac: float = 0.10) -> lgb.Booster:
    n = len(X_train)
    split = int(n * (1 - valid_frac))
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    p = LGBM_PARAMS_V2.copy()
    n_est  = p.pop('n_estimators', 1500)
    early  = p.pop('early_stopping_rounds', 100)

    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model

# ---- Evaluation helpers ----------------------------------------------------

def compute_fold_metrics(prob: np.ndarray, y_true: np.ndarray) -> dict:
    mask  = np.isfinite(prob) & np.isfinite(y_true)
    if mask.sum() < 20:
        return {'auc': np.nan, 'acc': np.nan, 'n': 0}
    p = prob[mask]
    y = y_true[mask].astype(int)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = np.nan
    acc = ((p >= 0.5).astype(int) == y).mean()
    return {'auc': float(auc), 'acc': float(acc), 'n': int(mask.sum())}

# ---- Main walk-forward -------------------------------------------------------

def run_walkforward_v2(df: pd.DataFrame) -> tuple:
    # Prepare features: fill NaN with sentinel, select top-10
    feat_cols = [c for c in TOP10_FEATURES if c in df.columns]
    missing   = [c for c in TOP10_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARN: missing features: {missing}")

    X_full = df[feat_cols].fillna(SENTINEL)
    y_full = df[TARGET_COL]

    # Drop rows where target is NaN
    valid  = y_full.notna()
    X_full = X_full.loc[valid]
    y_full = y_full.loc[valid]
    dates  = X_full.index

    folds = make_folds_rolling(dates)
    print(f"Walk-Forward v2: {len(folds)} folds | target={TARGET_COL} | features={len(feat_cols)}")
    print(f"{'Fold':>4} {'TrainWin':>21} {'OOS':>20}  {'AUC':>7} {'Acc%':>6} {'N':>5}")
    print("-" * 75)

    all_results  = []
    all_imp      = []

    for fi in folds:
        fold  = fi['fold']
        t_st  = fi['train_start']
        t_end = fi['train_end']
        o_st  = fi['oos_start']
        o_end = fi['oos_end']

        tr_mask  = (dates >= t_st)  & (dates <= t_end)
        oos_mask = (dates >= o_st)  & (dates <= o_end)

        if tr_mask.sum() < 300 or oos_mask.sum() < 20:
            continue

        X_tr = X_full.loc[tr_mask]
        y_tr = y_full.loc[tr_mask]
        X_oo = X_full.loc[oos_mask]
        y_oo = y_full.loc[oos_mask]

        model = train_binary_lgbm(X_tr, y_tr)
        prob  = model.predict(X_oo)

        m = compute_fold_metrics(prob, y_oo.values)
        print(f"{fold:>4}  {t_st.date()!s} to {t_end.date()!s}"
              f"  {o_st.date()!s}..{o_end.date()!s}  "
              f"{m['auc']:>7.4f} {m['acc']*100:>6.1f} {m['n']:>5}")

        for d, p_val, y_val in zip(X_oo.index, prob, y_oo.values):
            raw_lev = df.at[d, 'dh_raw_leverage'] if 'dh_raw_leverage' in df.columns else np.nan
            all_results.append({
                'date': d, 'fold': fold,
                'meta_prob': p_val, 'y_true': y_val,
                'raw_leverage': raw_lev,
                # Adjusted leverage: if ML confident (prob>0.6) use full DH, else dampen
                'adj_leverage': float(np.clip(
                    raw_lev * (0.5 + p_val),   # prob=1 → full, prob=0 → 50% cut
                    0.0, 1.2
                )) if not np.isnan(raw_lev) else np.nan,
            })

        imp = get_feature_importance(model, feat_cols, top_n=len(feat_cols))
        imp['fold'] = fold
        all_imp.append(imp)

    print("-" * 75)
    return pd.DataFrame(all_results).set_index('date'), all_imp


# ---- OOS evaluation summary -------------------------------------------------

def evaluate_v2(result_df: pd.DataFrame) -> dict:
    print("\n=== OOS Evaluation v2 (Meta-labeling) ===")
    fold_aucs = []
    for fold, grp in result_df.groupby('fold'):
        m = compute_fold_metrics(grp['meta_prob'].values, grp['y_true'].values)
        fold_aucs.append(m['auc'])

    overall = compute_fold_metrics(
        result_df['meta_prob'].values, result_df['y_true'].values
    )
    mean_auc     = np.nanmean(fold_aucs)
    auc_gt_50    = sum(1 for a in fold_aucs if a > 0.50)
    n_folds      = len(fold_aucs)

    print(f"  Overall AUC         : {overall['auc']:>7.4f}")
    print(f"  Mean fold AUC       : {mean_auc:>7.4f}")
    print(f"  Folds with AUC>0.50 : {auc_gt_50}/{n_folds}")
    print(f"  Overall Accuracy    : {overall['acc']*100:>6.1f}%")
    print(f"  Fold AUCs           : {[f'{a:.3f}' for a in fold_aucs]}")

    print("\n=== Adoption Criteria (v2) ===")
    # AUC>0.55 average AND 6/9 folds > 0.50
    pass1 = mean_auc >= 0.55
    pass2 = auc_gt_50 >= 6
    print(f"  [{'PASS' if pass1 else 'FAIL'}] mean_auc >= 0.55 : {mean_auc:.4f}")
    print(f"  [{'PASS' if pass2 else 'FAIL'}] folds AUC>0.50 >= 6/9 : {auc_gt_50}/{n_folds}")
    adopted = pass1 and pass2
    print(f"\n  => ML signal v2: {'ADOPTED' if adopted else 'REJECTED'}")

    return {
        'overall_auc': overall['auc'],
        'mean_fold_auc': mean_auc,
        'fold_aucs': fold_aucs,
        'auc_gt_50': auc_gt_50,
        'n_folds': n_folds,
        'adopted': adopted,
    }


# ---- Save outputs -----------------------------------------------------------

def save_outputs_v2(result_df: pd.DataFrame, importances: list):
    pred_path = os.path.join(DATA_DIR, 'ml_v2_oos_predictions.csv')
    result_df.to_csv(pred_path, float_format='%.6f')
    print(f"\nSaved: {pred_path} ({len(result_df):,} rows)")

    lev_df = result_df[['raw_leverage', 'meta_prob', 'adj_leverage', 'fold']].dropna()
    lev_path = os.path.join(DATA_DIR, 'ml_v2_adjusted_leverage.csv')
    lev_df.to_csv(lev_path, float_format='%.6f')
    print(f"Saved: {lev_path} ({len(lev_df):,} rows)")

    if importances:
        imp_all = pd.concat(importances, ignore_index=True)
        imp_agg = (imp_all.groupby('feature')['gain']
                   .agg(['mean', 'std', 'count'])
                   .sort_values('mean', ascending=False).reset_index())
        imp_path = os.path.join(DATA_DIR, 'ml_v2_feature_importance.csv')
        imp_agg.to_csv(imp_path, float_format='%.2f', index=False)
        print(f"Saved: {imp_path}")
        print("\nFeature importance (mean gain):")
        print(imp_agg.to_string(index=False))


# ---- Entry point ------------------------------------------------------------

def main():
    print("Loading ml_features.csv ...")
    ml_path = os.path.join(DATA_DIR, 'ml_features.csv')
    df = pd.read_csv(ml_path, parse_dates=['date'], index_col='date').sort_index()
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Re-run generate_ml_features.py.")

    class_rate = df[TARGET_COL].mean()
    print(f"  Meta-label positive rate: {class_rate:.3f} ({class_rate*100:.1f}% DH signal correct)")

    result_df, importances = run_walkforward_v2(df)
    metrics = evaluate_v2(result_df)
    save_outputs_v2(result_df, importances)

    return result_df, metrics


if __name__ == '__main__':
    main()
