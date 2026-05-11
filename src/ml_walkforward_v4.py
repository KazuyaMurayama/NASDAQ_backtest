"""
Walk-Forward v4: stride=5 (weekly non-overlapping evaluation).

Changes from v3:
  - EVAL_STRIDE: 21 → 5 (weekly sampling, N≈150/fold vs 36)
  - All other settings identical to v3 (MAE, min_leaf=40, NaN native, strategy target)

IC は stride=5 なので 21d target と部分重複するが、
実効サンプル数は 36 → 150 になり IC 推定が安定。
"""
import os
import sys
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import get_feature_importance

# ---- Configuration -----------------------------------------------------------
TRAIN_WINDOW_YEARS = 10
FIRST_OOS_START    = '2000-01-01'
OOS_YEARS          = 3
EMBARGO_DAYS       = 22
TARGET_COL         = 'target_strat_21d'
EVAL_STRIDE        = 5     # ← KEY CHANGE from v3 (was 21)

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

LGBM_PARAMS_V4 = {
    "objective":         "regression_l1",
    "metric":            "mae",
    "num_leaves":        15,
    "max_depth":         4,
    "min_data_in_leaf":  40,
    "learning_rate":     0.02,
    "n_estimators":      1500,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.5,
    "lambda_l2":         1.0,
    "min_gain_to_split": 0.01,
    "verbose":           -1,
    "random_state":      42,
    "use_missing":       True,
    "zero_as_missing":   False,
}

# ---- Strategy return target (reuse from v3) ----------------------------------

def compute_strategy_target(df_ml: pd.DataFrame) -> pd.DataFrame:
    idx = df_ml.index
    base = pd.read_csv(
        os.path.join(DATA_DIR, 'base_dataset.csv'),
        parse_dates=['date'], index_col='date',
    ).sort_index()
    dh = pd.read_csv(
        os.path.join(BASE_DIR, 'leverage_daily_detail.csv'),
        parse_dates=['date'], index_col='date',
    ).sort_index()

    r_nas_log  = base['nasdaq_ret'].reindex(idx)
    r_gold_log = base['gold_ret'].reindex(idx)
    r_bond     = base['bond_ret'].reindex(idx)
    vix        = base['vix'].reindex(idx)
    raw_lev    = dh['raw_leverage'].reindex(idx, method='ffill')

    r_nas  = np.exp(r_nas_log)  - 1.0
    r_gold = np.exp(r_gold_log) - 1.0

    vix_ma  = vix.rolling(252, min_periods=63).mean()
    vix_std = vix.rolling(252, min_periods=63).std().replace(0, 0.001)
    vz      = ((vix - vix_ma) / vix_std).clip(lower=0)

    wn = (0.55 + 0.25 * raw_lev - 0.10 * vz).clip(0.30, 0.90)
    wg = (1 - wn) * 0.50
    wb = (1 - wn) * 0.50

    lev_s = raw_lev.shift(2)
    wn_s  = wn.shift(2)
    wg_s  = wg.shift(2)
    wb_s  = wb.shift(2)

    dc = 0.0086 / 252
    gc = 0.0050 / 252
    bc = 0.0091 / 252

    r_strat = (
        wn_s * lev_s * (r_nas * 3.0 - dc)
        + wg_s * (r_gold * 2.0 - gc)
        + wb_s * (r_bond * 3.0 - bc)
    )
    target = r_strat.rolling(21).sum().shift(-21)

    df_out = df_ml.copy()
    df_out[TARGET_COL]           = target
    df_out['dh_raw_leverage_ml'] = raw_lev
    return df_out


# ---- Fold generation ---------------------------------------------------------

def make_folds_rolling(dates: pd.DatetimeIndex) -> list:
    folds     = []
    fold_num  = 1
    oos_start = pd.Timestamp(FIRST_OOS_START)
    max_date  = dates.max()
    while True:
        oos_end = oos_start + relativedelta(years=OOS_YEARS) - pd.Timedelta(days=1)
        if oos_start >= max_date:
            break
        oos_idx = dates.searchsorted(oos_start)
        emb_idx = max(0, oos_idx - EMBARGO_DAYS)
        train_end   = dates[emb_idx] if emb_idx < len(dates) else max_date
        train_start = oos_start - relativedelta(years=TRAIN_WINDOW_YEARS)
        if train_end <= train_start or train_end <= dates[0]:
            oos_start += relativedelta(years=OOS_YEARS)
            fold_num  += 1
            continue
        folds.append({
            'fold': fold_num, 'train_start': train_start, 'train_end': train_end,
            'oos_start': oos_start, 'oos_end': min(oos_end, max_date),
        })
        oos_start += relativedelta(years=OOS_YEARS)
        fold_num  += 1
    return folds


# ---- Training ----------------------------------------------------------------

def train_lgbm_v4(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    n = len(X_train); split = int(n * 0.90)
    dtrain = lgb.Dataset(X_train.iloc[:split], label=y_train.iloc[:split])
    dvalid = lgb.Dataset(X_train.iloc[split:], label=y_train.iloc[split:], reference=dtrain)
    p = LGBM_PARAMS_V4.copy()
    n_est = p.pop('n_estimators', 1500)
    return lgb.train(
        p, dtrain, num_boost_round=n_est, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )


# ---- Metrics -----------------------------------------------------------------

def compute_ic(pred, y_true):
    mask = np.isfinite(pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return {'ic': np.nan, 'hit': np.nan, 'n': 0}
    ic, _ = spearmanr(pred[mask], y_true[mask])
    hit = ((pred[mask] > 0) == (y_true[mask] > 0)).mean()
    return {'ic': float(ic), 'hit': float(hit), 'n': int(mask.sum())}


# ---- Walk-forward loop -------------------------------------------------------

def run_walkforward_v4(df: pd.DataFrame) -> tuple:
    feat_cols = [c for c in TOP10_FEATURES if c in df.columns]
    X_full = df[feat_cols]
    y_full = df[TARGET_COL]
    valid  = y_full.notna()
    X_full, y_full = X_full.loc[valid], y_full.loc[valid]
    dates  = X_full.index

    folds = make_folds_rolling(dates)
    print(f"Walk-Forward v4: {len(folds)} folds | target={TARGET_COL} | stride={EVAL_STRIDE}")
    print(f"\n{'Fold':>4} {'TrainWin':>21} {'OOS':>20}  {'IC':>7} {'Hit%':>6} {'N_eval':>7}")
    print("-" * 75)

    all_results = []
    all_imp     = []

    for fi in folds:
        fold  = fi['fold']
        t_st, t_end = fi['train_start'], fi['train_end']
        o_st, o_end = fi['oos_start'],   fi['oos_end']

        tr_mask  = (dates >= t_st) & (dates <= t_end)
        oos_mask = (dates >= o_st) & (dates <= o_end)
        if tr_mask.sum() < 300 or oos_mask.sum() < 20:
            continue

        X_tr, y_tr = X_full.loc[tr_mask], y_full.loc[tr_mask]
        X_oo, y_oo = X_full.loc[oos_mask], y_full.loc[oos_mask]

        model = train_lgbm_v4(X_tr, y_tr)
        pred  = model.predict(X_oo)

        stride_idx = np.arange(0, len(X_oo), EVAL_STRIDE)
        m = compute_ic(pred[stride_idx], y_oo.values[stride_idx])

        print(f"{fold:>4}  {t_st.date()!s} to {t_end.date()!s}"
              f"  {o_st.date()!s}..{o_end.date()!s}  "
              f"{m['ic']:>7.4f} {m['hit']*100:>6.1f} {m['n']:>7}")

        for d, pv, yv in zip(X_oo.index, pred, y_oo.values):
            all_results.append({'date': d, 'fold': fold, 'pred': pv, 'y_true': yv})

        imp = get_feature_importance(model, feat_cols, top_n=len(feat_cols))
        imp['fold'] = fold
        all_imp.append(imp)

    print("-" * 75)
    return pd.DataFrame(all_results).set_index('date'), all_imp


# ---- OOS evaluation ----------------------------------------------------------

def evaluate_v4(result_df: pd.DataFrame) -> dict:
    print("\n=== OOS Evaluation v4 (stride=5) ===")
    fold_ics = []
    for fold, grp in result_df.groupby('fold'):
        idx = np.arange(0, len(grp), EVAL_STRIDE)
        m = compute_ic(grp['pred'].values[idx], grp['y_true'].values[idx])
        fold_ics.append(m['ic'])

    all_idx = np.arange(0, len(result_df), EVAL_STRIDE)
    overall = compute_ic(result_df['pred'].values[all_idx], result_df['y_true'].values[all_idx])
    mean_ic = np.nanmean(fold_ics)
    ic_gt_0 = sum(1 for ic in fold_ics if ic > 0)
    n_folds = len(fold_ics)

    print(f"  Overall IC (stride-{EVAL_STRIDE})  : {overall['ic']:>7.4f}")
    print(f"  Mean fold IC             : {mean_ic:>7.4f}")
    print(f"  Folds with IC > 0        : {ic_gt_0}/{n_folds}")
    print(f"  Overall Hit Rate         : {overall['hit']*100:>6.1f}%")
    print(f"  Fold ICs: {[f'{a:.3f}' for a in fold_ics]}")
    print(f"  Total eval samples       : {overall['n']:,}")

    # Compare to v3 baseline
    print(f"\n  [v3 reference] mean IC=-0.006, folds>0: 4/9 (stride=21, N=36/fold)")
    print(f"  [v4 current  ] mean IC={mean_ic:.3f}, folds>0: {ic_gt_0}/9 (stride=5, N~150/fold)")

    print("\n=== Adoption Criteria (v4) ===")
    pass1 = mean_ic >= 0.03
    pass2 = ic_gt_0 >= 6
    print(f"  [{'PASS' if pass1 else 'FAIL'}] mean_ic >= 0.03 : {mean_ic:.4f}")
    print(f"  [{'PASS' if pass2 else 'FAIL'}] folds IC>0 >= 6/9 : {ic_gt_0}/{n_folds}")
    adopted = pass1 and pass2
    print(f"\n  => ML signal v4: {'ADOPTED' if adopted else 'REJECTED'}")

    return {
        'overall_ic': overall['ic'], 'mean_fold_ic': mean_ic,
        'fold_ics': fold_ics, 'ic_gt_0': ic_gt_0, 'n_folds': n_folds, 'adopted': adopted,
    }


# ---- Save outputs ------------------------------------------------------------

def save_outputs_v4(result_df, importances):
    pred_path = os.path.join(DATA_DIR, 'ml_v4_oos_predictions.csv')
    result_df.to_csv(pred_path, float_format='%.6f')
    print(f"\nSaved: {pred_path} ({len(result_df):,} rows)")
    if importances:
        imp_all = pd.concat(importances, ignore_index=True)
        imp_agg = (imp_all.groupby('feature')['gain']
                   .agg(['mean', 'std', 'count'])
                   .sort_values('mean', ascending=False).reset_index())
        imp_path = os.path.join(DATA_DIR, 'ml_v4_feature_importance.csv')
        imp_agg.to_csv(imp_path, float_format='%.2f', index=False)
        print(f"Saved: {imp_path}")
        print("\nFeature importance (mean gain):")
        print(imp_agg.to_string(index=False))


# ---- Entry point -------------------------------------------------------------

def main():
    print("Loading ml_features.csv ...")
    df = pd.read_csv(
        os.path.join(DATA_DIR, 'ml_features.csv'),
        parse_dates=['date'], index_col='date',
    ).sort_index()
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")

    print("\nBuilding actual strategy return target ...")
    df = compute_strategy_target(df)
    print(f"  target_strat_21d: {df[TARGET_COL].notna().sum():,} non-NaN rows")

    result_df, importances = run_walkforward_v4(df)
    metrics = evaluate_v4(result_df)
    save_outputs_v4(result_df, importances)
    return result_df, metrics


if __name__ == '__main__':
    main()
