"""
Purged Walk-Forward CV for NASDAQ ML strategy.

Fold schedule (expanding train window, 3-year OOS, 22-day embargo):
  Fold 1: train 1986-01 to 1999-12  |  OOS 2000-02 to 2002-12
  Fold 2: train 1986-01 to 2002-12  |  OOS 2003-02 to 2005-12
  ...
  Fold 9: train 1986-01 to 2023-12  |  OOS 2024-02 to 2026-03

After all folds:
  - Prints IC, Hit Rate, RMSE per fold + overall
  - Saves OOS predictions → data/ml_oos_predictions.csv
  - Saves adjusted leverage   → data/ml_adjusted_leverage.csv
  - Saves feature importance  → data/ml_feature_importance.csv
"""
import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ml_model import (
    LGBM_PARAMS,
    prepare_features,
    train_lgbm,
    predict,
    evaluate_fold,
    pred_to_leverage_multiplier,
    apply_ml_to_leverage,
    get_feature_importance,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Walk-Forward configuration
TRAIN_START    = '1986-01-02'
FIRST_OOS_START= '2000-01-01'
OOS_YEARS      = 3
EMBARGO_DAYS   = 22      # business days, matches 21-day target horizon


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def make_folds(dates: pd.DatetimeIndex) -> list:
    """
    Returns list of dicts:
      { 'fold': int, 'train_end': Timestamp, 'oos_start': Timestamp, 'oos_end': Timestamp }
    OOS windows are non-overlapping 3-year blocks.
    """
    folds = []
    fold_num = 1
    oos_start = pd.Timestamp(FIRST_OOS_START)
    train_start = pd.Timestamp(TRAIN_START)
    max_date = dates.max()

    while True:
        oos_end = oos_start + relativedelta(years=OOS_YEARS) - pd.Timedelta(days=1)
        if oos_start >= max_date:
            break

        # train_end = day before embargo before OOS start (in business days)
        oos_start_idx = dates.searchsorted(oos_start)
        embargo_idx = max(0, oos_start_idx - EMBARGO_DAYS)
        train_end = dates[embargo_idx] if embargo_idx < len(dates) else max_date

        if train_end <= train_start:
            oos_start = oos_start + relativedelta(years=OOS_YEARS)
            fold_num += 1
            continue

        folds.append({
            'fold': fold_num,
            'train_start': train_start,
            'train_end': train_end,
            'oos_start': oos_start,
            'oos_end': min(oos_end, max_date),
        })
        oos_start = oos_start + relativedelta(years=OOS_YEARS)
        fold_num += 1

    return folds


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------

def run_walkforward(df: pd.DataFrame, params: dict = None,
                    target_col: str = 'target_ret_21d') -> pd.DataFrame:
    """
    Run full walk-forward evaluation.
    Returns OOS result dataframe with columns:
      date, fold, pred, y_true, raw_leverage
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    X_full, y_full = prepare_features(df, target_col=target_col)
    dates = X_full.index

    folds = make_folds(dates)
    print(f"Walk-Forward: {len(folds)} folds, target={target_col}")
    print(f"{'Fold':>4} {'Train':>11} {'TrainEnd':>10} {'OOS':>20}  "
          f"{'IC':>7} {'Hit%':>6} {'RMSE':>8} {'N':>5}")
    print("-" * 80)

    all_results = []
    all_importances = []

    for fold_info in folds:
        fold   = fold_info['fold']
        t_st   = fold_info['train_start']
        t_end  = fold_info['train_end']
        o_st   = fold_info['oos_start']
        o_end  = fold_info['oos_end']

        train_mask = (dates >= t_st) & (dates <= t_end)
        oos_mask   = (dates >= o_st) & (dates <= o_end)

        if train_mask.sum() < 500 or oos_mask.sum() < 20:
            continue

        X_tr = X_full.loc[train_mask]
        y_tr = y_full.loc[train_mask]
        X_oo = X_full.loc[oos_mask]
        y_oo = y_full.loc[oos_mask]

        # Train
        model = train_lgbm(X_tr, y_tr, params=params)

        # Predict (train + OOS, train needed for z-score normalization)
        pred_tr = predict(model, X_tr)
        pred_oo = predict(model, X_oo)

        # Evaluate
        metrics = evaluate_fold(pred_oo, y_oo.values)
        print(f"{fold:>4}  {t_st.date()!s:>10} {t_end.date()!s:>10} "
              f"  {o_st.date()!s}..{o_end.date()!s}  "
              f"{metrics['ic']:>7.4f} {metrics['hit_rate']*100:>6.1f} "
              f"{metrics['rmse']:>8.4f} {metrics['n']:>5}")

        # Leverage multiplier
        ml_mult = pred_to_leverage_multiplier(pred_oo, pred_tr)

        # Store results
        for i, (date, p, yt, mm) in enumerate(
                zip(X_oo.index, pred_oo, y_oo.values, ml_mult)):
            raw_lev = df.loc[date, 'dh_raw_leverage'] if 'dh_raw_leverage' in df.columns else np.nan
            all_results.append({
                'date': date,
                'fold': fold,
                'pred': p,
                'y_true': yt,
                'ml_mult': mm,
                'raw_leverage': raw_lev,
                'adj_leverage': apply_ml_to_leverage(
                    np.array([raw_lev]), np.array([mm])
                )[0] if not np.isnan(raw_lev) else np.nan,
            })

        # Feature importance (last fold or every fold)
        imp = get_feature_importance(model, list(X_full.columns), top_n=30)
        imp['fold'] = fold
        all_importances.append(imp)

    print("-" * 80)

    result_df = pd.DataFrame(all_results).set_index('date')
    return result_df, all_importances


# ---------------------------------------------------------------------------
# OOS Evaluation Summary
# ---------------------------------------------------------------------------

def evaluate_oos(result_df: pd.DataFrame) -> dict:
    """Print and return fold-level and overall evaluation metrics."""
    overall_ic  = evaluate_fold(result_df['pred'].values, result_df['y_true'].values)
    n_folds     = result_df['fold'].nunique()
    fold_ics    = []

    print("\n=== OOS Evaluation Summary ===")
    for fold, grp in result_df.groupby('fold'):
        m = evaluate_fold(grp['pred'].values, grp['y_true'].values)
        fold_ics.append(m['ic'])

    ic_gt_zero   = sum(1 for ic in fold_ics if ic > 0)
    mean_ic      = np.nanmean(fold_ics)

    print(f"  Overall IC (Spearman) : {overall_ic['ic']:>8.4f}")
    print(f"  Mean fold IC          : {mean_ic:>8.4f}")
    print(f"  Folds with IC > 0     : {ic_gt_zero}/{n_folds}")
    print(f"  Overall Hit Rate      : {overall_ic['hit_rate']*100:>7.1f}%")
    print(f"  Overall RMSE          : {overall_ic['rmse']:>8.4f}")
    print(f"  Total OOS rows        : {overall_ic['n']:>7}")

    # Adoption check
    print("\n=== Adoption Criteria Check ===")
    crit = {
        'mean_ic >= 0.03':     (mean_ic >= 0.03,   f"{mean_ic:.4f}"),
        'folds_ic_pos >= 6/9': (ic_gt_zero >= 6,  f"{ic_gt_zero}/{n_folds}"),
    }
    all_pass = all(v[0] for v in crit.values())
    for name, (passed, val) in crit.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {val}")

    print(f"\n  => Preliminary ML signal: {'ADOPTED' if all_pass else 'REJECTED (IC criteria)'}")

    return {
        'overall_ic':   overall_ic['ic'],
        'mean_fold_ic': mean_ic,
        'fold_ics':     fold_ics,
        'hit_rate':     overall_ic['hit_rate'],
        'rmse':         overall_ic['rmse'],
        'n_oos':        overall_ic['n'],
        'ic_gt_zero':   ic_gt_zero,
        'n_folds':      n_folds,
        'adopted':      all_pass,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(result_df: pd.DataFrame, importances: list):
    # OOS predictions
    pred_path = os.path.join(DATA_DIR, 'ml_oos_predictions.csv')
    result_df.to_csv(pred_path, float_format='%.6f')
    print(f"\nSaved OOS predictions: {pred_path} ({len(result_df):,} rows)")

    # Adjusted leverage (only where raw_leverage exists)
    lev_df = result_df[['raw_leverage', 'ml_mult', 'adj_leverage', 'fold']].dropna()
    lev_path = os.path.join(DATA_DIR, 'ml_adjusted_leverage.csv')
    lev_df.to_csv(lev_path, float_format='%.6f')
    print(f"Saved adjusted leverage: {lev_path} ({len(lev_df):,} rows)")

    # Feature importance (aggregated across folds)
    if importances:
        imp_all = pd.concat(importances, ignore_index=True)
        imp_agg = (imp_all.groupby('feature')['gain']
                   .agg(['mean', 'std', 'count'])
                   .sort_values('mean', ascending=False)
                   .reset_index())
        imp_path = os.path.join(DATA_DIR, 'ml_feature_importance.csv')
        imp_agg.to_csv(imp_path, float_format='%.2f', index=False)
        print(f"Saved feature importance: {imp_path}")
        print("\nTop 20 features by mean gain:")
        print(imp_agg.head(20).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading ml_features.csv ...")
    ml_path = os.path.join(DATA_DIR, 'ml_features.csv')
    df = pd.read_csv(ml_path, parse_dates=['date'], index_col='date').sort_index()
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")

    # Run walk-forward
    result_df, importances = run_walkforward(df)

    # Evaluate
    metrics = evaluate_oos(result_df)

    # Save
    save_outputs(result_df, importances)

    return result_df, metrics


if __name__ == '__main__':
    main()
