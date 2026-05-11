"""
Walk-Forward v3: 4 simultaneous fixes.

  ① Target: actual DH Dyn 2x3x [A] strategy return (Gold 2x + Bond 3x included),
             21-day forward; OOS IC evaluated on stride-21 (non-overlapping)
  ② NaN native to LightGBM (SENTINEL -9999 removed; use_missing=True)
  ③ min_data_in_leaf: 200 → 40
  ④ objective: RMSE → MAE (regression_l1)

Fold schedule: rolling 10-year train, 3-year OOS, 22-day embargo (same as v2).
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
EVAL_STRIDE        = 21     # non-overlapping IC evaluation step

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

LGBM_PARAMS_V3 = {
    "objective":         "regression_l1",  # ④ MAE (was RMSE)
    "metric":            "mae",
    "num_leaves":        15,
    "max_depth":         4,
    "min_data_in_leaf":  40,               # ③ was 200
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
    "use_missing":       True,   # ② NaN native (no SENTINEL)
    "zero_as_missing":   False,
}

# ---- Strategy return target (fix ①) -----------------------------------------

def compute_strategy_target(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Compute actual DH Dyn 2x3x [A] daily strategy return, then build
    21-day forward cumulative target.

    Formula (Approach A):
      daily = wn_s * lev_s * (r_nas*3 - dc) + wg_s * (r_gold*2 - gc) + wb_s * (r_bond*3 - bc)
      weights:  wn = clip(0.55 + 0.25*lev - 0.10*max(vz,0), 0.30, 0.90)
                wg = wb = (1 - wn) * 0.50
      lag: lev, wn, wg, wb all shifted +2 days (execution delay)
    """
    idx = df_ml.index

    base = pd.read_csv(
        os.path.join(DATA_DIR, 'base_dataset.csv'),
        parse_dates=['date'], index_col='date',
    ).sort_index()

    dh = pd.read_csv(
        os.path.join(BASE_DIR, 'leverage_daily_detail.csv'),
        parse_dates=['date'], index_col='date',
    ).sort_index()

    # Align to ml_features index
    r_nas_log  = base['nasdaq_ret'].reindex(idx)
    r_gold_log = base['gold_ret'].reindex(idx)
    r_bond     = base['bond_ret'].reindex(idx)   # already simple return
    vix        = base['vix'].reindex(idx)
    raw_lev    = dh['raw_leverage'].reindex(idx, method='ffill')

    # Convert log returns to simple for correct leverage compounding
    r_nas  = np.exp(r_nas_log)  - 1.0
    r_gold = np.exp(r_gold_log) - 1.0

    # VIX z-score for dynamic weight (clip at 0 — only penalise high-vol regimes)
    vix_ma  = vix.rolling(252, min_periods=63).mean()
    vix_std = vix.rolling(252, min_periods=63).std().replace(0, 0.001)
    vz      = ((vix - vix_ma) / vix_std).clip(lower=0)

    # Dynamic portfolio weights (raw_leverage signal, before execution lag)
    wn = (0.55 + 0.25 * raw_lev - 0.10 * vz).clip(0.30, 0.90)
    wg = (1 - wn) * 0.50
    wb = (1 - wn) * 0.50

    # 2-day execution lag (same as original strategy)
    lev_s = raw_lev.shift(2)
    wn_s  = wn.shift(2)
    wg_s  = wg.shift(2)
    wb_s  = wb.shift(2)

    # Daily costs
    dc = 0.0086 / 252   # NASDAQ 3x annual cost
    gc = 0.0050 / 252   # Gold  2x annual cost
    bc = 0.0091 / 252   # Bond  3x annual cost

    # Daily strategy return (Approach A: Gold/Bond NOT multiplied by lev)
    r_strat = (
        wn_s * lev_s * (r_nas * 3.0 - dc)
        + wg_s * (r_gold * 2.0 - gc)
        + wb_s * (r_bond * 3.0 - bc)
    )

    # 21-day FORWARD cumulative strategy return
    # At day t: sum(r_strat[t+1], ..., r_strat[t+21])
    target = r_strat.rolling(21).sum().shift(-21)

    df_out = df_ml.copy()
    df_out[TARGET_COL]       = target
    df_out['r_strat_daily']  = r_strat
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
            'fold':        fold_num,
            'train_start': train_start,
            'train_end':   train_end,
            'oos_start':   oos_start,
            'oos_end':     min(oos_end, max_date),
        })
        oos_start += relativedelta(years=OOS_YEARS)
        fold_num  += 1

    return folds


# ---- Training ----------------------------------------------------------------

def train_lgbm_v3(X_train: pd.DataFrame, y_train: pd.Series,
                   valid_frac: float = 0.10) -> lgb.Booster:
    n     = len(X_train)
    split = int(n * (1 - valid_frac))
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    p = LGBM_PARAMS_V3.copy()
    n_est = p.pop('n_estimators', 1500)

    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


# ---- Metrics -----------------------------------------------------------------

def compute_ic(pred: np.ndarray, y_true: np.ndarray) -> dict:
    mask = np.isfinite(pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return {'ic': np.nan, 'hit': np.nan, 'n': 0}
    p, y = pred[mask], y_true[mask]
    ic, _   = spearmanr(p, y)
    hit     = ((p > 0) == (y > 0)).mean()
    return {'ic': float(ic), 'hit': float(hit), 'n': int(mask.sum())}


# ---- Walk-forward loop -------------------------------------------------------

def run_walkforward_v3(df: pd.DataFrame) -> tuple:
    feat_cols = [c for c in TOP10_FEATURES if c in df.columns]
    missing   = [c for c in TOP10_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARN: missing features: {missing}")

    # ② NaN native: do NOT fillna (let LightGBM handle NaN via use_missing=True)
    X_full = df[feat_cols]
    y_full = df[TARGET_COL]

    valid  = y_full.notna()
    X_full = X_full.loc[valid]
    y_full = y_full.loc[valid]
    dates  = X_full.index

    folds = make_folds_rolling(dates)
    print(f"Walk-Forward v3: {len(folds)} folds | target={TARGET_COL} | features={len(feat_cols)}")
    print(f"  Fixes: ① strat target ② NaN native ③ min_leaf=40 ④ MAE | eval_stride={EVAL_STRIDE}")
    print(f"\n{'Fold':>4} {'TrainWin':>21} {'OOS':>20}  {'IC':>7} {'Hit%':>6} {'N':>5}")
    print("-" * 75)

    all_results = []
    all_imp     = []

    for fi in folds:
        fold  = fi['fold']
        t_st  = fi['train_start']
        t_end = fi['train_end']
        o_st  = fi['oos_start']
        o_end = fi['oos_end']

        tr_mask  = (dates >= t_st) & (dates <= t_end)
        oos_mask = (dates >= o_st) & (dates <= o_end)

        if tr_mask.sum() < 300 or oos_mask.sum() < 20:
            continue

        X_tr = X_full.loc[tr_mask]
        y_tr = y_full.loc[tr_mask]
        X_oo = X_full.loc[oos_mask]
        y_oo = y_full.loc[oos_mask]

        model = train_lgbm_v3(X_tr, y_tr)
        pred  = model.predict(X_oo)

        # ① Non-overlapping IC: every EVAL_STRIDE-th OOS sample
        stride_idx = np.arange(0, len(X_oo), EVAL_STRIDE)
        m = compute_ic(pred[stride_idx], y_oo.values[stride_idx])

        print(f"{fold:>4}  {t_st.date()!s} to {t_end.date()!s}"
              f"  {o_st.date()!s}..{o_end.date()!s}  "
              f"{m['ic']:>7.4f} {m['hit']*100:>6.1f} {m['n']:>5}")

        for d, pv, yv in zip(X_oo.index, pred, y_oo.values):
            raw_lev = df.at[d, 'dh_raw_leverage_ml'] if 'dh_raw_leverage_ml' in df.columns else np.nan
            all_results.append({
                'date': d, 'fold': fold,
                'pred': pv, 'y_true': yv,
                'raw_leverage': raw_lev,
            })

        imp = get_feature_importance(model, feat_cols, top_n=len(feat_cols))
        imp['fold'] = fold
        all_imp.append(imp)

    print("-" * 75)
    return pd.DataFrame(all_results).set_index('date'), all_imp


# ---- OOS evaluation summary --------------------------------------------------

def evaluate_v3(result_df: pd.DataFrame) -> dict:
    print("\n=== OOS Evaluation v3 ===")
    fold_ics = []
    for fold, grp in result_df.groupby('fold'):
        stride_idx = np.arange(0, len(grp), EVAL_STRIDE)
        m = compute_ic(grp['pred'].values[stride_idx], grp['y_true'].values[stride_idx])
        fold_ics.append(m['ic'])

    overall_stride = np.arange(0, len(result_df), EVAL_STRIDE)
    overall = compute_ic(
        result_df['pred'].values[overall_stride],
        result_df['y_true'].values[overall_stride],
    )
    mean_ic   = np.nanmean(fold_ics)
    ic_gt_0   = sum(1 for ic in fold_ics if ic > 0)
    n_folds   = len(fold_ics)

    print(f"  Overall IC (stride-{EVAL_STRIDE}) : {overall['ic']:>7.4f}")
    print(f"  Mean fold IC            : {mean_ic:>7.4f}")
    print(f"  Folds with IC > 0       : {ic_gt_0}/{n_folds}")
    print(f"  Overall Hit Rate        : {overall['hit']*100:>6.1f}%")
    print(f"  Fold ICs: {[f'{a:.3f}' for a in fold_ics]}")

    print("\n=== Adoption Criteria (v3) ===")
    pass1 = mean_ic >= 0.05
    pass2 = ic_gt_0 >= 6
    print(f"  [{'PASS' if pass1 else 'FAIL'}] mean_ic >= 0.05 : {mean_ic:.4f}")
    print(f"  [{'PASS' if pass2 else 'FAIL'}] folds IC>0 >= 6/9 : {ic_gt_0}/{n_folds}")
    adopted = pass1 and pass2
    print(f"\n  => ML signal v3: {'ADOPTED' if adopted else 'REJECTED'}")

    return {
        'overall_ic':   overall['ic'],
        'mean_fold_ic': mean_ic,
        'fold_ics':     fold_ics,
        'ic_gt_0':      ic_gt_0,
        'n_folds':      n_folds,
        'adopted':      adopted,
    }


# ---- Leverage sleeve verification -------------------------------------------

def verify_leverage_sleeves(df: pd.DataFrame):
    """
    Confirm Gold 2x and Bond 3x sleeves provide CAGR uplift vs 1x.
    Also shows raw_leverage distribution to verify DH signal is active.
    """
    idx = df.index
    base = pd.read_csv(
        os.path.join(DATA_DIR, 'base_dataset.csv'),
        parse_dates=['date'], index_col='date',
    ).sort_index()

    r_gold_log = base['gold_ret'].reindex(idx).fillna(0)
    r_bond     = base['bond_ret'].reindex(idx).fillna(0)
    r_gold     = np.exp(r_gold_log) - 1.0

    gc = 0.0050 / 252
    bc = 0.0091 / 252

    gold_1x = (1 + r_gold).cumprod()
    gold_2x = (1 + r_gold * 2.0 - gc).cumprod()
    bond_1x = (1 + r_bond).cumprod()
    bond_3x = (1 + r_bond * 3.0 - bc).cumprod()

    n_years = len(idx) / 252.0
    period_start = idx[0].date()
    period_end   = idx[-1].date()

    print(f"\n=== Leverage Sleeve Verification ===")
    print(f"  Period: {period_start} to {period_end} ({n_years:.1f} yrs)")
    print(f"  {'Sleeve':<12} {'CAGR':>8} {'MaxDD':>8} {'FinalNAV':>10}")
    print("  " + "-" * 42)
    for name, nav in [
        ('Gold 1x',  gold_1x),
        ('Gold 2x',  gold_2x),
        ('Bond 1x',  bond_1x),
        ('Bond 3x',  bond_3x),
    ]:
        cagr  = float(nav.iloc[-1]) ** (1.0 / n_years) - 1.0
        maxdd = (nav / nav.cummax() - 1.0).min()
        print(f"  {name:<12} {cagr:>7.1%} {maxdd:>8.1%} {nav.iloc[-1]:>10.2f}")

    # DH raw_leverage distribution
    raw_lev = df['dh_raw_leverage_ml'].dropna() if 'dh_raw_leverage_ml' in df.columns else pd.Series(dtype=float)
    if len(raw_lev) > 0:
        print(f"\n  DH raw_leverage stats (n={len(raw_lev):,}):")
        pcts = raw_lev.quantile([0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
        print(f"  min={pcts[0]:.3f} p10={pcts[0.10]:.3f} p25={pcts[0.25]:.3f} "
              f"median={pcts[0.50]:.3f} p75={pcts[0.75]:.3f} p90={pcts[0.90]:.3f} max={pcts[1.0]:.3f}")
        frac_invest = (raw_lev > 0.15).mean()
        print(f"  Fraction lev>0.15 (threshold A): {frac_invest:.1%}")

    # Recent 3-year sleeve performance (2023-2026)
    recent = idx[idx >= '2023-01-01']
    if len(recent) > 0:
        r0 = idx.searchsorted(recent[0])
        print(f"\n  Recent period {recent[0].date()} to {recent[-1].date()}:")
        n_r = len(recent) / 252.0
        for name, nav in [
            ('Gold 2x', gold_2x),
            ('Bond 3x', bond_3x),
        ]:
            nav_r = nav.iloc[r0:]
            nav_r = nav_r / nav_r.iloc[0]
            cagr_r = float(nav_r.iloc[-1]) ** (1.0 / n_r) - 1.0
            dd_r   = (nav_r / nav_r.cummax() - 1.0).min()
            print(f"    {name:<10} CAGR={cagr_r:.1%}  MaxDD={dd_r:.1%}")


# ---- Save outputs ------------------------------------------------------------

def save_outputs_v3(result_df: pd.DataFrame, importances: list):
    pred_path = os.path.join(DATA_DIR, 'ml_v3_oos_predictions.csv')
    result_df.to_csv(pred_path, float_format='%.6f')
    print(f"\nSaved: {pred_path} ({len(result_df):,} rows)")

    if importances:
        imp_all = pd.concat(importances, ignore_index=True)
        imp_agg = (imp_all.groupby('feature')['gain']
                   .agg(['mean', 'std', 'count'])
                   .sort_values('mean', ascending=False)
                   .reset_index())
        imp_path = os.path.join(DATA_DIR, 'ml_v3_feature_importance.csv')
        imp_agg.to_csv(imp_path, float_format='%.2f', index=False)
        print(f"Saved: {imp_path}")
        print("\nFeature importance (mean gain):")
        print(imp_agg.to_string(index=False))


# ---- Entry point -------------------------------------------------------------

def main():
    print("Loading ml_features.csv ...")
    ml_path = os.path.join(DATA_DIR, 'ml_features.csv')
    df = pd.read_csv(ml_path, parse_dates=['date'], index_col='date').sort_index()
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")

    print("\nBuilding actual strategy return target (DH Dyn 2x3x [A]) ...")
    df = compute_strategy_target(df)
    valid_target = df[TARGET_COL].notna().sum()
    print(f"  target_strat_21d: {valid_target:,} non-NaN rows")
    print(f"  target mean={df[TARGET_COL].mean():.4f}  std={df[TARGET_COL].std():.4f}")

    result_df, importances = run_walkforward_v3(df)
    metrics = evaluate_v3(result_df)
    verify_leverage_sleeves(df)
    save_outputs_v3(result_df, importances)

    return result_df, metrics


if __name__ == '__main__':
    main()
