"""
S2_VZGated Rolling Window Cross-Validation (2026-05-17)
========================================================
S2確定パラメータ（k_vz=0.30, gate_min=0.50, target_vol=0.80）の
OOS安定性を 5yr IS → 1yr OOS ローリングウィンドウで検証。

比較ベースライン: P2 (k_vz=0, gate_min=0, target_vol=0.80)

設計:
  - L_t・NAVは全期間で一度だけ計算し、ウィンドウごとにスライス
  - vz/raw_a2のウォームアップは全期間で一貫して保持（look-ahead汚染なし）
  - 固定パラメータの評価（グリッドサーチなし）

出力:
  - コンソール: 全ウィンドウ結果 + 集計統計
  - ファイル: S2_ROLLING_CV_2026-05-17.md
"""

import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    TRADING_DAYS, THRESHOLD,
)
from test_portfolio_diversification import prepare_gold_data
from cfd_leverage_backtest import (
    build_nav_strategy, CFD_SPREAD_LOW,
)
from dynamic_leverage_strategies import (
    compute_L_vol_target,
    compute_L_s2_vz_gated,
)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

# S2確定パラメータ (CFD_DYNAMIC_LEVERAGE_GUIDE.md)
S2_PARAMS = dict(target_vol=0.80, k_vz=0.30, gate_min=0.50, l_min=1.0, l_max=7.0)

# P2ベースライン (VIXゲートなし)
P2_TARGET_VOL = 0.80

IS_YEARS     = 5
OOS_YEARS    = 1
WARMUP_START = '1975-01-01'   # vz/raw_a2の252日ウォームアップ後の有効開始点


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calc_window_metrics(nav_dated: pd.Series, start: str, end: str) -> dict:
    """指定期間のCAGR/Sharpe/MaxDDを計算（起点を1.0にリベース）。"""
    t_start = pd.Timestamp(start)
    t_end   = pd.Timestamp(end)
    mask = (nav_dated.index >= t_start) & (nav_dated.index <= t_end)
    if mask.sum() < 20:
        return {'CAGR': np.nan, 'Sharpe': np.nan, 'MaxDD': np.nan, 'n_days': 0}
    ns = nav_dated[mask].copy()
    ns = ns / ns.iloc[0]
    r  = ns.pct_change().fillna(0)
    n  = len(ns)
    yrs = n / TRADING_DAYS
    if yrs <= 0 or ns.iloc[-1] <= 0:
        return {'CAGR': np.nan, 'Sharpe': np.nan, 'MaxDD': np.nan, 'n_days': n}
    cagr   = float(ns.iloc[-1]) ** (1 / yrs) - 1
    std    = float(r.std())
    sharpe = (float(r.mean()) * TRADING_DAYS) / (std * TRADING_DAYS ** 0.5) if std > 0 else np.nan
    maxdd  = float((ns / ns.cummax() - 1).min())
    return {'CAGR': cagr, 'Sharpe': sharpe, 'MaxDD': maxdd, 'n_days': n}


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------

def generate_windows(dates: pd.Series):
    """IS=5yr, OOS=1yr, slide=1yr のウィンドウリストを返す。"""
    warmup    = pd.Timestamp(WARMUP_START)
    data_end  = dates.iloc[-1]
    windows   = []
    yr = int(WARMUP_START[:4])
    while True:
        is_start  = pd.Timestamp(f'{yr}-01-01')
        if is_start < warmup:
            is_start = warmup
        is_end    = pd.Timestamp(f'{yr + IS_YEARS}-01-01') - pd.Timedelta(days=1)
        oos_start = is_end + pd.Timedelta(days=1)
        oos_end   = pd.Timestamp(f'{yr + IS_YEARS + OOS_YEARS}-01-01') - pd.Timedelta(days=1)
        if oos_start > data_end:
            break
        if oos_end > data_end:
            oos_end = data_end
        # OOS最低200日以上
        oos_days = ((dates >= oos_start) & (dates <= oos_end)).sum()
        if oos_days < 200:
            yr += OOS_YEARS
            continue
        windows.append((
            str(is_start.date()), str(is_end.date()),
            str(oos_start.date()), str(oos_end.date()),
        ))
        yr += OOS_YEARS
    return windows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fp(v, d=1):
    return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v*100:+.{d}f}%'

def _ff(v, d=3):
    return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.{d}f}'


def generate_report(df_s2: pd.DataFrame, df_p2: pd.DataFrame, windows) -> str:
    lines = []
    lines.append('# S2_VZGated Rolling Window CV 検証結果')
    lines.append('')
    lines.append('作成日: 2026-05-17')
    lines.append('')
    lines.append('## 検証設定')
    lines.append('')
    lines.append('| 項目 | 値 |')
    lines.append('|---|---|')
    lines.append(f'| IS窓長 | {IS_YEARS}年 |')
    lines.append(f'| OOS窓長 | {OOS_YEARS}年 |')
    lines.append('| スライド幅 | 1年 |')
    lines.append(f'| 総ウィンドウ数 | {len(windows)} |')
    lines.append('| S2パラメータ | target_vol=0.80, k_vz=0.30, gate_min=0.50, l_max=7.0 |')
    lines.append('| P2パラメータ | target_vol=0.80（VIXゲートなし baseline） |')
    lines.append('')
    lines.append('---')
    lines.append('')

    def window_table(df, title):
        lines.append(f'## {title}')
        lines.append('')
        lines.append('| # | OOS期間 | CAGR(IS) | CAGR(OOS) | Sharpe(OOS) | MaxDD(OOS) | IS-OOS Gap |')
        lines.append('|---|---|---|---|---|---|---|')
        for _, r in df.iterrows():
            oos_period = f'{r["OOS_start"]}〜{r["OOS_end"]}'
            lines.append(
                f'| {int(r["Window"])} | {oos_period}'
                f' | {_fp(r["CAGR_IS"])} | {_fp(r["CAGR_OOS"])}'
                f' | {_ff(r["Sharpe_OOS"])} | {_fp(r["MaxDD_OOS"])}'
                f' | {_fp(r["IS_OOS_Gap"])} |'
            )
        lines.append('')

    window_table(df_s2, 'S2_VZGated 全ウィンドウ結果')
    window_table(df_p2, 'P2 (baseline) 全ウィンドウ結果')

    lines.append('## 集計統計（S2 vs P2）')
    lines.append('')
    lines.append('| 指標 | S2 平均 | S2 中央値 | S2 std | S2 min | P2 平均 | P2 中央値 | P2 std | P2 min |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    metrics_cfg = [
        ('CAGR_OOS',   'CAGR(OOS)',    True),
        ('Sharpe_OOS', 'Sharpe(OOS)',  False),
        ('MaxDD_OOS',  'MaxDD(OOS)',   True),
        ('IS_OOS_Gap', 'IS-OOS Gap',   True),
    ]
    for col, label, is_pct in metrics_cfg:
        s2v = df_s2[col].dropna()
        p2v = df_p2[col].dropna()
        fmt = _fp if is_pct else _ff
        lines.append(
            f'| {label}'
            f' | {fmt(s2v.mean())} | {fmt(s2v.median())} | {fmt(s2v.std())} | {fmt(s2v.min())}'
            f' | {fmt(p2v.mean())} | {fmt(p2v.median())} | {fmt(p2v.std())} | {fmt(p2v.min())} |'
        )
    lines.append('')

    lines.append('## S2 vs P2 勝率（各OOSウィンドウ）')
    lines.append('')
    n_windows = len(df_s2)
    for col, label, higher_is_better in [
        ('CAGR_OOS',   'CAGR(OOS)',   True),
        ('Sharpe_OOS', 'Sharpe(OOS)', True),
        ('MaxDD_OOS',  'MaxDD(OOS)',  True),
    ]:
        s2v = df_s2[col].values
        p2v = df_p2[col].values
        wins = int((s2v > p2v).sum()) if higher_is_better else int((s2v < p2v).sum())
        lines.append(f'- **{label}**: S2がP2を上回った窓 = **{wins}/{n_windows}** ({wins/n_windows*100:.0f}%)')
    lines.append('')

    pos_s2 = int((df_s2['CAGR_OOS'].dropna() > 0).sum())
    pos_p2 = int((df_p2['CAGR_OOS'].dropna() > 0).sum())
    n = len(df_s2['CAGR_OOS'].dropna())
    lines.append('## OOS CAGR プラス率')
    lines.append('')
    lines.append(f'- S2: **{pos_s2}/{n}** ({pos_s2/n*100:.0f}%) のウィンドウでCAGR > 0')
    lines.append(f'- P2: **{pos_p2}/{n}** ({pos_p2/n*100:.0f}%) のウィンドウでCAGR > 0')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/s2_rolling_cv.py`*')
    lines.append('*関連正典: [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md)*')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 70)
    print(f'S2_VZGated Rolling Window CV (IS={IS_YEARS}yr, OOS={OOS_YEARS}yr, slide=1yr)')
    print('=' * 70)

    df      = load_data(DATA_PATH)
    close   = df['Close']
    dates   = df['Date']
    returns = close.pct_change().fillna(0)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building bond/gold...')
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, True)
    gold_1x = prepare_gold_data(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)

    print('Building A2 signal (full period)...')
    raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    print('Computing L_t (full period)...')
    L_s2 = compute_L_s2_vz_gated(returns, vz, **S2_PARAMS)
    L_p2 = compute_L_vol_target(returns, target_vol=P2_TARGET_VOL)

    print('Building full NAVs (S2 & P2)...')
    nav_s2_raw = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_s2), CFD_SPREAD_LOW,
    )
    nav_p2_raw = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, 'CFD', np.asarray(L_p2), CFD_SPREAD_LOW,
    )
    date_index = pd.to_datetime(dates.values)
    nav_s2 = pd.Series(nav_s2_raw.values, index=date_index)
    nav_p2 = pd.Series(nav_p2_raw.values, index=date_index)

    print('Generating rolling windows...')
    windows = generate_windows(dates)
    print(f'  Total windows: {len(windows)}')

    rows_s2 = []
    rows_p2 = []

    print(f'\n{"Win":>3} {"OOS期間":^25} {"S2 CAGR":>9} {"S2 Sharpe":>10} | {"P2 CAGR":>9} {"P2 Sharpe":>10}')
    print('-' * 75)

    for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        ms2_is  = calc_window_metrics(nav_s2, is_s, is_e)
        ms2_oos = calc_window_metrics(nav_s2, oos_s, oos_e)
        mp2_is  = calc_window_metrics(nav_p2, is_s, is_e)
        mp2_oos = calc_window_metrics(nav_p2, oos_s, oos_e)

        gap_s2 = ms2_is['CAGR'] - ms2_oos['CAGR']
        gap_p2 = mp2_is['CAGR'] - mp2_oos['CAGR']

        rows_s2.append({
            'Window': i + 1,
            'IS_start': is_s, 'IS_end': is_e,
            'OOS_start': oos_s, 'OOS_end': oos_e,
            'CAGR_IS':   ms2_is['CAGR'],
            'CAGR_OOS':  ms2_oos['CAGR'],
            'Sharpe_OOS': ms2_oos['Sharpe'],
            'MaxDD_OOS':  ms2_oos['MaxDD'],
            'IS_OOS_Gap': gap_s2,
        })
        rows_p2.append({
            'Window': i + 1,
            'IS_start': is_s, 'IS_end': is_e,
            'OOS_start': oos_s, 'OOS_end': oos_e,
            'CAGR_IS':   mp2_is['CAGR'],
            'CAGR_OOS':  mp2_oos['CAGR'],
            'Sharpe_OOS': mp2_oos['Sharpe'],
            'MaxDD_OOS':  mp2_oos['MaxDD'],
            'IS_OOS_Gap': gap_p2,
        })

        s2c = ms2_oos['CAGR']
        s2sh = ms2_oos['Sharpe']
        p2c = mp2_oos['CAGR']
        p2sh = mp2_oos['Sharpe']
        s2c_s  = f'{s2c*100:+.1f}%'  if not np.isnan(s2c)  else '—'
        s2sh_s = f'{s2sh:.3f}'       if not np.isnan(s2sh) else '—'
        p2c_s  = f'{p2c*100:+.1f}%'  if not np.isnan(p2c)  else '—'
        p2sh_s = f'{p2sh:.3f}'       if not np.isnan(p2sh) else '—'
        win_mark = '✓' if (not np.isnan(s2sh) and not np.isnan(p2sh) and s2sh > p2sh) else ' '
        print(f'{i+1:>3} {oos_s}〜{oos_e}  {s2c_s:>9} {s2sh_s:>10} |{win_mark}{p2c_s:>9} {p2sh_s:>10}')

    df_s2 = pd.DataFrame(rows_s2)
    df_p2 = pd.DataFrame(rows_p2)

    print('\n' + '=' * 70)
    print('集計統計')
    print('=' * 70)
    for col, label in [('CAGR_OOS', 'CAGR(OOS)'), ('Sharpe_OOS', 'Sharpe(OOS)'), ('MaxDD_OOS', 'MaxDD(OOS)'), ('IS_OOS_Gap', 'IS-OOS Gap')]:
        s2v = df_s2[col].dropna()
        p2v = df_p2[col].dropna()
        is_pct = col in ('CAGR_OOS', 'MaxDD_OOS', 'IS_OOS_Gap')
        def _s(v): return f'{v*100:+.1f}%' if is_pct else f'{v:.3f}'
        print(f'{label}:')
        print(f'  S2: mean={_s(s2v.mean())}  median={_s(s2v.median())}  std={_s(s2v.std())}  min={_s(s2v.min())}')
        print(f'  P2: mean={_s(p2v.mean())}  median={_s(p2v.median())}  std={_s(p2v.std())}  min={_s(p2v.min())}')

    wins_sharpe = int((df_s2['Sharpe_OOS'].values > df_p2['Sharpe_OOS'].values).sum())
    print(f'\nS2 Sharpe勝率: {wins_sharpe}/{len(df_s2)} ({wins_sharpe/len(df_s2)*100:.0f}%)')
    pos_s2 = int((df_s2['CAGR_OOS'] > 0).sum())
    pos_p2 = int((df_p2['CAGR_OOS'] > 0).sum())
    print(f'CAGR>0率: S2={pos_s2}/{len(df_s2)} ({pos_s2/len(df_s2)*100:.0f}%)  P2={pos_p2}/{len(df_p2)} ({pos_p2/len(df_p2)*100:.0f}%)')

    print('\nGenerating report...')
    md = generate_report(df_s2, df_p2, windows)
    out = os.path.join(BASE, 'S2_ROLLING_CV_2026-05-17.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
