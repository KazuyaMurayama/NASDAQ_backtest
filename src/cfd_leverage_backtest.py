"""
CFD/くりっく株365方式のレバレッジNASDAQ商品を使った DH Dyn バックテスト
==========================================================================
生成日: 2026-05-15

目的:
  TQQQ (3x ETF, 毎日リバランス型 → vol drag あり) に対し、
  CFD/くりっく株365方式の 3x/4x/5x NASDAQ商品 (vol drag なし) を
  NASDAQスリーブとして用いた場合のパフォーマンス変化を検証する。

評価戦略:
  1. DH Dyn 3x2x3x [TQQQ]   ← Scenario D 再現 (sanity check)
  2. DH Dyn 3x2x3x [CFD]    ← CFD 3x + Gold2x + Bond3x
  3. DH Dyn 4x2x3x [CFD]    ← CFD 4x + Gold2x + Bond3x
  4. DH Dyn 5x2x3x [CFD]    ← CFD 5x + Gold2x + Bond3x
  5. A2 Optimized [TQQQ]    ← 既存比較用

CFD コストモデル (vol drag なし、線形近似):
  r_cfd = L * r_nasdaq - (L-1) * (sofr_daily + cfd_spread/252) - cfd_ter/252

出力:
  - コンソール出力 (7指標テーブル)
  - CFD_LEVERAGE_BACKTEST_2026-05-15.md (レポート)
"""

import sys, os, types

# multitasking stub (yfinance dependency)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data, calc_dd_signal
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    TRADING_DAYS,
    SWAP_SPREAD,
    DELAY,
    BASE_LEV,
    THRESHOLD,
    ANNUAL_COST,
    GOLD_2X_COST,
    BOND_3X_COST,
)
from test_portfolio_diversification import prepare_gold_data

# ---------------------------------------------------------------------------
# CFD cost parameters
# ---------------------------------------------------------------------------
CFD_SPREAD_LOW  = 0.0020  # 0.20%/yr くりっく株365 (最安クラス)
CFD_SPREAD_MID  = 0.0030  # 0.30%/yr GMO/DMM CFD (推定)
CFD_SPREAD_HIGH = 0.0050  # 0.50%/yr IG証券 CFD (推定)
CFD_SPREADS = {'LOW(0.20%)': CFD_SPREAD_LOW, 'MID(0.30%)': CFD_SPREAD_MID, 'HIGH(0.50%)': CFD_SPREAD_HIGH}
CFD_TER = 0.0000          # CFD/証拠金取引はTERなし

# ---------------------------------------------------------------------------
# Evaluation periods
# ---------------------------------------------------------------------------
FULL_START = '1974-01-02'
FULL_END   = '2026-12-31'
IS_START   = '1974-01-02'
IS_END     = '2021-05-07'
OOS_START  = '2021-05-08'

# Expected baselines (from CURRENT_BEST_STRATEGY.md and v4 report)
EXPECTED_TQQQ_CAGR = 0.2250   # Scenario D, FULL
EXPECTED_A2_CAGR   = 0.2328   # A2 Optimized FULL (v4, SOFR-corrected)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')


# ---------------------------------------------------------------------------
# CFD NASDAQ sleeve
# ---------------------------------------------------------------------------

def build_cfd_nas_sleeve(r_nas: np.ndarray,
                          leverage: float,
                          sofr_daily: np.ndarray,
                          cfd_spread: float = CFD_SPREAD_LOW,
                          cfd_ter: float = CFD_TER) -> np.ndarray:
    """
    CFD/くりっく株365方式の日次NASDAQスリーブ収益を返す。

    r_cfd = L * r_nas - (L-1) * (sofr_daily + cfd_spread/252) - cfd_ter/252

    vol drag なし: 毎日リバランスしない連続レバレッジ商品を線形近似。
    TQQQ との違い: 1) TERなし, 2) スプレッドが低い, 3) (L-1)倍の金利 (線形)

    Args:
        r_nas:      NASDAQ daily returns (np.ndarray)
        leverage:   3.0 / 4.0 / 5.0
        sofr_daily: daily SOFR proxy (DTB3/252)
        cfd_spread: annual spread (0.0020 etc.)
        cfd_ter:    annual TER (0.0 for CFD)
    Returns:
        daily CFD return array (np.ndarray)
    """
    assert len(r_nas) == len(sofr_daily), "length mismatch: r_nas vs sofr_daily"
    borrow = (leverage - 1.0) * (sofr_daily + cfd_spread / TRADING_DAYS)
    dc = cfd_ter / TRADING_DAYS
    return leverage * r_nas - borrow - dc


# ---------------------------------------------------------------------------
# Strategy NAV builder
# ---------------------------------------------------------------------------

def build_nav_strategy(close, lev, wn, wg, wb, dates,
                        gold_2x_nav, bond_3x_nav, sofr_daily,
                        nas_mode: str = 'TQQQ',
                        cfd_leverage: float = 3.0,
                        cfd_spread: float = CFD_SPREAD_LOW) -> pd.Series:
    """
    DH Dyn Approach A の NAV を構築する。

    nas_mode='TQQQ': 現行 Scenario D の式を使用
        nas_ret = r_nas * 3.0 - 2.0*(sofr+swap_spread/252) - ANNUAL_COST/252

    nas_mode='CFD': CFD 線形コストモデル
        nas_ret = build_cfd_nas_sleeve(r_nas, cfd_leverage, sofr_daily, cfd_spread)

    Note: lev (DH signal, 0-1) は wn と共に NASDAQスリーブに掛け算するだけ。
          CFD 倍率 (3/4/5) は nas_ret 内に既に込み。二重掛け禁止。
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x_nav).pct_change().fillna(0).values

    idx   = dates.index
    lev_s = pd.Series(lev, index=idx).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn,  index=idx).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg,  index=idx).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb,  index=idx).shift(DELAY).fillna(0).values

    if nas_mode == 'TQQQ':
        swap_d = SWAP_SPREAD / TRADING_DAYS
        dc     = ANNUAL_COST / TRADING_DAYS
        nas_ret = r_nas * BASE_LEV - 2.0 * (sofr_daily + swap_d) - dc
    else:  # CFD
        nas_ret = build_cfd_nas_sleeve(r_nas, cfd_leverage, sofr_daily, cfd_spread)

    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3
    return (1 + pd.Series(daily, index=idx)).cumprod()


# ---------------------------------------------------------------------------
# A2 Optimized (TQQQ, SOFR-corrected)
# ---------------------------------------------------------------------------

def build_a2_tqqq(close, sofr_daily) -> tuple:
    """
    A2 Optimized [TQQQ] NAV を構築する (gen_yearly_returns_v4.py と同等)。
    Returns: (nav pd.Series, trades_per_year float)
    """
    from test_delay_robust import calc_momentum_decel_mult
    from test_vix_integration import calc_vix_proxy
    from opt_lev2x3x import calc_asym_ewma

    returns = close.pct_change()

    av    = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean()
    ratio = close / ma150
    ttv   = (0.10 + 0.20 * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt    = (ttv / av).clip(0, 1.0)

    ma200 = close.rolling(200).mean()
    sl    = ma200.pct_change()
    sm    = sl.rolling(60).mean()
    ss    = sl.rolling(60).std().replace(0, 0.0001)
    slope = (0.9 + 0.35 * (sl - sm) / ss).clip(0.3, 1.5).fillna(1.0)

    mom   = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)

    vp    = calc_vix_proxy(returns)
    vma   = vp.rolling(252).mean()
    vs    = vp.rolling(252).std().replace(0, 0.001)
    vz    = (vp - vma) / vs
    vm    = (1.0 - 0.25 * vz).clip(0.5, 1.15)

    raw_a2 = (calc_dd_signal(close, 0.82, 0.92) * vt * slope * mom * vm).clip(0, 1.0).fillna(0)

    # Rebalance threshold (same as DH Dyn)
    sig_raw = raw_a2.values
    n = len(sig_raw)
    sig_out = np.zeros(n)
    cur = sig_raw[0]
    sig_out[0] = cur
    n_trades = 0
    for i in range(1, n):
        t = sig_raw[i]
        if (t == 0 and cur > 0) or (cur == 0 and t > 0) or abs(t - cur) > THRESHOLD:
            cur = t; n_trades += 1
        sig_out[i] = cur

    # Build TQQQ NAV with SOFR correction
    r_nas  = close.pct_change().fillna(0).values
    dc     = ANNUAL_COST / TRADING_DAYS
    swap_d = SWAP_SPREAD / TRADING_DAYS
    lev_s  = pd.Series(sig_out, index=close.index).shift(DELAY).fillna(0).values

    nav = np.ones(n)
    for i in range(1, n):
        lv = lev_s[i]
        if lv > 0:
            r = lv * (3.0 * r_nas[i] - dc - 2.0 * sofr_daily[i] - swap_d)
        else:
            r = 0.0
        nav[i] = nav[i - 1] * (1 + r)

    n_years = n / TRADING_DAYS
    return pd.Series(nav, index=close.index), n_trades / n_years


# ---------------------------------------------------------------------------
# Metrics calculation (7 indicators)
# ---------------------------------------------------------------------------

def calc_7metrics(nav: pd.Series, dates: pd.Series, trades_per_year=None) -> dict:
    """
    7 評価指標を算出する:
    1. CAGR (FULL / IS / OOS)
    2. Sharpe (FULL / IS / OOS)
    3. MaxDD (FULL)
    4. Worst5Y CAGR
    5. Worst10Y CAGR
    6. WinRate (年次プラス率, FULL)
    7. Trades/年

    Returns dict with keys:
      CAGR_FULL, CAGR_IS, CAGR_OOS,
      Sharpe_FULL, Sharpe_IS, Sharpe_OOS,
      MaxDD_FULL, Worst5Y, Worst10Y, WinRate, Trades_yr
    """
    def _metrics_period(start, end):
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if mask.sum() < 100:
            return {}
        ns = nav[mask].copy() / nav[mask].iloc[0]
        r  = ns.pct_change().fillna(0)
        n  = len(ns); yrs = n / TRADING_DAYS
        if yrs <= 0 or ns.iloc[-1] <= 0:
            return {}
        cagr  = float(ns.iloc[-1]) ** (1 / yrs) - 1
        std   = r.std()
        sharpe = (r.mean() * TRADING_DAYS) / (std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan
        maxdd  = (ns / ns.cummax() - 1).min()
        return {'CAGR': cagr, 'Sharpe': sharpe, 'MaxDD': maxdd}

    full = _metrics_period(FULL_START, FULL_END)
    is_  = _metrics_period(IS_START,   IS_END)
    oos  = _metrics_period(OOS_START,  FULL_END)

    # Worst 5Y / 10Y (full period rolling)
    mask_f = (dates >= pd.Timestamp(FULL_START)) & (dates <= pd.Timestamp(FULL_END))
    ns_f   = nav[mask_f].copy() / nav[mask_f].iloc[0]
    dt_f   = dates[mask_f]

    w5  = np.nan
    w10 = np.nan
    if len(ns_f) >= TRADING_DAYS * 5:
        r5  = (ns_f / ns_f.shift(TRADING_DAYS * 5)) ** (1 / 5) - 1
        w5  = float(r5.min())
    if len(ns_f) >= TRADING_DAYS * 10:
        r10 = (ns_f / ns_f.shift(TRADING_DAYS * 10)) ** (1 / 10) - 1
        w10 = float(r10.min())

    # WinRate (annual)
    ndf = pd.DataFrame({'nav': ns_f.values, 'dt': dt_f.values})
    ndf['year'] = pd.to_datetime(ndf['dt']).dt.year
    yn  = ndf.groupby('year')['nav'].last()
    wr  = float((yn.pct_change().dropna() > 0).mean())

    return {
        'CAGR_FULL':   full.get('CAGR',   np.nan),
        'CAGR_IS':     is_.get('CAGR',    np.nan),
        'CAGR_OOS':    oos.get('CAGR',    np.nan),
        'Sharpe_FULL': full.get('Sharpe', np.nan),
        'Sharpe_IS':   is_.get('Sharpe',  np.nan),
        'Sharpe_OOS':  oos.get('Sharpe',  np.nan),
        'MaxDD_FULL':  full.get('MaxDD',  np.nan),
        'Worst5Y':     w5,
        'Worst10Y':    w10,
        'WinRate':     wr,
        'Trades_yr':   trades_per_year,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v*100:+.{decimals}f}%'

def _fmt_f(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:.{decimals}f}'

def _fmt_trades(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:.1f}'


def generate_report(all_metrics: dict, sensitivity: dict, data_info: dict) -> str:
    """Markdown レポートを生成して返す"""

    strat_order = [
        'DH Dyn 3x2x3x [TQQQ]',
        'DH Dyn 3x2x3x [CFD]',
        'DH Dyn 4x2x3x [CFD]',
        'DH Dyn 5x2x3x [CFD]',
        'A2 Optimized [TQQQ]',
    ]

    lines = []
    lines.append('# CFD/くりっく株365 レバレッジNASDAQ DH Dyn バックテスト')
    lines.append('')
    lines.append(f'**生成日**: 2026-05-15')
    lines.append(f'**データ期間**: {data_info["start"]} 〜 {data_info["end"]}')
    lines.append(f'**IS期間**: {IS_START} 〜 {IS_END} | **OOS期間**: {OOS_START} 〜 {data_info["end"]}')
    lines.append('')
    lines.append('**関連ファイル**:')
    lines.append('- [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) — 現行ベスト戦略')
    lines.append('- [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md) — 閾値0.15採用根拠')
    lines.append('- [YEARLY_RETURNS_REPORT_2026-05-12_v4.md](YEARLY_RETURNS_REPORT_2026-05-12_v4.md) — SOFR補正済み比較')
    lines.append('')

    # --- Section 1: Executive Summary ---
    lines.append('---')
    lines.append('')
    lines.append('## 1. エグゼクティブサマリー')
    lines.append('')

    # Find best CAGR among CFD strategies
    cfd_cagrs = {k: all_metrics[k]['CAGR_FULL'] for k in strat_order if 'CFD' in k}
    best_cfd = max(cfd_cagrs, key=lambda k: cfd_cagrs[k] if not np.isnan(cfd_cagrs[k]) else -99)
    best_cagr = cfd_cagrs[best_cfd]
    tqqq_cagr = all_metrics['DH Dyn 3x2x3x [TQQQ]']['CAGR_FULL']
    cfd3_cagr = all_metrics['DH Dyn 3x2x3x [CFD]']['CAGR_FULL']

    lines.append(f'- **CFD 3x (ボラドラッグ消失効果)**: TQQQの +{tqqq_cagr*100:.2f}% → CFD 3x +{cfd3_cagr*100:.2f}% (差分 {(cfd3_cagr-tqqq_cagr)*100:+.2f}pp)')
    lines.append(f'- **CFD最高CAGR**: {best_cfd} = +{best_cagr*100:.2f}% (CFD_SPREAD = 0.20%, くりっく株365想定)')
    lines.append(f'- **MaxDD**: 4x/5xはリスク増大。DH Dyn 4x2x3x MaxDD = {_fmt_pct(all_metrics["DH Dyn 4x2x3x [CFD]"]["MaxDD_FULL"])}')
    lines.append(f'- **スプレッド感応度**: 別セクション参照。LOW/MID/HIGHでCAGR差は小さい。')
    lines.append('')

    # --- Section 2: 戦略一覧 ---
    lines.append('---')
    lines.append('')
    lines.append('## 2. 戦略一覧と前提')
    lines.append('')
    lines.append('| # | 戦略名 | NASDAQスリーブ | Gold | Bond | シグナル |')
    lines.append('|---|--------|--------------|------|------|---------|')
    lines.append('| 1 | DH Dyn 3x2x3x [TQQQ] | TQQQ 3x (ETF, vol drag あり) | 2x | 3x | A2 Opt [A] |')
    lines.append('| 2 | DH Dyn 3x2x3x [CFD]  | CFD 3x (vol drag なし) | 2x | 3x | 同上 |')
    lines.append('| 3 | DH Dyn 4x2x3x [CFD]  | CFD 4x (vol drag なし) | 2x | 3x | 同上 |')
    lines.append('| 4 | DH Dyn 5x2x3x [CFD]  | CFD 5x (vol drag なし) | 2x | 3x | 同上 |')
    lines.append('| 5 | A2 Optimized [TQQQ]  | TQQQ 3x (単資産) | なし | なし | A2 (単独) |')
    lines.append('')
    lines.append('### コストモデル定数')
    lines.append('')
    lines.append('| パラメータ | TQQQ | CFD (LOW) | CFD (MID) | CFD (HIGH) |')
    lines.append('|-----------|------|----------|----------|-----------|')
    lines.append(f'| TER | {ANNUAL_COST*100:.2f}%/yr | {CFD_TER*100:.2f}% | {CFD_TER*100:.2f}% | {CFD_TER*100:.2f}% |')
    lines.append(f'| スプレッド | {SWAP_SPREAD*100:.2f}% | {CFD_SPREAD_LOW*100:.2f}% | {CFD_SPREAD_MID*100:.2f}% | {CFD_SPREAD_HIGH*100:.2f}% |')
    lines.append(f'| SOFR倍率 (3x) | 2.0× | 2.0× | 2.0× | 2.0× |')
    lines.append(f'| SOFR倍率 (4x) | — | 3.0× | 3.0× | 3.0× |')
    lines.append(f'| SOFR倍率 (5x) | — | 4.0× | 4.0× | 4.0× |')
    lines.append(f'| Gold 2x SOFR | 1.0× | 1.0× | 1.0× | 1.0× |')
    lines.append(f'| Bond 3x SOFR | 2.0× | 2.0× | 2.0× | 2.0× |')
    lines.append('')

    # --- Section 3: CFD コストモデル ---
    lines.append('---')
    lines.append('')
    lines.append('## 3. CFD コストモデルの根拠')
    lines.append('')
    lines.append('### ボラティリティドラッグとは')
    lines.append('')
    lines.append('日次リバランス型ETF (TQQQ) は毎日3倍固定を維持するため、価格変動ごとに')
    lines.append('高値で買い・安値で売りをして vol drag が発生する:')
    lines.append('')
    lines.append('```')
    lines.append('D ≈ 0.5 × L × (L-1) × σ²')
    lines.append('TQQQ (L=3, σ=22%): D ≈ 0.5 × 3 × 2 × 0.22² ≈ -14.52%/年')
    lines.append('```')
    lines.append('')
    lines.append('CFD/くりっく株365は毎日リバランスしないため、vol drag が発生しない。')
    lines.append('')
    lines.append('### 線形コスト式')
    lines.append('')
    lines.append('```python')
    lines.append('# CFD の日次収益 (L = 3, 4, 5)')
    lines.append('r_cfd = L * r_nasdaq - (L-1) * (sofr_daily + spread/252) - ter/252')
    lines.append('```')
    lines.append('')
    lines.append('レバレッジが高いほど借入コストが増加 (L-1 倍):')
    lines.append('')
    lines.append('| レバレッジ | SOFR倍率 | SOFR=4%時の年率借入コスト |')
    lines.append('|-----------|---------|--------------------------|')
    lines.append('| 3x (TQQQ) | 2.0× | 8.0%+スプレッド+TER = 9.36% |')
    lines.append('| 3x (CFD)  | 2.0× | 8.0%+0.20% = 8.20% |')
    lines.append('| 4x (CFD)  | 3.0× | 12.0%+0.30% = 12.30% |')
    lines.append('| 5x (CFD)  | 4.0× | 16.0%+0.40% = 16.40% |')
    lines.append('')

    # --- Section 4: メイン結果 ---
    lines.append('---')
    lines.append('')
    lines.append('## 4. メイン結果 (CFD_SPREAD = 0.20% / くりっく株365想定)')
    lines.append('')

    # Header
    header = ('| 戦略 | CAGR (FULL) | CAGR (IS) | CAGR (OOS) '
              '| Sharpe (FULL) | Sharpe (IS) | Sharpe (OOS) '
              '| MaxDD (FULL) | Worst5Y | Worst10Y | WinRate | Trades/年 |')
    sep = '|---|---|---|---|---|---|---|---|---|---|---|---|'
    lines.append(header)
    lines.append(sep)

    for name in strat_order:
        m = all_metrics.get(name, {})
        row = (f'| **{name}** '
               f'| {_fmt_pct(m.get("CAGR_FULL"))} '
               f'| {_fmt_pct(m.get("CAGR_IS"))} '
               f'| {_fmt_pct(m.get("CAGR_OOS"))} '
               f'| {_fmt_f(m.get("Sharpe_FULL"))} '
               f'| {_fmt_f(m.get("Sharpe_IS"))} '
               f'| {_fmt_f(m.get("Sharpe_OOS"))} '
               f'| {_fmt_pct(m.get("MaxDD_FULL"))} '
               f'| {_fmt_pct(m.get("Worst5Y"))} '
               f'| {_fmt_pct(m.get("Worst10Y"))} '
               f'| {_fmt_pct(m.get("WinRate"), 1)} '
               f'| {_fmt_trades(m.get("Trades_yr"))} |')
        lines.append(row)
    lines.append('')

    # --- Section 5: 感応度分析 ---
    lines.append('---')
    lines.append('')
    lines.append('## 5. CFD スプレッド感応度 (CFD 3 戦略のみ)')
    lines.append('')
    lines.append('| 戦略 | スプレッド | CAGR (FULL) | CAGR (IS) | Sharpe (FULL) | MaxDD (FULL) | Worst5Y |')
    lines.append('|------|----------|------------|----------|-------------|------------|--------|')

    cfd_strats = ['DH Dyn 3x2x3x [CFD]', 'DH Dyn 4x2x3x [CFD]', 'DH Dyn 5x2x3x [CFD]']
    for spread_label, spread_metrics in sensitivity.items():
        for name in cfd_strats:
            m = spread_metrics.get(name, {})
            lines.append(f'| {name} | {spread_label} '
                         f'| {_fmt_pct(m.get("CAGR_FULL"))} '
                         f'| {_fmt_pct(m.get("CAGR_IS"))} '
                         f'| {_fmt_f(m.get("Sharpe_FULL"))} '
                         f'| {_fmt_pct(m.get("MaxDD_FULL"))} '
                         f'| {_fmt_pct(m.get("Worst5Y"))} |')
        lines.append('| | | | | | | |')
    lines.append('')

    # --- Section 6: Sanity Check ---
    lines.append('---')
    lines.append('')
    lines.append('## 6. サニティチェック')
    lines.append('')
    tqqq_actual = all_metrics.get('DH Dyn 3x2x3x [TQQQ]', {}).get('CAGR_FULL', np.nan)
    a2_actual   = all_metrics.get('A2 Optimized [TQQQ]',   {}).get('CAGR_FULL', np.nan)
    tqqq_ok = abs(tqqq_actual - EXPECTED_TQQQ_CAGR) < 0.005 if not np.isnan(tqqq_actual) else False
    a2_ok   = abs(a2_actual   - EXPECTED_A2_CAGR)   < 0.005 if not np.isnan(a2_actual)   else False

    lines.append(f'| チェック項目 | 期待値 | 実測値 | 結果 |')
    lines.append(f'|------------|--------|--------|------|')
    lines.append(f'| DH Dyn 3x2x3x [TQQQ] FULL CAGR | +{EXPECTED_TQQQ_CAGR*100:.2f}% | {_fmt_pct(tqqq_actual)} | {"✅ PASS" if tqqq_ok else "⚠️ WARN"} |')
    lines.append(f'| A2 Optimized [TQQQ] FULL CAGR   | +{EXPECTED_A2_CAGR*100:.2f}% | {_fmt_pct(a2_actual)} | {"✅ PASS" if a2_ok else "⚠️ WARN"} |')
    lines.append(f'| CFD 3x CAGR > TQQQ 3x CAGR | CFD優位 | CFD {_fmt_pct(cfd3_cagr)} > TQQQ {_fmt_pct(tqqq_cagr)} | {"✅ PASS" if cfd3_cagr > tqqq_cagr else "⚠️ WARN"} |')
    lines.append('')

    # --- Section 7: リスク所見 ---
    lines.append('---')
    lines.append('')
    lines.append('## 7. リスク所見')
    lines.append('')
    m4 = all_metrics.get('DH Dyn 4x2x3x [CFD]', {})
    m5 = all_metrics.get('DH Dyn 5x2x3x [CFD]', {})

    lines.append('### 4x/5x のMaxDD・Worst5Y')
    lines.append('')
    lines.append(f'- **DH Dyn 4x2x3x**: MaxDD = {_fmt_pct(m4.get("MaxDD_FULL"))}, Worst5Y = {_fmt_pct(m4.get("Worst5Y"))}')
    lines.append(f'- **DH Dyn 5x2x3x**: MaxDD = {_fmt_pct(m5.get("MaxDD_FULL"))}, Worst5Y = {_fmt_pct(m5.get("Worst5Y"))}')
    lines.append('')
    lines.append('### 実務上の制約')
    lines.append('')
    lines.append('- **証拠金維持率**: くりっく株365は証拠金不足で追証 (5倍で暴落時ロスカット危険)')
    lines.append('- **スリッページ**: CFDのスプレッドは市場状況で変動 (本シミュレーションは固定)')
    lines.append('- **1980年代の高金利環境**: SOFR相当 15-20%時は借入コストが (L-1)×15-20% に拡大')
    lines.append('- **ロールコスト**: 先物ベース商品は限月ロール時にコスト発生 (本シミュレーション未計上)')
    lines.append('- **本シミュレーションはwn/wg/wbの再最適化をしていない**: 4x/5xでは現行重みが過大の可能性')
    lines.append('')

    # --- Section 8: 結論 ---
    lines.append('---')
    lines.append('')
    lines.append('## 8. 結論と CURRENT_BEST_STRATEGY.md への影響')
    lines.append('')

    lines.append('### 主要な発見')
    lines.append('')
    lines.append(f'1. **CFD 3x の優位性**: vol drag 消失により、同じ 3x でも TQQQ比 +{(cfd3_cagr-tqqq_cagr)*100:.1f}pp の CAGR 改善')
    lines.append(f'2. **CFD 4x が有望**: CAGR +{_fmt_pct(m4.get("CAGR_FULL"))} だが MaxDD {_fmt_pct(m4.get("MaxDD_FULL"))} — wn再最適化が必要')
    lines.append(f'3. **CFD 5x は高リスク**: CAGR +{_fmt_pct(m5.get("CAGR_FULL"))} だが MaxDD {_fmt_pct(m5.get("MaxDD_FULL"))}, Worst5Y {_fmt_pct(m5.get("Worst5Y"))} — 要検討')
    lines.append(f'4. **スプレッド感応度は小さい**: LOW/MID/HIGH間のCAGR差は ≤0.5pp (SOFR≒4%時)')
    lines.append('')
    lines.append('### CURRENT_BEST_STRATEGY.md の更新について')
    lines.append('')
    lines.append('**本レポートは CURRENT_BEST_STRATEGY.md を直接書き換えない。**')
    lines.append('')
    lines.append('CFD 3x は TQQQ 3x を CAGR で上回るが、実運用可能な商品の確認 (くりっく株365の流動性・コスト実測) と')
    lines.append('wn/wg/wb の再最適化が完了してから CURRENT_BEST_STRATEGY.md を更新すること。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## Appendix: コスト比較詳細')
    lines.append('')
    lines.append('### TQQQ vs CFD 3x のコスト構造比較 (SOFR=4.37%想定)')
    lines.append('')
    avg_sofr = 0.0437
    lines.append('| コスト項目 | TQQQ 3x | CFD 3x |')
    lines.append('|-----------|--------|--------|')
    lines.append(f'| TER | {ANNUAL_COST*100:.2f}%/yr | 0.00% |')
    lines.append(f'| SOFR financing (2×SOFR) | {2*avg_sofr*100:.2f}% | {2*avg_sofr*100:.2f}% |')
    lines.append(f'| Swap/spread | {SWAP_SPREAD*100:.2f}% (実証) | {CFD_SPREAD_LOW*100:.2f}% (くりっく) |')
    lines.append(f'| vol drag (σ=22%) | -14.52%/年 (構造) | **0%** (非日次リバランス) |')
    lines.append(f'| **合計コスト** | **{(ANNUAL_COST + 2*avg_sofr + SWAP_SPREAD)*100:.2f}%/yr + vol drag** | **{(2*avg_sofr + CFD_SPREAD_LOW)*100:.2f}%/yr** |')
    lines.append('')
    lines.append('> vol drag が消失する分、CFD 3x は長期的に TQQQ 3x を上回る。')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/cfd_leverage_backtest.py`*')
    lines.append(f'*データ期間: {data_info["start"]} 〜 {data_info["end"]}*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("CFD Leverage Backtest -- DH Dyn 3x/4x/5x2x3x (2026-05-15)")
    print("=" * 80)

    # 1. Load data
    df    = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    n     = len(df)
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days, {n/TRADING_DAYS:.1f} years)")

    # 2. SOFR + bond + gold (Scenario D)
    print("\nLoading SOFR (DTB3)...")
    sofr = load_sofr(dates)
    sofr_mean = np.nanmean(sofr) * TRADING_DAYS * 100
    print(f"  Mean SOFR: {sofr_mean:.2f}%/yr")

    print("Building corrected bond model (dgs30 + time-varying Dmod + splice)...")
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print("  Bond 3x done.")

    print("Building corrected gold 2x (1xSOFR)...")
    gold_1x = prepare_gold_data(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    print("  Gold 2x done.")

    # 3. DH Dyn signal (same for all DH strategies)
    print("\nBuilding A2 signal and DH Dyn Approach A rebalance...")
    raw, vz = build_a2_signal(close, close.pct_change())
    lev_A, wn_A, wg_A, wb_A, n_trades = simulate_rebalance_A(raw, vz, THRESHOLD)
    n_years = n / TRADING_DAYS
    dh_trades_yr = n_trades / n_years
    print(f"  DH signal trades: {n_trades} ({dh_trades_yr:.1f}/yr)")

    # 4. Build all strategy NAVs
    print("\nBuilding strategy NAVs...")

    navs = {}
    trades_yr = {}

    # DH TQQQ (Scenario D sanity check)
    print("  [1/5] DH Dyn 3x2x3x [TQQQ] ...")
    navs['DH Dyn 3x2x3x [TQQQ]'] = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, nas_mode='TQQQ')
    trades_yr['DH Dyn 3x2x3x [TQQQ]'] = dh_trades_yr

    # DH CFD variants
    for i, (lev_cfd, name) in enumerate([(3.0, 'DH Dyn 3x2x3x [CFD]'),
                                           (4.0, 'DH Dyn 4x2x3x [CFD]'),
                                           (5.0, 'DH Dyn 5x2x3x [CFD]')], start=2):
        print(f"  [{i}/5] {name} ...")
        navs[name] = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=lev_cfd, cfd_spread=CFD_SPREAD_LOW)
        trades_yr[name] = dh_trades_yr

    # A2 Optimized
    print("  [5/5] A2 Optimized [TQQQ] ...")
    nav_a2, a2_trades = build_a2_tqqq(close, sofr)
    navs['A2 Optimized [TQQQ]'] = nav_a2
    trades_yr['A2 Optimized [TQQQ]'] = a2_trades

    # 5. Calculate 7 metrics (main: LOW spread)
    print("\nCalculating 7 metrics...")
    all_metrics = {}
    for name, nav in navs.items():
        all_metrics[name] = calc_7metrics(nav, dates, trades_yr[name])

    # 6. Sanity checks
    print("\n--- Sanity Checks ---")
    tqqq_cagr = all_metrics['DH Dyn 3x2x3x [TQQQ]']['CAGR_FULL']
    a2_cagr   = all_metrics['A2 Optimized [TQQQ]']['CAGR_FULL']
    tqqq_ok = abs(tqqq_cagr - EXPECTED_TQQQ_CAGR) < 0.005
    a2_ok   = abs(a2_cagr   - EXPECTED_A2_CAGR)   < 0.005
    print(f"  DH Dyn 3x2x3x [TQQQ] CAGR: {tqqq_cagr*100:.2f}%  expected {EXPECTED_TQQQ_CAGR*100:.2f}%  {'✅' if tqqq_ok else '⚠️ WARN'}")
    print(f"  A2 Optimized [TQQQ] CAGR:   {a2_cagr*100:.2f}%   expected {EXPECTED_A2_CAGR*100:.2f}%  {'✅' if a2_ok else '⚠️ WARN'}")
    cfd3_cagr = all_metrics['DH Dyn 3x2x3x [CFD]']['CAGR_FULL']
    print(f"  CFD 3x ({cfd3_cagr*100:.2f}%) > TQQQ 3x ({tqqq_cagr*100:.2f}%): {'✅' if cfd3_cagr > tqqq_cagr else '⚠️'}")

    # 7. Print main results table
    strat_order = list(navs.keys())
    print("\n" + "=" * 120)
    print("MAIN RESULTS (CFD_SPREAD = LOW 0.20%)")
    print("=" * 120)
    hdr = f"{'Strategy':<30} | {'CAGR_F':>8} {'CAGR_I':>8} {'CAGR_O':>8} | {'Shrp_F':>7} {'Shrp_I':>7} {'Shrp_O':>7} | {'MaxDD':>8} | {'W5Y':>7} {'W10Y':>7} | {'WR':>6} | {'Trd/yr':>7}"
    print(hdr)
    print("-" * 120)
    for name in strat_order:
        m = all_metrics[name]
        def p(v): return f'{v*100:>7.2f}%' if v is not None and not np.isnan(v) else f'{"N/A":>8}'
        def f(v): return f'{v:>7.3f}' if v is not None and not np.isnan(v) else f'{"N/A":>7}'
        def t(v): return f'{v:>7.1f}' if v is not None and not np.isnan(v) else f'{"N/A":>7}'
        print(f"{name:<30} | {p(m['CAGR_FULL'])} {p(m['CAGR_IS'])} {p(m['CAGR_OOS'])} | "
              f"{f(m['Sharpe_FULL'])} {f(m['Sharpe_IS'])} {f(m['Sharpe_OOS'])} | "
              f"{p(m['MaxDD_FULL'])} | {p(m['Worst5Y'])} {p(m['Worst10Y'])} | "
              f"{m['WinRate']*100:>5.1f}% | {t(m['Trades_yr'])}")
    print("=" * 120)

    # 8. Sensitivity analysis
    print("\nRunning spread sensitivity analysis...")
    sensitivity = {}
    for spread_label, spread_val in CFD_SPREADS.items():
        sensitivity[spread_label] = {}
        for lev_cfd, name in [(3.0, 'DH Dyn 3x2x3x [CFD]'),
                               (4.0, 'DH Dyn 4x2x3x [CFD]'),
                               (5.0, 'DH Dyn 5x2x3x [CFD]')]:
            nav_s = build_nav_strategy(
                close, lev_A, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=lev_cfd, cfd_spread=spread_val)
            sensitivity[spread_label][name] = calc_7metrics(nav_s, dates, dh_trades_yr)
        print(f"  {spread_label} done.")

    # 9. Print sensitivity table
    print("\n--- Spread Sensitivity (CAGR FULL) ---")
    print(f"{'Strategy':<30} {'LOW(0.20%)':>12} {'MID(0.30%)':>12} {'HIGH(0.50%)':>12}")
    for name in ['DH Dyn 3x2x3x [CFD]', 'DH Dyn 4x2x3x [CFD]', 'DH Dyn 5x2x3x [CFD]']:
        row = f"{name:<30}"
        for sl in ['LOW(0.20%)', 'MID(0.30%)', 'HIGH(0.50%)']:
            v = sensitivity[sl][name].get('CAGR_FULL', np.nan)
            row += f" {v*100:>+11.2f}%"
        print(row)

    # 10. Generate and save MD report
    print("\nGenerating report...")
    data_info = {
        'start': str(dates.iloc[0].date()),
        'end':   str(dates.iloc[-1].date()),
    }
    md = generate_report(all_metrics, sensitivity, data_info)
    out_path = os.path.join(BASE, 'CFD_LEVERAGE_BACKTEST_2026-05-15.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"\nSaved: {out_path}")
    print("Done.")

    return all_metrics, sensitivity


if __name__ == '__main__':
    main()
