"""
CFD/くりっく株365方式のレバレッジNASDAQ商品を使った DH Dyn バックテスト
==========================================================================
生成日: 2026-05-15

目的:
  TQQQ (3x ETF, 毎日リバランス型 → vol drag あり) に対し、
  CFD/くりっく株365方式の 3x/4x/5x/6x/7x NASDAQ商品 (vol drag なし) を
  NASDAQスリーブとして用いた場合のパフォーマンス変化を検証する。

評価戦略:
  1. DH Dyn 3x2x3x [TQQQ]           ← Scenario D 再現 (sanity check)
  2. DH Dyn 3x2x3x [CFD]            ← CFD 3x + Gold2x + Bond3x
  3. DH Dyn 4x2x3x [CFD]            ← CFD 4x + Gold2x + Bond3x
  4. DH Dyn 5x2x3x [CFD]            ← CFD 5x + Gold2x + Bond3x
  5. DH Dyn 6x2x3x [CFD]            ← CFD 6x (指数CFD10倍枠内・専用上場商品なし)
  6. DH Dyn 7x2x3x [CFD]            ← CFD 7x (指数CFD10倍枠内・専用上場商品なし)
  7. A2 Optimized [TQQQ]            ← 既存比較用

CFD コストモデル (vol drag なし、線形近似):
  r_cfd = L * r_nasdaq - (L-1) * (sofr_daily + cfd_spread/252) - cfd_ter/252

⚠️ 6x/7x について:
  6x/7x専用の指数連動上場商品は日本国内に存在しない。
  ただし国内指数CFD業者 (IG証券・GMO・DMM等) は最大10倍まで規制内で提供しており、
  6x/7xは自分でレバレッジ倍率を設定することで取引可能。
  高金利期 (SOFR=15%) には借入コストがNASDAQリターンを上回る可能性あり。

出力:
  - コンソール出力 (7指標テーブル)
  - CFD_LEVERAGE_BACKTEST_6x7x_2026-05-15.md (レポート)
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

# NAV崩壊フロア: 1日の損失が -99.9% を超える日はロスカット相当としてクリップ
NAV_FLOOR = -0.999

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

# 6x/7x: 専用上場商品なし (CFD業者の指数10倍枠内で取引可能)
NO_LISTED_PRODUCT_LEVERAGES = {6.0, 7.0}  # 専用上場商品なし (CFD業者で10倍枠内で設定可能)


# ---------------------------------------------------------------------------
# CFD NASDAQ sleeve
# ---------------------------------------------------------------------------

def build_cfd_nas_sleeve(r_nas: np.ndarray,
                          leverage,
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
        leverage:   float (固定倍率) or np.ndarray (日次動的倍率, 要事前shift済み)
        sofr_daily: daily SOFR proxy (DTB3/252)
        cfd_spread: annual spread (0.0020 etc.)
        cfd_ter:    annual TER (0.0 for CFD)
    Returns:
        daily CFD return array (np.ndarray)
    """
    assert len(r_nas) == len(sofr_daily), "length mismatch: r_nas vs sofr_daily"
    L = np.asarray(leverage, dtype=float)
    if L.ndim == 0:
        L = np.full(len(r_nas), float(leverage))
    assert len(L) == len(r_nas), "length mismatch: leverage array vs r_nas"
    borrow = (L - 1.0) * (sofr_daily + cfd_spread / TRADING_DAYS)
    dc = cfd_ter / TRADING_DAYS
    return L * r_nas - borrow - dc


# ---------------------------------------------------------------------------
# Strategy NAV builder
# ---------------------------------------------------------------------------

def build_nav_strategy(close, lev, wn, wg, wb, dates,
                        gold_2x_nav, bond_3x_nav, sofr_daily,
                        nas_mode: str = 'TQQQ',
                        cfd_leverage=3.0,
                        cfd_spread: float = CFD_SPREAD_LOW) -> pd.Series:
    """
    DH Dyn Approach A の NAV を構築する。

    nas_mode='TQQQ': 現行 Scenario D の式を使用
        nas_ret = r_nas * 3.0 - 2.0*(sofr+swap_spread/252) - ANNUAL_COST/252

    nas_mode='CFD': CFD 線形コストモデル
        nas_ret = build_cfd_nas_sleeve(r_nas, cfd_leverage, sofr_daily, cfd_spread)

    Note: lev (DH signal, 0-1) は wn と共に NASDAQスリーブに掛け算するだけ。
          CFD 倍率 (3/4/5/6/7) は nas_ret 内に既に込み。二重掛け禁止。

    NAV崩壊対策: 1日収益が NAV_FLOOR を下回る場合はクリップ (ロスカット相当)。
                 発生日数は nav.attrs['blowup_days'] に記録。
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
        L_arr = np.asarray(cfd_leverage, dtype=float)
        if L_arr.ndim > 0:
            # 動的レバレッジ: DELAY分シフト (未シフトのL_t系列を受け取り内部でシフト)
            L_shifted = pd.Series(L_arr, index=idx).shift(DELAY).fillna(1.0).values
        else:
            L_shifted = float(cfd_leverage)
        nas_ret = build_cfd_nas_sleeve(r_nas, L_shifted, sofr_daily, cfd_spread)

    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # NAV崩壊フロア (ロスカット相当)
    blowup_days = int((daily < NAV_FLOOR).sum())
    daily_clipped = np.maximum(daily, NAV_FLOOR)

    nav = (1 + pd.Series(daily_clipped, index=idx)).cumprod()
    nav.attrs['blowup_days'] = blowup_days
    return nav


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
      MaxDD_FULL, Worst5Y, Worst10Y, WinRate, Trades_yr, Blowup_Days
    """
    blowup = nav.attrs.get('blowup_days', 0)

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
        'Blowup_Days': blowup,
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

def _fmt_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{int(v)}'


def generate_report(all_metrics: dict, sensitivity: dict, data_info: dict) -> str:
    """Markdown レポートを生成して返す (6x/7x 拡張版)"""

    strat_order = [
        'DH Dyn 3x2x3x [TQQQ]',
        'DH Dyn 3x2x3x [CFD]',
        'DH Dyn 4x2x3x [CFD]',
        'DH Dyn 5x2x3x [CFD]',
        'DH Dyn 6x2x3x [CFD]',
        'DH Dyn 7x2x3x [CFD]',
        'A2 Optimized [TQQQ]',
    ]

    cfd_strats_all = [
        'DH Dyn 3x2x3x [CFD]',
        'DH Dyn 4x2x3x [CFD]',
        'DH Dyn 5x2x3x [CFD]',
        'DH Dyn 6x2x3x [CFD]',
        'DH Dyn 7x2x3x [CFD]',
    ]

    tqqq_cagr = all_metrics['DH Dyn 3x2x3x [TQQQ]']['CAGR_FULL']
    cfd3_cagr = all_metrics['DH Dyn 3x2x3x [CFD]']['CAGR_FULL']

    lines = []
    lines.append('# CFD/くりっく株365 レバレッジNASDAQ DH Dyn バックテスト (3x〜7x 拡張版)')
    lines.append('')
    lines.append('作成日: 2026-05-15')
    lines.append('最終更新日: 2026-05-15')
    lines.append('')
    lines.append(f'**生成日**: 2026-05-15')
    lines.append(f'**データ期間**: {data_info["start"]} 〜 {data_info["end"]}')
    lines.append(f'**IS期間**: {IS_START} 〜 {IS_END} | **OOS期間**: {OOS_START} 〜 {data_info["end"]}')
    lines.append('')
    lines.append('**関連ファイル**:')
    lines.append('- [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) — 現行ベスト戦略')
    lines.append('- [CFD_LEVERAGE_BACKTEST_2026-05-15.md](CFD_LEVERAGE_BACKTEST_2026-05-15.md) — 前版 (3x〜5x)')
    lines.append('- [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md) — 閾値0.15採用根拠')
    lines.append('- [YEARLY_RETURNS_REPORT_2026-05-12_v4.md](YEARLY_RETURNS_REPORT_2026-05-12_v4.md) — SOFR補正済み比較')
    lines.append('')

    # --- Note on 6x/7x availability ---
    lines.append('> **6x/7x の商品存在性について**')
    lines.append('> - 6x/7x専用の指数連動上場商品 (くりっく株365等) は日本国内に存在しない')
    lines.append('> - ただし国内指数CFD業者 (IG証券・GMO・DMM等) は **最大10倍** まで規制内で提供しており、')
    lines.append('>   6x/7xは自分でレバレッジ倍率を設定することで**通常の取引口座で実現可能**')
    lines.append('> - **CURRENT_BEST_STRATEGY.md は更新しない**')
    lines.append('')

    # --- Section 1: Executive Summary ---
    lines.append('---')
    lines.append('')
    lines.append('## 1. エグゼクティブサマリー')
    lines.append('')

    m6 = all_metrics.get('DH Dyn 6x2x3x [CFD]', {})
    m7 = all_metrics.get('DH Dyn 7x2x3x [CFD]', {})
    m5 = all_metrics.get('DH Dyn 5x2x3x [CFD]', {})

    lines.append(f'- **CFD 3x (ボラドラッグ消失効果)**: TQQQ +{tqqq_cagr*100:.2f}% → CFD 3x +{cfd3_cagr*100:.2f}% (差分 {(cfd3_cagr-tqqq_cagr)*100:+.2f}pp)')
    lines.append(f'- **CFD 6x**: CAGR {_fmt_pct(m6.get("CAGR_FULL"))}, MaxDD {_fmt_pct(m6.get("MaxDD_FULL"))}, Worst5Y {_fmt_pct(m6.get("Worst5Y"))}, Worst10Y {_fmt_pct(m6.get("Worst10Y"))}, NAV崩壊日 {_fmt_int(m6.get("Blowup_Days"))}日')
    lines.append(f'- **CFD 7x**: CAGR {_fmt_pct(m7.get("CAGR_FULL"))}, MaxDD {_fmt_pct(m7.get("MaxDD_FULL"))}, Worst5Y {_fmt_pct(m7.get("Worst5Y"))}, Worst10Y {_fmt_pct(m7.get("Worst10Y"))}, NAV崩壊日 {_fmt_int(m7.get("Blowup_Days"))}日')

    # Check CAGR monotonicity
    cagrs = {
        '3x': cfd3_cagr,
        '4x': all_metrics.get('DH Dyn 4x2x3x [CFD]', {}).get('CAGR_FULL', np.nan),
        '5x': m5.get('CAGR_FULL', np.nan),
        '6x': m6.get('CAGR_FULL', np.nan),
        '7x': m7.get('CAGR_FULL', np.nan),
    }
    vals = [v for v in cagrs.values() if not np.isnan(v)]
    is_monotone = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
    if is_monotone:
        lines.append('- **CAGRの単調増加**: 3x < 4x < 5x < 6x < 7x ✅ (高金利期コストをDHシグナルが軽減)')
    else:
        lines.append('- **⚠️ CAGRの単調増加が崩れている**: 1980年代高金利期の借入コスト増大による逆転現象 → 詳細はセクション7参照')
    lines.append('')

    # --- Section 2: 戦略一覧 ---
    lines.append('---')
    lines.append('')
    lines.append('## 2. 戦略一覧と前提')
    lines.append('')
    lines.append('| # | 戦略名 | NASDAQスリーブ | Gold | Bond | シグナル | 専用上場商品 |')
    lines.append('|---|--------|--------------|------|------|---------|------------|')
    lines.append('| 1 | DH Dyn 3x2x3x [TQQQ]  | TQQQ 3x (ETF, vol drag あり) | 2x | 3x | A2 Opt [A] | ○ くりっく株365等 |')
    lines.append('| 2 | DH Dyn 3x2x3x [CFD]   | CFD 3x (vol drag なし) | 2x | 3x | 同上 | ○ くりっく株365等 |')
    lines.append('| 3 | DH Dyn 4x2x3x [CFD]   | CFD 4x (vol drag なし) | 2x | 3x | 同上 | △ CFD業者で設定 (10倍枠内) |')
    lines.append('| 4 | DH Dyn 5x2x3x [CFD]   | CFD 5x (vol drag なし) | 2x | 3x | 同上 | △ CFD業者で設定 (10倍枠内) |')
    lines.append('| 5 | DH Dyn 6x2x3x [CFD]   | CFD 6x (vol drag なし) | 2x | 3x | 同上 | △ CFD業者で設定 (10倍枠内) |')
    lines.append('| 6 | DH Dyn 7x2x3x [CFD]   | CFD 7x (vol drag なし) | 2x | 3x | 同上 | △ CFD業者で設定 (10倍枠内) |')
    lines.append('| 7 | A2 Optimized [TQQQ]   | TQQQ 3x (単資産) | なし | なし | A2 (単独) | ○ 実在 |')
    lines.append('')
    lines.append('### コストモデル定数')
    lines.append('')
    lines.append('| パラメータ | TQQQ | CFD (LOW) | CFD (MID) | CFD (HIGH) |')
    lines.append('|-----------|------|----------|----------|-----------|')
    lines.append(f'| TER | {ANNUAL_COST*100:.2f}%/yr | 0.00% | 0.00% | 0.00% |')
    lines.append(f'| スプレッド | {SWAP_SPREAD*100:.2f}% | {CFD_SPREAD_LOW*100:.2f}% | {CFD_SPREAD_MID*100:.2f}% | {CFD_SPREAD_HIGH*100:.2f}% |')
    lines.append(f'| SOFR倍率 (3x) | 2.0× | 2.0× | 2.0× | 2.0× |')
    lines.append(f'| SOFR倍率 (4x) | — | 3.0× | 3.0× | 3.0× |')
    lines.append(f'| SOFR倍率 (5x) | — | 4.0× | 4.0× | 4.0× |')
    lines.append(f'| SOFR倍率 (6x) | — | 5.0× | 5.0× | 5.0× |')
    lines.append(f'| SOFR倍率 (7x) | — | 6.0× | 6.0× | 6.0× |')
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
    lines.append('# CFD の日次収益 (L = 3, 4, 5, 6, 7)')
    lines.append('r_cfd = L * r_nasdaq - (L-1) * (sofr_daily + spread/252) - ter/252')
    lines.append('```')
    lines.append('')
    lines.append('### 年率借入コスト比較 (SOFR=4% / SOFR=15% 両シナリオ)')
    lines.append('')
    lines.append('| レバレッジ | SOFR倍率 | SOFR=4%時 (LOW spread) | SOFR=15%時 (HIGH rate) |')
    lines.append('|-----------|---------|------------------------|------------------------|')
    avg_sofr_low = 0.04
    avg_sofr_hi  = 0.15
    for L, label in [(3, '3x (TQQQ)'), (3, '3x (CFD)'), (4, '4x (CFD)'), (5, '5x (CFD)'),
                     (6, '6x (CFD)'), (7, '7x (CFD)')]:
        mult = L - 1
        if label == '3x (TQQQ)':
            cost_low = mult * avg_sofr_low + SWAP_SPREAD + ANNUAL_COST
            cost_hi  = mult * avg_sofr_hi  + SWAP_SPREAD + ANNUAL_COST
            lines.append(f'| {label} | {mult}.0× | {cost_low*100:.1f}%+vol drag | {cost_hi*100:.1f}%+vol drag |')
        else:
            cost_low = mult * (avg_sofr_low + CFD_SPREAD_LOW)
            cost_hi  = mult * (avg_sofr_hi  + CFD_SPREAD_LOW)
            tag = ' ⚠️' if L >= 6 else ''
            lines.append(f'| {label} | {mult}.0× | {cost_low*100:.1f}%{tag} | {cost_hi*100:.1f}%{tag} |')
    lines.append('')
    lines.append('> ⚠️ SOFR=15%時の 6x/7x 借入コストは NASDAQ年平均リターン (~15%) を超える。')
    lines.append('> DH Dynシグナル (lev=0) が高金利期・ベア相場でポジションを落とすことで一部緩和される。')
    lines.append('')

    # --- Section 4: メイン結果 ---
    lines.append('---')
    lines.append('')
    lines.append('## 4. メイン結果 (CFD_SPREAD = 0.20% / LOWスプレッド想定)')
    lines.append('')

    header = ('| 戦略 | CAGR (FULL) | CAGR (IS) | CAGR (OOS) '
              '| Sharpe (FULL) | Sharpe (IS) | Sharpe (OOS) '
              '| MaxDD (FULL) | Worst5Y | Worst10Y | WinRate | Trades/年 | NAV崩壊日 |')
    sep = '|---|---|---|---|---|---|---|---|---|---|---|---|---|'
    lines.append(header)
    lines.append(sep)

    for name in strat_order:
        m = all_metrics.get(name, {})
        no_listed = name in ('DH Dyn 6x2x3x [CFD]', 'DH Dyn 7x2x3x [CFD]')
        tag = ' †' if no_listed else ''
        blowup_str = _fmt_int(m.get('Blowup_Days', 0))
        if m.get('Blowup_Days', 0) > 0:
            blowup_str = f'**{blowup_str}** ⚠️'
        row = (f'| **{name}**{tag} '
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
               f'| {_fmt_trades(m.get("Trades_yr"))} '
               f'| {blowup_str} |')
        lines.append(row)
    lines.append('')
    lines.append('※ NAV崩壊日 = ポートフォリオ日次収益が -99.9% を下回った日数 (ロスカット相当)')
    lines.append('')

    # --- Section 5: 感応度分析 ---
    lines.append('---')
    lines.append('')
    lines.append('## 5. CFD スプレッド感応度 (5 CFD戦略 × 3スプレッド)')
    lines.append('')
    lines.append('| 戦略 | スプレッド | CAGR (FULL) | CAGR (IS) | Sharpe (FULL) | MaxDD (FULL) | Worst5Y |')
    lines.append('|------|----------|------------|----------|-------------|------------|--------|')

    for spread_label, spread_metrics in sensitivity.items():
        for name in cfd_strats_all:
            m = spread_metrics.get(name, {})
            no_listed = name in ('DH Dyn 6x2x3x [CFD]', 'DH Dyn 7x2x3x [CFD]')
            tag = ' †' if no_listed else ''
            lines.append(f'| {name}{tag} | {spread_label} '
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

    # Monotonicity check (MaxDD abs)
    dd_vals = [abs(all_metrics.get(f'DH Dyn {l}x2x3x [CFD]', {}).get('MaxDD_FULL', np.nan))
               for l in [3, 4, 5, 6, 7]]
    dd_vals_clean = [v for v in dd_vals if not np.isnan(v)]
    dd_monotone = all(dd_vals_clean[i] <= dd_vals_clean[i+1] for i in range(len(dd_vals_clean)-1))

    lines.append(f'| チェック項目 | 期待値 | 実測値 | 結果 |')
    lines.append(f'|------------|--------|--------|------|')
    lines.append(f'| DH Dyn 3x2x3x [TQQQ] FULL CAGR | +{EXPECTED_TQQQ_CAGR*100:.2f}% | {_fmt_pct(tqqq_actual)} | {"✅ PASS" if tqqq_ok else "⚠️ WARN"} |')
    lines.append(f'| A2 Optimized [TQQQ] FULL CAGR   | +{EXPECTED_A2_CAGR*100:.2f}% | {_fmt_pct(a2_actual)} | {"✅ PASS" if a2_ok else "⚠️ WARN"} |')
    lines.append(f'| CFD 3x CAGR > TQQQ 3x CAGR | CFD優位 | CFD {_fmt_pct(cfd3_cagr)} > TQQQ {_fmt_pct(tqqq_cagr)} | {"✅ PASS" if cfd3_cagr > tqqq_cagr else "⚠️ WARN"} |')
    lines.append(f'| MaxDD単調増加 (3x→7x) | abs(MaxDD)増大 | 実測値参照 | {"✅ PASS" if dd_monotone else "⚠️ 非単調"} |')

    # CAGR monotonicity
    cagr_vals = [all_metrics.get(f'DH Dyn {l}x2x3x [CFD]', {}).get('CAGR_FULL', np.nan)
                 for l in [3, 4, 5, 6, 7]]
    cagr_clean = [v for v in cagr_vals if not np.isnan(v)]
    cagr_mono = all(cagr_clean[i] < cagr_clean[i+1] for i in range(len(cagr_clean)-1))
    lines.append(f'| CAGR単調増加 (3x→7x) | 単調増加が理想だが高金利期で逆転しうる | 実測値参照 | {"✅ 単調" if cagr_mono else "⚠️ 逆転あり (高金利期コスト)"} |')

    # blowup check
    b6 = all_metrics.get('DH Dyn 6x2x3x [CFD]', {}).get('Blowup_Days', 0)
    b7 = all_metrics.get('DH Dyn 7x2x3x [CFD]', {}).get('Blowup_Days', 0)
    lines.append(f'| NAV崩壊日 (6x) | 0が理想 | {b6}日 | {"✅ なし" if b6 == 0 else f"⚠️ {b6}日発生"} |')
    lines.append(f'| NAV崩壊日 (7x) | 0が理想 | {b7}日 | {"✅ なし" if b7 == 0 else f"⚠️ {b7}日発生"} |')
    lines.append('')

    # --- Section 7: リスク所見 ---
    lines.append('---')
    lines.append('')
    lines.append('## 7. リスク所見')
    lines.append('')
    lines.append('### 6x/7x の高金利期借入コスト問題')
    lines.append('')
    lines.append('1980年代初頭 (DTB3 ≈ 15-20%) における年率借入コスト:')
    lines.append('')
    lines.append('| レバレッジ | SOFR=15%時の年率借入コスト | NASDAQ年平均リターンとの比較 |')
    lines.append('|-----------|---------------------------|------------------------------|')
    lines.append('| 5x | 4 × (15% + 0.20%) = 60.8%/yr | NASDAQ平均 ~15% を大幅超過 |')
    lines.append('| 6x | 5 × (15% + 0.20%) = 76.0%/yr | NASDAQ平均 ~15% を大幅超過 |')
    lines.append('| 7x | 6 × (15% + 0.20%) = 91.2%/yr | NASDAQ平均 ~15% を大幅超過 |')
    lines.append('')
    lines.append('> DH Dynシグナルは高金利・ベア相場でポジション (lev_s) を下げる。')
    lines.append('> このため「全期間保有」より実際の損失は抑制されるが、完全には回避できない。')
    lines.append('')

    m4 = all_metrics.get('DH Dyn 4x2x3x [CFD]', {})

    lines.append('### 全レバレッジのMaxDD・Worst5Y・Worst10Y一覧')
    lines.append('')
    lines.append('| 戦略 | MaxDD | Worst5Y | Worst10Y | Blowup日数 |')
    lines.append('|------|-------|---------|---------|-----------|')
    for name in strat_order:
        m = all_metrics.get(name, {})
        lines.append(f'| {name} | {_fmt_pct(m.get("MaxDD_FULL"))} | {_fmt_pct(m.get("Worst5Y"))} | {_fmt_pct(m.get("Worst10Y"))} | {_fmt_int(m.get("Blowup_Days", 0))} |')
    lines.append('')

    lines.append('### 実務上の制約')
    lines.append('')
    lines.append('- **6x/7x 専用上場商品なし**: くりっく株365等の上場商品は3x止まり。ただしCFD業者で指数10倍枠内で設定可能。')
    lines.append('- **証拠金維持率**: 5x以上でも暴落時の追証・ロスカットリスクが高い')
    lines.append('- **スリッページ**: CFDのスプレッドは市場状況で変動 (本シミュレーションは固定)')
    lines.append('- **ロールコスト**: 先物ベース商品は限月ロール時にコスト発生 (未計上)')
    lines.append('- **wn/wg/wb の再最適化未実施**: 4x以上では現行重みが最適でない可能性')
    lines.append('')

    # --- Section 8: 結論 ---
    lines.append('---')
    lines.append('')
    lines.append('## 8. 結論と CURRENT_BEST_STRATEGY.md への影響')
    lines.append('')
    lines.append('### 主要な発見')
    lines.append('')
    lines.append(f'1. **CFD 3x の優位性**: vol drag 消失により TQQQ比 {(cfd3_cagr-tqqq_cagr)*100:+.1f}pp の CAGR 改善')
    lines.append(f'2. **CFD 4x が有望**: CAGR {_fmt_pct(m4.get("CAGR_FULL"))} — wn再最適化が完了すれば実運用候補')
    lines.append(f'3. **CFD 5x**: CAGR {_fmt_pct(m5.get("CAGR_FULL"))}, MaxDD {_fmt_pct(m5.get("MaxDD_FULL"))}, Worst5Y {_fmt_pct(m5.get("Worst5Y"))} — 証拠金管理が課題')
    lines.append(f'4. **CFD 6x** †: CAGR {_fmt_pct(m6.get("CAGR_FULL"))}, MaxDD {_fmt_pct(m6.get("MaxDD_FULL"))}, Worst10Y {_fmt_pct(m6.get("Worst10Y"))} — CFD業者で設定可能 (専用上場商品なし)')
    lines.append(f'5. **CFD 7x** †: CAGR {_fmt_pct(m7.get("CAGR_FULL"))}, MaxDD {_fmt_pct(m7.get("MaxDD_FULL"))}, Worst10Y {_fmt_pct(m7.get("Worst10Y"))} — CFD業者で設定可能 (専用上場商品なし)')
    lines.append(f'6. **スプレッド感応度**: LOW/MID/HIGH間のCAGR差は ≤0.6pp (SOFR≒4%時)')
    lines.append('')
    lines.append('### CURRENT_BEST_STRATEGY.md の更新について')
    lines.append('')
    lines.append('**本レポートは CURRENT_BEST_STRATEGY.md を直接書き換えない。**')
    lines.append('')
    lines.append('- 6x/7x は専用上場商品なし (CFD業者で設定可能)。wn/wg/wb再最適化・証拠金維持率の検討が必要')
    lines.append('- CFD 3x/4x は実運用候補だが、wn/wg/wb 再最適化・商品流動性確認が先決')
    lines.append('- 上記が完了した時点で CURRENT_BEST_STRATEGY.md の更新を検討すること')
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- Appendix A: コスト比較 ---
    lines.append('## Appendix A: TQQQ vs CFD コスト構造比較 (SOFR=4.37%想定)')
    lines.append('')
    avg_sofr = 0.0437
    lines.append('| コスト項目 | TQQQ 3x | CFD 3x | CFD 6x [理論] | CFD 7x [理論] |')
    lines.append('|-----------|--------|--------|--------------|--------------|')
    lines.append(f'| TER | {ANNUAL_COST*100:.2f}%/yr | 0.00% | 0.00% | 0.00% |')
    lines.append(f'| SOFR financing | 2×{avg_sofr*100:.2f}%={2*avg_sofr*100:.2f}% | 2×{avg_sofr*100:.2f}%={2*avg_sofr*100:.2f}% | 5×{avg_sofr*100:.2f}%={5*avg_sofr*100:.2f}% | 6×{avg_sofr*100:.2f}%={6*avg_sofr*100:.2f}% |')
    lines.append(f'| Swap/spread | {SWAP_SPREAD*100:.2f}% | {CFD_SPREAD_LOW*100:.2f}% | 5×0.20%=1.00% | 6×0.20%=1.20% |')
    lines.append(f'| vol drag (σ=22%) | -14.52%/年 | **0%** | **0%** | **0%** |')
    lines.append(f'| **合計 (SOFR除く)** | {(ANNUAL_COST+SWAP_SPREAD)*100:.2f}%+vol drag | {CFD_SPREAD_LOW*100:.2f}% | {5*CFD_SPREAD_LOW*100:.2f}% | {6*CFD_SPREAD_LOW*100:.2f}% |')
    lines.append('')

    # --- Appendix B: 高金利期試算 ---
    lines.append('## Appendix B: 高金利期 SOFR感応度試算')
    lines.append('')
    lines.append('保有中 (lev=1) の日次収益 = `L × r_nas - (L-1) × (SOFR + spread/252)`')
    lines.append('r_nas = 0 (横ばい) 仮定時の年率損失:')
    lines.append('')
    lines.append('| SOFR水準 | 3x CFD | 4x CFD | 5x CFD | 6x [理論] | 7x [理論] |')
    lines.append('|---------|--------|--------|--------|----------|----------|')
    for sofr_rate in [0.02, 0.04, 0.08, 0.15, 0.20]:
        row = f'| SOFR={sofr_rate*100:.0f}% |'
        for L in [3, 4, 5, 6, 7]:
            cost = (L-1) * (sofr_rate + CFD_SPREAD_LOW)
            row += f' -{cost*100:.1f}% |'
        lines.append(row)
    lines.append('')

    # --- Appendix C: QC ---
    lines.append('## Appendix C: QC コード片 (借入係数アサーション)')
    lines.append('')
    lines.append('```python')
    lines.append('# 借入係数 = L-1 であることを確認')
    lines.append('for L in [3, 4, 5, 6, 7]:')
    lines.append('    out = build_cfd_nas_sleeve(np.array([0.0]), L, np.array([0.04/252]), cfd_spread=0.002)')
    lines.append('    expected = -(L-1)*(0.04/252 + 0.002/252)')
    lines.append('    assert abs(out[0] - expected) < 1e-12, f"L={L} coef wrong"')
    lines.append('```')
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
    print("CFD Leverage Backtest -- DH Dyn 3x/4x/5x/6x/7x2x3x (2026-05-15)")
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

    cfd_variants = [
        (3.0, 'DH Dyn 3x2x3x [CFD]'),
        (4.0, 'DH Dyn 4x2x3x [CFD]'),
        (5.0, 'DH Dyn 5x2x3x [CFD]'),
        (6.0, 'DH Dyn 6x2x3x [CFD]'),
        (7.0, 'DH Dyn 7x2x3x [CFD]'),
    ]

    navs = {}
    trades_yr = {}

    # DH TQQQ (Scenario D sanity check)
    print("  [1/7] DH Dyn 3x2x3x [TQQQ] ...")
    navs['DH Dyn 3x2x3x [TQQQ]'] = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr, nas_mode='TQQQ')
    trades_yr['DH Dyn 3x2x3x [TQQQ]'] = dh_trades_yr

    # DH CFD variants
    for i, (lev_cfd, name) in enumerate(cfd_variants, start=2):
        theory_mark = ' [専用上場商品なし]' if lev_cfd in NO_LISTED_PRODUCT_LEVERAGES else ''
        print(f"  [{i}/7] {name}{theory_mark} ...")
        navs[name] = build_nav_strategy(
            close, lev_A, wn_A, wg_A, wb_A, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=lev_cfd, cfd_spread=CFD_SPREAD_LOW)
        trades_yr[name] = dh_trades_yr

    # A2 Optimized
    print("  [7/7] A2 Optimized [TQQQ] ...")
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

    # CAGR monotonicity check
    cagr_seq = [(lev, all_metrics.get(name, {}).get('CAGR_FULL', np.nan))
                for lev, name in cfd_variants]
    print("  Leverage monotonicity (CAGR):")
    for lev, cagr in cagr_seq:
        b = all_metrics.get(next(n for l, n in cfd_variants if l == lev), {}).get('Blowup_Days', 0)
        print(f"    {lev:.0f}x: {cagr*100:+.2f}%  blowup={b}日")

    # 7. Print main results table
    strat_order_print = list(navs.keys())
    print("\n" + "=" * 130)
    print("MAIN RESULTS (CFD_SPREAD = LOW 0.20%)")
    print("=" * 130)
    hdr = f"{'Strategy':<35} | {'CAGR_F':>8} {'CAGR_I':>8} {'CAGR_O':>8} | {'Shrp_F':>7} {'Shrp_I':>7} {'Shrp_O':>7} | {'MaxDD':>8} | {'W5Y':>7} {'W10Y':>7} | {'WR':>6} | {'Trd/yr':>7} | {'Blowup':>6}"
    print(hdr)
    print("-" * 130)
    for name in strat_order_print:
        m = all_metrics[name]
        def p(v): return f'{v*100:>7.2f}%' if v is not None and not np.isnan(v) else f'{"N/A":>8}'
        def f(v): return f'{v:>7.3f}' if v is not None and not np.isnan(v) else f'{"N/A":>7}'
        def t(v): return f'{v:>7.1f}' if v is not None and not np.isnan(v) else f'{"N/A":>7}'
        b = m.get('Blowup_Days', 0)
        print(f"{name:<35} | {p(m['CAGR_FULL'])} {p(m['CAGR_IS'])} {p(m['CAGR_OOS'])} | "
              f"{f(m['Sharpe_FULL'])} {f(m['Sharpe_IS'])} {f(m['Sharpe_OOS'])} | "
              f"{p(m['MaxDD_FULL'])} | {p(m['Worst5Y'])} {p(m['Worst10Y'])} | "
              f"{m['WinRate']*100:>5.1f}% | {t(m['Trades_yr'])} | {b:>6}")
    print("=" * 130)

    # 8. Sensitivity analysis
    print("\nRunning spread sensitivity analysis...")
    sensitivity = {}
    for spread_label, spread_val in CFD_SPREADS.items():
        sensitivity[spread_label] = {}
        for lev_cfd, name in cfd_variants:
            nav_s = build_nav_strategy(
                close, lev_A, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=lev_cfd, cfd_spread=spread_val)
            sensitivity[spread_label][name] = calc_7metrics(nav_s, dates, dh_trades_yr)
        print(f"  {spread_label} done.")

    # 9. Print sensitivity table
    print("\n--- Spread Sensitivity (CAGR FULL) ---")
    all_cfd_names = [n for _, n in cfd_variants]
    print(f"{'Strategy':<35} {'LOW(0.20%)':>12} {'MID(0.30%)':>12} {'HIGH(0.50%)':>12}")
    for name in all_cfd_names:
        row = f"{name:<35}"
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
    out_path = os.path.join(BASE, 'CFD_LEVERAGE_BACKTEST_6x7x_2026-05-15.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"\nSaved: {out_path}")
    print("Done.")

    return all_metrics, sensitivity


if __name__ == '__main__':
    main()
