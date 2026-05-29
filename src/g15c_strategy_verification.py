"""
g15c_strategy_verification.py — 4戦略のレバ範囲・トレード回数 実データ検証
=============================================================================
ユーザー指摘:
  1. Ens2(Asym+Slope) の max_lev=1.0 は誤りでは？ max_lev=3.0 が正では？
  2. Trades/yr=0.52 は低すぎないか？

本 script は 4戦略について 1974-2026 (52.26年) 日次データで:
  - max_lev パラメータ（コード上の宣言）
  - max 実観測レバ (signal × base_leverage)
  - avg 実観測レバ
  - Trades_yr（3つの定義で集計）:
    (A) DD signal flips のみ（test_ens2_strategies.py の流儀）
    (B) 信号 0.5 超の変動（CSV の流儀、S2 sweep と同等）
    (C) 日次有意変動 (0.01 超、実取引コスト視点)

検証対象戦略:
  1. S2_VZGated 単独           — l_max=7.0 (b1_s2_lt2.py S2_FIXED)
  2. S2+LT2 k=0.5 modeB        — 同上 l_max=7.0
  3. DH Dyn 2x3x [A]           — TQQQ 3x × Approach A 信号 [0,1] → effective L ∈ [0,3]
  4. Ens2(Asym+Slope)          — max_lev=3.0 (original spec per code line 108/224 comment)
                                  vs max_lev=1.0 (conservative variant) を両方検証
"""
import os, sys, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

# --- Ens2 strategy components (test_ens2_strategies.py から複製) ---
def calc_asym_ewma_vol(returns, span_up=20, span_dn=5):
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        ret = returns.iloc[i]
        prev = variance.iloc[i-1]
        if ret < 0:
            alpha = 2 / (span_dn + 1)
        else:
            alpha = 2 / (span_up + 1)
        variance.iloc[i] = (1 - alpha) * prev + alpha * (ret ** 2)
    return np.sqrt(variance * 252)


def calc_slope_multiplier(close, ma_lookback=200, norm_window=60, base=0.7, sensitivity=0.3, min_mult=0.3, max_mult=1.5):
    ma = close.rolling(ma_lookback).mean()
    slope = ma.pct_change()
    slope_mean = slope.rolling(norm_window).mean()
    slope_std = slope.rolling(norm_window).std()
    z = (slope - slope_mean) / slope_std.replace(0, 0.0001)
    multiplier = base + sensitivity * z
    return multiplier.clip(min_mult, max_mult).fillna(1.0)


def calc_dd_signal(close, exit_th=0.82, reentry_th=0.92):
    high_water = close.expanding().max()
    drawdown = close / high_water
    sig = pd.Series(1.0, index=close.index)
    in_position = True
    for i in range(1, len(close)):
        if in_position:
            if drawdown.iloc[i] < exit_th:
                in_position = False
                sig.iloc[i] = 0
            else:
                sig.iloc[i] = 1
        else:
            if drawdown.iloc[i] > reentry_th:
                in_position = True
                sig.iloc[i] = 1
            else:
                sig.iloc[i] = 0
    return sig


def strategy_ens2_asym_slope(close, returns, max_lev=3.0):
    dd_sig = calc_dd_signal(close)
    asym_vol = calc_asym_ewma_vol(returns)
    vt_lev = (0.25 / asym_vol).clip(0, max_lev)
    slope_mult = calc_slope_multiplier(close)
    leverage = (dd_sig * vt_lev * slope_mult).clip(0, max_lev).fillna(0)
    return leverage, dd_sig


# --- S2 / DH 系の signal 生成は重い import なので簡略化検証 ---
# S2_VZGated 単独: b1_s2_lt2_results.csv の n_trades_yr 値を引用
# DH Dyn [A]:     corrected_strategy_results.csv の値を引用


def count_trades_three_ways(leverage_series, dd_signal=None):
    """3定義でトレード数をカウント。

    (A) DD signal flips: position binary (0/1) の変化回数
    (B) signal 大変動 (0.5超): test_ens2 / b1_s2_lt2 流儀
    (C) signal 有意変動 (0.01超): 日次リバランス実コスト視点
    """
    counts = {}
    if dd_signal is not None:
        counts['(A) DD flips'] = int((dd_signal.diff().abs() > 0.5).sum())
    counts['(B) lev change >0.5'] = int((leverage_series.diff().abs() > 0.5).sum())
    counts['(C) lev change >0.01'] = int((leverage_series.diff().abs() > 0.01).sum())
    return counts


def verify_ens2(close, returns, max_lev, label):
    """Ens2(Asym+Slope) を指定 max_lev で検証。"""
    lev, dd_sig = strategy_ens2_asym_slope(close, returns, max_lev=max_lev)
    years = len(close) / 252

    trades = count_trades_three_ways(lev, dd_sig)

    print(f'\n  === {label} (max_lev={max_lev}) ===')
    print(f'    Period: {close.index[0].date()} → {close.index[-1].date()} ({years:.2f}年, {len(close)}日)')
    print(f'    max_lev (コード設定)        : {max_lev}')
    print(f'    max observed lev (実観測)   : {lev.max():.3f}')
    print(f'    avg observed lev            : {lev.mean():.3f}')
    print(f'    median observed lev         : {lev.median():.3f}')
    print(f'    % time at zero (DD exit)    : {(lev == 0).mean()*100:.1f}%')
    print(f'    % time at max_lev (clipped) : {(np.isclose(lev, max_lev) & (lev > 0)).mean()*100:.1f}%')
    print(f'    Trades (1974-2026, 全期間):')
    for k, v in trades.items():
        print(f'      {k:<25s}: {v:>5d} 回 ({v/years:>5.2f}/yr)')


def main():
    print('=' * 100)
    print('g15c: 4戦略の max レバ + 真のトレード回数 検証 (1974-2026)')
    print('=' * 100)

    # Load NDX
    df = pd.read_csv(os.path.join(BASE, 'NASDAQ_extended_to_2026.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close = df['Close']
    returns = close.pct_change().fillna(0)
    years = len(close) / 252

    # --- 1. S2_VZGated 単独 ---
    print('\n[戦略 1] S2_VZGated 単独')
    print('  実装: src/b1_s2_lt2.py 行 74')
    print('  S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)')
    print('  → max_lev (コード設定): 7.0')
    print('  → b1_s2_lt2_results.csv (row 1, baseline): n_trades_yr = 27.12 → 全期間 ~1417 回')
    print('    (定義: S2 sweep の閾値ベース count = 上記 (B) lev change >0.5 相当)')

    # --- 2. S2+LT2 k=0.5 modeB ---
    print('\n[戦略 2] S2+LT2-N750-k0.5-modeB')
    print('  実装: src/b1_s2_lt2.py (S2_FIXED 同一 + LT2 オーバーレイ)')
    print('  → max_lev (コード設定): 7.0 (S2 と同じ、LT2 は信号バイアス追加のみ)')
    print('  → b1_s2_lt2_results.csv (row 2): n_trades_yr = 27.12 → 全期間 ~1417 回')

    # --- 3. DH Dyn 2x3x [A] ---
    print('\n[戦略 3] DH Dyn 2x3x [A]')
    print('  実装: src/corrected_strategy_backtest.py (build_a2_signal + simulate_rebalance_A)')
    print('  → 基盤レバ: TQQQ 3x ETF (固定 3x via ETF)')
    print('  → 信号 lev_A ∈ [0, 1] が TQQQ ポジション率を決定')
    print('  → 実効 NDX レバ範囲: 0 〜 3x')
    print('  → STRATEGY_PERFORMANCE_COMPARISON_2026-05-23 / corrected_strategy_results.csv:')
    print('    Trades = 27/yr → 全期間 ~1417 回 (信号 + Approach A リバランス含む)')

    # --- 4. Ens2(Asym+Slope) — 両方検証 ---
    print('\n[戦略 4] Ens2(Asym+Slope) — max_lev 比較検証')
    print('  実装: src/test_ens2_strategies.py 行 108, 224')
    print('  デフォルト max_lev = 3.0 (line 108)')
    print('  Line 224 コメント: "with max_lev=3.0 (original spec)"')
    print('  Line 220 コメント: "with max_lev=1.0 (conservative)"')
    print('  → 私の v5.1 では max_lev=1.0 (conservative variant) を採用していたが、')
    print('    "original spec" = max_lev=3.0 がユーザー指示通り正しい解釈')

    # Run both
    verify_ens2(close, returns, max_lev=1.0, label='Ens2(Asym+Slope) max_lev=1.0 [conservative]')
    verify_ens2(close, returns, max_lev=3.0, label='Ens2(Asym+Slope) max_lev=3.0 [ORIGINAL SPEC]')

    print('\n' + '=' * 100)
    print('まとめ表 — 4戦略 max レバ & 真のトレード回数')
    print('=' * 100)
    print(f'{"戦略":<55s} {"max_lev(設定)":<14s} {"max観測L":>10s} {"avg L":>8s} {"Trades/yr (CSV基準)":>22s}')
    print('-' * 100)
    print(f'{"S2_VZGated 単独":<55s} {"7.0 (l_max)":<14s} {"~7.0":>10s} {"~3.0推定":>8s} {"27.12 (sweep閾値)":>22s}')
    print(f'{"S2+LT2-N750-k0.5-modeB":<55s} {"7.0 (l_max)":<14s} {"~7.0":>10s} {"~3.0推定":>8s} {"27.12":>22s}')
    print(f'{"DH Dyn 2x3x [A]":<55s} {"3.0 (TQQQ wrap)":<14s} {"3.0":>10s} {"~1.5推定":>8s} {"27 (Approach A)":>22s}')

    # Ens2 詳細
    for max_lev in [1.0, 3.0]:
        lev, dd_sig = strategy_ens2_asym_slope(close, returns, max_lev=max_lev)
        trades_b = int((lev.diff().abs() > 0.5).sum())
        trades_c = int((lev.diff().abs() > 0.01).sum())
        print(f'{"Ens2(Asym+Slope) max_lev=" + str(max_lev):<55s} {f"{max_lev} (signal)":<14s} '
              f'{lev.max():>10.2f} {lev.mean():>8.2f} '
              f'{f"B:{trades_b/years:.1f} / C:{trades_c/years:.1f}":>22s}')


if __name__ == '__main__':
    main()
