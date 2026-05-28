"""
g12_sbi_cfd_cost_comparison.py
================================
CFDスプレッド感度分析: バックテスト前提（0.20%/yr）vs SBI CFD実績（3.0%/yr）

背景:
  D5/E4 バックテストは cfd_spread=0.0020（くりっく株365・0.20%/yr）で計算されているが、
  くりっく株365にはNASDAQ製品が存在しない。
  実際に使用可能なSBI CFD (NQ100) は業者マージン≈3.0%/yr（SOFR除く）であり、
  このコスト差が CAGR にどの程度影響するかを定量化する。

CFDコストモデル:
  r_cfd = L_t * r_nasdaq - (L_t - 1) * (sofr_daily + cfd_spread/252)
  ※ cfd_spread は SOFR 超過分（業者スプレッド）のみ

出力:
  - g12_sbi_cfd_cost_comparison.csv
  - G12_SBI_CFD_COST_COMPARISON.md
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics,
    CFD_SPREAD_LOW, IS_START, IS_END, OOS_START,
)
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from a2_dyn_lmax import (
    compute_L_s2_dyn_lmax, signal_to_bias_dynamic,
    compute_p10_5y, calc_all_metrics,
    S2_BASE, K_LO, K_HI, K_MID,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# CFDスプレッド定義（SOFR超過分のみ / 年率）
# ---------------------------------------------------------------------------
SPREADS = {
    'Backtest前提 (0.20%/yr)':   0.0020,   # くりっく株365想定（NASDAQ非対応）
    'SBI CFD NQ100 (3.0%/yr)':  0.0300,   # SBI CFD 業者マージン実績
    'GMO/IG CFD (2.5%/yr)':     0.0250,   # 他社CFD推定
}

# ---------------------------------------------------------------------------
# テスト対象戦略設定
# ---------------------------------------------------------------------------
# E4固定値
K_LO_E4 = 0.1; K_HI_E4 = 0.8; K_MID_E4 = 0.50
LT_N = 750

KEY_CONFIGS = [
    ('E4 / vz=0.70 / lmax=7.0 [現行ベスト]',  0.70, 7.0),
    ('D5 vz=0.65 / lmax=5.0 [MaxDD最良候補]', 0.65, 5.0),
    ('D5 vz=0.65 / lmax=5.5',                 0.65, 5.5),
    ('D5 vz=0.65 / lmax=6.0',                 0.65, 6.0),
    ('D5 vz=0.65 / lmax=7.0',                 0.65, 7.0),
    ('D5 vz=0.60 / lmax=4.5 [MaxDD最良]',      0.60, 4.5),
    ('D5 vz=0.60 / lmax=5.0',                 0.60, 5.0),
]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 75)
    print('G12: CFDスプレッド感度分析（バックテスト前提 vs SBI CFD実績）')
    print('=' * 75)

    # --- データロード ---
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}  ({n:,}日 / {n_years:.1f}年)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years
    print(f'A2 signal Trades/yr: {n_trades_yr:.1f}')

    lt_sig_raw = build_lt_signal(close, 'LT2', LT_N)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values

    # --- 平均レバレッジ統計（参考） ---
    print('\n[参考] 各設定での平均L_s2（IS+OOS全期間）')
    for label, vz_thr, l_max in KEY_CONFIGS:
        regime_hi = vz_arr > +vz_thr
        regime_lo = vz_arr < -vz_thr
        k_dyn     = np.where(regime_hi, K_HI_E4, np.where(regime_lo, K_LO_E4, K_MID_E4))
        lt_bias   = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_dyn),
                              index=lt_sig_raw.index)
        lev_mod   = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

        l_max_series = pd.Series(l_max, index=ret.index)
        L_s2 = compute_L_s2_dyn_lmax(ret, vz, l_max_series, **S2_BASE)

        # lev_mod × L_s2 が実効レバレッジ（wn_Aを乗じる前）
        eff_lev = pd.Series(np.asarray(lev_mod) * L_s2.values, index=ret.index)
        print(f'  {label[:40]:40s} | L_s2_avg={L_s2.mean():.2f} | lev_mod_avg={lev_mod.mean():.3f} '
              f'| eff_L_avg={eff_lev.mean():.2f} (max={eff_lev.max():.2f})')

    print('\n')

    # --- メインシミュレーション ---
    results = []

    for spread_label, cfd_spread in SPREADS.items():
        print(f'\n■ {spread_label}')
        print(f'  cfd_spread = {cfd_spread*100:.2f}%/yr')

        for label, vz_thr, l_max in KEY_CONFIGS:
            regime_hi = vz_arr > +vz_thr
            regime_lo = vz_arr < -vz_thr
            k_dyn     = np.where(regime_hi, K_HI_E4, np.where(regime_lo, K_LO_E4, K_MID_E4))
            lt_bias   = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_dyn),
                                  index=lt_sig_raw.index)
            lev_mod   = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

            l_max_series = pd.Series(l_max, index=ret.index)
            L_s2 = compute_L_s2_dyn_lmax(ret, vz, l_max_series, **S2_BASE)

            nav = build_nav_strategy(
                close, lev_mod, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=cfd_spread,
            )
            m = calc_all_metrics(nav, dates, n_trades_yr)

            row = {
                'spread_label': spread_label,
                'cfd_spread_pct': cfd_spread * 100,
                'strategy': label,
                'vz_thr': vz_thr,
                'l_max': l_max,
                'CAGR_OOS': m.get('CAGR_OOS', float('nan')),
                'CAGR_IS':  m.get('CAGR_IS',  float('nan')),
                'Sharpe_OOS': m.get('Sharpe_OOS', float('nan')),
                'MaxDD_FULL': m.get('MaxDD_FULL', float('nan')),
                'Worst10Y_star': m.get('Worst10Y_star', float('nan')),
                'P10_5Y': m.get('P10_5Y', float('nan')),
                'IS_OOS_gap': m.get('IS_OOS_gap', float('nan')),
                'Trades_yr': n_trades_yr,
            }
            results.append(row)

            print(f'  {label[:50]:50s} | CAGR_OOS={m["CAGR_OOS"]*100:+.1f}%  '
                  f'Sharpe={m["Sharpe_OOS"]:+.3f}  MaxDD={m["MaxDD_FULL"]*100:+.1f}%')

    # --- CSV出力 ---
    df_out = pd.DataFrame(results)
    csv_path = os.path.join(BASE, 'g12_sbi_cfd_cost_comparison.csv')
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\nCSV: {csv_path}')

    # --- MD出力 ---
    _write_md(df_out, n_trades_yr)


def _write_md(df, n_trades_yr):
    lines = []
    W = lines.append

    W('# G12: CFDスプレッド感度分析レポート')
    W('')
    W('> **分析目的**: D5/E4バックテストが仮定したCFDスプレッド（0.20%/yr）は')
    W('> くりっく株365に基づくが、NASDAQ製品は存在しない。')
    W('> 実際のSBI CFD NQ100（業者マージン≈3.0%/yr）を使用した場合のCAGR変化を定量化。')
    W('')
    W('## コスト前提の比較')
    W('')
    W('| CFDスプレッド | 年率 | 実在性 | L=5時の年間追加コスト | L=7時の年間追加コスト |')
    W('|--------------|------|--------|----------------------|----------------------|')
    W('| バックテスト前提 | **0.20%/yr** | ❌ NASDAQ非対応 | — | — |')
    W('| GMO/IG CFD | **2.50%/yr** | ✅ 利用可能 | +9.2%/yr | +13.8%/yr |')
    W('| **SBI CFD (NQ100)** | **3.00%/yr** | ✅ 利用可能 | +11.2%/yr | +16.8%/yr |')
    W('')
    W('> 追加コスト = (L-1) × (新スプレッド - 旧スプレッド) = (L-1) × 差分%')
    W('')
    W('---')
    W('')
    W('## CAGR比較表（戦略 × CFDスプレッド）')
    W('')

    spread_labels = df['spread_label'].unique().tolist()

    for strat_label in df['strategy'].unique():
        df_s = df[df['strategy'] == strat_label]
        W(f'### {strat_label}')
        W('')
        W('| CFDスプレッド | CAGR_OOS | Sharpe_OOS | MaxDD | Worst10Y★ | IS-OOS gap |')
        W('|--------------|----------|------------|-------|----------|-----------|')
        for _, row in df_s.iterrows():
            W(f'| {row["spread_label"]} '
              f'| {row["CAGR_OOS"]*100:+.1f}% '
              f'| {row["Sharpe_OOS"]:+.3f} '
              f'| {row["MaxDD_FULL"]*100:+.1f}% '
              f'| {row["Worst10Y_star"]*100:+.1f}% '
              f'| {row["IS_OOS_gap"]*100:+.2f}pp |')
        W('')

    W('---')
    W('')
    W('## サマリー: バックテスト前提 vs SBI CFD実績の差分')
    W('')
    W('| 戦略 | lmax | CAGR_OOS<br>バックテスト | CAGR_OOS<br>SBI CFD | 差分 | Sharpe<br>SBI CFD |')
    W('|------|------|------------------------|---------------------|------|-----------------|')

    ref_spread = 'Backtest前提 (0.20%/yr)'
    sbi_spread = 'SBI CFD NQ100 (3.0%/yr)'

    for strat_label in df['strategy'].unique():
        df_s = df[df['strategy'] == strat_label]
        ref_row = df_s[df_s['spread_label'] == ref_spread]
        sbi_row = df_s[df_s['spread_label'] == sbi_spread]
        if ref_row.empty or sbi_row.empty:
            continue
        r = ref_row.iloc[0]; s = sbi_row.iloc[0]
        diff = s['CAGR_OOS'] - r['CAGR_OOS']
        lmax = r['l_max']
        short = strat_label.split('/')[0].strip()[:30]
        W(f'| {short} | {lmax:.1f} '
          f'| {r["CAGR_OOS"]*100:+.1f}% '
          f'| {s["CAGR_OOS"]*100:+.1f}% '
          f'| **{diff*100:+.1f}pp** '
          f'| {s["Sharpe_OOS"]:+.3f} |')

    W('')
    W('---')
    W('')
    W('*生成: `src/g12_sbi_cfd_cost_comparison.py` | IS=1974-2021 / OOS=2021-2026*')

    md_path = os.path.join(BASE, 'G12_SBI_CFD_COST_COMPARISON.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'MD:  {md_path}')


if __name__ == '__main__':
    main()
