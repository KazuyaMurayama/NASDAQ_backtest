"""
G19D: F10 Trades/yr=52 の内訳分解 (wn-tilt 由来 vs L_s2 由来)
=================================================================
v6.1 で F10 は E4 比 +24 Trades/yr (52 vs 28)。この差分が「wn-tilt 由来の
日次微変動」なのか「L_s2 由来の再現」なのか定量分解する。

g14 標準 (lev_change>0.5 流儀):
  trade = (wn 変化) OR (wb 変化) OR (lev_mod 変化)

成分:
  - L_s2 変化: S2_VZGated レバ (1〜7x) の日次変化
  - lev_raw 変化: DH Dyn A シグナル (0/1 or [0,1]) の閾値ベース変化
  - lev_mod 変化: LT2 bias 重畳後の lev_mod の日次変化
  - wn 変化 (E4): wn_A の閾値ベース変化
  - wn 変化 (F10): wn_A + ε-deadband tilt の変化
  - wb 変化 (E4 / F10): bond 配分の変化

決定: F10 の「+24 Trades/yr 追加」 = wn_f10/wb_f10 の差分由来 を実証
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import (
    load_shared_assets, BASE, TRADING_DAYS,
)


def count_changes(arr):
    a = np.asarray(arr)
    return int((np.diff(a) != 0).sum())


def count_combo(arrs):
    """複数 array いずれかに変化があった日をカウント。"""
    masks = [np.diff(np.asarray(x)) != 0 for x in arrs]
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return int(combined.sum())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G19D: F10 Trades/yr=52 内訳分解')
    print('=' * 80)

    a = load_shared_assets()
    n_years = a['n_years']

    # 各成分の日次変化回数
    lev_raw = np.asarray(a['lev_raw'])         # DH Dyn A signal
    lev_mod_e4 = np.asarray(a['lev_mod_e4'])    # E4 (LT2 overlay)
    L_s2_lmax7 = np.asarray(a['L_s2_lmax7'].values)
    wn_A = np.asarray(a['wn_A'])               # E4 base
    wb_A = np.asarray(a['wb_A'])
    wn_f10 = np.asarray(a['wn_f10'])           # F10 (wn_A + tilt)
    wb_f10 = np.asarray(a['wb_f10'])

    # 各単体変化回数
    n_lev_raw = count_changes(lev_raw)
    n_lev_mod_e4 = count_changes(lev_mod_e4)
    n_L_s2 = count_changes(L_s2_lmax7)
    n_wn_A = count_changes(wn_A)
    n_wb_A = count_changes(wb_A)
    n_wn_f10 = count_changes(wn_f10)
    n_wb_f10 = count_changes(wb_f10)

    print(f'\n[単体成分の日次変化回数]')
    print(f'  L_s2 (lmax=7) only:         {n_L_s2:>5d} / {n_years:.1f}y = {n_L_s2/n_years:>6.1f}/yr')
    print(f'  lev_raw (DH Dyn A) only:    {n_lev_raw:>5d} / {n_years:.1f}y = {n_lev_raw/n_years:>6.1f}/yr')
    print(f'  lev_mod_e4 (LT2 overlay):   {n_lev_mod_e4:>5d} / {n_years:.1f}y = {n_lev_mod_e4/n_years:>6.1f}/yr')
    print(f'  wn_A (E4 base):             {n_wn_A:>5d} / {n_years:.1f}y = {n_wn_A/n_years:>6.1f}/yr')
    print(f'  wb_A (E4 base):             {n_wb_A:>5d} / {n_years:.1f}y = {n_wb_A/n_years:>6.1f}/yr')
    print(f'  wn_f10 (F10 with tilt):     {n_wn_f10:>5d} / {n_years:.1f}y = {n_wn_f10/n_years:>6.1f}/yr')
    print(f'  wb_f10 (F10 with tilt):     {n_wb_f10:>5d} / {n_years:.1f}y = {n_wb_f10/n_years:>6.1f}/yr')

    # E4 trades = wn_A OR wb_A OR lev_mod_e4 のいずれか変化した日数
    n_E4_trades = count_combo([wn_A, wb_A, lev_mod_e4])
    n_F10_trades = count_combo([wn_f10, wb_f10, lev_mod_e4])
    print(f'\n[戦略統合 trades/yr (g14 流儀)]')
    print(f'  E4 trades (wn_A | wb_A | lev_mod_e4):     {n_E4_trades} / {n_years:.1f}y = {n_E4_trades/n_years:.1f}/yr')
    print(f'  F10 trades (wn_f10 | wb_f10 | lev_mod_e4): {n_F10_trades} / {n_years:.1f}y = {n_F10_trades/n_years:.1f}/yr')
    print(f'  差分 (F10 - E4): {n_F10_trades - n_E4_trades} / {n_years:.1f}y = '
          f'{(n_F10_trades - n_E4_trades)/n_years:.1f}/yr')

    # 各成分の独立寄与度: 単独で trades count を増やしている部分
    # F10 trades の追加分が「wn_f10 だけ変化した日」or「wb_f10 だけ変化した日」か検証
    diff_wn_only = (np.diff(wn_f10) != 0) & (np.diff(wn_A) == 0) & (np.diff(wb_A) == 0) & (np.diff(lev_mod_e4) == 0)
    diff_wb_only = (np.diff(wb_f10) != 0) & (np.diff(wn_A) == 0) & (np.diff(wb_A) == 0) & (np.diff(lev_mod_e4) == 0)
    n_diff_wn_only = int(diff_wn_only.sum())
    n_diff_wb_only = int(diff_wb_only.sum())
    n_diff_either = int((diff_wn_only | diff_wb_only).sum())

    print(f'\n[F10 追加 trades 内訳]')
    print(f'  wn_f10 単独変化 (E4ベース不変):  {n_diff_wn_only} ({n_diff_wn_only/n_years:.1f}/yr)')
    print(f'  wb_f10 単独変化 (E4ベース不変):  {n_diff_wb_only} ({n_diff_wb_only/n_years:.1f}/yr)')
    print(f'  上記 OR (重複除く):            {n_diff_either} ({n_diff_either/n_years:.1f}/yr)')

    # F10 - E4 trades 差分計算
    print(f'\n[結論]')
    print(f'  F10 vs E4 trade 差: {(n_F10_trades - n_E4_trades)/n_years:.1f}/yr')
    print(f'  → F10 追加 trades は F8-R5 (CALM_BOOST tilt 連続関数) を ε=0.015 でデッドバンドした上で')
    print(f'    残った wn-tilt 由来の更新が {(n_F10_trades - n_E4_trades)/n_years:.1f}/yr 発生')
    print(f'  → 実取引コスト視点では、これらは「微小Δ」の rebalance (tilt step毎に発火)')
    print(f'  → daily_cost = |Δposition| × spread で評価すると小さい (g18 で検証済み)')

    # OOS specific
    dates = a['dates']
    oos_mask = dates >= pd.Timestamp('2021-05-08')
    oos_years = oos_mask.sum() / TRADING_DAYS
    n_F10_trades_oos = count_combo([wn_f10[oos_mask.values], wb_f10[oos_mask.values],
                                     lev_mod_e4[oos_mask.values]])
    n_E4_trades_oos = count_combo([wn_A[oos_mask.values], wb_A[oos_mask.values],
                                    lev_mod_e4[oos_mask.values]])
    print(f'\n[OOS (2021-2026) 内訳]')
    print(f'  E4 trades OOS:  {n_E4_trades_oos} / {oos_years:.1f}y = {n_E4_trades_oos/oos_years:.1f}/yr')
    print(f'  F10 trades OOS: {n_F10_trades_oos} / {oos_years:.1f}y = {n_F10_trades_oos/oos_years:.1f}/yr')
    print(f'  差分 OOS: {(n_F10_trades_oos - n_E4_trades_oos)/oos_years:.1f}/yr')

    # CSV 保存
    out = pd.DataFrame([
        dict(component='L_s2 (S2_VZGated lmax=7)', changes=n_L_s2, per_yr=n_L_s2/n_years),
        dict(component='lev_raw (DH Dyn A)', changes=n_lev_raw, per_yr=n_lev_raw/n_years),
        dict(component='lev_mod_e4 (LT2 overlay)', changes=n_lev_mod_e4, per_yr=n_lev_mod_e4/n_years),
        dict(component='wn_A (E4 base)', changes=n_wn_A, per_yr=n_wn_A/n_years),
        dict(component='wb_A (E4 base)', changes=n_wb_A, per_yr=n_wb_A/n_years),
        dict(component='wn_f10 (F10 with tilt)', changes=n_wn_f10, per_yr=n_wn_f10/n_years),
        dict(component='wb_f10 (F10 with tilt)', changes=n_wb_f10, per_yr=n_wb_f10/n_years),
        dict(component='E4 strategy (combo)', changes=n_E4_trades, per_yr=n_E4_trades/n_years),
        dict(component='F10 strategy (combo)', changes=n_F10_trades, per_yr=n_F10_trades/n_years),
        dict(component='F10 - E4 (additional)', changes=n_F10_trades - n_E4_trades,
              per_yr=(n_F10_trades - n_E4_trades)/n_years),
    ])
    csv_out = os.path.join(BASE, 'g19d_f10_trade_decomp_results.csv')
    out.to_csv(csv_out, index=False)
    print(f'\n→ CSV saved: {csv_out}')


if __name__ == '__main__':
    main()
