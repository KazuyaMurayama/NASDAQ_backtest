"""G22A: DH 改善「配分比率 × タイミング 2 軸変動」5 変種 NAV 構築
=================================================================
DH-Z1: DH base 連続配分 + binary vz_gate (lev_mod_065 ≥ 0.5)
DH-Z2: F10 ε tilt 配分 + binary vz_gate (NEW CANDIDATE 配分機構)
DH-Z3: DH base 連続配分 + 保守 composite (vz<0.65 ∧ lt_sig>0 ∧ raw_a2>0.15)
DH-Z4: 固定 bull preset (80/10/10) + binary vz_gate
DH-Z5: regime preset switch (raw_a2>0.5 → bull 80/10/10, else balanced 50/25/25) + 常時 IN

レバレッジ操作 (lev_mod 連続 multiplier, L_s2 cap) は一切使用しない。
peak_lev_eff = max(wn × lev_raw × 3) ≤ 3.0x を assert で機械検証。

商品: TQQQ + TMF + WisdomTree 2036（変更なし）
税: apply_tax_etf_decimal (×0.8273)
取引コスト: per_unit_cost = 0.0010 (moderate)

出力:
  - g22a_dh_alloc_timing_navs.csv  (6 列 [REF + Z1〜Z5] × ~13169 日)
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets, BASE
from g18_daily_trade_cost_wfa import build_dh_nav_with_cost

DH_PER_UNIT = 0.0010  # moderate


# ----- timing mask -----

def hold_mask_vz_binary(a):
    """Z1, Z2, Z4: lev_mod_065 ≥ 0.5 を binary HOLD signal にする"""
    lm = np.nan_to_num(np.asarray(a['lev_mod_065']), nan=0.0)
    return (lm >= 0.5).astype(float)

def hold_mask_composite(a):
    """Z3: vz<0.65 ∧ lt_sig>0 ∧ raw_a2>0.15 の AND 保守 composite"""
    vz_arr = np.nan_to_num(np.asarray(a['vz_arr']), nan=0.0)
    lt_arr = np.nan_to_num(np.asarray(a['lt_sig_arr']), nan=0.0)
    a2_arr = np.nan_to_num(np.asarray(a['raw_a2'].values if hasattr(a['raw_a2'], 'values') else a['raw_a2']), nan=0.0)
    vz_ok = vz_arr < 0.65
    lt_ok = lt_arr > 0.0
    a2_ok = a2_arr > 0.15
    return (vz_ok & lt_ok & a2_ok).astype(float)

def hold_mask_always(a):
    """Z5: 常時 IN"""
    return np.ones(len(a['close']))


# ----- allocation patterns -----

def alloc_dh_base(a):
    """Z1, Z3: DH 既存連続配分"""
    return np.asarray(a['wn_A']), np.asarray(a['wg_A']), np.asarray(a['wb_A'])

def alloc_f10_tilt(a):
    """Z2: F10 ε tilt 配分（NEW CANDIDATE と同一機構）"""
    return np.asarray(a['wn_f10']), np.asarray(a['wg_f10']), np.asarray(a['wb_f10'])

def alloc_fixed_bull(a):
    """Z4: 固定 bull preset (TQQQ 80% / TMF 10% / 2036 10%)
    wn=NASDAQ, wg=2036(Gold-position), wb=Bond(TMF) の対応"""
    n = len(a['close'])
    return np.full(n, 0.80), np.full(n, 0.10), np.full(n, 0.10)

def alloc_regime_switch(a):
    """Z5: raw_a2 > 0.5 → bull (80/10/10), else balanced (50/25/25)"""
    a2_arr = np.nan_to_num(np.asarray(a['raw_a2'].values if hasattr(a['raw_a2'], 'values') else a['raw_a2']), nan=0.0)
    bull_mask = (a2_arr > 0.5)
    wn = np.where(bull_mask, 0.80, 0.50)
    wg = np.where(bull_mask, 0.10, 0.25)
    wb = np.where(bull_mask, 0.10, 0.25)
    return wn, wg, wb


VARIANT_SPECS = {
    'DH-Z1 (DH base + binary vz)':               ('vz_binary',  'dh_base'),
    'DH-Z2 (F10 e tilt + binary vz)':            ('vz_binary',  'f10'),
    'DH-Z3 (DH base + composite)':               ('composite',  'dh_base'),
    'DH-Z4 (Fixed bull 80/10/10 + binary vz)':   ('vz_binary',  'fixed_bull'),
    'DH-Z5 (Regime preset switch, always IN)':   ('always_in',  'regime'),
}

TIMING_FN = {'vz_binary': hold_mask_vz_binary, 'composite': hold_mask_composite,
              'always_in': hold_mask_always}
ALLOC_FN = {'dh_base': alloc_dh_base, 'f10': alloc_f10_tilt,
             'fixed_bull': alloc_fixed_bull, 'regime': alloc_regime_switch}


def build_variant(a, timing_key, alloc_key):
    """variant 1 つの NAV を構築。mask × allocation を pre-multiply して既存
    build_dh_nav_with_cost に渡す。レバ操作なし。"""
    mask = TIMING_FN[timing_key](a)
    wn, wg, wb = ALLOC_FN[alloc_key](a)
    wn = np.asarray(wn, dtype=float); wg = np.asarray(wg, dtype=float); wb = np.asarray(wb, dtype=float)
    lev_raw = np.asarray(a['lev_raw'], dtype=float)

    wn_m = wn * mask
    wg_m = wg * mask
    wb_m = wb * mask
    lev_m = lev_raw * mask

    nav_adj, yr_cost = build_dh_nav_with_cost(
        a['close'], lev_m, wn_m, wg_m, wb_m,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    return nav_adj, yr_cost, mask, wn_m, lev_m


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G22A: DH 「配分 × タイミング 2 軸変動」5 変種 NAV (moderate cost = 0.10%)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']

    # REF: 現行 DH (timing=常時 IN, 配分=DH base, mask=ones)
    nav_ref, cost_ref, _, _, _ = build_variant(a, 'always_in', 'dh_base')
    print(f'\n[REF] DH Dyn 2x3x [A] (現行 = always IN + DH base)')
    print(f'  NAV final: {nav_ref.iloc[-1]:.2f}')
    print(f'  Avg yr cost: {cost_ref*100:.3f}% / yr')

    navs = {'DH-REF (現行)': nav_ref}
    sanity_rows = []

    print('\n[Variants]')
    for label, (tkey, akey) in VARIANT_SPECS.items():
        nav, cost, mask, wn_m, lev_m = build_variant(a, tkey, akey)
        peak_lev_eff = float(np.nanmax(wn_m * lev_m * 3.0))
        hold_ratio = float(mask.mean()) * 100
        # peak leverage 検証 (mandatory)
        assert peak_lev_eff <= 3.0 + 1e-9, (
            f'PEAK LEV VIOLATION {label}: peak_lev_eff={peak_lev_eff:.4f} > 3.0')
        navs[label] = nav
        sanity_rows.append(dict(
            Strategy=label, timing=tkey, allocation=akey,
            hold_ratio_pct=hold_ratio,
            peak_lev_eff=peak_lev_eff,
            nav_final=float(nav.iloc[-1]),
            yr_cost_pct=cost*100,
        ))
        print(f'  {label:48s}  hold={hold_ratio:5.1f}%  peak_lev={peak_lev_eff:.2f}x  '
              f'NAV={nav.iloc[-1]:>10,.0f}  cost={cost*100:.3f}%/yr')

    # CSV: NAVs
    out = pd.DataFrame(navs)
    out.index.name = 'date'
    csv = os.path.join(BASE, 'g22a_dh_alloc_timing_navs.csv')
    out.to_csv(csv, float_format='%.6f')
    print(f'\n→ NAVs CSV: {csv}')
    print(f'  shape: {out.shape}')

    # CSV: sanity summary
    sanity_df = pd.DataFrame(sanity_rows)
    csv_s = os.path.join(BASE, 'g22a_dh_alloc_timing_sanity.csv')
    sanity_df.to_csv(csv_s, index=False, float_format='%.4f')
    print(f'→ Sanity CSV: {csv_s}')

    # Distinctness check
    print('\n[NAV final distinctness vs REF]')
    ref_final = float(nav_ref.iloc[-1])
    for col in out.columns:
        final = float(out[col].iloc[-1])
        diff_pct = (final / ref_final - 1.0) * 100
        flag = '✓REF' if col == 'DH-REF (現行)' else (
            '✓DISTINCT' if abs(diff_pct) > 0.1 else '⚠IDENTICAL')
        print(f'  {col:48s}  final={final:>12,.0f}  vs REF {diff_pct:+7.2f}%  {flag}')


if __name__ == '__main__':
    main()
