"""G23A: DH 改善精製版 3 候補 NAV 構築 (W1, W2, W3)
=================================================================
DH-Z2 (F10 ε + binary vz) の OOS 改善時に IS が大幅低下した問題を、
より洗練された timing/allocation で解消する 3 候補を検証。

DH-W1 (Asymmetric+Hysteresis):
  状態機械。Enter HOLD when lev_mod_065 ≥ 0.7, Exit when ≤ 0.3、中間は現状態維持
  HOLD 時 DH base 配分、OUT 時 cash
  → Z2 の whipsaw (2021/2025 で bull 逸失) を解消する狙い

DH-W2 (Z2 + TMF rotation):
  Z2 と同じ binary vz HOLD mask。HOLD 時 F10 tilt、**OUT 時 100% TMF**
  → risk-off 時の bond rally (2008/2020 style) を取りに行く

DH-W3 (3-state allocation preset switch):
  raw_a2 > 0.7 ∧ vz < 0 → bull 80/10/10 (full confidence)
  raw_a2 ≤ 0.3 ∨ vz > 0.65 → 100% TMF (defensive rotation)
  else → DH base (middle confidence, preserve DH)
  → preset 切替で IS bull capture と OOS bear defense を両立

商品: TQQQ + TMF + WisdomTree 2036（変更なし）
コスト前提: DH per_unit_cost = 0.0010 (moderate), ETF 税 ×0.8273
peak_lev_eff = max(wn × lev_raw × 3) ≤ 3.0x を assert で検証
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
ENTER_THR_W1 = 0.7
EXIT_THR_W1 = 0.3


def _get_a2(a):
    return a['raw_a2'].values if hasattr(a['raw_a2'], 'values') else a['raw_a2']


# ----- timing/allocation builders -----

def hold_mask_W1(a, enter_thr=ENTER_THR_W1, exit_thr=EXIT_THR_W1):
    """Asymmetric + hysteresis state machine.
    Start OUT. Enter HOLD when lev_mod ≥ enter_thr (≥0.7 = confirmed calm).
    Exit OUT when lev_mod ≤ exit_thr (≤0.3 = confirmed regime change).
    Between: stay in current state (hysteresis band)."""
    lm = np.nan_to_num(np.asarray(a['lev_mod_065']), nan=0.0)
    n = len(lm)
    mask = np.zeros(n, dtype=float)
    state = 0  # 0=OUT, 1=HOLD; start OUT
    for i in range(n):
        if state == 0 and lm[i] >= enter_thr:
            state = 1
        elif state == 1 and lm[i] <= exit_thr:
            state = 0
        mask[i] = float(state)
    return mask


def build_W1(a):
    """W1: hysteresis HOLD/OUT × DH base allocation"""
    mask = hold_mask_W1(a)
    wn = np.asarray(a['wn_A']) * mask
    wg = np.asarray(a['wg_A']) * mask
    wb = np.asarray(a['wb_A']) * mask
    lev_raw = np.asarray(a['lev_raw']) * mask
    nav, cost = build_dh_nav_with_cost(
        a['close'], lev_raw, wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    return nav, cost, mask, wn, lev_raw


def build_W2(a):
    """W2: Z2 binary HOLD mask × (HOLD: F10 tilt / OUT: 100% TMF)"""
    lm = np.nan_to_num(np.asarray(a['lev_mod_065']), nan=0.0)
    mask = (lm >= 0.5).astype(float)
    out_mask = 1.0 - mask
    # HOLD: F10 tilted weights. OUT: wn=0, wg=0, wb=1
    wn = np.asarray(a['wn_f10']) * mask
    wg = np.asarray(a['wg_f10']) * mask
    wb_held = np.asarray(a['wb_f10']) * mask
    wb = wb_held + out_mask  # +1 when OUT
    lev_raw = np.asarray(a['lev_raw']) * mask  # 0 when OUT
    nav, cost = build_dh_nav_with_cost(
        a['close'], lev_raw, wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    return nav, cost, mask, wn, lev_raw


def build_W3(a):
    """W3: 3-state preset switch
    - High (raw_a2>0.7 ∧ vz<0): TQQQ 80% / 2036 10% / TMF 10%, lev_raw=1.0
    - Low (raw_a2≤0.3 ∨ vz>0.65): TQQQ 0% / 2036 0% / TMF 100%, lev_raw=0
    - Medium: DH base wn/wb/wg with DH lev_raw"""
    raw_a2 = _get_a2(a)
    vz = np.asarray(a['vz_arr'])
    n = len(a['close'])

    high = (raw_a2 > 0.7) & (vz < 0)
    low  = (raw_a2 <= 0.3) | (vz > 0.65)
    med  = ~(high | low)

    wn_dh = np.asarray(a['wn_A']); wg_dh = np.asarray(a['wg_A']); wb_dh = np.asarray(a['wb_A'])
    lev_dh = np.asarray(a['lev_raw'])

    wn = np.where(high, 0.80, np.where(med, wn_dh, 0.0))
    wg = np.where(high, 0.10, np.where(med, wg_dh, 0.0))
    wb = np.where(high, 0.10, np.where(med, wb_dh, 1.0))
    lev_raw = np.where(high, 1.0, np.where(med, lev_dh, 0.0))

    nav, cost = build_dh_nav_with_cost(
        a['close'], lev_raw, wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    # for sanity: mask = 1 if high or med, 0 if low
    sanity_mask = (~low).astype(float)
    return nav, cost, sanity_mask, wn, lev_raw


VARIANT_BUILDERS = {
    'DH-W1 (Asymm+Hysteresis, DH base)': build_W1,
    'DH-W2 (Z2 binary + TMF rotation)':   build_W2,
    'DH-W3 (3-state preset switch)':       build_W3,
}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G23A: DH 改善精製版 3 候補 (W1, W2, W3) NAV (moderate cost = 0.10%)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']

    # REF: 現行 DH
    from g22a_dh_alloc_timing_variants import build_variant as build_z_variant
    nav_ref, cost_ref, _, _, _ = build_z_variant(a, 'always_in', 'dh_base')
    nav_z2, cost_z2, _, _, _ = build_z_variant(a, 'vz_binary', 'f10')
    print(f'\n[Reference]')
    print(f'  DH-REF (現行): NAV={nav_ref.iloc[-1]:,.0f}, cost={cost_ref*100:.3f}%/yr')
    print(f'  DH-Z2 (v4 採用): NAV={nav_z2.iloc[-1]:,.0f}, cost={cost_z2*100:.3f}%/yr')

    navs = {'DH-REF (現行)': nav_ref, 'DH-Z2 (v4 採用)': nav_z2}
    sanity_rows = []

    print('\n[New candidates W1/W2/W3]')
    for label, build_fn in VARIANT_BUILDERS.items():
        nav, cost, mask, wn_m, lev_m = build_fn(a)
        peak_lev = float(np.nanmax(wn_m * lev_m * 3.0))
        hold_ratio = float(mask.mean()) * 100
        assert peak_lev <= 3.0 + 1e-9, f'PEAK LEV VIOLATION {label}: {peak_lev:.4f}'
        navs[label] = nav
        sanity_rows.append(dict(
            Strategy=label,
            hold_ratio_pct=hold_ratio,
            peak_lev_eff=peak_lev,
            nav_final=float(nav.iloc[-1]),
            yr_cost_pct=cost*100,
        ))
        print(f'  {label:46s}  hold={hold_ratio:5.1f}%  peak_lev={peak_lev:.2f}x  '
              f'NAV={nav.iloc[-1]:>10,.0f}  cost={cost*100:.3f}%/yr')

    out = pd.DataFrame(navs)
    out.index.name = 'date'
    csv = os.path.join(BASE, 'g23a_dh_refinement_navs.csv')
    out.to_csv(csv, float_format='%.6f')
    print(f'\n→ NAVs CSV: {csv}')

    sanity_df = pd.DataFrame(sanity_rows)
    csv_s = os.path.join(BASE, 'g23a_dh_refinement_sanity.csv')
    sanity_df.to_csv(csv_s, index=False, float_format='%.4f')
    print(f'→ Sanity CSV: {csv_s}')


if __name__ == '__main__':
    main()
