"""
G16: WFA — Legacy 戦略拡張 (S2_VZGated alone)
================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-06-01)

目的:
  v4 統合版 §2-B の Legacy 4戦略のうち、SBI CFD フレームワーク内で
  WFA 実施可能な S2_VZGated 単独 を追加で WFA 再評価。
  DH Dyn 2x3x [A] / Ens2 は TQQQ ETF フレームワークなので別途扱い。
  S2+LT2 k=0.5 modeB は g14-g3-REF-N750 として既実施。

新規追加 (2戦略):
  S2-alone-lmax7 : S2_VZGated 単独 (LT2 なし, l_max=7.0)
  S2-alone-lmax5 : S2_VZGated 単独 (LT2 なし, l_max=5.0) — 比較用

g14 のインフラを完全に流用し、lev_mod として LT2 オーバーレイ無しの
lev_raw クリップを使用するのみ差分。

出力:
  - g16_wfa_legacy_ext_per_window.csv
  - g16_wfa_legacy_ext_summary.csv
"""

import sys
import os
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# g14 インフラを完全流用
from g14_wfa_sbi_cfd import (
    load_shared_assets,
    generate_windows,
    compute_window_metrics,
    compute_summary_stats,
    evaluate_criteria,
    _make_nav,
    S2_LMAX7, S2_LMAX5,
    SBI_CFD_SPREAD,
    TODAY,
    BASE,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b


def build_s2_alone(a, l_max):
    """S2_VZGated 単独 (LT2 オーバーレイ無し) で NAV 構築。

    lev_mod として LT2 bias=0 の clip(lev_raw, 0, 1) を使用。
    """
    # LT2 bias=0 で apply_lt_mode_b: 実質 clip(lev_raw, 0, 1)
    zero_bias = pd.Series(np.zeros(len(a['lt_sig_raw'])), index=a['lt_sig_raw'].index)
    lev_mod_alone = apply_lt_mode_b(a['lev_raw'], zero_bias, l_min=0.0, l_max=1.0)

    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_LMAX7, 'l_max': l_max} if l_max != 7.0 else S2_LMAX7)

    nav = _make_nav(
        a['close'], lev_mod_alone, a['wn_A'], a['wg_A'], a['wb_A'],
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
        L_s2.values,
    )
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


STRATEGY_BUILDERS = [
    ('g16-S2-VZG-alone-lmax7', lambda a: build_s2_alone(a, l_max=7.0)),
    ('g16-S2-VZG-alone-lmax5', lambda a: build_s2_alone(a, l_max=5.0)),
]


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 72)
    print(f'G16: WFA — Legacy 拡張 (S2_VZGated alone, SBI CFD spread={SBI_CFD_SPREAD})')
    print(f'実行日: {TODAY}')
    print('=' * 72)

    print('\n[S1] Loading shared assets (via g14)...')
    assets = load_shared_assets()
    dates  = assets['dates']

    print('\n[S2] Generating windows...')
    windows = generate_windows(dates)
    print(f'  {len(windows)} windows: '
          f'{windows[0]["start_date"].date()} ... {windows[-1]["end_date"].date()}')

    all_pw_rows = []
    all_sm_rows = []
    results = {}

    print(f'\n[S3] Evaluating {len(STRATEGY_BUILDERS)} legacy strategies...')
    print('-' * 72)
    for sid, builder in STRATEGY_BUILDERS:
        print(f'  [{sid}] ', end='', flush=True)
        try:
            nav, wn, wb, lev_arr = builder(assets)
        except Exception as exc:
            print(f'FAILED: {exc}')
            raise

        per_rows = []
        for w in windows:
            m = compute_window_metrics(nav, w, wn=wn, wb=wb, lev_arr=lev_arr)
            m.update(dict(
                strategy=sid, window_id=w['window_id'],
                start_date=w['start_date'], end_date=w['end_date'],
                short_flag=w['short_flag'],
            ))
            per_rows.append(m)
            all_pw_rows.append(m)

        per_df  = pd.DataFrame(per_rows)
        summary = compute_summary_stats(per_df)
        verdict, crits = evaluate_criteria(summary)
        results[sid] = dict(per_window=per_df, summary=summary,
                            verdict=verdict, criteria=crits)
        sm_row = {'strategy': sid, 'verdict': verdict,
                  **summary, **{f'crit_{k}': v for k, v in crits.items()}}
        all_sm_rows.append(sm_row)

        ci_lo  = summary.get('WFA_CI95_lo', np.nan)
        ci_hi  = summary.get('WFA_CI95_hi', np.nan)
        wfe    = summary.get('WFA_WFE', np.nan)
        mean_c = summary.get('mean_CAGR', np.nan)
        tp     = summary.get('t_pvalue', np.nan)
        print(f'CAGR={mean_c*100:+6.2f}%  '
              f'CI95=[{ci_lo*100:+6.2f}%, {ci_hi*100:+6.2f}%]  '
              f't_p={tp:.4f}  WFE={wfe:+.3f}  => {verdict}')

    print('-' * 72)

    pw_path = os.path.join(BASE, 'g16_wfa_legacy_ext_per_window.csv')
    sm_path = os.path.join(BASE, 'g16_wfa_legacy_ext_summary.csv')
    pd.DataFrame(all_pw_rows).to_csv(pw_path, index=False, float_format='%.6f')
    pd.DataFrame(all_sm_rows).to_csv(sm_path, index=False, float_format='%.6f')
    print(f'\n  Per-window: {pw_path}')
    print(f'  Summary:    {sm_path}')

    print('\nDone.')
    return results


if __name__ == '__main__':
    main()
