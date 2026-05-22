"""H4 wg/wb 比率スイープ (2026-05-17)
====================================
H4 (TOCOM金先物 × TMF軽減) のパラメータグリッドサーチ。
採用基準: CAGR_IS/OOS≥25%, Sharpe_IS/OOS≥0.70, Worst5Y≥0%, IS-OOS Gap≤8pp

グリッド:
  wg_frac ∈ {0.30, 0.35, ..., 0.80}  (11値)
  L_g     ∈ {2.0, 3.0}               (2値)
  bond    ∈ {TMF_3x_nodrag, TMF_3x_drag} (2値)
  合計: 44コンボ"""

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
    TRADING_DAYS, THRESHOLD, SWAP_SPREAD,
)
from test_portfolio_diversification import prepare_gold_data
from cfd_leverage_backtest import (
    build_nav_strategy, calc_7metrics, CFD_SPREAD_LOW,
)
from sleeves_extended import (
    build_gold_tocom, build_bond_3x_with_drag,
)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
OUT_MD    = os.path.join(BASE, 'H4_WGWB_SWEEP_2026-05-17.md')
OUT_CSV   = os.path.join(BASE, 'H4_WGWB_SWEEP_2026-05-17.csv')

WG_FRACTIONS  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
L_G_VALUES    = [2.0, 3.0]
BOND_VARIANTS = ['TMF_3x_nodrag', 'TMF_3x_drag']

CAGR_IS_MIN    = 0.25
CAGR_OOS_MIN   = 0.25
SHARPE_IS_MIN  = 0.70
SHARPE_OOS_MIN = 0.70
WORST5Y_MIN    = 0.00
GAP_MAX        = 8.0


def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v*100:+.{d}f}%'


def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{d}f}'


def make_weights(wn_A, wg_frac):
    rest = 1.0 - wn_A
    wg   = rest * wg_frac
    wb   = rest * (1.0 - wg_frac)
    return wn_A, wg, wb


def generate_report(df_sorted, passing, sanity_row=None):
    lines = []
    lines.append('# H4 wg/wb 比率スイープ結果')
    lines.append('')
    lines.append('作成日: 2026-05-17')
    lines.append('最終更新日: 2026-05-17')
    lines.append('')
    lines.append('参照: [H1_H5_SUMMARY_2026-05-17.md](H1_H5_SUMMARY_2026-05-17.md)')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 採用基準 (更新版)')
    lines.append('')
    lines.append('| 指標 | 基準 |')
    lines.append('|---|---|')
    lines.append(f'| CAGR_IS | ≥ {CAGR_IS_MIN*100:.0f}% |')
    lines.append(f'| CAGR_OOS | ≥ {CAGR_OOS_MIN*100:.0f}% |')
    lines.append(f'| Sharpe_IS | ≥ {SHARPE_IS_MIN:.2f} |')
    lines.append(f'| Sharpe_OOS | ≥ {SHARPE_OOS_MIN:.2f} |')
    lines.append(f'| Worst5Y | ≥ {WORST5Y_MIN*100:.0f}% |')
    lines.append(f'| IS-OOS Gap | ≤ {GAP_MAX:.0f}pp |')
    lines.append('')

    if sanity_row is not None:
        lines.append('## サニティチェック (H4ベースライン再現)')
        lines.append('')
        lines.append('(wg_frac=0.65, L_g=3.0, TMF_3x_nodrag は既存H4と同一構成)')
        lines.append('')
        lines.append(f'- Sharpe_OOS: {_ff(sanity_row.get("Sharpe_OOS"))} (期待値 ≈0.837)')
        lines.append(f'- Worst5Y: {_fp(sanity_row.get("Worst5Y"))} (期待値 ≈-2.24%)')
        lines.append('')

    lines.append('## グリッド仕様')
    lines.append('')
    lines.append(f'- wg_frac: {WG_FRACTIONS} ({len(WG_FRACTIONS)}値)')
    lines.append(f'- L_g: {L_G_VALUES} ({len(L_G_VALUES)}値)')
    lines.append(f'- Bond: {BOND_VARIANTS} ({len(BOND_VARIANTS)}値)')
    lines.append(f'- 合計: {len(WG_FRACTIONS)*len(L_G_VALUES)*len(BOND_VARIANTS)}コンボ')
    lines.append('')

    lines.append('## 全結果 (Sharpe_OOS 降順)')
    lines.append('')
    lines.append('| wg_frac | L_g | Bond | CAGR_IS | CAGR_OOS | Sh_IS | Sh_OOS | MaxDD | W5Y | Gap | Pass |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in df_sorted.iterrows():
        p = '✅' if r.pass_all else ('⚡' if r.pass_w5_sh else '❌')
        lines.append(
            f'| {r.wg_frac:.2f} | {r.L_g:.1f}x | {r.bond_variant} '
            f'| {_fp(r.CAGR_IS)} | {_fp(r.CAGR_OOS)} '
            f'| {_ff(r.Sharpe_IS)} | {_ff(r.Sharpe_OOS)} '
            f'| {_fp(r.MaxDD_FULL)} | {_fp(r.Worst5Y)} '
            f'| {r.IS_OOS_Gap:.1f}pp | {p} |'
        )
    lines.append('')
    lines.append('凡例: ✅ = 全基準パス, ⚡ = Sharpe+W5Y パス (CAGR/Gap 未達), ❌ = 不採用')
    lines.append('')

    if not passing.empty:
        lines.append('## 採用候補 (全基準パス)')
        lines.append('')
        for _, r in passing.iterrows():
            lines.append(f'### wg_frac={r.wg_frac:.2f}, L_g={r.L_g:.1f}x, {r.bond_variant}')
            lines.append('')
            lines.append(f'- CAGR: IS {_fp(r.CAGR_IS)}, OOS {_fp(r.CAGR_OOS)}, FULL {_fp(r.CAGR_FULL)}')
            lines.append(f'- Sharpe: IS {_ff(r.Sharpe_IS)}, OOS {_ff(r.Sharpe_OOS)}')
            lines.append(f'- MaxDD: {_fp(r.MaxDD_FULL)}, Worst5Y: {_fp(r.Worst5Y)}, Gap: {r.IS_OOS_Gap:.1f}pp')
            lines.append('')
    else:
        lines.append('## 採用候補なし — ニアミス分析')
        lines.append('')
        lines.append('全基準を同時に満たすコンボなし。')
        lines.append('')
        lines.append('### Worst5Y ニアミスTop5 (Worst5Y 降順)')
        lines.append('')
        lines.append('| wg_frac | L_g | Bond | Sharpe_OOS | Worst5Y | CAGR_OOS | Gap |')
        lines.append('|---|---|---|---|---|---|---|')
        top5_w5 = df_sorted.nlargest(5, 'Worst5Y')
        for _, r in top5_w5.iterrows():
            lines.append(
                f'| {r.wg_frac:.2f} | {r.L_g:.1f}x | {r.bond_variant} '
                f'| {_ff(r.Sharpe_OOS)} | {_fp(r.Worst5Y)} '
                f'| {_fp(r.CAGR_OOS)} | {r.IS_OOS_Gap:.1f}pp |'
            )
        lines.append('')
        lines.append('### Sharpe_OOS ニアミスTop5 (Worst5Y≥-1%条件付き)')
        lines.append('')
        lines.append('| wg_frac | L_g | Bond | Sharpe_OOS | Worst5Y | CAGR_OOS | Gap |')
        lines.append('|---|---|---|---|---|---|---|')
        cond = df_sorted[df_sorted.Worst5Y >= -0.01].nlargest(5, 'Sharpe_OOS')
        for _, r in cond.iterrows():
            lines.append(
                f'| {r.wg_frac:.2f} | {r.L_g:.1f}x | {r.bond_variant} '
                f'| {_ff(r.Sharpe_OOS)} | {_fp(r.Worst5Y)} '
                f'| {_fp(r.CAGR_OOS)} | {r.IS_OOS_Gap:.1f}pp |'
            )
        lines.append('')

    lines.append('## 軸効果分析')
    lines.append('')
    lines.append('### A: wg_frac別 平均Worst5Y (L_g=3.0, nodrag固定)')
    lines.append('')
    lines.append('| wg_frac | mean Sharpe_OOS | mean Worst5Y |')
    lines.append('|---|---|---|')
    subset = df_sorted[(df_sorted.L_g == 3.0) & (df_sorted.bond_variant == 'TMF_3x_nodrag')]
    for wgf, grp in subset.groupby('wg_frac'):
        lines.append(f'| {wgf:.2f} | {_ff(grp.Sharpe_OOS.mean())} | {_fp(grp.Worst5Y.mean())} |')
    lines.append('')

    lines.append('### B: L_g別 平均Worst5Y (wg_frac=0.65固定)')
    lines.append('')
    lines.append('| L_g | Bond | Sharpe_OOS | Worst5Y |')
    lines.append('|---|---|---|---|')
    subset_b = df_sorted[df_sorted.wg_frac == 0.65]
    for _, r in subset_b.sort_values(['L_g', 'bond_variant']).iterrows():
        lines.append(f'| {r.L_g:.1f}x | {r.bond_variant} | {_ff(r.Sharpe_OOS)} | {_fp(r.Worst5Y)} |')
    lines.append('')

    lines.append('### C: Bond variant別 平均Worst5Y (全コンボ)')
    lines.append('')
    lines.append('| Bond | mean Sharpe_OOS | mean Worst5Y | mean CAGR_OOS |')
    lines.append('|---|---|---|---|')
    for bv, grp in df_sorted.groupby('bond_variant'):
        lines.append(f'| {bv} | {_ff(grp.Sharpe_OOS.mean())} | {_fp(grp.Worst5Y.mean())} | {_fp(grp.CAGR_OOS.mean())} |')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/h4_wgwb_sweep.py`*')
    lines.append('*参照: [H1_H5_SUMMARY_2026-05-17.md](H1_H5_SUMMARY_2026-05-17.md)*')
    return '\n'.join(lines)


def main():
    print('=' * 70)
    print('H4 wg/wb 比率スイープ (44コンボ)')
    print('=' * 70)

    df     = load_data(DATA_PATH)
    close  = df['Close']
    dates  = df['Date']
    returns = close.pct_change().fillna(0)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)')

    print('Loading SOFR...')
    sofr = load_sofr(dates)

    print('Building bond sleeves...')
    bond_1x = build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0
    )
    bond_3x_nodrag = build_bond_3x(bond_1x, sofr, True)
    bond_3x_drag_arr = build_bond_3x_with_drag(
        np.asarray(bond_1x, dtype=float), sofr, SWAP_SPREAD
    )

    print('Building gold sleeves...')
    gold_1x = prepare_gold_data(dates)
    gold_1x_arr = np.asarray(gold_1x, dtype=float)
    gold_navs = {
        2.0: build_gold_tocom(gold_1x_arr, 2.0, sofr, roll_cost_annual=0.02),
        3.0: build_gold_tocom(gold_1x_arr, 3.0, sofr, roll_cost_annual=0.02),
    }
    bond_navs = {
        'TMF_3x_nodrag': bond_3x_nodrag,
        'TMF_3x_drag':   bond_3x_drag_arr,
    }

    print('Building A2 signal (DH Dyn [A])...')
    raw_a2, vz, _ = build_a2_signal(close, returns, return_components=True)
    lev_A, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)

    records = []
    total = len(WG_FRACTIONS) * len(L_G_VALUES) * len(BOND_VARIANTS)
    done = 0

    print(f'\n合計 {total} コンボ実行:')
    print('-' * 70)

    for wg_frac in WG_FRACTIONS:
        for L_g in L_G_VALUES:
            for bond_var in BOND_VARIANTS:
                done += 1
                print(f'  [{done:2d}/{total}] wg={wg_frac:.2f} Lg={L_g:.1f} {bond_var:<16}', end=' ')

                wn, wg, wb = make_weights(wn_A, wg_frac)
                nav = build_nav_strategy(
                    close, lev_A, wn, wg, wb, dates,
                    gold_navs[L_g], bond_navs[bond_var],
                    sofr, 'TQQQ', 3.0, CFD_SPREAD_LOW,
                )
                m = calc_7metrics(nav, dates)

                cagr_is  = m.get('CAGR_IS',  float('nan'))
                cagr_oos = m.get('CAGR_OOS', float('nan'))
                gap = abs(cagr_is - cagr_oos) * 100 if not (np.isnan(cagr_is) or np.isnan(cagr_oos)) else float('nan')

                rec = {
                    'wg_frac':      wg_frac,
                    'L_g':          L_g,
                    'bond_variant': bond_var,
                    'CAGR_FULL':    m.get('CAGR_FULL',   float('nan')),
                    'CAGR_IS':      cagr_is,
                    'CAGR_OOS':     cagr_oos,
                    'Sharpe_FULL':  m.get('Sharpe_FULL', float('nan')),
                    'Sharpe_IS':    m.get('Sharpe_IS',   float('nan')),
                    'Sharpe_OOS':   m.get('Sharpe_OOS',  float('nan')),
                    'MaxDD_FULL':   m.get('MaxDD_FULL',  float('nan')),
                    'Worst5Y':      m.get('Worst5Y',     float('nan')),
                    'IS_OOS_Gap':   gap,
                }

                p_cagr_is  = rec['CAGR_IS']   >= CAGR_IS_MIN
                p_cagr_oos = rec['CAGR_OOS']  >= CAGR_OOS_MIN
                p_sh_is    = rec['Sharpe_IS']  >= SHARPE_IS_MIN
                p_sh_oos   = rec['Sharpe_OOS'] >= SHARPE_OOS_MIN
                p_w5       = rec['Worst5Y']    >= WORST5Y_MIN
                p_gap      = rec['IS_OOS_Gap'] <= GAP_MAX
                rec['pass_all']    = all([p_cagr_is, p_cagr_oos, p_sh_is, p_sh_oos, p_w5, p_gap])
                rec['pass_w5_sh']  = p_sh_oos and p_w5

                status = '✅' if rec['pass_all'] else ('⚡' if rec['pass_w5_sh'] else '❌')
                print(f'Sh_OOS={rec["Sharpe_OOS"]:.3f} W5Y={rec["Worst5Y"]*100:+.2f}% {status}')
                records.append(rec)

    df_results = pd.DataFrame(records)
    df_sorted  = df_results.sort_values('Sharpe_OOS', ascending=False).reset_index(drop=True)
    passing    = df_sorted[df_sorted.pass_all].copy()
    pass_w5_sh = df_sorted[df_sorted.pass_w5_sh & ~df_sorted.pass_all].copy()

    sanity = df_sorted[
        (df_sorted.wg_frac == 0.65) &
        (df_sorted.L_g == 3.0) &
        (df_sorted.bond_variant == 'TMF_3x_nodrag')
    ]
    sanity_row = sanity.iloc[0].to_dict() if not sanity.empty else None

    print('\n' + '=' * 70)
    print('サニティチェック (H4ベースライン = wg=0.65, Lg=3.0, nodrag):')
    if sanity_row:
        print(f'  Sharpe_OOS={sanity_row["Sharpe_OOS"]:.3f} (expect ≈0.837)')
        print(f'  Worst5Y={sanity_row["Worst5Y"]*100:+.2f}% (expect ≈-2.24%)')

    print(f'\n採用基準パス (全6条件): {len(passing)} / {len(df_sorted)}')
    print(f'Sharpe_OOS+Worst5Y パス: {len(passing) + len(pass_w5_sh)} / {len(df_sorted)}')

    if not passing.empty:
        best = passing.iloc[0]
        print(f'\nBEST採用候補:')
        print(f'  wg={best.wg_frac:.2f}, Lg={best.L_g:.1f}x, {best.bond_variant}')
        print(f'  Sharpe_OOS={best.Sharpe_OOS:.3f}, W5Y={best.Worst5Y*100:+.2f}%, CAGR_OOS={best.CAGR_OOS*100:+.2f}%')
    else:
        top5 = df_sorted.nlargest(5, 'Worst5Y')
        print('\nWorst5Y 上位5 (採用候補なし — ニアミス):')
        print(f'{"wg":>6} {"Lg":>5} {"bond":<18} {"Sh_OOS":>8} {"W5Y":>8} {"CAGR_OOS":>10}')
        for _, r in top5.iterrows():
            print(f'  {r.wg_frac:.2f} {r.L_g:.1f}x {r.bond_variant:<18} '
                  f'{r.Sharpe_OOS:>8.3f} {r.Worst5Y*100:>+7.2f}% {r.CAGR_OOS*100:>+9.2f}%')

    df_sorted.to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f'\nCSV saved: {OUT_CSV}')

    md = generate_report(df_sorted, passing, sanity_row)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'MD saved: {OUT_MD}')
    print('Done.')


if __name__ == '__main__':
    main()
