"""
年次リターン比較レポート — B9候補 vs 現行ベスト (2026-05-23)
=============================================================
対象: B9-Winner / S2+LT2-N750◆ / B9-Stable / S2+LT2-N1500 / BH 1x
出力: B9_YEARLY_RETURNS_2026-05-23.md
"""

import sys, os, types

# multitasking スタブ (yfinance 依存回避)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    calc_7metrics,
    CFD_SPREAD_LOW,
    FULL_START, FULL_END, IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import (
    prepare_gold_local,
    nav_to_annual,
    rolling_nY_cagr,
)
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)
LT2_N750_FIXED  = dict(N=750,  k_lt=0.5)
LT2_N1500_FIXED = dict(N=1500, k_lt=0.5)

CONFIGS = [
    dict(label='B9-Winner',     gold_frac=0.65, wn_min=0.20, lt_N=750),
    dict(label='S2+LT2-N750◆', gold_frac=0.50, wn_min=0.30, lt_N=750),
    dict(label='B9-Stable',     gold_frac=0.60, wn_min=0.30, lt_N=750),
    dict(label='S2+LT2-N1500', gold_frac=0.50, wn_min=0.30, lt_N=1500),
    dict(label='BH 1x',         gold_frac=None, wn_min=None, lt_N=None),
]

# サニティ参照値 (CURRENT_BEST_STRATEGY.md 確定値)
REF_CAGR_OOS  = 0.3116   # S2+LT2-N750◆ (gold_frac=0.50, wn_min=0.30)
REF_SHARPE_OOS = 0.858


# ---------------------------------------------------------------------------
# ヘルパー: wn_min/gold_frac 引数化 rebalance (f1_alloc_sweep.py からコピー)
# ---------------------------------------------------------------------------

def simulate_rebalance_A_wmin(raw, vz, threshold=THRESHOLD,
                               wn_min: float = 0.30, wn_max: float = 0.90,
                               gold_frac: float = 0.50):
    """simulate_rebalance_A と同一ロジックだが wn_min/wn_max/gold_frac を引数化。"""
    n = len(raw)
    raw_v = raw.values; vz_v = vz.values
    lev = np.zeros(n); wn = np.zeros(n); wg = np.zeros(n); wb = np.zeros(n)
    cur_lev = raw_v[0]
    cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[0], 0), wn_min, wn_max))
    cur_wg = (1 - cur_wn) * gold_frac
    cur_wb = (1 - cur_wn) * (1 - gold_frac)
    lev[0] = cur_lev; wn[0] = cur_wn; wg[0] = cur_wg; wb[0] = cur_wb
    n_trades = 0
    for i in range(1, n):
        t = raw_v[i]
        if (t == 0 and cur_lev > 0) or (cur_lev == 0 and t > 0) or abs(t - cur_lev) > threshold:
            cur_lev = t
            cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[i], 0), wn_min, wn_max))
            cur_wg = (1 - cur_wn) * gold_frac
            cur_wb = (1 - cur_wn) * (1 - gold_frac)
            n_trades += 1
        lev[i] = cur_lev; wn[i] = cur_wn; wg[i] = cur_wg; wb[i] = cur_wb
    return lev, wn, wg, wb, n_trades


# ---------------------------------------------------------------------------
# ヘルパー: 年次リターン / 統計
# ---------------------------------------------------------------------------

def nav_to_annual_returns(nav: pd.Series, dates: pd.Series) -> pd.Series:
    """日次NAV → 年次リターン (%) — gen_s2_yearly_returns.py と同一実装"""
    df = pd.DataFrame({'nav': nav.values, 'dt': dates.values})
    df['year'] = pd.to_datetime(df['dt']).dt.year
    last_val = df.groupby('year')['nav'].last()
    ret = (last_val / last_val.shift(1) - 1).dropna()
    first_year = last_val.index[0]
    ret[first_year] = last_val[first_year] / nav.iloc[0] - 1
    return (ret * 100).round(1).sort_index()


def compute_worst5y(nav, trading_days=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    r = (s / s.shift(trading_days * 5)) ** 0.2 - 1
    return float(r.dropna().min())


def calc_stats_extended(nav: pd.Series, dates: pd.Series) -> dict:
    """FULL / IS / OOS 統計 + Worst5Y, 年次サマリー"""
    m = calc_7metrics(nav, dates)
    ar = nav_to_annual_returns(nav, dates)
    # Worst5Y via rolling nY CAGR (b1_s2_lt2.py パターン)
    ann_series = nav_to_annual(nav, dates)
    r5 = rolling_nY_cagr(ann_series, 5)
    worst5y_cagr = float(r5.min()) if len(r5) > 0 else np.nan
    return {
        'CAGR_FULL':   m.get('CAGR_FULL',   np.nan) * 100,
        'CAGR_IS':     m.get('CAGR_IS',     np.nan) * 100,
        'CAGR_OOS':    m.get('CAGR_OOS',    np.nan) * 100,
        'Sharpe_FULL': m.get('Sharpe_FULL', np.nan),
        'Sharpe_IS':   m.get('Sharpe_IS',   np.nan),
        'Sharpe_OOS':  m.get('Sharpe_OOS',  np.nan),
        'MaxDD_FULL':  m.get('MaxDD_FULL',  np.nan) * 100,
        'Worst5Y':     worst5y_cagr * 100 if not np.isnan(worst5y_cagr) else np.nan,
        'Median':      float(ar.median()),
        'Max':         float(ar.max()),
        'Min':         float(ar.min()),
        'Plus':        int((ar > 0).sum()),
        'Minus':       int((ar <= 0).sum()),
    }


# ---------------------------------------------------------------------------
# Markdown 生成
# ---------------------------------------------------------------------------

def generate_md(all_annual: dict, all_stats: dict, data_info: dict) -> str:
    strat_names = list(all_annual.keys())
    all_years   = sorted(set().union(*[set(s.index) for s in all_annual.values()]))

    lines = []
    lines.append('# B9候補 vs 現行ベスト 年次リターン比較 (1974-2026)')
    lines.append('')
    lines.append('作成日: 2026-05-23')
    lines.append('EVALUATION_STANDARD: v1.1 | コスト: Scenario D')
    lines.append('')
    lines.append(f'**生成日**: 2026-05-23')
    lines.append(f'**データ期間**: {data_info["start"]} 〜 {data_info["end"]}')
    lines.append(f'**IS期間**: {IS_START} 〜 {IS_END}')
    lines.append(f'**OOS期間**: {OOS_START} 〜 {data_info["end"]}')
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- 補正条件 ---
    lines.append('## 補正条件')
    lines.append('')
    lines.append('| 戦略 | コスト条件 |')
    lines.append('|------|-----------|')
    lines.append('| B9-Winner / S2+LT2-N750◆ / B9-Stable / S2+LT2-N1500 | Scenario D: CFD (L-1)×SOFR + (L-1)×0.20%スプレッド |')
    lines.append('| BH 1x | 補正なし（ベンチマーク） |')
    lines.append('| SOFR proxy | DTB3 (FRED 3M T-bill) |')
    lines.append('| LT2 modeB | lt_bias = signal_to_bias(lt_sig, k=0.5) / lev_mod = clip(lev_A + lt_bias, 0, 1) |')
    lines.append('| S2パラメータ | target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0 |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- 統計サマリー (FULL) ---
    hdr = '| 統計量 | ' + ' | '.join(n for n in strat_names) + ' |'
    sep = '|--------|' + ':----------:|' * len(strat_names)

    lines.append('## 統計サマリー (FULL期間: 1974-2026)')
    lines.append('')
    lines.append(hdr)
    lines.append(sep)

    full_rows = [
        ('**CAGR (FULL)**', lambda s: f'{s["CAGR_FULL"]:+.2f}%'),
        ('Sharpe (FULL)',    lambda s: f'{s["Sharpe_FULL"]:.3f}'),
        ('MaxDD (FULL)',     lambda s: f'{s["MaxDD_FULL"]:.1f}%'),
        ('Worst5Y CAGR',    lambda s: f'{s["Worst5Y"]:+.2f}%'),
        ('中央値',           lambda s: f'{s["Median"]:+.1f}%'),
        ('最大',             lambda s: f'{s["Max"]:+.1f}%'),
        ('最小',             lambda s: f'{s["Min"]:+.1f}%'),
        ('プラス年数',       lambda s: str(s['Plus'])),
        ('マイナス年数',     lambda s: str(s['Minus'])),
    ]
    for label, fmt in full_rows:
        row = f'| {label} | ' + ' | '.join(fmt(all_stats[n]) for n in strat_names) + ' |'
        lines.append(row)
    lines.append('')

    # --- IS統計 ---
    lines.append(f'## IS統計 ({IS_START} 〜 {IS_END})')
    lines.append('')
    lines.append(hdr)
    lines.append(sep)
    is_rows = [
        ('**CAGR (IS)**',  lambda s: f'{s["CAGR_IS"]:+.2f}%'),
        ('Sharpe (IS)',    lambda s: f'{s["Sharpe_IS"]:.3f}'),
    ]
    for label, fmt in is_rows:
        row = f'| {label} | ' + ' | '.join(fmt(all_stats[n]) for n in strat_names) + ' |'
        lines.append(row)
    lines.append('')

    # --- OOS統計 ---
    lines.append(f'## OOS統計 ({OOS_START} 〜 {data_info["end"]})')
    lines.append('')
    lines.append(hdr)
    lines.append(sep)
    oos_rows = [
        ('**CAGR (OOS)**',  lambda s: f'{s["CAGR_OOS"]:+.2f}%'),
        ('Sharpe (OOS)',    lambda s: f'{s["Sharpe_OOS"]:.3f}'),
    ]
    for label, fmt in oos_rows:
        row = f'| {label} | ' + ' | '.join(fmt(all_stats[n]) for n in strat_names) + ' |'
        lines.append(row)
    lines.append('')
    lines.append('---')
    lines.append('')

    # --- 年次リターン表 ---
    lines.append('## 年次リターン表 (1974-2026) [単位: %]')
    lines.append('')
    lines.append('> `[OOS]` = OOS期間 (2021年以降)')
    lines.append('')
    yr_hdr = '| 年 | ' + ' | '.join(n for n in strat_names) + ' |'
    yr_sep = '|----|' + ':---:|' * len(strat_names)
    lines.append(yr_hdr)
    lines.append(yr_sep)

    for yr in all_years:
        yr_str = f'{yr} [OOS]' if yr >= 2021 else str(yr)
        cells = []
        for n in strat_names:
            v = all_annual[n].get(yr, np.nan)
            cells.append('—' if np.isnan(v) else f'{v:+.1f}')
        lines.append(f'| {yr_str} | ' + ' | '.join(cells) + ' |')

    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*生成スクリプト: `src/gen_b9_yearly_returns.py`*')
    lines.append(f'*データ期間: {data_info["start"]} 〜 {data_info["end"]}*')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('B9 Yearly Returns -- B9-Winner / S2+LT2-N750◆ / B9-Stable / S2+LT2-N1500 / BH 1x')
    print('実行日: 2026-05-23')
    print('=' * 70)

    # S1: データロード
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days)')

    # S2: 共有資産（1回のみ生成）
    print('\nBuilding shared assets...')
    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Shared assets done.')

    # S3: DH Dyn シグナル（1回のみ）
    print('Building DH Dyn signal...')
    raw_a2, vz = build_a2_signal(close, ret)
    # ref: simulate_rebalance_A (gold_frac=0.50, wn_min=0.30 相当)
    lev_A_ref, wn_A_ref, wg_A_ref, wb_A_ref, n_tr_ref = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_years = n / TRADING_DAYS
    print(f'  DH Dyn signal: {n_tr_ref} trades, {n_tr_ref / n_years:.1f}/yr')

    # S4: S2 CFD レバレッジ系列（1回のみ）
    print('Building S2 CFD leverage series...')
    L_s2 = compute_L_s2_vz_gated(
        ret, vz,
        target_vol=S2_FIXED['target_vol'],
        k_vz=S2_FIXED['k_vz'],
        gate_min=S2_FIXED['gate_min'],
        n=S2_FIXED['n'],
        l_min=S2_FIXED['l_min'],
        l_max=S2_FIXED['l_max'],
        step=S2_FIXED['step'],
    )
    print(f'  L_s2 range: [{L_s2.min():.1f}, {L_s2.max():.1f}]  median={L_s2.median():.1f}')

    # S5: LT2 シグナルキャッシュ (N=750, N=1500)
    print('Building LT2 signals...')
    lt_sig_750  = build_lt_signal(close, 'LT2', 750)
    lt_sig_1500 = build_lt_signal(close, 'LT2', 1500)
    lt_bias_750  = signal_to_bias(lt_sig_750,  0.5)
    lt_bias_1500 = signal_to_bias(lt_sig_1500, 0.5)
    print(f'  LT2-N750  bias range: [{lt_bias_750.min():+.3f}, {lt_bias_750.max():+.3f}]')
    print(f'  LT2-N1500 bias range: [{lt_bias_1500.min():+.3f}, {lt_bias_1500.max():+.3f}]')

    # S6: 各 CONFIG の NAV 構築
    print('\nBuilding NAVs...')
    navs = {}

    # --- S2+LT2-N750◆ (REF) をまず構築してサニティチェック ---
    cfg_ref = next(c for c in CONFIGS if c['label'] == 'S2+LT2-N750◆')
    print(f'  [REF] S2+LT2-N750◆ (gold_frac={cfg_ref["gold_frac"]}, wn_min={cfg_ref["wn_min"]})...')
    lev_ref, wn_ref, wg_ref, wb_ref, _ = simulate_rebalance_A_wmin(
        raw_a2, vz, THRESHOLD,
        wn_min=cfg_ref['wn_min'], wn_max=0.90, gold_frac=cfg_ref['gold_frac']
    )
    lev_mod_ref = apply_lt_mode_b(lev_ref, lt_bias_750, l_min=0.0, l_max=1.0)
    nav_ref = build_nav_strategy(
        close, lev_mod_ref, wn_ref, wg_ref, wb_ref, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    navs['S2+LT2-N750◆'] = nav_ref

    # Sanity check: simulate_rebalance_A vs simulate_rebalance_A_wmin(wn_min=0.30, gold_frac=0.50)
    m_ref_quick = calc_7metrics(nav_ref, dates)
    sanity_diff = (m_ref_quick.get('CAGR_OOS', np.nan) - REF_CAGR_OOS) * 100
    sanity_ok   = abs(sanity_diff) <= 0.10
    print(f'  [SANITY] S2+LT2-N750◆ CAGR_OOS = {m_ref_quick.get("CAGR_OOS",np.nan)*100:+.2f}%  '
          f'(ref +31.16%, diff {sanity_diff:+.2f} pp)  {"OK" if sanity_ok else "WARN"}')

    # --- 残りの CONFIG をループ ---
    for cfg in CONFIGS:
        label = cfg['label']
        if label == 'S2+LT2-N750◆':
            continue  # 上で構築済み

        print(f'  [{label}]...')
        if label == 'BH 1x':
            r = close.pct_change().fillna(0)
            nav_bh = (1 + r).cumprod()
            nav_bh.attrs['blowup_days'] = 0
            navs[label] = nav_bh
            continue

        # 通常の S2+LT2 戦略
        lt_bias = lt_bias_750 if cfg['lt_N'] == 750 else lt_bias_1500
        lev_c, wn_c, wg_c, wb_c, _ = simulate_rebalance_A_wmin(
            raw_a2, vz, THRESHOLD,
            wn_min=cfg['wn_min'], wn_max=0.90, gold_frac=cfg['gold_frac']
        )
        lev_mod_c = apply_lt_mode_b(lev_c, lt_bias, l_min=0.0, l_max=1.0)
        nav_c = build_nav_strategy(
            close, lev_mod_c, wn_c, wg_c, wb_c, dates,
            gold_2x, bond_3x, sofr,
            nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
        )
        navs[label] = nav_c

    # CONFIGS 順に並び替え
    navs_ordered = {cfg['label']: navs[cfg['label']] for cfg in CONFIGS}

    # S7: 年次リターン & 統計
    print('\nComputing annual returns & stats...')
    all_annual = {}
    all_stats  = {}
    for name, nav in navs_ordered.items():
        ar = nav_to_annual_returns(nav, dates)
        all_annual[name] = ar
        all_stats[name]  = calc_stats_extended(nav, dates)
        s = all_stats[name]
        print(f'  {name}: CAGR_FULL={s["CAGR_FULL"]:+.2f}%  CAGR_OOS={s["CAGR_OOS"]:+.2f}%  '
              f'Sharpe_OOS={s["Sharpe_OOS"]:.3f}  MaxDD={s["MaxDD_FULL"]:.1f}%')

    # S8: サニティ詳細
    print('\n--- Sanity Check Summary ---')
    s_ref = all_stats['S2+LT2-N750◆']
    print(f'  S2+LT2-N750◆ CAGR_OOS:   {s_ref["CAGR_OOS"]:+.2f}%  (expect +31.16%)')
    print(f'  S2+LT2-N750◆ Sharpe_OOS: {s_ref["Sharpe_OOS"]:.3f}     (expect 0.858)')
    for name in ['B9-Winner', 'B9-Stable', 'S2+LT2-N1500']:
        s = all_stats[name]
        delta_sh = s['Sharpe_OOS'] - s_ref['Sharpe_OOS']
        delta_cagr = s['CAGR_OOS'] - s_ref['CAGR_OOS']
        print(f'  {name} vs N750◆: ΔSharpe_OOS={delta_sh:+.3f}  ΔCAGR_OOS={delta_cagr:+.2f}pp')

    # S9: 年次リターン抜粋コンソール表示
    print('\n--- Annual Returns (OOS 2021-2026) ---')
    names_short = ['B9-Win', 'N750◆', 'B9-Stab', 'N1500', 'BH1x']
    header = f'{"Year":<12}' + ''.join(f'{n:>10}' for n in names_short)
    print(header)
    print('-' * (12 + 10 * 5))
    oos_years = [yr for yr in sorted(all_annual['BH 1x'].index) if yr >= 2021]
    for yr in oos_years:
        yr_str = f'{yr}[OOS]'
        row = f'{yr_str:<12}'
        for name in [cfg['label'] for cfg in CONFIGS]:
            v = all_annual[name].get(yr, float('nan'))
            row += f'{v:>+9.1f}%' if not np.isnan(v) else f'{"—":>10}'
        print(row)

    # S10: 最悪年
    print('\n--- Worst Year per Strategy ---')
    for name in [cfg['label'] for cfg in CONFIGS]:
        ar = all_annual[name]
        worst_yr = ar.idxmin()
        worst_val = ar[worst_yr]
        print(f'  {name}: worst year = {worst_yr} ({worst_val:+.1f}%)')

    # S11: MD生成・保存
    print('\nGenerating MD report...')
    data_info = {'start': str(dates.iloc[0].date()), 'end': str(dates.iloc[-1].date())}
    md  = generate_md(all_annual, all_stats, data_info)
    out = os.path.join(BASE, 'B9_YEARLY_RETURNS_2026-05-23.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
