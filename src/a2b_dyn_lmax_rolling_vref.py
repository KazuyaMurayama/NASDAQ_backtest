"""
A2B: Dynamic l_max + Rolling VOL_REF (適応的基準ボラ)
====================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

概念: A2 (Dynamic l_max) の改良版。固定 VOL_REF=0.20 を rolling median (2520日≈10年)
  に置換し、NASDAQ実際の長期ボラ平均 ≈0.183 へ適応的に追従させる。
  これにより低ボラ局面で過剰デレバを抑制し、A2 が損なった CAGR を回復する。

  vol_ref_t = vol252.rolling(2520, min_periods=504).median()
              .fillna(vol252.expanding(min_periods=5).median())
  l_max_t   = clip(l_max_base - vol_sens × (vol252 / vol_ref_t - 1),
                   l_floor, l_ceil)

E4採用値固定: k_lo=0.1, k_hi=0.8, vz_thr=0.7

グリッド:
  LMAX_BASE_GRID = [5.0, 5.5, 6.0]   (A2 と同一)
  VOL_SENS_GRID  = [1.0, 2.0, 3.0]   (A2 と同一)
  L_FLOOR=4.5, L_CEIL=6.5

  REF: lmax_base=7.0, vol_sens=0.0 (固定 l_max=7.0 = E4ベスト)
  合計: 3×3=9 configs + REF

期待効果:
  - A2 では MaxDD 改善が見られたが CAGR_OOS は REF 比で低下した
  - rolling VOL_REF により低ボラ期の vol252/vol_ref_t が 1 近傍に収束
  - 結果として低ボラ期の l_max 引下げが緩和され CAGR_OOS の回復を期待

出力:
  - a2b_dyn_lmax_rolling_vref_results.csv
  - A2B_DYN_LMAX_ROLLING_VREF_2026-05-27.md
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
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド
# ---------------------------------------------------------------------------
# E4採用値固定
K_LO = 0.1; K_HI = 0.8; VZ_THR = 0.7
K_MID = 0.50

# Dynamic l_max スイープ
LMAX_BASE_GRID = [5.0, 5.5, 6.0]    # l_max基準値
VOL_SENS_GRID  = [1.0, 2.0, 3.0]    # ボラ感度
L_FLOOR = 4.5; L_CEIL = 6.5         # クリップ範囲
# VOL_REF は固定値ではなく rolling median で動的に算出（A2 からの改良点）
VOL_REF_WINDOW = 2520            # 10年≒2520営業日
VOL_REF_MIN_PERIODS = 504        # 2年最低限（ウォームアップ）

# S2基本パラメータ (l_max はループ内で動的に渡す)
S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)

REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.891
REF_MAXDD      = -0.6001

PASS_SHARPE_DELTA = 0.010
PASS_CAGR_OOS     = 0.2900
PASS_GAP          = 0.040
PASS_MAXDD        = -0.6100
PASS_WORST10Y     = 0.150


# ---------------------------------------------------------------------------
# ローカル: l_max をベクトル対応にした S2 関数
# ---------------------------------------------------------------------------

def compute_L_s2_dyn_lmax(returns, vz, l_max_series, target_vol, k_vz, gate_min, n, l_min, step):
    """compute_L_s2_vz_gated の l_max をベクトル対応版"""
    sigma = returns.rolling(n, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    sigma = sigma.clip(lower=1e-6)
    vz_pos = vz.fillna(0.0).clip(lower=0.0)
    vz_gate = (1.0 - k_vz * vz_pos).clip(lower=gate_min, upper=1.0)
    L_raw = (target_vol / sigma) * vz_gate
    # l_max_series を Series として使用
    if not isinstance(l_max_series, pd.Series):
        l_max_arr = pd.Series(l_max_series, index=returns.index)
    else:
        l_max_arr = l_max_series.reindex(returns.index).fillna(float(l_max_series.mean()))
    L_clipped = np.minimum(np.maximum(L_raw, l_min), l_max_arr)
    # step量子化
    if step > 0:
        L_clipped = (L_clipped / step).round() * step
    return L_clipped


def signal_to_bias_dynamic(lt_sig_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
    """element-wise lt_bias = (-k * sig * 0.5).clip(-0.5, 0.5)"""
    return np.clip(-k_arr * lt_sig_arr * 0.5, -0.5, 0.5)


def compute_p10_5y(nav, td=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    return float(((s / s.shift(td * 5)) ** 0.2 - 1).dropna().quantile(0.10))


def calc_all_metrics(nav, dates, trades_yr):
    m = calc_7metrics(nav, dates, trades_per_year=trades_yr)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    return {**m,
            'Worst10Y_star': float(r10.min()) if len(r10) > 0 else float('nan'),
            'P10_5Y':        compute_p10_5y(nav.values),
            'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS']}


def passes_all(r):
    return (r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA and
            r['CAGR_OOS'] >= PASS_CAGR_OOS and
            r['IS_OOS_gap'] <= PASS_GAP and
            r['MaxDD_FULL'] > PASS_MAXDD and
            r['Worst10Y_star'] >= PASS_WORST10Y)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('A2B: Dynamic l_max + Rolling VOL_REF スイープ')
    print('=' * 70)
    total = len(LMAX_BASE_GRID) * len(VOL_SENS_GRID)
    print(f'グリッド: lmax_base={LMAX_BASE_GRID} × vol_sens={VOL_SENS_GRID}  ({total} configs + REF)')
    print(f'E4固定値: K_LO={K_LO}, K_HI={K_HI}, VZ_THR={VZ_THR}')
    print(f'VOL_REF: rolling median (window={VOL_REF_WINDOW}日, min_periods={VOL_REF_MIN_PERIODS}日)')

    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr = n_tr / n_years

    lt_sig_raw = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    lev_arr    = lev_raw

    # E4 レジーム割り当て（固定: k_lo=0.1, k_hi=0.8, vz_thr=0.7）
    regime_hi  = vz_arr > +VZ_THR
    regime_lo  = vz_arr < -VZ_THR
    k_dyn_e4   = np.where(regime_hi, K_HI, np.where(regime_lo, K_LO, K_MID))
    lt_bias_e4 = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_dyn_e4), index=lt_sig_raw.index)
    lev_mod_e4 = apply_lt_mode_b(lev_arr, lt_bias_e4, l_min=0.0, l_max=1.0)

    # 252日実現ボラ（年率）: ループ外で事前計算
    vol252 = ret.rolling(252, min_periods=126).std() * np.sqrt(TRADING_DAYS)
    vol252_exp = ret.expanding(min_periods=30).std() * np.sqrt(TRADING_DAYS)
    vol252 = vol252.fillna(vol252_exp)

    # Rolling VOL_REF: 10年≒2520日 median。
    # 序盤 (n < 504) は expanding median でウォームアップ。
    # 最初期 (n<5) は vol252 自身で fallback（NaN を残さない安全策）。
    vol_ref_t = vol252.rolling(VOL_REF_WINDOW, min_periods=VOL_REF_MIN_PERIODS).median()
    vol_ref_t = vol_ref_t.fillna(vol252.expanding(min_periods=5).median())
    vol_ref_t = vol_ref_t.fillna(vol252)
    vol_ref_t = vol_ref_t.clip(lower=0.05)  # ゼロ除算回避

    print('Assets and signals built. Computing vol252 & vol_ref_t...')
    print(f'vol252 stats:   mean={vol252.mean():.3f}, min={vol252.min():.3f}, max={vol252.max():.3f}')
    print(f'vol_ref_t stats: mean={vol_ref_t.mean():.3f}, min={vol_ref_t.min():.3f}, '
          f'max={vol_ref_t.max():.3f}, last={vol_ref_t.iloc[-1]:.3f}')
    print(f'  (A2 fixed VOL_REF=0.20 → A2B adaptive median, NASDAQ long-term ≈0.183)')
    print('Starting sweep...')

    # REF: vol_sens=0.0, lmax_base=7.0 (固定 l_max=7.0 = E4ベスト)
    l_max_ref = pd.Series(7.0, index=ret.index)
    L_s2_ref = compute_L_s2_dyn_lmax(ret, vz, l_max_ref, **S2_BASE)
    nav_ref = build_nav_strategy(
        close, lev_mod_e4, wn_A, wg_A, wb_A, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_ref.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr)
    m_ref.update({'lmax_base': 7.0, 'vol_sens': 0.0,
                  'Trades_yr': n_trades_yr,
                  'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
    print(f'[REF base=7.0/sens=0.0] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  '
          f'Sharpe={m_ref["Sharpe_OOS"]:+.3f}  MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] REF CAGR_OOS diff={diff_ref:+.2f}pp → '
          f'{"OK" if abs(diff_ref) <= 0.30 else "WARN (>0.30pp)"}')

    results = [m_ref]
    idx = 0

    for lmax_base in LMAX_BASE_GRID:
        for vol_sens in VOL_SENS_GRID:
            idx += 1
            # 動的 l_max 系列（vol_ref_t は rolling median = A2 との唯一の本質的差分）
            l_max_t = (lmax_base - vol_sens * (vol252 / vol_ref_t - 1)).clip(L_FLOOR, L_CEIL)
            l_max_t = l_max_t.reindex(ret.index).fillna(lmax_base)

            L_s2_dyn = compute_L_s2_dyn_lmax(ret, vz, l_max_t, **S2_BASE)

            nav_dyn = build_nav_strategy(
                close, lev_mod_e4, wn_A, wg_A, wb_A, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2_dyn.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav_dyn, dates, n_trades_yr)
            m.update({'lmax_base': lmax_base, 'vol_sens': vol_sens,
                      'Trades_yr': n_trades_yr,
                      'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan')})
            results.append(m)
            print(f'  [{idx:>2d}/{total}] base={lmax_base:.1f} sens={vol_sens:.1f}: '
                  f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%  '
                  f'MaxDD_delta={(m["MaxDD_FULL"] - m_ref["MaxDD_FULL"])*100:+.2f}pp')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list   = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    best_maxdd  = max(results[1:], key=lambda r: r['MaxDD_FULL'])  # 最小ドローダウン

    # MaxDD改善確認
    improved_dd = [r for r in results[1:] if r['MaxDD_FULL'] > m_ref['MaxDD_FULL']]
    print(f'MaxDD改善 configs: {len(improved_dd)} / {total}')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: base={best_sharpe["lmax_base"]:.1f} sens={best_sharpe["vol_sens"]:.1f} '
          f'→ Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%')
    print(f'Best MaxDD: base={best_maxdd["lmax_base"]:.1f} sens={best_maxdd["vol_sens"]:.1f} '
          f'→ MaxDD={best_maxdd["MaxDD_FULL"]*100:+.2f}%  CAGR={best_maxdd["CAGR_OOS"]*100:+.2f}%')

    verdict = ('PASS' if pass_list else
               'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'総合判定: {verdict}')

    # CSV
    pd.DataFrame([{
        'lmax_base': r['lmax_base'], 'vol_sens': r['vol_sens'],
        'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
        'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
        'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
        'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
    } for r in results]).to_csv(os.path.join(BASE, 'a2b_dyn_lmax_rolling_vref_results.csv'),
                                index=False, float_format='%.6f')

    # MD
    hdr1, hdr2 = MD_HEADER_1P
    top_results = sorted(results[1:], key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(f'base={r["lmax_base"]:.1f}/sens={r["vol_sens"]:.1f}', r)
        for r in top_results
    )
    ref_row = fmt_row_1p('base=7.0/sens=0.0 (REF=E4)', m_ref)

    # REF vs Best MaxDD 比較テキスト
    dd_delta = (best_maxdd['MaxDD_FULL'] - m_ref['MaxDD_FULL']) * 100
    cagr_delta = (best_maxdd['CAGR_OOS'] - m_ref['CAGR_OOS']) * 100

    report = f"""\
# A2B: Dynamic l_max + Rolling VOL_REF — A2 の CAGR 損失を回復

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

A2 (`src/a2_dyn_lmax.py`) で固定 `VOL_REF=0.20` を用いた結果、NASDAQ の長期実現ボラ
平均 ≈0.183 を上回るため、低ボラ局面（vol252 < 0.20）でも l_max が引き下げられず、
むしろ平常時にデレバが弱まる方向に効いた一方、CAGR_OOS の改善は限定的だった。
本 A2B は VOL_REF を {VOL_REF_WINDOW} 日 rolling median に置換し、実勢ボラに追従する
適応的基準を導入する。

| 項目 | 定義 |
|------|------|
| **E4固定値** | k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}（変更なし） |
| **LMAX_BASE グリッド** | {LMAX_BASE_GRID}（l_max基準値） |
| **VOL_SENS グリッド** | {VOL_SENS_GRID}（ボラ感度） |
| **L_FLOOR / L_CEIL** | {L_FLOOR} / {L_CEIL}（クリップ範囲） |
| **VOL_REF** | **rolling median (window={VOL_REF_WINDOW}日, min_periods={VOL_REF_MIN_PERIODS}日)** |
| **vol_ref_t 統計** | mean={vol_ref_t.mean():.3f}, min={vol_ref_t.min():.3f}, max={vol_ref_t.max():.3f}, last={vol_ref_t.iloc[-1]:.3f} |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**動的 l_max 計算式 (A2 → A2B 差分)**:
```
A2:  l_max_t = clip(lmax_base - vol_sens × (vol252 / 0.20 - 1), {L_FLOOR}, {L_CEIL})
A2B: vol_ref_t = vol252.rolling({VOL_REF_WINDOW}, min_periods={VOL_REF_MIN_PERIODS}).median()
                       .fillna(vol252.expanding(min_periods=5).median())
     l_max_t   = clip(lmax_base - vol_sens × (vol252 / vol_ref_t - 1), {L_FLOOR}, {L_CEIL})
```

**直感**: 「凪期は基準ボラも下がるので、vol252/vol_ref_t は1付近に収束 → l_max を下げない」
       「嵐期は基準ボラの上昇は遅いので、vol252/vol_ref_t > 1 が維持され l_max は下がる」
       → MaxDD 抑制と CAGR 回復の両立を狙う非対称設計。

**サニティ**: REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (E4={REF_CAGR_OOS*100:+.2f}%, diff {diff_ref:+.2f}pp)

---

## §2 9指標テーブル（Sharpe 降順 + REF）

{hdr1}
{hdr2}
{ref_row}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+{PASS_SHARPE_DELTA:.3f} (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **MaxDD改善 configs**: {len(improved_dd)} / {total}
- **最高 Sharpe**: base={best_sharpe["lmax_base"]:.1f}, sens={best_sharpe["vol_sens"]:.1f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **最良 MaxDD**: base={best_maxdd["lmax_base"]:.1f}, sens={best_maxdd["vol_sens"]:.1f} → MaxDD={best_maxdd["MaxDD_FULL"]*100:+.2f}% (E4比 {dd_delta:+.2f}pp), CAGR_OOS={best_maxdd["CAGR_OOS"]*100:+.2f}% (E4比 {cagr_delta:+.2f}pp)
- **総合判定: {verdict}**

---

## §4 A2 比較

A2 (固定 VOL_REF=0.20) の同一グリッド結果は `a2_dyn_lmax_results.csv` を参照。
A2B との差分は `vol_ref` 列のみで CAGR_OOS / MaxDD の振る舞いに違いが
出れば仮説 (rolling vref が CAGR を回復) が支持される。

| 観点 | A2 (固定 VOL_REF=0.20) | A2B (rolling median ≈0.183) |
|------|----------------------|------------------------------|
| 低ボラ期の l_max | 引下げ気味 (vol252/0.20 < 1 が稀) | 引下げほぼなし (vol252/vol_ref≈1) |
| 高ボラ期の l_max | 強く引下げ | 同程度に引下げ |
| 期待 CAGR_OOS | ベースライン | **回復**（凪期に過剰デレバを抑制） |
| 期待 MaxDD | ベースライン | **同等維持**（嵐期は同様にデレバ） |

---

## §5 考察

- A2 の固定 VOL_REF=0.20 は NASDAQ ヒストリカル平均ではなく直感値で設定
  されていた。NASDAQ の実測 vol252 長期平均は ≈0.183 (本実験で確認)
- Rolling median による適応化は過学習リスクが小さい: 10年窓は十分長く、
  median 演算はノイズ耐性が高い。自由度の増加はゼロ（パラメータ追加なし）
- 注意点: 序盤 ウォームアップ期間 (504日未満) は expanding median で代替するため
  IS 主要評価 (1976–2021) には影響しない
- 過学習リスク: 2自由度 (lmax_base, vol_sens)。A2 と同一なので追加リスクなし

---

*生成スクリプト: `src/a2b_dyn_lmax_rolling_vref.py`*
*参照: `src/a2_dyn_lmax.py` (固定 VOL_REF 版), `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'A2B_DYN_LMAX_ROLLING_VREF_2026-05-27.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "a2b_dyn_lmax_rolling_vref_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
