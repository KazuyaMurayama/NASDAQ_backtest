"""
C3: Yang-Zhang Vol推定（ボラ推定量置換）
=========================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

概念: E4のボラティリティゾーン計算（vz）に使う日次ボラを
  close-to-close標準偏差からYang-Zhang推定量に変更。
  YZは高値・安値・始値を活用し、同じ窓サイズで5〜10倍精度のボラ推定ができる。

Yang-Zhang式:
  σ²_YZ = σ²_overnight + k×σ²_cc_open + (1-k)×σ²_rs
  σ²_overnight = mean((log(Open/Close.shift(1)))²)
  σ²_rs = Rogers-Satchell = mean(log(H/O)*log(H/C) + log(L/O)*log(L/C))
  σ²_cc_open = mean((log(Close/Open))²)
  k = 0.34/(1.34 + (n+1)/(n-1))

注記: 1974-1984のOHLCは合成データ（O=H=L=C）のため、
      この期間のYZ推定量はclose-to-closeと同等になる。

グリッド:
  YZ_LOOKBACK_GRID = [10, 20, 30]
  VZ_THR_GRID      = [0.5, 0.7, 1.0]
  K_LO=0.1, K_HI=0.8固定（E4採用値）
  REF: close-to-close vz（E4ベスト: n=20, vz_thr=0.7）

  合計: 3×3=9 configs + REF

出力:
  - c3_yang_zhang_results.csv
  - C3_YANG_ZHANG_VOL_2026-05-27.md
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
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from _sweep_format import MD_HEADER_1P, fmt_row_1p, MD_WFA_NOTE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# グリッド
# ---------------------------------------------------------------------------
YZ_LOOKBACK_GRID = [10, 20, 30]   # Yang-Zhangの窓サイズ
VZ_THR_GRID      = [0.5, 0.7, 1.0]  # vz閾値
K_LO_FIXED = 0.1   # E4採用値
K_HI_FIXED = 0.8   # E4採用値
K_MID      = 0.50

# E4ベスト（close-to-close REF）
REF_CAGR_OOS   = 0.3353
REF_SHARPE_OOS = 0.891
REF_MAXDD      = -0.6001

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)

PASS_SHARPE_DELTA = 0.020
PASS_CAGR_OOS     = 0.2916
PASS_GAP          = 0.030
PASS_MAXDD        = -0.6445
PASS_WORST10Y     = 0.150

OHLC_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')


# ---------------------------------------------------------------------------
# Yang-Zhang ボラティリティ推定量
# ---------------------------------------------------------------------------

def yang_zhang_vol(df_ohlc, n=20, trading_days=252):
    """
    Yang-Zhang ボラティリティ推定量（年率）
    df_ohlc: Open, High, Low, Close 列を持つDataFrame
    注記: 1974-1984のOHLCは合成データ（O=H=L=C）のため
          この期間のYZ推定量はclose-to-closeと同等になる。
    """
    o  = np.log(df_ohlc['Open'].astype(float))
    h  = np.log(df_ohlc['High'].astype(float))
    l  = np.log(df_ohlc['Low'].astype(float))
    c  = np.log(df_ohlc['Close'].astype(float))
    c1 = c.shift(1)  # 前日Close

    min_periods = max(5, n // 2)

    # overnight variance: log(Open_t / Close_{t-1})^2
    log_oc1   = o - c1
    sigma2_o  = (log_oc1 ** 2).rolling(n, min_periods=min_periods).mean()

    # close-to-open variance: log(Close_t / Open_t)^2
    log_co    = c - o
    sigma2_c  = (log_co ** 2).rolling(n, min_periods=min_periods).mean()

    # Rogers-Satchell variance
    # RS = log(H/O)*log(H/C) + log(L/O)*log(L/C)
    log_ho = h - o
    log_hc = h - c
    log_lo = l - o
    log_lc = l - c
    rs = log_ho * log_hc + log_lo * log_lc
    sigma2_rs = rs.rolling(n, min_periods=min_periods).mean()

    # Yang-Zhang weight
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    sigma2_yz = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs
    sigma2_yz = sigma2_yz.clip(lower=1e-8)  # Rogers-Satchell ゼロ除算対処
    return np.sqrt(sigma2_yz * trading_days)  # 年率ボラ


def load_ohlc(filepath):
    """OHLCデータをロードし、DateインデックスのDataFrameを返す"""
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def build_vz_from_yz(close, df_ohlc, dates, yz_n=20):
    """
    Yang-Zhang ボラからvz（Z-score）を計算。
    dates（Timestamp Series）でOHLCをアラインしてから計算。
    close.indexはRangeIndexのため、datesの値（Timestamp）でloc。
    cc-volがフォールバック（YZ=0またはNaN過多時）。
    """
    # OHLCをdatesの日付でアラインする（close.indexはRangeIndexのため日付Seriesを使う）
    dates_ts = pd.to_datetime(dates.values)
    ohlc_aligned = df_ohlc.loc[df_ohlc.index.isin(dates_ts)].copy()
    # DataFrameをRangeIndexに変換してcloseと同じ長さ・順序に
    ohlc_aligned = ohlc_aligned.reset_index(drop=True)

    # YZボラ計算
    yz_vol = yang_zhang_vol(ohlc_aligned, n=yz_n)

    # NaN率チェック（50%超でcc-volにフォールバック）
    nan_rate = yz_vol.isna().mean()
    if nan_rate > 0.5:
        print(f'  [WARN] YZ NaN rate={nan_rate:.1%} → cc-vol フォールバック')
        vp = close.pct_change().fillna(0)
        yz_vol = vp.rolling(yz_n).std() * np.sqrt(252)
        yz_vol = yz_vol.clip(lower=1e-8)

    # Z-score（vz）: 現在YZボラ vs 252日移動平均・標準偏差
    yz_mean = yz_vol.rolling(252).mean()
    yz_std  = yz_vol.rolling(252).std().replace(0, 0.001)
    vz_yz   = (yz_vol - yz_mean) / yz_std
    vz_yz   = vz_yz.fillna(0.0)

    return vz_yz


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def signal_to_bias_dynamic(lt_sig_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
    return np.clip(-k_arr * lt_sig_arr * 0.5, -0.5, 0.5)


def compute_p10_5y(nav, td=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    return float(((s / s.shift(td * 5)) ** 0.2 - 1).dropna().quantile(0.10))


def calc_all_metrics(nav, dates, trades_yr):
    m    = calc_7metrics(nav, dates, trades_per_year=trades_yr)
    ann  = nav_to_annual(nav, dates)
    r10  = rolling_nY_cagr(ann, 10)
    return {**m,
            'Worst10Y_star': float(r10.min()) if len(r10) > 0 else float('nan'),
            'P10_5Y':        compute_p10_5y(nav.values),
            'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS']}


def passes_all(r):
    return (r['Sharpe_OOS'] - REF_SHARPE_OOS >= PASS_SHARPE_DELTA and
            r['CAGR_OOS']   >= PASS_CAGR_OOS  and
            r['IS_OOS_gap'] <= PASS_GAP        and
            r['MaxDD_FULL'] >  PASS_MAXDD      and
            r['Worst10Y_star'] >= PASS_WORST10Y)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('C3: Yang-Zhang Vol推定 スイープ')
    print('=' * 70)
    total = len(YZ_LOOKBACK_GRID) * len(VZ_THR_GRID)
    print(f'グリッド: YZ_n={YZ_LOOKBACK_GRID} × vz_thr={VZ_THR_GRID}  ({total} configs + REF)')
    print(f'K_LO={K_LO_FIXED} / K_HI={K_HI_FIXED} 固定（E4採用値）')

    # データロード
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n:,} days)')

    # OHLCロード
    print(f'OHLC: {OHLC_PATH}')
    df_ohlc = load_ohlc(OHLC_PATH)
    print(f'OHLC columns: {df_ohlc.columns.tolist()}')

    # 合成データ期間確認
    synthetic_mask = (df_ohlc['Open'] == df_ohlc['High']) & \
                     (df_ohlc['High'] == df_ohlc['Low'])  & \
                     (df_ohlc['Low']  == df_ohlc['Close'])
    synth_days = synthetic_mask.sum()
    print(f'合成OHLC（O=H=L=C）日数: {synth_days} ({synth_days/len(df_ohlc)*100:.1f}%)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    # -------------------------------------------------------------------
    # REF: close-to-close vz を使ったE4ベスト（k_lo=0.1, k_hi=0.8, vz_thr=0.7）
    # -------------------------------------------------------------------
    print('\n--- REF: close-to-close vz (E4ベスト k_lo=0.1, k_hi=0.8, vz_thr=0.7) ---')
    raw_a2_ref, vz_ref = build_a2_signal(close, ret)
    lev_raw_ref, wn_A_ref, wg_A_ref, wb_A_ref, n_tr_ref = simulate_rebalance_A(
        raw_a2_ref, vz_ref, THRESHOLD)
    n_trades_yr_ref = n_tr_ref / n_years
    L_s2_ref = compute_L_s2_vz_gated(ret, vz_ref, **S2_FIXED)

    lt_sig_raw   = build_lt_signal(close, 'LT2', 750)
    lt_sig_arr   = lt_sig_raw.values
    vz_ref_arr   = vz_ref.values
    lev_ref_arr  = lev_raw_ref

    # E4ベスト設定でREFを構築
    VZ_THR_REF = 0.7
    regime_hi_ref  = vz_ref_arr > +VZ_THR_REF
    regime_lo_ref  = vz_ref_arr < -VZ_THR_REF
    k_dyn_ref = np.where(regime_hi_ref, K_HI_FIXED,
                         np.where(regime_lo_ref, K_LO_FIXED, K_MID))
    lt_bias_ref = pd.Series(signal_to_bias_dynamic(lt_sig_arr, k_dyn_ref),
                            index=lt_sig_raw.index)
    lev_mod_ref = apply_lt_mode_b(lev_ref_arr, lt_bias_ref, l_min=0.0, l_max=1.0)
    nav_ref = build_nav_strategy(
        close, lev_mod_ref, wn_A_ref, wg_A_ref, wb_A_ref, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_ref.values, cfd_spread=CFD_SPREAD_LOW,
    )
    m_ref = calc_all_metrics(nav_ref, dates, n_trades_yr_ref)
    m_ref.update({
        'yz_n': 0, 'vz_thr': VZ_THR_REF,
        'Trades_yr': n_trades_yr_ref,
        'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan'),
        'label': f'CC-REF (E4 k_lo={K_LO_FIXED}/k_hi={K_HI_FIXED}/vz={VZ_THR_REF})',
    })
    print(f'[CC-REF] CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}%  Sharpe={m_ref["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={m_ref["MaxDD_FULL"]*100:+.2f}%')
    diff_ref = (m_ref['CAGR_OOS'] - REF_CAGR_OOS) * 100
    print(f'[SANITY] diff={diff_ref:+.2f}pp → {"OK" if abs(diff_ref) <= 0.30 else "WARN"}')

    results = [m_ref]
    idx = 0

    # -------------------------------------------------------------------
    # YZ スイープ
    # -------------------------------------------------------------------
    print('\nStarting YZ sweep...')
    for yz_n in YZ_LOOKBACK_GRID:
        # YZボラからvzを計算
        vz_yz = build_vz_from_yz(close, df_ohlc, dates, yz_n=yz_n)

        # YZベースのシグナルビルド（A2 raw は同一; vzのみ変更）
        raw_a2_yz, _ = build_a2_signal(close, ret)
        # vz差し替え: simulate_rebalance_A はvzを使って wn を計算する
        lev_raw_yz, wn_A_yz, wg_A_yz, wb_A_yz, n_tr_yz = simulate_rebalance_A(
            raw_a2_yz, vz_yz, THRESHOLD)
        n_trades_yr_yz = n_tr_yz / n_years
        L_s2_yz = compute_L_s2_vz_gated(ret, vz_yz, **S2_FIXED)
        lev_yz_arr = lev_raw_yz
        vz_yz_arr  = vz_yz.values

        for vz_thr in VZ_THR_GRID:
            idx += 1
            # レジーム割り当て
            regime_hi  = vz_yz_arr > +vz_thr
            regime_lo  = vz_yz_arr < -vz_thr

            # K_LO / K_HI 固定（E4採用値）
            k_dyn = np.where(regime_hi, K_HI_FIXED,
                             np.where(regime_lo, K_LO_FIXED, K_MID))
            lt_bias_dyn = pd.Series(
                signal_to_bias_dynamic(lt_sig_arr, k_dyn),
                index=lt_sig_raw.index,
            )
            lev_mod_dyn = apply_lt_mode_b(lev_yz_arr, lt_bias_dyn, l_min=0.0, l_max=1.0)
            nav_dyn = build_nav_strategy(
                close, lev_mod_dyn, wn_A_yz, wg_A_yz, wb_A_yz, dates,
                gold_2x, bond_3x, sofr,
                nas_mode='CFD', cfd_leverage=L_s2_yz.values, cfd_spread=CFD_SPREAD_LOW,
            )
            m = calc_all_metrics(nav_dyn, dates, n_trades_yr_yz)
            label = f'YZ_n={yz_n}/vz={vz_thr}'
            m.update({
                'yz_n': yz_n, 'vz_thr': vz_thr,
                'Trades_yr': n_trades_yr_yz,
                'WFA_CI95_lo': float('nan'), 'WFA_WFE': float('nan'),
                'label': label,
            })
            results.append(m)
            print(f'  [{idx:>2d}/{total}] YZ_n={yz_n} vz_thr={vz_thr:.1f}: '
                  f'CAGR={m["CAGR_OOS"]*100:+.2f}%  Sharpe={m["Sharpe_OOS"]:+.3f}  '
                  f'MaxDD={m["MaxDD_FULL"]*100:+.2f}%')

    print(f'\n全 {len(results)-1} configs 完了。')

    pass_list   = [r for r in results[1:] if passes_all(r)]
    best_sharpe = max(results[1:], key=lambda r: r['Sharpe_OOS'])
    verdict     = ('PASS' if pass_list else
                   'WARN' if best_sharpe['Sharpe_OOS'] > REF_SHARPE_OOS else 'FAIL')
    print(f'PASS configs: {len(pass_list)}')
    print(f'Best Sharpe: YZ_n={best_sharpe["yz_n"]} vz_thr={best_sharpe["vz_thr"]:.1f} '
          f'→ Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}  '
          f'MaxDD={best_sharpe["MaxDD_FULL"]*100:+.2f}%  IS-OOS={best_sharpe["IS_OOS_gap"]*100:+.2f}pp')
    print(f'総合判定: {verdict}')

    # CSV
    csv_rows = []
    for r in results:
        csv_rows.append({
            'yz_n': r['yz_n'], 'vz_thr': r['vz_thr'],
            'CAGR_IS': r['CAGR_IS'], 'CAGR_OOS': r['CAGR_OOS'],
            'Sharpe_OOS': r['Sharpe_OOS'], 'MaxDD_FULL': r['MaxDD_FULL'],
            'Worst10Y_star': r['Worst10Y_star'], 'P10_5Y': r['P10_5Y'],
            'IS_OOS_gap': r['IS_OOS_gap'], 'Trades_yr': r['Trades_yr'],
        })
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(BASE, 'c3_yang_zhang_results.csv'),
        index=False, float_format='%.6f')

    # MD
    hdr1, hdr2 = MD_HEADER_1P
    all_yz_results = results[1:]  # REFを除く
    top_results = sorted(all_yz_results, key=lambda r: r['Sharpe_OOS'], reverse=True)
    rows_md = '\n'.join(
        fmt_row_1p(r['label'], r)
        for r in top_results
    )
    ref_row  = fmt_row_1p(m_ref['label'], m_ref)

    report = f"""\
# C3: Yang-Zhang Vol推定（ボラ推定量置換）スイープ

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **YZ_lookback グリッド** | {YZ_LOOKBACK_GRID}（Yang-Zhang推定窓サイズ） |
| **vz_thr グリッド** | {VZ_THR_GRID}（レジーム切替え閾値） |
| **k_lo** | {K_LO_FIXED} 固定（E4採用値） |
| **k_hi** | {K_HI_FIXED} 固定（E4採用値） |
| **k_mid** | {K_MID}（閾値外の中間域） |
| **合計 configs** | {total} + REF |
| **IS** | {IS_START} 〜 {IS_END} |
| **OOS** | {OOS_START} 〜 |

**Yang-Zhang式**:
```
σ²_YZ = σ²_overnight + k×σ²_cc_open + (1-k)×σ²_rs
σ²_overnight = mean((log(Open/Close_prev))²)
σ²_rs (Rogers-Satchell) = mean(log(H/O)×log(H/C) + log(L/O)×log(L/C))
σ²_cc_open = mean((log(Close/Open))²)
k = 0.34 / (1.34 + (n+1)/(n-1))
```

**注記: 1974-1984のOHLCは合成データ（O=H=L=C）のため、
この期間のYZ推定量はclose-to-closeと同等になる。**

**レジーム割り当て（k_lo/k_hi固定）**:
```
vz_yz > +vz_thr  → k = {K_HI_FIXED}  (高YZボラ: 強い防御バイアス)
vz_yz < -vz_thr  → k = {K_LO_FIXED}  (低YZボラ: 弱い防御バイアス)
otherwise        → k = {K_MID}  (中間域)
```
`lt_bias = (-k × lt_sig × 0.5).clip(-0.5, 0.5)`

**サニティ**: CC-REF CAGR_OOS={m_ref["CAGR_OOS"]*100:+.2f}% (diff {diff_ref:+.2f}pp)

---

## §2 9指標テーブル（全{total} configs + CC-REF）

> CC-REF = close-to-close vz を使ったE4ベスト（k_lo=0.1/k_hi=0.8/vz=0.7）
> YZ configs は Sharpe_OOS 降順

{hdr1}
{hdr2}
{ref_row}
{rows_md}

{MD_WFA_NOTE}

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ CC-REF+0.020 (≥{REF_SHARPE_OOS+PASS_SHARPE_DELTA:.3f}) |
| (ii) CAGR_OOS | ≥ {PASS_CAGR_OOS*100:.1f}% |
| (iii) IS-OOS gap | ≤ {PASS_GAP*100:.1f}pp |
| (iv) MaxDD | > {PASS_MAXDD*100:.1f}% |
| (v) Worst10Y★ | ≥ {PASS_WORST10Y*100:.1f}% |

- **PASS configs**: {len(pass_list)} / {total}
- **最高 Sharpe**: YZ_n={best_sharpe["yz_n"]}, vz_thr={best_sharpe["vz_thr"]:.1f} → Sharpe={best_sharpe["Sharpe_OOS"]:+.3f}
- **総合判定: {verdict}**

---

## §4 考察

YZ推定量はOHLC情報を活用してclose-to-closeより効率的なボラ推定を行う:
- 1974-1984の合成OHLC期間はclose-to-closeと同等（YZの優位性は実データ期間のみ有効）
- YZ Z-scoreの分布はcc Z-scoreと異なる可能性があるため、vz_thr=0.7だけでなく0.5/1.0も探索
- 改善が見られない場合: YZボラの「精度向上」がvzのZ-score計算で相殺される可能性がある
  （分母の252日rolling stdも変化するため）
- 次の実験候補: YZボラを直接パーセンタイルでゾーン分割する（Z-scoreを経由しない）

---

*生成スクリプト: `src/c3_yang_zhang_vol.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
"""
    md_path = os.path.join(BASE, 'C3_YANG_ZHANG_VOL_2026-05-27.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {os.path.join(BASE, "c3_yang_zhang_results.csv")}')
    print(f'Saved: {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
