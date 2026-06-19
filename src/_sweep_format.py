"""
_sweep_format.py — Sweep MDテーブル共有フォーマッタ
=====================================================
EVALUATION_STANDARD §3.12 / §5.6 準拠 (v2.0, 2026-06-19)

統一指標セット（新10列標準）:
  1. CAGR_IS   2. CAGR_OOS  3. Sharpe_Full  4. MaxDD
  5. 最悪単日   6. Worst10Y★  7. Worst5Y      8. P10▷
  9. Trade/年   10. 頑強性・過学習

頑強性・過学習セル判定ロジック:
  ❌過学習疑い: WFE<0.5 or >2 or CI95_lo<0 or |gap|>5pp
  ✅頑強:       WFE∈[0.5,2] and CI95_lo>0 and |gap|≤3pp
                and (CPCV_p10>0 or 未算出) and (t_p<0.05 or 未算出)
                and (Regime_min>-10 or 未算出)
  ⚠条件付:     上記以外
  (部分):       算出済でないゲートがある場合に付加

提供フォーマッタ:
  - MD_HEADER_1P  / fmt_row_1p   : 1パラメータ sweep
  - MD_HEADER_2P  / fmt_row_2p   : 2パラメータ sweep
  - MD_HEADER_STRAT / fmt_row_strat : 戦略横並び比較
  - MD_HEADER_INTEGRATED / fmt_row_integrated : 統合比較レポート
  - fmt_annual_table  : §5 年次リターン表
  - fmt_stats_table   : §6 統計サマリ

Sharpe マーカ: ◎ = Sharpe_Full > +0.934 / ★ = > +1.100（フル期間基準）
  ◎ 閾値 = E4 Active 実算出値 +0.9341 (2026-06-19 実バックテスト確定)
  ★ 閾値 = B3a_k365 実算出値 +1.1023 (同上)
  ※旧値 (◎=0.700 / ★=0.800) は推定値につき廃止
列名 Worst10Y★ の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）
IS=1974-2021-05-07 / OOS=2021-05-08-現在 / Full=全期間
"""
import numpy as np


# ---------------------------------------------------------------------------
# 値フォーマッタ
# ---------------------------------------------------------------------------

def _fp1(v):
    """百分率 1桁小数 例: +27.3%"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  —  '
    return f'{v * 100:+5.1f}%'


def _ff2(v):
    """浮動小数 2桁小数 例: +0.88"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  —  '
    return f'{v:+5.2f}'


def _gap_pp(v):
    """IS-OOS gap (pp表示) — 後方互換用・非推奨（v2.0以降は頑強性セル内部材料）"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '   —    '
    return f'{v * 100:+5.2f}pp'


def _tr(v):
    """Trades/yr 整数 例: 27"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ' — '
    return f'{int(round(v)):>3d}'


def _wfa(v):
    """WFA CI95_lo — 後方互換用・非推奨（v2.0以降は_robustness_cell使用）"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '   —   '
    return f'{v:+6.3f}'


def _ovfit_wfe(wfe):
    """過学習スコア — 後方互換用・非推奨（v2.0以降は_robustness_cell使用）"""
    if wfe is None or (isinstance(wfe, float) and np.isnan(wfe)):
        return '—'
    wfe = float(wfe)
    if wfe < 0.5:
        label = '❌ HIGH'
    elif wfe <= 2.0:
        label = '✅ LOW'
    else:
        label = '⚠ MED'
    return f'{label}<br>({wfe:.1f})'


def _worst1d(v, date):
    """最悪単日 + 発生日: −12.3%<br>(2020-03-16)"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  —  '
    date_str = f'({date})' if date else ''
    return f'{v * 100:+5.1f}%<br>{date_str}'


def _robustness_cell(r):
    """頑強性・過学習統合セル。NaN安全。

    Parameters
    ----------
    r : dict
        任意キー: WFA_WFE, WFA_CI95_lo, IS_OOS_gap_pp,
                  CPCV_p10, t_p, Regime_min
    """
    wfe     = r.get('WFA_WFE')
    ci95    = r.get('WFA_CI95_lo')
    gap     = r.get('IS_OOS_gap_pp')
    cpcv    = r.get('CPCV_p10')
    tp      = r.get('t_p')
    reg_min = r.get('Regime_min')

    def _isnan(x):
        return x is None or (isinstance(x, float) and np.isnan(x))

    has_wfe  = not _isnan(wfe)
    has_ci95 = not _isnan(ci95)
    has_gap  = not _isnan(gap)
    has_cpcv = not _isnan(cpcv)
    has_tp   = not _isnan(tp)
    has_reg  = not _isnan(reg_min)

    if not has_wfe and not has_ci95:
        return '—'

    # 過学習疑いゲート
    is_fail = False
    if has_wfe  and (float(wfe) < 0.5 or float(wfe) > 2.0):
        is_fail = True
    if has_ci95 and float(ci95) < 0:
        is_fail = True
    if has_gap  and abs(float(gap)) > 5:
        is_fail = True

    # 頑強性ゲート
    is_pass = True
    partial = False
    if has_wfe:
        if not (0.5 <= float(wfe) <= 2.0):
            is_pass = False
    else:
        is_pass = False
        partial = True
    if has_ci95:
        if float(ci95) <= 0:
            is_pass = False
    else:
        is_pass = False
        partial = True
    if has_gap:
        if abs(float(gap)) > 3:
            is_pass = False
    else:
        partial = True
    if has_cpcv:
        if float(cpcv) <= 0:
            is_pass = False
    else:
        partial = True
    if has_tp:
        if float(tp) >= 0.05:
            is_pass = False
    else:
        partial = True
    if has_reg:
        if float(reg_min) <= -0.10:   # reg_min は比率 (-0.10 = -10%)
            is_pass = False
    else:
        partial = True

    if is_fail:
        label = '❌過学習疑い'
    elif is_pass:
        label = '✅頑強'
        if partial:
            label += '(部分)'
    else:
        label = '⚠条件付'
        if partial:
            label += '(部分)'

    lines = [label]
    if has_wfe and has_ci95:
        lines.append(f'WFE{float(wfe):.2f} CI95lo{float(ci95)*100:+.0f}%')
    elif has_wfe:
        lines.append(f'WFE{float(wfe):.2f}')
    elif has_ci95:
        lines.append(f'CI95lo{float(ci95)*100:+.0f}%')
    if has_cpcv or has_tp:
        cpcv_str = f'CPCV{float(cpcv)*100:+.0f}%' if has_cpcv else ''
        tp_str   = f' t_p{float(tp):.3f}' if has_tp else ''
        lines.append(cpcv_str + tp_str)
    if has_reg:
        lines.append(f'Reg{float(reg_min)*100:+.1f}%')   # 比率→% 表示

    return '<br>'.join(lines)


# ---------------------------------------------------------------------------
# 新10列標準テーブルヘッダ (v2.0)
# ---------------------------------------------------------------------------

_METRIC_COLS_HEADER = (
    ' CAGR<br>IS<br>⓽ | CAGR<br>OOS<br>⓽ | Sharpe<br>Full<br>ⓒ'
    ' | Max<br>DD<br>ⓒ | 最悪<br>単日<br>ⓒ | Worst<br>10Y★<br>⓽'
    ' | Worst<br>5Y<br>⓽ | P10<br>5Y▷<br>⓽ | Trade<br>/年<br>ⓞ | 頑強性<br>過学習 |'
)
_METRIC_COLS_SEP = '---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:'

# 1パラメータ sweep (例: b6 — N のみ)
MD_HEADER_1P = (
    f'| Param |{_METRIC_COLS_HEADER}',
    f'|:------|{_METRIC_COLS_SEP}|',
)

# 2パラメータ sweep (例: b3/b4/b5/b7/b8 — N × k_lt)
MD_HEADER_2P = (
    f'| N | k_lt |{_METRIC_COLS_HEADER}',
    f'|--:|-----:|{_METRIC_COLS_SEP}|',
)

# 戦略横並び比較表
MD_HEADER_STRAT = (
    f'| Strategy |{_METRIC_COLS_HEADER}',
    f'|:---------|{_METRIC_COLS_SEP}|',
)

# 統合比較表（旧11列から新10列標準に統一）
MD_HEADER_INTEGRATED = (
    f'| Strategy |{_METRIC_COLS_HEADER}',
    f'|:---------|{_METRIC_COLS_SEP}|',
)

# WFA 未計算注記
MD_WFA_NOTE = (
    '*頑強性・過学習セル: `—` は WFA 未計算。'
    '計算後は `src/g2_wfa_shortlist.py` で補完予定。*  \n'
    '*進格条件: **WFA_CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）'
    '/ **|IS-OOS gap| ≤ 3pp**。'
    'Sharpe マーカ: ◎ = Sharpe_Full > +0.934 / ★ = > +1.100（フル期間基準）。'
    '◎閾値=E4 Active実算出値+0.9341 / ★閾値=B3a_k365実算出値+1.1023（2026-06-19確定）。'
    '列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*'
)

# 戦略比較用凡例
MD_METRIC_GLOSSARY = (
    '*◎ Sharpe_Full > +0.934（E4 Active実算出値・2026-06-19確定）/ ★ > +1.100（B3a_k365実算出値・2026-06-19確定）。'
    'Sharpe は**フル期間**（IS+OOS 全体）日次リターンの年率値（Rf=0）。'
    '頑強性セル: ✅頑強=WFE∈[0.5,2.0] かつ CI95_lo>0 かつ |gap|≤3pp'
    ' / ❌過学習疑い=WFE<0.5 or >2 or CI95_lo<0 or |gap|>5pp / ⚠それ以外。'
    '最悪単日ⓒ: 全期間の最悪1日騰落率と発生日。'
    '列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカの ◎/★ とは別意味）。'
    'IS=1974-2021-05-07 / OOS=2021-05-08-現在 / Full=全期間。*'
)


# ---------------------------------------------------------------------------
# 汎用 data-row ビルダ (v2.0 — 新10列標準)
# ---------------------------------------------------------------------------

def fmt_row_2p(n, k_lt, r, ref_s2=0.934, ref_lt2=1.100):
    """N × k_lt 型の1行を返す (b3/b4/b5/b7/b8 共通) — v2.0: 新10列標準"""
    sf = r.get('Sharpe_FULL', float('nan'))
    if isinstance(sf, float) and np.isnan(sf):
        mark = ''
    else:
        mark = ' ★' if sf > ref_lt2 else (' ◎' if sf > ref_s2 else '')
    return (
        f'| {n:>4d} | {k_lt:.1f} '
        f'| {_fp1(r.get("CAGR_IS"))} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_ff2(sf)}{mark} '
        f'| {_fp1(r["MaxDD_FULL"])} '
        f'| {_worst1d(r.get("Worst1D"), r.get("Worst1D_date"))} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r.get("Worst5Y"))} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_robustness_cell(r)} |'
    )


def fmt_row_1p(param_label, r, ref_s2=0.934, ref_lt2=1.100):
    """1パラメータ型の1行を返す (b6 等) — v2.0: 新10列標準"""
    sf = r.get('Sharpe_FULL', float('nan'))
    if isinstance(sf, float) and np.isnan(sf):
        mark = ''
    else:
        mark = ' ★' if sf > ref_lt2 else (' ◎' if sf > ref_s2 else '')
    return (
        f'| {param_label} '
        f'| {_fp1(r.get("CAGR_IS"))} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_ff2(sf)}{mark} '
        f'| {_fp1(r["MaxDD_FULL"])} '
        f'| {_worst1d(r.get("Worst1D"), r.get("Worst1D_date"))} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r.get("Worst5Y"))} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_robustness_cell(r)} |'
    )


def fmt_row_strat(label, r, ref_s2=0.934, ref_lt2=1.100,
                  sharpe_ref_mark=None, maxdd_ref_mark=None):
    """戦略横並び比較表の1行 — v2.0: 新10列標準"""
    sf = r.get('Sharpe_FULL', float('nan'))
    if isinstance(sf, float) and np.isnan(sf):
        mark = ''
    else:
        mark = ' ★' if sf > ref_lt2 else (' ◎' if sf > ref_s2 else '')
    s_sfx = sharpe_ref_mark or ''
    m_sfx = maxdd_ref_mark  or ''
    return (
        f'| {label} '
        f'| {_fp1(r.get("CAGR_IS"))} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_ff2(sf)}{mark}{s_sfx} '
        f'| {_fp1(r["MaxDD_FULL"])}{m_sfx} '
        f'| {_worst1d(r.get("Worst1D"), r.get("Worst1D_date"))} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r.get("Worst5Y"))} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_robustness_cell(r)} |'
    )


def fmt_row_integrated(label, r, ref_s2=0.934, ref_lt2=1.100,
                       sharpe_ref_mark=None, maxdd_ref_mark=None):
    """統合比較表の1行 — v2.0: 新10列標準（旧11列の累積CAGRセルを廃止）"""
    sf = r.get('Sharpe_FULL', float('nan'))
    if isinstance(sf, float) and np.isnan(sf):
        mark = ''
    else:
        mark = ' ★' if sf > ref_lt2 else (' ◎' if sf > ref_s2 else '')
    s_sfx = sharpe_ref_mark or ''
    m_sfx = maxdd_ref_mark  or ''
    return (
        f'| {label} '
        f'| {_fp1(r.get("CAGR_IS"))} '
        f'| {_fp1(r["CAGR_OOS"])} '
        f'| {_ff2(sf)}{mark}{s_sfx} '
        f'| {_fp1(r["MaxDD_FULL"])}{m_sfx} '
        f'| {_worst1d(r.get("Worst1D"), r.get("Worst1D_date"))} '
        f'| {_fp1(r["Worst10Y_star"])} '
        f'| {_fp1(r.get("Worst5Y"))} '
        f'| {_fp1(r["P10_5Y"])} '
        f'| {_tr(r.get("Trades_yr"))} '
        f'| {_robustness_cell(r)} |'
    )


# ---------------------------------------------------------------------------
# §5 年次リターン表（1977-2026 / moderate / 税後）フォーマッタ
# ---------------------------------------------------------------------------

def fmt_annual_table(strategies, yearly_returns, start_year=1977):
    """§5 年次リターン表を Markdown テーブル文字列として返す。

    Parameters
    ----------
    strategies : list[str]
        列ヘッダ（戦略名、順序そのまま）
    yearly_returns : dict[str, dict[int, float]]
        yearly_returns[strategy][year] = annual_return（比率。例: 0.20 = +20%）
        税後・moderate コスト後の値を渡すこと。
    start_year : int
        表示開始年（default=1977、LT2-N750 ウォームアップ後）
    """
    header = '| 年 | ' + ' | '.join(strategies) + ' |'
    sep = '|---:|' + ':---:|' * len(strategies)
    rows = [header, sep]
    all_years = sorted({y for s in yearly_returns.values() for y in s})
    for yr in all_years:
        if yr < start_year:
            continue
        cells = []
        for s in strategies:
            v = yearly_returns.get(s, {}).get(yr)
            cells.append('  —  ' if v is None else f'{v * 100:+.1f}')
        rows.append('| ' + str(yr) + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# §6 統計サマリ（1974-2026 / moderate / 税後）フォーマッタ
# ---------------------------------------------------------------------------

def fmt_stats_table(strategies, yearly_returns, start_year=1974):
    """§6 統計サマリ（7行: mean/median/std/max/min/プラス年/マイナス年）を返す。

    Parameters
    ----------
    strategies : list[str]
    yearly_returns : dict[str, dict[int, float]]
        return は比率。税後・moderate コスト後。
    start_year : int
        集計開始年（default=1974。§5 と異なり IS 期間全体を含む）
    """
    header = '| 統計 | ' + ' | '.join(strategies) + ' |'
    sep = '|:---|' + ':---:|' * len(strategies)
    rows = [header, sep]

    stat_defs = [
        ('平均 (mean)',     lambda a: f'{np.mean(a) * 100:+.2f}%'),
        ('中央値 (median)', lambda a: f'{np.median(a) * 100:+.2f}%'),
        ('標準偏差 (std)',  lambda a: f'{np.std(a) * 100:+.2f}%'),
        ('最大 (max)',      lambda a: f'{np.max(a) * 100:+.2f}%'),
        ('最小 (min)',      lambda a: f'{np.min(a) * 100:+.2f}%'),
        ('プラス年数',      lambda a: str(sum(1 for x in a if x > 0))),
        ('マイナス年数',    lambda a: str(sum(1 for x in a if x < 0))),
    ]

    for stat_name, fn in stat_defs:
        cells = []
        for s in strategies:
            rets = np.array([r for yr, r in yearly_returns.get(s, {}).items()
                             if yr >= start_year])
            cells.append('—' if len(rets) == 0 else fn(rets))
        rows.append(f'| {stat_name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)
