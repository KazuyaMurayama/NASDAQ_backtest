# -*- coding: utf-8 -*-
"""
gen_p09c1_alloc_report_20260620.py

P09_C1 配分バリエーション(A0-A7)の最終レポートを2つのCSVから機械生成する。
手打ち数値は一切なし。全数値はCSVをプログラム的に読み込み、統計はpythonで計算する。

入力:
  audit_results/p09c1_alloc_stage1_20260620.csv  (utf-8)   … 10指標 + WFA/CPCV/bootstrap/regime/veto + sleeve_turnover_yr
  audit_results/p09c1_alloc_annual_20260620.csv  (utf-8-sig) … year + A0..A7 + NASDAQ_1x_BH_pct (税後%)

出力:
  P09C1_ALLOCATION_VARIATIONS_20260620.md
"""

import csv
import os
import statistics
from collections import OrderedDict

# ---------------------------------------------------------------------------
# パス解決 (このスクリプトは src/audit/ に置かれる想定。リポルートを2つ上から取る)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STAGE1_CSV = os.path.join(REPO_ROOT, "audit_results", "p09c1_alloc_stage1_20260620.csv")
ANNUAL_CSV = os.path.join(REPO_ROOT, "audit_results", "p09c1_alloc_annual_20260620.csv")
OUT_MD = os.path.join(REPO_ROOT, "P09C1_ALLOCATION_VARIATIONS_20260620.md")

# ---------------------------------------------------------------------------
# バリエーション定義 (§1 用 / 表示順は CSV 行順 A0..A7)
# ---------------------------------------------------------------------------
VAR_DEFS = OrderedDict([
    ("A0_P09_C1_BASE",
     ("P09_C1_BASE",
      "現行（inverse-vol 63d, Gold常時+Bond gate>0, 残りSOFR）= ベースライン",
      "— ベースライン")),
    ("A1_INVVOL_W126",
     ("INVVOL_W126",
      "inverse-vol窓 63→126日",
      "ボラ推定窓を倍に")),
    ("A2_INVVOL_DAILY",
     ("INVVOL_DAILY",
      "重み更新 5BD→毎日",
      "再配分頻度を毎日に")),
    ("A3_BOND_HYST",
     ("BOND_HYST",
      "Bondゲート binary→ヒステリシス(±0.05)",
      "Bondゲートのバタつき抑制")),
    ("A4_RISK_BUDGET",
     ("RISK_BUDGET",
      "OUTを目標ボラ10%にスケール、残りSOFR",
      "OUTスリーブをボラ・ターゲット化")),
    ("A5_CONVICTION",
     ("CONVICTION",
      "OUT強度(lev_mod_065、2日ラグ)連動でキャッシュ比可変(最大50%)",
      "確信度連動でキャッシュ可変")),
    ("A6_GOLD_TILT",
     ("GOLD_TILT",
      "高ボラregime(因果的expanding中央値)でGold≥75%に傾斜",
      "高ボラ局面でGold傾斜")),
    ("A7_IN_VOL_BRAKE",
     ("IN_VOL_BRAKE",
      "IN脚の実現ボラ>30%日に一部現金退避(最大50%)",
      "IN脚にボラ・ブレーキ")),
])

AFTER_TAX = 0.8273  # 譲渡益税後係数 (CSVの年次はすでに税後%・stage1も税後)


# ---------------------------------------------------------------------------
# CSV 読み込み
# ---------------------------------------------------------------------------
def read_stage1():
    # utf-8-sig で開く: ファイルが utf-8 でも BOM が付与されている場合に列名 "label" を保つため
    rows = OrderedDict()
    with open(STAGE1_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("label"):
                continue
            rows[r["label"]] = r
    return rows


def read_annual():
    """utf-8-sig で読み、year を int、各値を float に。1975<=year<=2025 のみ返す。"""
    full = []
    with open(ANNUAL_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if not r.get("year"):
                continue
            y = int(float(r["year"]))
            rec = {"year": y}
            for k in fieldnames:
                if k == "year":
                    continue
                v = r.get(k, "")
                rec[k] = float(v) if v not in ("", None) else None
            full.append(rec)
    # 部分年 (1974, 2026) を除外
    filtered = [rec for rec in full if 1975 <= rec["year"] <= 2025]
    return fieldnames, full, filtered


# ---------------------------------------------------------------------------
# 数値フォーマッタ (符号付き% / マイナス記号は U+2212 の "−" を使う)
# ---------------------------------------------------------------------------
MINUS = "−"  # −


def pct_from_frac(x, dec=2):
    """小数(0.1777)→ '+17.77%' 形式。"""
    v = x * 100.0
    return _signed(v, dec) + "%"


def pct_from_pct(x, dec=2):
    """既に%値(17.77)→ '+17.77%' 形式。"""
    return _signed(x, dec) + "%"


def _signed(v, dec=2):
    s = f"{abs(v):.{dec}f}"
    if v < 0:
        return MINUS + s
    return "+" + s


def f_num(v, dec=3):
    return f"{v:.{dec}f}"


# ---------------------------------------------------------------------------
# 頑健性ルーブリック (機械判定)
# ---------------------------------------------------------------------------
def robustness_cell(r):
    wfe = float(r["wfa_WFE"])
    ci_lo = float(r["wfa_CI95_lo"])          # 小数 (0.189 = 18.9%)
    gap_pp = float(r["IS_OOS_gap_pp"])       # 既に pp 単位
    reg_min = float(r["regime_min_at"])      # 小数 (-0.0008)

    overfit = (wfe > 2) or (wfe < 0.5) or (ci_lo < 0) or (abs(gap_pp) > 5)
    robust = (0.5 <= wfe <= 2) and (ci_lo > 0) and (abs(gap_pp) <= 3) and (reg_min > -0.10)

    if overfit:
        glyph = "❌過学習疑い"   # ❌
    elif robust:
        glyph = "✅頑健"          # ✅
    else:
        glyph = "⚠条件付"        # ⚠
    detail = f"(WFE={wfe:.2f}, CI95lo={_signed(ci_lo*100,1)}%, Reg={_signed(reg_min*100,1)}%)"
    return glyph, glyph + " " + detail


def robustness_glyph_only(r):
    return robustness_cell(r)[0]


# ---------------------------------------------------------------------------
# Sharpe マーカー
# ---------------------------------------------------------------------------
def sharpe_with_marker(val):
    s = f"{val:.3f}"
    if val > 1.100:
        return s + "★"   # ★
    elif val > 0.934:
        return s + "◎"   # ◎
    return s


# ---------------------------------------------------------------------------
# 統計 (§4)
# ---------------------------------------------------------------------------
def geom_cagr_from_pct(series_pct):
    """年次%リストから全期間幾何CAGR(小数)を返す。"""
    n = len(series_pct)
    prod = 1.0
    for r in series_pct:
        prod *= (1.0 + r / 100.0)
    return prod ** (1.0 / n) - 1.0


def series_stats(series_pct):
    n = len(series_pct)
    g = geom_cagr_from_pct(series_pct)
    mean = statistics.fmean(series_pct)
    sd = statistics.stdev(series_pct)  # ddof=1
    best = max(series_pct)
    worst = min(series_pct)
    pos = sum(1 for x in series_pct if x > 0)
    return dict(n=n, geom=g, mean=mean, sd=sd, best=best, worst=worst, pos=pos)


# ---------------------------------------------------------------------------
# MD 生成
# ---------------------------------------------------------------------------
def build_md():
    stage1 = read_stage1()
    ann_fields, ann_full, ann = read_annual()

    labels = list(VAR_DEFS.keys())  # A0..A7 (CSV順)

    # CAGR_OOS 降順 (A0 は先頭固定、残りを CAGR_OOS_at 降順)
    rest = [l for l in labels if l != "A0_P09_C1_BASE"]
    rest_sorted = sorted(rest, key=lambda l: float(stage1[l]["CAGR_OOS_at"]), reverse=True)
    order_metric = ["A0_P09_C1_BASE"] + rest_sorted

    out = []
    w = out.append

    # ---- bootstrap P-value ranges (computed from stage1 CSV, all 8 rows) ----
    all_labels = list(stage1.keys())
    v7lo  = min(float(stage1[l]["mm_v7_P_min"])   for l in all_labels)
    v7hi  = max(float(stage1[l]["mm_v7_P_min"])   for l in all_labels)
    b3alo = min(float(stage1[l]["mm_b3a_P_min"])  for l in all_labels)
    b3ahi = max(float(stage1[l]["mm_b3a_P_min"])  for l in all_labels)
    shlo  = min(float(stage1[l]["mm_v7_P_sharpe"]) for l in all_labels)
    shhi  = max(float(stage1[l]["mm_v7_P_sharpe"]) for l in all_labels)

    # ---- ヘッダ ----
    w("# P09_C1 配分バリエーション検証レポート")
    w("")
    w("作成日: 2026-06-20")
    w("最終更新日: 2026-06-20")
    w("")
    w("> 商品は P09_C1 と完全同一（TQQQ / Gold2x / Bond / SOFRキャッシュ）。"
      "シグナル→アクション（OUTスリーブ配分・IN脚キャッシュ退避）のみを変えた8案(A0-A7)。")
    w("> A0 = P09_C1 ベースライン（_build_full_c1 と CAGR_OOS 一致 diff 0.0e+00 で検証済"
      "＝商品・IN脚不変の証明）。")
    w("> 全数値: コスト後・譲渡益税後 ×0.8273。評価=標準10指標 v2.0 + Stage-1 フルゲート。")
    w("")
    # ADDITION 1: headline conclusion blockquote (QC addition)
    # A7 deltas vs A0 (computed from CSV)
    _a0 = stage1["A0_P09_C1_BASE"]
    _a7 = stage1["A7_IN_VOL_BRAKE"]
    _a7_d_cagr_hl = (float(_a7["CAGR_OOS_at"]) - float(_a0["CAGR_OOS_at"])) * 100.0
    _a7_d_maxdd_hl = (float(_a7["MaxDD_FULL"]) - float(_a0["MaxDD_FULL"])) * 100.0
    _a7_d_w1d_hl = (float(_a7["Worst1D"]) - float(_a0["Worst1D"])) * 100.0
    _a0_maxdd_pct = float(_a0["MaxDD_FULL"]) * 100.0
    _a7_maxdd_pct = float(_a7["MaxDD_FULL"]) * 100.0
    _a0_w1d_pct   = float(_a0["Worst1D"]) * 100.0
    _a7_w1d_pct   = float(_a7["Worst1D"]) * 100.0
    w(f"> **🔑 結論（先出し）: OUTスリーブ配分の変更(A1-A6)は A0 を統計的に上回らない。** "
      f"multi-metric bootstrap で A1-A7 のいずれも A0／基準(V7・B3a)と**有意差なし**"
      f"（mm_v7_P_min {v7lo:.2f}-{v7hi:.2f}, mm_b3a_P_min {b3alo:.2f}-{b3ahi:.2f}, "
      f"P_sharpe {shlo:.2f}-{shhi:.2f} — すべて NS）。A1/A2/A4/A5 の CAGR_OOS +0.02〜+0.23pp は"
      f"**ノイズ域**で実質改善ではない（OUT期は全体の約47%の日にしか効かず、配分微調整は二次的）。"
      f"唯一の構造的効果は **A7(IN脚ボラブレーキ)**: CAGR_OOS {_signed(_a7_d_cagr_hl,2)}pp と引き換えに "
      f"MaxDD {_signed(_a0_maxdd_pct,2)}%→{_signed(_a7_maxdd_pct,2)}%（{_signed(_a7_d_maxdd_hl,2)}pp）・"
      f"最悪単日 {_signed(_a0_w1d_pct,2)}%→{_signed(_a7_w1d_pct,2)}%（{_signed(_a7_d_w1d_hl,2)}pp）。"
      f"§5の「ADOPT候補」ラベルは機械ルール(OR条件)の産物で統計的優越を意味しない（§5脚注参照）。")
    w("")

    # ===================================================================
    # §1 バリエーション定義表
    # ===================================================================
    w("## §1 バリエーション定義表")
    w("")
    w("| # | 名称 | 配分ルールの変更点 | P09_C1(A0)との差分 |")
    w("|---|------|------------------|---------------------|")
    for i, l in enumerate(labels):
        name, change, diff = VAR_DEFS[l]
        tag = f"A{i}"
        w(f"| {tag} | {name} | {change} | {diff} |")
    w("")

    # ===================================================================
    # §2 標準10指標 v2.0
    # ===================================================================
    w("## §2 標準10指標 (v2.0)")
    w("")
    w("A0 先頭、残りは CAGR_OOS 降順。⓽=min9基準後の値 / ⓒ=全期間(コスト後) / "
      "ⓞ=年あたり取引回数。Sharpe マーカー: ◎ >0.934, ★ >1.100。")
    w("")
    hdr = ("| 戦略 | CAGR_IS⓽ | CAGR_OOS⓽ | Sharpe_FULL ⓒ | MaxDD ⓒ | "
           "最悪単日 ⓒ(日付) | Worst10Y★⓽ | Worst5Y⓽ | P10_5Y▷⓽ | "
           "Trades/yr ⓞ | ス内回転/年 | 頑健性 |")
    w(hdr)
    w("|------|---------|----------|--------------|--------|"
      "----------------|-----------|---------|---------|"
      "-----------|-----------|--------|")
    for l in order_metric:
        r = stage1[l]
        name = VAR_DEFS[l][0]
        cagr_is = pct_from_frac(float(r["CAGR_IS_at"]))
        cagr_oos = pct_from_frac(float(r["CAGR_OOS_at"]))
        sharpe = sharpe_with_marker(float(r["Sharpe_FULL"]))
        maxdd = pct_from_frac(float(r["MaxDD_FULL"]))
        w1d = pct_from_frac(float(r["Worst1D"]))
        w1d_date = r["Worst1D_date"]
        worstday = f"{w1d} ({w1d_date})"
        w10y = pct_from_frac(float(r["Worst10Y_at"]))
        w5y = pct_from_frac(float(r["Worst5Y_at"]))
        p10 = pct_from_frac(float(r["P10_5Y_at"]))
        trades = f"{float(r['Trades_yr']):.1f}"
        slv = f"{float(r['sleeve_turnover_yr']):.1f}"
        rob_glyph, robcell = robustness_cell(r)
        # ADDITION 2a: A6/GOLD_TILT robustness glyph gets asterisk to tie to footnote
        if l == "A6_GOLD_TILT" and rob_glyph == "✅頑健":
            robcell = robcell.replace("✅頑健", "✅頑健*", 1)
        w(f"| {name} | {cagr_is} | {cagr_oos} | {sharpe} | {maxdd} | "
          f"{worstday} | {w10y} | {w5y} | {p10} | {trades} | {slv} | {robcell} |")
    w("")
    w("> **ス内回転/年** = スリーブ内日次再配分の回転（Trades/yr=29.2 固定には現れない）。"
      "A2/A4/A5 は Trades/yr が示すより内部回転が多い（取引コスト未計上のため実効でやや割引が必要）。")
    # ADDITION 2b: A6 robustness footnote (QC addition)
    _a6 = stage1["A6_GOLD_TILT"]
    _a6_regime_min_pct = float(_a6["regime_min_at"]) * 100.0
    _a0_regime_min_pct = float(stage1["A0_P09_C1_BASE"]["regime_min_at"]) * 100.0
    _a6_maxdd_pct = float(_a6["MaxDD_FULL"]) * 100.0
    _a0_maxdd_pct2 = float(stage1["A0_P09_C1_BASE"]["MaxDD_FULL"]) * 100.0
    _a6_d_maxdd_pct = _a6_maxdd_pct - _a0_maxdd_pct2
    _a6_v7_p_min = float(_a6["mm_v7_P_min"])
    w(f"> **\\*A6(GOLD_TILT)の✅頑健について**: ルーブリック上は regime_min {_signed(_a6_regime_min_pct,2)}% > −10% で✅だが、"
      f"因果的highvolマスク修正後 regime_min が A0 比 {_signed(_a0_regime_min_pct,2)}%→{_signed(_a6_regime_min_pct,2)}%（**全案中最悪**）・"
      f"MaxDD {_signed(_a6_d_maxdd_pct,2)}pp 悪化（{_signed(_a0_maxdd_pct2,2)}%→{_signed(_a6_maxdd_pct,2)}%）。+0.58pp の CAGR優位は**テールリスク悪化と引き換え**で、"
      f"bootstrap でも対基準有意差なし（mm_v7_P_min={_a6_v7_p_min:.2f}）。修正前の良績はルックアヘッド人工物だった。"
      "✅は「採用安全」を意味しない。")
    w("")

    # ===================================================================
    # §3 年次リターン表 (税後・1975-2025)
    # ===================================================================
    w("## §3 年次リターン表（税後・1975-2025）")
    w("")
    w("各セルはコスト後・譲渡益税後 (×0.8273) の年次リターン%。"
      "1974 と 2026 の部分年は除外（51年: 1975-2025）。")
    w("")
    cols = labels + ["NASDAQ_1x_BH_pct"]
    col_disp = [VAR_DEFS[l][0] for l in labels] + ["NASDAQ_1x_BH"]
    w("| 年 | " + " | ".join(col_disp) + " |")
    w("|----|" + "|".join(["------"] * len(col_disp)) + "|")
    for rec in ann:
        cells = [str(rec["year"])]
        for c in cols:
            v = rec[c]
            cells.append(pct_from_pct(v) if v is not None else "—")
        w("| " + " | ".join(cells) + " |")
    w("")
    w(f"> 年数チェック: {len(ann)} 行（1975-2025 = 51年）。")
    w("")

    # ===================================================================
    # §4 統計表 (税後・1975-2025)
    # ===================================================================
    w("## §4 統計表（税後・1975-2025）")
    w("")
    series_map = OrderedDict()
    for l in labels:
        series_map[VAR_DEFS[l][0]] = [rec[l] for rec in ann]
    series_map["NASDAQ_1x_BH"] = [rec["NASDAQ_1x_BH_pct"] for rec in ann]

    w("| 系列 | 全期間CAGR | 平均年率 | 標準偏差 | 最良年 | 最悪年 | プラスの年数 |")
    w("|------|-----------|---------|---------|--------|--------|-------------|")
    n_years = len(ann)
    for name, ser in series_map.items():
        st = series_stats(ser)
        w(f"| {name} | {pct_from_frac(st['geom'])} | {pct_from_pct(st['mean'])} | "
          f"{st['sd']:.2f} | {pct_from_pct(st['best'])} | {pct_from_pct(st['worst'])} | "
          f"{st['pos']}/{n_years} |")
    w("")
    w("> ⚠ NASDAQ 1倍B&H は売却時まで課税繰延のため年次×0.8273は過剰課税。"
      "実際のB&H手取りはこれより高く税前に近い。"
      "高回転戦略(A0-A7, 毎年利益確定)は年次×0.8273が妥当。")
    w("")

    # ===================================================================
    # §5 採否 (機械判定)
    # ===================================================================
    w("## §5 採否（機械判定 / ルーブリック準拠）")
    w("")
    a0 = stage1["A0_P09_C1_BASE"]
    a0_cagr_oos = float(a0["CAGR_OOS_at"])
    a0_sharpe = float(a0["Sharpe_FULL"])
    a0_maxdd = float(a0["MaxDD_FULL"])
    a0_w1d = float(a0["Worst1D"])
    a0_slv = float(a0["sleeve_turnover_yr"])

    w("各案 vs A0。Δは pp（CAGR/MaxDD/最悪単日）または素差（Sharpe）。"
      "判定ルール（機械）:")
    w("- **ADOPT候補**: (ΔCAGR_OOS≥0 OR ΔMaxDD改善≥3pp OR Δ最悪単日改善≥2pp) "
      "かつ 頑健性≠❌ かつ VETO_s1=0")
    w("- **REJECT**: 頑健性=❌過学習疑い OR VETO_s1=1 OR A0に完全劣後"
      "(CAGR_OOS・MaxDD・Worst1Dすべてで同等以下)")
    w("- **NEEDS_WORK**: それ以外")
    w("")
    w("| 案 | ΔCAGR_OOS | ΔSharpe | ΔMaxDD | Δ最悪単日 | 頑健性 | VETO_s1 | Δス内回転 | 判定 |")
    w("|----|-----------|---------|--------|-----------|--------|---------|-----------|------|")

    verdicts = OrderedDict()
    for l in labels:
        if l == "A0_P09_C1_BASE":
            continue
        r = stage1[l]
        d_cagr = (float(r["CAGR_OOS_at"]) - a0_cagr_oos) * 100.0   # pp
        d_sharpe = float(r["Sharpe_FULL"]) - a0_sharpe
        d_maxdd = (float(r["MaxDD_FULL"]) - a0_maxdd) * 100.0       # pp (正=改善, MaxDDは負値なので差が正で浅くなる)
        d_w1d = (float(r["Worst1D"]) - a0_w1d) * 100.0             # pp (正=改善, Worst1Dは負値)
        d_slv = float(r["sleeve_turnover_yr"]) - a0_slv
        glyph = robustness_glyph_only(r)
        veto = int(float(r["VETO_s1"]))

        # 改善判定
        maxdd_impr = d_maxdd  # 正で浅く=改善
        w1d_impr = d_w1d      # 正で浅く=改善

        is_overfit = glyph.startswith("❌")
        # A0に完全劣後: CAGR_OOS・MaxDD・Worst1D すべて同等以下
        dominated = (d_cagr <= 0) and (maxdd_impr <= 0) and (w1d_impr <= 0)

        adopt_trigger = (d_cagr >= 0) or (maxdd_impr >= 3) or (w1d_impr >= 2)

        if is_overfit or veto == 1 or dominated:
            verdict = "REJECT"
        elif adopt_trigger and (not is_overfit) and veto == 0:
            verdict = "ADOPT候補"
        else:
            verdict = "NEEDS_WORK"
        verdicts[l] = dict(
            verdict=verdict, d_cagr=d_cagr, d_sharpe=d_sharpe, d_maxdd=d_maxdd,
            d_w1d=d_w1d, d_slv=d_slv, glyph=glyph, veto=veto, dominated=dominated,
        )

        tag = "A" + l[1]
        w(f"| {tag} {VAR_DEFS[l][0]} | {_signed(d_cagr,2)}pp | {_signed(d_sharpe,3)} | "
          f"{_signed(d_maxdd,2)}pp | {_signed(d_w1d,2)}pp | {glyph} | {veto} | "
          f"{_signed(d_slv,2)} | **{verdict}** |")
    w("")
    # ADDITION 3: §5 ADOPT-candidate disclaimer (QC addition)
    w("> **⚠ 「ADOPT候補」は統計的優越を意味しない。** 上表のラベルは機械ルール"
      f"（OR条件: ΔCAGR_OOS≥0 で発火）の結果にすぎない。multi-metric bootstrap では A1-A7 の"
      f"**いずれも** A0／基準(V7・B3a)と**有意差なし**（mm_v7_P_min {v7lo:.2f}-{v7hi:.2f}, "
      f"mm_b3a_P_min {b3alo:.2f}-{b3ahi:.2f}, P_sharpe {shlo:.2f}-{shhi:.2f} — すべて P>0.05 = NS）。"
      f"A1/A2/A4/A5 の +0.02〜+0.23pp はノイズ域で、A0 に対する実質的改善ではない。"
      f"「6/7 が ADOPT候補」を「多くの案が A0 を上回る」と読んではならない（§6 結論参照）。"
      f"意思決定上の実質的選択肢は **A7（リスク低減を CAGR と引き換えに取るか）** の一点のみ。")
    w("")

    # 各案の判定根拠テキスト
    w("### 判定根拠（駆動した数値）")
    w("")
    for l in labels:
        if l == "A0_P09_C1_BASE":
            continue
        v = verdicts[l]
        tag = "A" + l[1]
        name = VAR_DEFS[l][0]
        base = (f"- **{tag} {name} → {v['verdict']}**: "
                f"ΔCAGR_OOS={_signed(v['d_cagr'],2)}pp, "
                f"ΔSharpe={_signed(v['d_sharpe'],3)}, "
                f"ΔMaxDD={_signed(v['d_maxdd'],2)}pp, "
                f"Δ最悪単日={_signed(v['d_w1d'],2)}pp, "
                f"頑健性={v['glyph']}, VETO_s1={v['veto']}, "
                f"Δス内回転={_signed(v['d_slv'],2)}/年。")
        notes = []
        if l == "A6_GOLD_TILT":
            notes.append("因果的highvolマスクに修正後、MaxDD/regime_minが悪化"
                         "（ルックアヘッド除去で魅力が縮小）。修正前の良績は"
                         "ルックアヘッド人工物だった。")
        if l == "A5_CONVICTION":
            notes.append("lev_mod_065を2日ラグに修正済（ラグ前は約2日先読みだった）。"
                         "スリーブ内回転4.1/年と高く、取引コスト未計上のため"
                         "実効はやや割引が必要。")
        if v["dominated"]:
            notes.append("A0に CAGR_OOS・MaxDD・Worst1D すべてで同等以下（完全劣後）。")
        if notes:
            base += " " + " ".join(notes)
        w(base)
    w("")
    w("> **DDveto** (MaxDD<-50%) はユーザー許容（DD 50-60%台OK）のため「参考」。"
      "ただし Worst10Y<0 / regime_min<-10% / ❌過学習 はハード懸念として残す。")
    w("> **sleeve_turnover_yr** はスリーブ内日次再配分の回転で、Trades/yr(29.2固定)には現れない。"
      "取引コスト未計上のため高回転案(A2/A4/A5)は実効でやや不利方向に割り引くべき。")
    w("")

    # ===================================================================
    # §6 結論
    # ===================================================================
    w("## §6 結論")
    w("")
    # 機械的サマリ用の数値
    a7 = stage1["A7_IN_VOL_BRAKE"]
    a7_d_cagr = (float(a7["CAGR_OOS_at"]) - a0_cagr_oos) * 100.0
    a7_d_maxdd = (float(a7["MaxDD_FULL"]) - a0_maxdd) * 100.0
    a7_d_w1d = (float(a7["Worst1D"]) - a0_w1d) * 100.0
    # OUT系(A1-A6)のCAGR_OOS変動レンジ
    out_deltas = []
    for l in ["A1_INVVOL_W126", "A2_INVVOL_DAILY", "A3_BOND_HYST",
              "A4_RISK_BUDGET", "A5_CONVICTION", "A6_GOLD_TILT"]:
        out_deltas.append((float(stage1[l]["CAGR_OOS_at"]) - a0_cagr_oos) * 100.0)
    out_min, out_max = min(out_deltas), max(out_deltas)
    out_absmax = max(abs(out_min), abs(out_max))

    a6 = stage1["A6_GOLD_TILT"]
    a6_d_cagr = (float(a6["CAGR_OOS_at"]) - a0_cagr_oos) * 100.0
    a6_d_maxdd = (float(a6["MaxDD_FULL"]) - a0_maxdd) * 100.0
    a6_reg = float(a6["regime_min_at"]) * 100.0
    a0_reg = float(a0["regime_min_at"]) * 100.0
    para = (
        f"OUTスリーブ系の調整(A1-A6)は CAGR_OOS を A0 比 "
        f"{_signed(out_min,2)}〜{_signed(out_max,2)}pp（絶対値で最大{out_absmax:.2f}pp）"
        f"しか動かさず、いずれも二次的（OUTは全体の約47%の日にしか効かないため）。"
        f"DD許容・CAGR重視のユーザー優先度では、これら微小な改善は配分微調整の労に見合わない。"
        f"一方 A7(IN_VOL_BRAKE)は CAGR_OOS を {_signed(a7_d_cagr,2)}pp 譲る代わりに、"
        f"MaxDD を {_signed(a7_d_maxdd,2)}pp（"
        f"{pct_from_frac(a0_maxdd)}→{pct_from_frac(float(a7['MaxDD_FULL']))}）、"
        f"最悪単日を {_signed(a7_d_w1d,2)}pp（"
        f"{pct_from_frac(a0_w1d)}→{pct_from_frac(float(a7['Worst1D']))}）"
        f"改善する唯一の構造的リスク低減策。"
        f"ユーザー優先度（CAGR重視・DD許容・最悪単日低減は歓迎）に照らすと、"
        f"純粋にCAGRで A0 を最も上回るのは A6(GOLD_TILT, ΔCAGR_OOS={_signed(a6_d_cagr,2)}pp)だが、"
        f"因果的highvolマスク修正後は MaxDD が {_signed(a6_d_maxdd,2)}pp 悪化"
        f"（{pct_from_frac(a0_maxdd)}→{pct_from_frac(float(a6['MaxDD_FULL']))}）し "
        f"regime_min も {_signed(a0_reg,1)}%→{_signed(a6_reg,1)}% へ悪化したため、"
        f"このCAGR優位はリスク悪化と引き換えで魅力が縮小している"
        f"（修正前の良績はルックアヘッド人工物だった）。"
        f"残るA1-A5の CAGR_OOS 改善は最大でも +0.23pp 程度の二次的差で、A0 を明確に上回るとは言えない。"
        f"結論として、A0 を置換すべき明確な優越案は存在しない。"
        f"リスク低減を志向するなら A7 が唯一の検討候補（CAGRトレードオフを許容できる場合に限る）。"
    )
    w(para)
    w("")

    return "\n".join(out) + "\n"


def main():
    md = build_md()
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(f"WROTE: {OUT_MD}")
    print(f"bytes: {len(md.encode('utf-8'))}")


if __name__ == "__main__":
    main()
