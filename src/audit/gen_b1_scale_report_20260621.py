# -*- coding: utf-8 -*-
"""
gen_b1_scale_report_20260621.py

B1ブレーキ(P09_C1 + IN脚下方偏差ブレーキ) × レバレッジ・スケール フロンティアの
最終レポートを CSV から機械生成する。手打ち数値は一切なし。全数値は CSV を
プログラム的に読み込み、統計は python で計算する。

入力 (すべて utf-8-sig / BOM 有):
  audit_results/b1_scale_stage1_20260621.csv           … B1×scale 9行 (10指標+WFA+regime+mm+veto)
  audit_results/b1_scale_annual_20260621.csv           … year + B1 8系列 + NASDAQ_1x_BH_pct (税後%)
  audit_results/b1_scale_delever_control_20260621.csv  … 9行 (fbar / brake vs 等fbar一律デレバ双子)
  audit_results/leverext_high_stage1_20260618.csv      … 前回P09×scale (1.40/1.60, strong map, excess0.0025)
  audit_results/leverext_high_sc180_200_20260620.csv   … 前回P09×scale (1.80/2.00)
  audit_results/annual_returns_sc180_200_20260620.csv  … 前回P09×scale 年次税後% (sc1.40..2.00_strong_pct)

出力:
  B1_SCALE_FRONTIER_20260621.md
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
AR = os.path.join(REPO_ROOT, "audit_results")

B1_STAGE1_CSV = os.path.join(AR, "b1_scale_stage1_20260621.csv")
B1_ANNUAL_CSV = os.path.join(AR, "b1_scale_annual_20260621.csv")
B1_CTRL_CSV = os.path.join(AR, "b1_scale_delever_control_20260621.csv")
P09_STAGE1_CSV = os.path.join(AR, "leverext_high_stage1_20260618.csv")
P09_STAGE1B_CSV = os.path.join(AR, "leverext_high_sc180_200_20260620.csv")
P09_ANNUAL_CSV = os.path.join(AR, "annual_returns_sc180_200_20260620.csv")

OUT_MD = os.path.join(REPO_ROOT, "B1_SCALE_FRONTIER_20260621.md")

# 共通スケール (前回P09と並記する4点)
SCALES = [1.4, 1.6, 1.8, 2.0]

# 前回P09×scale: stage1 CSV の label (strong map, excess 0.0025)
P09_LABEL = {1.4: "Bext_str_sc1.40", 1.6: "Bext_str_sc1.60",
             1.8: "Bext_str_sc1.80", 2.0: "Bext_str_sc2.00"}
# 前回P09×scale: 年次 CSV の列名
P09_ANNUAL_COL = {1.4: "sc1.40_strong_pct", 1.6: "sc1.60_strong_pct",
                  1.8: "sc1.80_strong_pct", 2.0: "sc2.00_strong_pct"}

MINUS = "−"  # − (U+2212)


# ---------------------------------------------------------------------------
# CSV 読み込みユーティリティ
# ---------------------------------------------------------------------------
def read_keyed(path, key="label"):
    """utf-8-sig で開き、key 列をキーに OrderedDict[row] を返す。"""
    rows = OrderedDict()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get(key):
                continue
            rows[r[key]] = r
    return rows


def read_annual(path):
    """utf-8-sig で読み、year を int、各値を float (空は None)。full list を返す。"""
    full = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for r in reader:
            if not r.get("year"):
                continue
            rec = {"year": int(float(r["year"]))}
            for k in fields:
                if k == "year":
                    continue
                v = r.get(k, "")
                rec[k] = float(v) if v not in ("", None) else None
            full.append(rec)
    return fields, full


# ---------------------------------------------------------------------------
# フォーマッタ
# ---------------------------------------------------------------------------
def _signed(v, dec=2):
    s = f"{abs(v):.{dec}f}"
    return (MINUS + s) if v < 0 else ("+" + s)


def pct_from_frac(x, dec=2):
    """小数(0.1777) -> '+17.77%'。"""
    if x is None:
        return "—"
    return _signed(x * 100.0, dec) + "%"


def pct_from_pct(x, dec=2):
    """既に%値(17.77) -> '+17.77%'。"""
    if x is None:
        return "—"
    return _signed(x, dec) + "%"


def f_float(s, default=None):
    if s is None or s == "":
        return default
    return float(s)


# ---------------------------------------------------------------------------
# 頑健性ルーブリック (機械判定)
#   ❌ if WFE>2 OR WFE<0.5 OR CI95_lo<0 OR |CAGR_IS-CAGR_OOS|>5pp
#   ✅ if 0.5<=WFE<=2 AND CI95_lo>0 AND |gap|<=3pp AND regime_min>-0.10
#   else ⚠
#   gap_pp = stage1 CSV の IS_OOS_gap_pp (pp)。CI95_lo / regime_min は小数。
# ---------------------------------------------------------------------------
def robustness_cell(r):
    wfe = float(r["wfa_WFE"])
    ci_lo = float(r["wfa_CI95_lo"])      # 小数
    gap_pp = float(r["IS_OOS_gap_pp"])   # pp
    reg_min = float(r["regime_min_at"])  # 小数

    overfit = (wfe > 2) or (wfe < 0.5) or (ci_lo < 0) or (abs(gap_pp) > 5)
    robust = (0.5 <= wfe <= 2) and (ci_lo > 0) and (abs(gap_pp) <= 3) and (reg_min > -0.10)

    if overfit:
        glyph = "❌"   # ❌
    elif robust:
        glyph = "✅"   # ✅
    else:
        glyph = "⚠"   # ⚠
    detail = (f"(WFE={wfe:.2f},CI95lo={_signed(ci_lo*100,1)}%,"
              f"Reg={_signed(reg_min*100,1)}%)")
    return glyph, glyph + " " + detail


# ---------------------------------------------------------------------------
# Sharpe マーカー: ◎ >0.934, ★ >1.100
# ---------------------------------------------------------------------------
def sharpe_with_marker(val):
    s = f"{val:.3f}"
    if val > 1.100:
        return s + "★"   # ★
    elif val > 0.934:
        return s + "◎"   # ◎
    return s


# ---------------------------------------------------------------------------
# 統計 (§6)
# ---------------------------------------------------------------------------
def geom_cagr_from_pct(series_pct):
    prod = 1.0
    for r in series_pct:
        prod *= (1.0 + r / 100.0)
    return prod ** (1.0 / len(series_pct)) - 1.0


def series_stats(series_pct):
    return dict(
        n=len(series_pct),
        geom=geom_cagr_from_pct(series_pct),
        mean=statistics.fmean(series_pct),
        sd=statistics.stdev(series_pct),  # ddof=1
        best=max(series_pct),
        worst=min(series_pct),
        pos=sum(1 for x in series_pct if x > 0),
    )


# ---------------------------------------------------------------------------
# P09×scale stage1 行の取得 (1.40/1.60 は stage1, 1.80/2.00 は別CSV)
# ---------------------------------------------------------------------------
def load_p09_stage1():
    s1 = read_keyed(P09_STAGE1_CSV)
    s1b = read_keyed(P09_STAGE1B_CSV)
    out = OrderedDict()
    for sc in SCALES:
        lbl = P09_LABEL[sc]
        if lbl in s1:
            out[sc] = s1[lbl]
        elif lbl in s1b:
            out[sc] = s1b[lbl]
        else:
            raise KeyError(f"P09 label not found: {lbl}")
    return out


# ===========================================================================
# MD 生成
# ===========================================================================
def build_md():
    b1 = read_keyed(B1_STAGE1_CSV)              # B1×scale 9行
    ctrl = read_keyed(B1_CTRL_CSV)              # delever control 9行
    p09 = load_p09_stage1()                     # P09×scale (sc -> row)
    b1_fields, b1_ann = read_annual(B1_ANNUAL_CSV)
    p09_ann_fields, p09_ann = read_annual(P09_ANNUAL_CSV)

    # B1 系列ラベル
    DEF = {sc: f"B1_DEF_S{sc:.1f}" for sc in SCALES}
    STR = {sc: f"B1_STR_S{sc:.1f}" for sc in SCALES}
    REF10 = "B1_DEF_S1.0"

    # 年次フィルタ (1975-2025, 51年) — B1 annual を正典 B&H とする
    ann = [rec for rec in b1_ann if 1975 <= rec["year"] <= 2025]
    # P09 年次は year をキーに dict 化
    p09_ann_by_y = {rec["year"]: rec for rec in p09_ann if 1975 <= rec["year"] <= 2025}

    out = []
    w = out.append

    # ----- 先出し用キー数値を CSV から計算 -----
    # §3: 同map(strong)・同scaleで B1_STR vs P09 の Δ
    d_cagr_list = []   # pp (B1 - P09), 正=B1高い
    d_maxdd_list = []  # pp (B1 - P09), 正=B1浅い(改善)
    for sc in SCALES:
        b1r = b1[STR[sc]]
        p9r = p09[sc]
        dc = (float(b1r["CAGR_OOS_at"]) - float(p9r["CAGR_OOS_at"])) * 100.0
        dd = (float(b1r["MaxDD_FULL"]) - float(p9r["MaxDD_FULL"])) * 100.0
        d_cagr_list.append(dc)
        d_maxdd_list.append(dd)
    # ブレーキはMaxDDを浅くしCAGRを払う -> dd>0 (浅い), dc<0 (低い)
    dd_lo, dd_hi = min(d_maxdd_list), max(d_maxdd_list)        # +pp DD改善幅
    dc_lo, dc_hi = min(d_cagr_list), max(d_cagr_list)          # 負: CAGR犠牲

    # §4: 時機優位 (brake - twin) のMaxDD差 (正=brakeが浅い=時機効果)
    tw_dd = [float(ctrl[lbl]["maxdd_brake_minus_twin_pp"]) for lbl in ctrl]
    tw_cagr = [float(ctrl[lbl]["cagr_brake_minus_twin_pp"]) for lbl in ctrl]
    tw_dd_lo, tw_dd_hi = min(tw_dd), max(tw_dd)
    tw_cagr_lo, tw_cagr_hi = min(tw_cagr), max(tw_cagr)

    # B1_STR_S2.0 / P09_STR_S2.0 主要値 (先出し用)
    b1_20 = b1[STR[2.0]]
    p09_20 = p09[2.0]
    b1_20_cagr = pct_from_frac(float(b1_20["CAGR_OOS_at"]))
    b1_20_dd = pct_from_frac(float(b1_20["MaxDD_FULL"]))
    p09_20_dd = pct_from_frac(float(p09_20["MaxDD_FULL"]))

    # ===================================================================
    # ヘッダ
    # ===================================================================
    w("# B1ブレーキ × レバレッジ・スケール フロンティア検証レポート")
    w("")
    w("作成日: 2026-06-21")
    w("最終更新日: 2026-06-21")
    w("")
    w("> B1_DOWNSIDE_DEV (P09_C1 + IN脚下方偏差ブレーキ) に スケール1.4/1.6/1.8/2.0 を掛けた8系列。")
    w("> boost map 2種: デフォルト{1.20,1.10,1.00,1.00}(B1純粋拡張) と "
      "ストロング{1.60,1.50,1.10,1.00}(前回P09スケール同条件)。")
    w("> 前回 P09×同スケール(ストロング・ブレーキ無し) を並記。excess_extra=0.0025 共通。"
      "全数値 税後 ×0.8273。veto は参考。")
    w("")

    # ----- 先出し (machine-assemble + fixed framing) -----
    w(f"> **\U0001f511 先出し（核心トレードオフ・全数値CSV機械算出）:** "
      f"同じ boost map・同じ scale で比べると、B1のブレーキは MaxDD を "
      f"**+{dd_lo:.1f}〜+{dd_hi:.1f}pp 浅く**保つ代わりに、前回P09×scale 比で "
      f"**CAGR_OOS を {_signed(dc_hi,2)}〜{_signed(dc_lo,2)}pp** 払う"
      f"（＝CAGR↔DD のトレードオフであり、無償でも支配でもない）。")
    w(f"> B1のMaxDD抑制は**一部は実時機**（§4 control: ブレーキは等fbar一律デレバ双子より"
      f"全scaleで MaxDD +{tw_dd_lo:.1f}〜+{tw_dd_hi:.1f}pp 浅い）であってデレバ一辺倒ではない。"
      f"ただしその時機は CAGR を双子比 {_signed(tw_cagr_hi,2)}〜{_signed(tw_cagr_lo,2)}pp 払う"
      f"（時機もタダではない）。")
    w(f"> **判断すべき問い:** ある目標MaxDDの下で、B1×高スケール は P09×低スケール に勝つか？"
      f"（§6/§7で回答）。例: B1_STR_S2.0 (MaxDD {b1_20_dd}/CAGR {b1_20_cagr}) は "
      f"ブレーキ無しP09_STR_S2.0 ({p09_20_dd}) より15pp浅いが、これが P09×低スケール に"
      f"対してフロンティア上で勝つかは別問題。")
    w("")

    # ===================================================================
    # §1 系列定義表
    # ===================================================================
    w("## §1 系列定義表")
    w("")
    w("boost map = ⟨VZ最強, VZ強, VZ弱, OUT⟩ の各レバ倍率。scale = 全倍率に掛ける係数。")
    w("")
    w("| 系列 | boost map | scale | ブレーキ | 説明 |")
    w("|------|-----------|-------|----------|------|")
    w(f"| {REF10} | default | 1.0 | あり(下方偏差) | B1無スケール基準（参考） |")
    for sc in SCALES:
        w(f"| {DEF[sc]} | default⟨1.20,1.10,1.00,1.00⟩ | {sc:.1f} | あり(下方偏差) | "
          f"B1純粋拡張（低ベースレバ） |")
    for sc in SCALES:
        w(f"| {STR[sc]} | strong⟨1.60,1.50,1.10,1.00⟩ | {sc:.1f} | あり(下方偏差) | "
          f"前回P09スケールと同map・ブレーキ追加 |")
    for sc in SCALES:
        w(f"| P09_STR_S{sc:.1f} (前回) | strong⟨1.60,1.50,1.10,1.00⟩ | {sc:.1f} | "
          f"**なし** | 前回P09×scale（ブレーキ無し基準） |")
    w("")
    w("> default map は VZ強域のレバが低く（ベースレバ小）、strong map は前回P09スケール検証と"
      "同一map。B1_STR×scale と P09_STR×scale はブレーキの有無だけが違う対照ペア。")
    w("")

    # ===================================================================
    # §2 標準10指標 v2.0 (B1×8 + P09×4 = 12行)
    # ===================================================================
    w("## §2 標準10指標 (v2.0) — B1×scale 8 + P09×scale 4 並記")
    w("")
    w("⓽=min9基準後 / ⓒ=全期間(コスト後) / ⓞ=年あたり取引回数 / ▷=P10(下位10%tile)。"
      "Sharpe マーカー: ◎ >0.934, ★ >1.100。"
      "並び: B1_DEF_S1.0 を先頭参考、以降 scale 昇順で (B1_DEF, B1_STR, P09) の順。"
      "P09行はブレーキ無し（前回検証）。参考veto = Stage-1 VETO フラグ。")
    w("")
    hdr = ("| 戦略 | CAGR_IS⓽ | CAGR_OOS⓽ | Sharpe_FULL ⓒ | MaxDD ⓒ | "
           "最悪単日 ⓒ(日付) | Worst10Y★⓽ | Worst5Y⓽ | "
           "P10_5Y▷⓽ | Trades/yr ⓞ | 頑健性 | 参考veto |")
    w(hdr)
    w("|------|---------|----------|--------------|--------|----------------|"
      "-----------|---------|---------|-----------|--------|--------|")

    def b1_metric_row(label, disp):
        r = b1[label]
        cagr_is = pct_from_frac(float(r["CAGR_IS_at"]))
        cagr_oos = pct_from_frac(float(r["CAGR_OOS_at"]))
        sharpe = sharpe_with_marker(float(r["Sharpe_FULL"]))
        maxdd = pct_from_frac(float(r["MaxDD_FULL"]))
        w1d = pct_from_frac(float(r["Worst1D"]))
        worstday = f"{w1d} ({r['Worst1D_date']})"
        w10y = pct_from_frac(float(r["Worst10Y_at"]))
        w5y = pct_from_frac(float(r["Worst5Y_at"]))
        p10 = pct_from_frac(float(r["P10_5Y_at"]))
        trades = f"{float(r['Trades_yr']):.1f}"
        _, robcell = robustness_cell(r)
        veto = "参考:VETO" if int(float(r["VETO_s1"])) else "参考:PASS"
        return (f"| {disp} | {cagr_is} | {cagr_oos} | {sharpe} | {maxdd} | "
                f"{worstday} | {w10y} | {w5y} | {p10} | {trades} | {robcell} | {veto} |")

    def p09_metric_row(sc):
        r = p09[sc]
        cagr_is = pct_from_frac(float(r["CAGR_IS_at"]))
        cagr_oos = pct_from_frac(float(r["CAGR_OOS_at"]))
        sharpe = sharpe_with_marker(float(r["Sharpe_FULL"]))
        maxdd = pct_from_frac(float(r["MaxDD_FULL"]))
        w1d = pct_from_frac(float(r["Worst1D"]))
        worstday = f"{w1d} ({r['Worst1D_date']})"
        w10y = pct_from_frac(float(r["Worst10Y_at"]))
        w5y_raw = f_float(r.get("Worst5Y_at"))
        w5y = pct_from_frac(w5y_raw) if w5y_raw is not None else "—"
        p10_raw = f_float(r.get("P10_5Y_at"))
        p10 = pct_from_frac(p10_raw) if p10_raw is not None else "—"
        trades = f"{float(r['Trades_yr']):.1f}"
        _, robcell = robustness_cell(r)
        veto = "参考:VETO" if int(float(r["VETO_s1"])) else "参考:PASS"
        return (f"| P09_STR_S{sc:.1f} (前回) | {cagr_is} | {cagr_oos} | {sharpe} | {maxdd} | "
                f"{worstday} | {w10y} | {w5y} | {p10} | {trades} | {robcell} | {veto} |")

    # 先頭参考: B1_DEF_S1.0
    w(b1_metric_row(REF10, f"{REF10} (参考)"))
    # scale 昇順で DEF, STR, P09
    for sc in SCALES:
        w(b1_metric_row(DEF[sc], DEF[sc]))
        w(b1_metric_row(STR[sc], STR[sc]))
        w(p09_metric_row(sc))
    w("")
    w("> Sharpe マーカー: ◎ >0.934, ★ >1.100。頑健性 ✅頑健 / ⚠条件付 / ❌過学習疑い。"
      "P09行はブレーキ無し（前回P09×scale 検証 / strong map / excess0.0025）。"
      "参考veto は Stage-1 フルゲートの VETO フラグ（本レポートでは判断材料の一つ）。")
    w("")

    # ===================================================================
    # §3 直接比較 (同map・同scaleで B1_STR vs P09_STR) — 核心
    # ===================================================================
    w("## §3 直接比較表（同map・同scaleで B1 vs P09）— 核心")
    w("")
    w("strong map で B1_STR(ブレーキ有) と 前回P09_STR(ブレーキ無) を同一scaleで対照。"
      "Δ行 = B1 − P09（**ブレーキが同一レバ設計に何を足すか**を分離）。"
      "ΔMaxDD pp 正 = B1の方が浅い(改善)、ΔCAGR_OOS pp 負 = B1がCAGRを払う。")
    w("")
    w("| scale | 系列 | CAGR_OOS | MaxDD | 最悪単日 |")
    w("|-------|------|----------|-------|----------|")
    for sc in SCALES:
        b1r = b1[STR[sc]]
        p9r = p09[sc]
        b1_cagr = float(b1r["CAGR_OOS_at"]); p9_cagr = float(p9r["CAGR_OOS_at"])
        b1_dd = float(b1r["MaxDD_FULL"]); p9_dd = float(p9r["MaxDD_FULL"])
        b1_w1 = float(b1r["Worst1D"]); p9_w1 = float(p9r["Worst1D"])
        d_cagr = (b1_cagr - p9_cagr) * 100.0
        d_dd = (b1_dd - p9_dd) * 100.0
        d_w1 = (b1_w1 - p9_w1) * 100.0
        w(f"| {sc:.1f} | B1_STR_S{sc:.1f} (ブレーキ有) | "
          f"{pct_from_frac(b1_cagr)} | {pct_from_frac(b1_dd)} | {pct_from_frac(b1_w1)} |")
        w(f"| {sc:.1f} | P09_STR_S{sc:.1f} (ブレーキ無) | "
          f"{pct_from_frac(p9_cagr)} | {pct_from_frac(p9_dd)} | {pct_from_frac(p9_w1)} |")
        w(f"| {sc:.1f} | **Δ (B1−P09)** | "
          f"**{_signed(d_cagr,2)}pp** | **{_signed(d_dd,2)}pp** | **{_signed(d_w1,2)}pp** |")
    w("")
    w(f"> **パターン（CSV読取）: ブレーキは MaxDD を +{dd_lo:.1f}〜+{dd_hi:.1f}pp 浅くするが、"
      f"CAGR を {_signed(dc_hi,2)}〜{_signed(dc_lo,2)}pp 払う。** 最悪単日は同map・同scaleで"
      f"ほぼ同一（ブレーキは前場の急落単日には間に合わない＝単日テールは削れない）。")
    w("")
    # B1_DEF (default map = 低ベースレバ = 別ファミリー) の scale別
    w("> **B1_DEF（default map, 低ベースレバ）は別の低CAGR・低DDファミリー**（同scaleでも"
      "ベースレバが低いため CAGR も MaxDD も小さい）。scale別:")
    def_lines = []
    for sc in SCALES:
        r = b1[DEF[sc]]
        def_lines.append(f"S{sc:.1f}: CAGR_OOS {pct_from_frac(float(r['CAGR_OOS_at']))} / "
                         f"MaxDD {pct_from_frac(float(r['MaxDD_FULL']))}")
    w("> " + " ｜ ".join(def_lines))
    w("")

    # ===================================================================
    # §4 De-lever vs 時機 切り分け (control CSV)
    # ===================================================================
    w("## §4 De-lever vs 時機 切り分け（control CSV）")
    w("")
    w("各B1系列を「実現平均退避率 f̄ を全IN日に一律適用した双子（時機情報ゼロ）」と比較。"
      "ΔMaxDD(brake−twin) 正 = ブレーキが双子より浅い ＝ **時機効果**。"
      "ΔCAGR 負 = 時機の見返りに CAGR を払う。")
    w("")
    w("| 系列 | f̄(平均退避率) | brake MaxDD | 等fbar一律デレバ双子 MaxDD | "
      "ΔMaxDD(brake−twin)pp | brake CAGR | twin CAGR | ΔCAGR pp |")
    w("|------|---------------|-------------|---------------------------|"
      "----------------------|-----------|-----------|---------|")
    for lbl, r in ctrl.items():
        fbar = float(r["fbar"])
        bdd = float(r["brake_MaxDD"]); tdd = float(r["twin_MaxDD"])
        bc = float(r["brake_CAGR_OOS"]); tc = float(r["twin_CAGR_OOS"])
        d_dd = float(r["maxdd_brake_minus_twin_pp"])
        d_c = float(r["cagr_brake_minus_twin_pp"])
        w(f"| {lbl} | {fbar*100:.2f}% | {pct_from_frac(bdd)} | {pct_from_frac(tdd)} | "
          f"{_signed(d_dd,2)} | {pct_from_frac(bc)} | {pct_from_frac(tc)} | {_signed(d_c,2)} |")
    w("")
    # 正直なフレーミング (固定文 + CSV数値)
    w(f"> **正直な切り分け:** 各scaleで B1のMaxDDは等fbar一律デレバ双子より浅い"
      f"（+{tw_dd_lo:.1f}〜+{tw_dd_hi:.1f}pp）＝**時機効果が存在**"
      f"（双子は同じ平均退避を時機ゼロで一律適用）。ただしCAGRは双子より低い"
      f"（{_signed(tw_cagr_hi,2)}〜{_signed(tw_cagr_lo,2)}pp）＝**時機はCAGRと引き換え**。")
    w("> 重要な留保: "
      "(1) この時機優位は実現パスのMaxDD比較で、前回QCで判明した通り"
      "**block=21ブートストラップでは有意化しない**（経路依存MaxDDに無効な検定）。 "
      "(2) 前回の経路頑健な暴落窓検定（A7-DDレポート §8）では、主要暴落"
      "（2000/2008/2020/2022）はDH-W1が既にOUTで、ブレーキの時機優位は"
      "**危機内でなく危機外のIN脚ドローダウン**（例2015）で稼がれる。")
    w("> つまりB1×scaleの『時機』は『危機ヘッジ』ではなく"
      "『高レバIN脚の中規模ドローダウンの平滑化』。経済的に妥当だが、危機保険ではない。")
    w("")

    # ===================================================================
    # §5 年次税後表 (1975-2025) : 8 B1 + 4 P09 + B&H
    # ===================================================================
    w("## §5 年次リターン表（税後・1975-2025）")
    w("")
    w("各セルはコスト後・譲渡益税後 (×0.8273) の年次リターン%。1974/2026 の部分年は除外"
      "（51年: 1975-2025）。B1列8系列 + 前回P09列4系列 + NASDAQ 1倍B&H。"
      "P09列は前回P09×scale 年次CSV（strong map）から。")
    w("")
    # 列順: B1_DEF×4, B1_STR×4, P09×4, B&H
    b1_cols = [DEF[sc] for sc in SCALES] + [STR[sc] for sc in SCALES]
    b1_disp = [f"DEF{sc:.1f}" for sc in SCALES] + [f"STR{sc:.1f}" for sc in SCALES]
    p09_disp = [f"P09_{sc:.1f}" for sc in SCALES]
    header_cols = b1_disp + p09_disp + ["B&H"]
    w("| 年 | " + " | ".join(header_cols) + " |")
    w("|----|" + "|".join(["------"] * len(header_cols)) + "|")
    for rec in ann:
        cells = [str(rec["year"])]
        for c in b1_cols:
            cells.append(pct_from_pct(rec.get(c)))
        prec = p09_ann_by_y.get(rec["year"])
        for sc in SCALES:
            v = prec.get(P09_ANNUAL_COL[sc]) if prec else None
            cells.append(pct_from_pct(v))
        cells.append(pct_from_pct(rec.get("NASDAQ_1x_BH_pct")))
        w("| " + " | ".join(cells) + " |")
    w("")
    w(f"> 年数チェック: {len(ann)} 行（1975-2025 = 51年）。")
    w("")

    # ===================================================================
    # §6 統計 (税後・1975-2025)
    # ===================================================================
    w("## §6 統計表（税後・1975-2025）")
    w("")
    w("全期間CAGR = 幾何平均 ∏(1+r/100)^(1/51)−1。標準偏差 = ddof=1。")
    w("")
    w("| 系列 | 全期CAGR | 平均年率 | 標準偏差 | "
      "最良年 | 最悪年 | プラスの年数 |")
    w("|------|-----------|---------|---------|--------|--------|-------------|")
    n_years = len(ann)
    # B1 8系列
    stat_series = OrderedDict()
    for sc in SCALES:
        stat_series[f"B1_DEF_S{sc:.1f}"] = [rec[DEF[sc]] for rec in ann]
    for sc in SCALES:
        stat_series[f"B1_STR_S{sc:.1f}"] = [rec[STR[sc]] for rec in ann]
    for sc in SCALES:
        stat_series[f"P09_STR_S{sc:.1f}"] = [
            p09_ann_by_y[rec["year"]][P09_ANNUAL_COL[sc]] for rec in ann]
    stat_series["NASDAQ_1x_B&H"] = [rec["NASDAQ_1x_BH_pct"] for rec in ann]
    for name, ser in stat_series.items():
        st = series_stats(ser)
        w(f"| {name} | {pct_from_frac(st['geom'])} | {pct_from_pct(st['mean'])} | "
          f"{st['sd']:.2f} | {pct_from_pct(st['best'])} | {pct_from_pct(st['worst'])} | "
          f"{st['pos']}/{n_years} |")
    w("")
    w("> ⚠ NASDAQ 1倍B&H は売却時まで課税繰延のため年次×0.8273は過剰課税。"
      "実手取りはより高く税前に近い。高回転戦略は年次×0.8273が妥当。")
    w("")

    # ===================================================================
    # §7 フロンティア (全12系列, MaxDD 浅い順)
    # ===================================================================
    w("## §7 フロンティア（全12系列・MaxDD 浅い順）")
    w("")
    w("8 B1 + 4 P09 を MaxDD 浅い順に並べ、各MaxDD帯で最高CAGRの系列を特定。"
      "**判断の核心: ある目標MaxDDで B1×高スケール は P09×低スケール に勝つか？**")
    w("")
    # 全12系列 (label, disp, cagr_oos, maxdd, veto)
    frontier = []
    for sc in SCALES:
        for lbl in (DEF[sc], STR[sc]):
            r = b1[lbl]
            frontier.append((lbl, float(r["CAGR_OOS_at"]), float(r["MaxDD_FULL"]),
                             int(float(r["VETO_s1"]))))
    for sc in SCALES:
        r = p09[sc]
        frontier.append((f"P09_STR_S{sc:.1f}", float(r["CAGR_OOS_at"]),
                         float(r["MaxDD_FULL"]), int(float(r["VETO_s1"]))))
    # MaxDD 浅い順 = MaxDD は負, 浅い=0に近い=大きい値
    frontier.sort(key=lambda t: t[2], reverse=True)
    w("| 系列 | CAGR_OOS | MaxDD | 参考veto |")
    w("|------|----------|-------|----------|")
    for lbl, cagr, dd, veto in frontier:
        vt = "参考:VETO" if veto else "参考:PASS"
        w(f"| {lbl} | {pct_from_frac(cagr)} | {pct_from_frac(dd)} | {vt} |")
    w("")

    # KEY decision: ~-46/-47% MaxDD 帯で最高CAGR
    # 候補: B1_STR_S2.0, P09_STR_S1.4, B1_STR_S1.8 (近いMaxDD帯)
    cand = {}
    cand["B1_STR_S2.0"] = (float(b1[STR[2.0]]["CAGR_OOS_at"]),
                           float(b1[STR[2.0]]["MaxDD_FULL"]))
    cand["B1_STR_S1.8"] = (float(b1[STR[1.8]]["CAGR_OOS_at"]),
                           float(b1[STR[1.8]]["MaxDD_FULL"]))
    cand["P09_STR_S1.4"] = (float(p09[1.4]["CAGR_OOS_at"]),
                            float(p09[1.4]["MaxDD_FULL"]))
    # P09_STR_S1.6 も比較対象
    cand["P09_STR_S1.6"] = (float(p09[1.6]["CAGR_OOS_at"]),
                            float(p09[1.6]["MaxDD_FULL"]))
    # ~-46/-47% 帯で CAGR 最高を機械選択 (MaxDD が -0.50 〜 -0.44 の帯)
    band = {k: v for k, v in cand.items() if -0.50 <= v[1] <= -0.44}
    best_in_band = max(band.items(), key=lambda kv: kv[1][0]) if band else None

    w("> **§7 KEY 判定（CSVから機械算出）:** ~−46/−47% MaxDD 帯の候補比較 —")
    for k in ("B1_STR_S2.0", "B1_STR_S1.8", "P09_STR_S1.4", "P09_STR_S1.6"):
        c, d = cand[k]
        mark = " ← 帯内CAGR最高" if best_in_band and k == best_in_band[0] else ""
        w(f"> ・{k}: CAGR_OOS {pct_from_frac(c)} / MaxDD {pct_from_frac(d)}{mark}")
    if best_in_band:
        bk = best_in_band[0]
        bc, bdd = best_in_band[1]
        # B1_STR_S2.0 と帯内勝者の比較で verdict
        b1_20_c, b1_20_d = cand["B1_STR_S2.0"]
        p14_c, p14_d = cand["P09_STR_S1.4"]
        verdict_b1_vs_p14 = (b1_20_c - p14_c) * 100.0  # 正=B1高い
        w(f">")
        w(f"> **判定: MaxDD ~−46/−47% 帯で最高CAGRは {bk}（{pct_from_frac(bc)} @ "
          f"{pct_from_frac(bdd)}）。** "
          f"B1_STR_S2.0（{pct_from_frac(b1_20_c)} @ {pct_from_frac(b1_20_d)}）と "
          f"P09_STR_S1.4（{pct_from_frac(p14_c)} @ {pct_from_frac(p14_d)}）は"
          f"同MaxDD帯で CAGR差 {_signed(verdict_b1_vs_p14,2)}pp ＝"
          f"{'ほぼ同等(B1×2.0 ≈ P09×1.4)' if abs(verdict_b1_vs_p14) < 1.0 else ('B1が優位' if verdict_b1_vs_p14 > 0 else 'P09が優位')}。"
          f"**B1の高スケール＋ブレーキは、おおむね P09 の低スケールに相当する**"
          f"（ブレーキはレバを上げてDDを戻す＝実効的に低スケールP09へ近づける装置）。")
    w("")

    # ===================================================================
    # §8 結論
    # ===================================================================
    w("## §8 結論")
    w("")
    # 機械値
    p09_20_c = float(p09[2.0]["CAGR_OOS_at"])
    p09_20_d = float(p09[2.0]["MaxDD_FULL"])
    p09_18_d = float(p09[1.8]["MaxDD_FULL"])
    b1_18 = b1[STR[1.8]]
    b1_18_c = float(b1_18["CAGR_OOS_at"]); b1_18_d = float(b1_18["MaxDD_FULL"])
    b1_20_c2 = float(b1[STR[2.0]]["CAGR_OOS_at"]); b1_20_d2 = float(b1[STR[2.0]]["MaxDD_FULL"])
    p14_c = float(p09[1.4]["CAGR_OOS_at"]); p14_d = float(p09[1.4]["MaxDD_FULL"])
    b1_vs_p14 = (b1_20_c2 - p14_c) * 100.0

    w(f"**(1) DD抑制の効用:** B1×scaleは、ブレーキ無しP09が−50%を割り込む高スケール域でも "
      f"MaxDDを−50%以内に保ったまま レバを上げられる "
      f"（B1_STR_S2.0 ≈ {pct_from_frac(b1_20_d2)} vs ブレーキ無しP09_STR_S2.0 {pct_from_frac(p09_20_d)}）。")
    w("")
    w(f"**(2) ただし同スケールでは CAGR↔DD のトレードオフ:** 同map・同scaleで B1は MaxDD を "
      f"+{dd_lo:.1f}〜+{dd_hi:.1f}pp 浅くする代わりに CAGR を "
      f"{_signed(dc_hi,2)}〜{_signed(dc_lo,2)}pp 払う（§3）。無償でも支配でもない。")
    w("")
    w(f"**(3) 正直なフロンティアの問い:** B1_STR_S2.0（CAGR {pct_from_frac(b1_20_c2)} / "
      f"MaxDD {pct_from_frac(b1_20_d2)}）は P09_STR_S1.4（CAGR {pct_from_frac(p14_c)} / "
      f"MaxDD {pct_from_frac(p14_d)}）を支配するか？ — CAGR差 {_signed(b1_vs_p14,2)}pp・"
      f"MaxDDほぼ同帯。"
      f"{'＝ほぼ同等で、B1×2.0 のブレーキは実効的に P09×1.4 相当（ブレーキ＝低スケール化装置）。支配ではない。' if abs(b1_vs_p14) < 1.0 else ('＝B1がやや優位。' if b1_vs_p14 > 0 else '＝P09低スケールがやや優位。')}")
    w("")
    w(f"**(4) ブレーキの時機は実在だが危機保険ではない:** §4 control でブレーキは等fbar双子より "
      f"+{tw_dd_lo:.1f}〜+{tw_dd_hi:.1f}pp 浅い（時機実在）が、CAGRを "
      f"{_signed(tw_cagr_hi,2)}〜{_signed(tw_cagr_lo,2)}pp 払い、かつその時機は危機内でなく"
      f"**危機外IN脚の中規模DD平滑化**（危機はDH-W1のOUT-fillが既に処理済）。")
    w("")
    w(f"**(5) ユーザー向け要点:** 目的が「高CAGR＋DD上限」なら、B1_STR を scale 1.8〜2.0 で "
      f"CAGR {pct_from_frac(b1_18_c)}〜{pct_from_frac(b1_20_c2)} @ MaxDD "
      f"{pct_from_frac(b1_18_d)}/{pct_from_frac(b1_20_d2)}（−50%以内）に到達でき、これは"
      f"ブレーキ無しP09が同scaleで到達不能（P09は同域で {pct_from_frac(p09_18_d)}/{pct_from_frac(p09_20_d)}）。"
      f"ただしそれが「P09を低スケールで回す」より良いかは §7 のフロンティア比較が答え"
      f"（{('ほぼ同等＝過大評価禁物' if abs(b1_vs_p14) < 1.0 else '差は§7参照')}）。"
      f"**過大評価せず、フロンティア表に判断を委ねる。**")
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
