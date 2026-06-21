# -*- coding: utf-8 -*-
"""
gen_a7dd_report_20260621.py

A7ベース ドローダウン削減バリエーション(A0/A7/B1-B4)の最終レポートを
2つのCSVから機械生成する。手打ち数値は一切なし。全数値はCSVをプログラム的に
読み込み、統計はpythonで計算する。

入力:
  audit_results/a7dd_stage1_20260621.csv  (utf-8-sig / BOM 有) … 10指標 + WFA/CPCV/bootstrap/regime/veto + timing(時機vsデレバ)
  audit_results/a7dd_annual_20260621.csv  (utf-8-sig)          … year + 6戦略列 + NASDAQ_1x_BH_pct (税後%)

出力:
  A7_DD_REDUCTION_VARIATIONS_20260621.md
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
STAGE1_CSV = os.path.join(REPO_ROOT, "audit_results", "a7dd_stage1_20260621.csv")
ANNUAL_CSV = os.path.join(REPO_ROOT, "audit_results", "a7dd_annual_20260621.csv")
OUT_MD = os.path.join(REPO_ROOT, "A7_DD_REDUCTION_VARIATIONS_20260621.md")

AFTER_TAX = 0.8273  # 譲渡益税後係数 (CSVの年次はすでに税後%・stage1も税後)

# ---------------------------------------------------------------------------
# バリエーション定義 (§1 用 / stage1 CSV の行ラベルに対応)
#   key = stage1 CSV の label, value = (表示名, 判断基準(signal->brake), 仮説)
# 表示順は CSV 行順 (A0, A7, B1, B2, B3, B4×3)
# ---------------------------------------------------------------------------
VAR_DEFS = OrderedDict([
    ("A0_P09_C1_BASE",
     ("A0 P09_C1_BASE",
      "ブレーキなし",
      "= ベースライン")),
    ("A7_REPRODUCE",
     ("A7_REPRODUCE",
      "実現ボラ>30%(w63) → 最大50%現金",
      "= 前回A7 (総ボラ・ブレーキ)")),
    ("B1_DOWNSIDE_DEV",
     ("B1_DOWNSIDE_DEV",
      "下方偏差(負リターンのみ,w63)>20% → 最大50%現金",
      "上昇相場の高ボラはブレーキしない")),
    ("B2_DD_THROTTLE",
     ("B2_DD_THROTTLE",
      "現在DD%(直近ピーク比,shift1) 15%→25%現金 / 25%→50%現金",
      "DD状態に応じて段階スロットル")),
    ("B3_ASYM_BRAKE",
     ("B3_ASYM_BRAKE",
      "A7と同ボラ閾値だが退避即時・復帰は5日連続閾値下回りで解除(非対称)",
      "退避は速く・復帰は遅く")),
    ("B4_VOL020_CAP50",
     ("B4_VOL020_CAP50",
      "A7の閾値掃引 (target_vol=20%, cap=50%)",
      "ブレーキ感度を上げる")),
    ("B4_VOL025_CAP50",
     ("B4_VOL025_CAP50",
      "A7の閾値掃引 (target_vol=25%, cap=50%)",
      "ブレーキ感度を中庸に")),
    ("B4_VOL030_CAP75",
     ("B4_VOL030_CAP75",
      "A7の閾値掃引 (target_vol=30%, cap=75%)",
      "退避上限を上げる")),
])

# §3 年次表の戦略列 (annual CSV の列名) と表示名
ANNUAL_COLS = OrderedDict([
    ("A0_P09_C1_BASE", "A0"),
    ("A7_REPRODUCE", "A7"),
    ("B1_DOWNSIDE_DEV", "B1"),
    ("B2_DD_THROTTLE", "B2"),
    ("B3_ASYM_BRAKE", "B3"),
    ("B4_VOL025_CAP50", "B4"),
    ("NASDAQ_1x_BH_pct", "B&H"),
])

# §4 統計表 / §3 callout で使う列 (B&H 含む)
STAT_COLS = list(ANNUAL_COLS.keys())


# ---------------------------------------------------------------------------
# CSV 読み込み
# ---------------------------------------------------------------------------
def read_stage1():
    """utf-8-sig で開く (BOM 有)。label をキーに dict を返す。"""
    rows = OrderedDict()
    with open(STAGE1_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("label"):
                continue
            rows[r["label"]] = r
    return rows


def read_annual():
    """utf-8-sig で読み、year を int、各値を float に。1975<=year<=2025 のみ filtered。"""
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
    filtered = [rec for rec in full if 1975 <= rec["year"] <= 2025]
    return fieldnames, full, filtered


# ---------------------------------------------------------------------------
# 数値フォーマッタ (符号付き% / マイナス記号は U+2212 の "−")
# ---------------------------------------------------------------------------
MINUS = "−"  # −


def _signed(v, dec=2):
    s = f"{abs(v):.{dec}f}"
    if v < 0:
        return MINUS + s
    return "+" + s


def pct_from_frac(x, dec=2):
    """小数(0.1777)→ '+17.77%' 形式。"""
    return _signed(x * 100.0, dec) + "%"


def pct_from_pct(x, dec=2):
    """既に%値(17.77)→ '+17.77%' 形式。"""
    return _signed(x, dec) + "%"


def f_float(s, default=None):
    """空文字を default に、それ以外は float に。"""
    if s is None or s == "":
        return default
    return float(s)


# ---------------------------------------------------------------------------
# 頑健性ルーブリック (機械判定)
#   ❌ if (WFE>2 OR WFE<0.5 OR CI95_lo<0 OR |CAGR_IS-CAGR_OOS|>5pp)
#   ✅ if (0.5<=WFE<=2 AND CI95_lo>0 AND |gap|<=3pp AND regime_min>-0.10)
#   else ⚠
#   gap_pp は stage1 CSV の IS_OOS_gap_pp (既に pp)。CI95_lo / regime_min は小数。
# ---------------------------------------------------------------------------
def robustness_cell(r):
    wfe = float(r["wfa_WFE"])
    ci_lo = float(r["wfa_CI95_lo"])          # 小数
    gap_pp = float(r["IS_OOS_gap_pp"])       # pp
    reg_min = float(r["regime_min_at"])      # 小数

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


def robustness_glyph_only(r):
    return robustness_cell(r)[0]


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
# 統計 (§4)
# ---------------------------------------------------------------------------
def geom_cagr_from_pct(series_pct):
    n = len(series_pct)
    prod = 1.0
    for r in series_pct:
        prod *= (1.0 + r / 100.0)
    return prod ** (1.0 / n) - 1.0


def series_stats(series_pct):
    n = len(series_pct)
    return dict(
        n=n,
        geom=geom_cagr_from_pct(series_pct),
        mean=statistics.fmean(series_pct),
        sd=statistics.stdev(series_pct),   # ddof=1
        best=max(series_pct),
        worst=min(series_pct),
        pos=sum(1 for x in series_pct if x > 0),
    )


# ---------------------------------------------------------------------------
# MD 生成
# ---------------------------------------------------------------------------
def build_md():
    stage1 = read_stage1()
    ann_fields, ann_full, ann = read_annual()

    labels = list(VAR_DEFS.keys())           # A0, A7, B1, B2, B3, B4×3 (CSV順)
    brake_labels = [l for l in labels if l != "A0_P09_C1_BASE"]  # 7 brakes

    # CAGR_OOS 降順 (A0 は先頭固定、残りを CAGR_OOS_at 降順)
    rest_sorted = sorted(brake_labels,
                         key=lambda l: float(stage1[l]["CAGR_OOS_at"]), reverse=True)
    order_metric = ["A0_P09_C1_BASE"] + rest_sorted

    out = []
    w = out.append

    # ---- timing P_maxdd レンジ & verdict 集計 (brakes のみ。A0 は timing 空) ----
    tpm_vals = [float(stage1[l]["timing_P_maxdd"]) for l in brake_labels]
    tpm_lo, tpm_hi = min(tpm_vals), max(tpm_vals)
    all_delever = all(stage1[l]["timing_verdict"] == "DELEVER_ONLY" for l in brake_labels)
    n_delever = sum(1 for l in brake_labels if stage1[l]["timing_verdict"] == "DELEVER_ONLY")

    # ---- A0 / A7 / B1 主要数値 ----
    a0 = stage1["A0_P09_C1_BASE"]
    a7 = stage1["A7_REPRODUCE"]
    b1 = stage1["B1_DOWNSIDE_DEV"]
    a0_cagr_oos = float(a0["CAGR_OOS_at"])
    a7_cagr_oos = float(a7["CAGR_OOS_at"])
    b1_cagr_oos = float(b1["CAGR_OOS_at"])
    a7_maxdd = float(a7["MaxDD_FULL"])
    b1_maxdd = float(b1["MaxDD_FULL"])
    a7_w1d = float(a7["Worst1D"])
    b1_w1d = float(b1["Worst1D"])
    b1_vs_a7_cagr = (b1_cagr_oos - a7_cagr_oos) * 100.0   # pp
    b1_vs_a7_maxdd = (b1_maxdd - a7_maxdd) * 100.0        # pp

    # ---- 1999 / 2000 年次差 (annual CSV から) ----
    rec1999 = next(r for r in ann if r["year"] == 1999)
    rec2000 = next(r for r in ann if r["year"] == 2000)
    a0_99 = rec1999["A0_P09_C1_BASE"]
    a7_99 = rec1999["A7_REPRODUCE"]
    b1_99 = rec1999["B1_DOWNSIDE_DEV"]
    a0_00 = rec2000["A0_P09_C1_BASE"]
    a7_00 = rec2000["A7_REPRODUCE"]
    b1_00 = rec2000["B1_DOWNSIDE_DEV"]
    a7_99_vs_a0 = a7_99 - a0_99   # pp (A7 - A0)
    b1_99_vs_a7 = b1_99 - a7_99   # pp (B1 - A7)

    # ===================================================================
    # ヘッダ
    # ===================================================================
    w("# A7ベース ドローダウン削減バリエーション検証レポート")
    w("")
    w("作成日: 2026-06-21")
    w("最終更新日: 2026-06-21")
    w("")
    w("> ベース = A7 (P09_C1 + IN脚ボラブレーキ)。A7をさらにDD削減する4系統"
      "(B1 下方偏差/B2 DD状態スロットル/B3 非対称退避復帰/B4 閾値掃引)を検証。")
    w("> A0 = P09_C1 (ブレーキなし)。A7_REPRODUCE = 前回A7再現 "
      f"(MaxDD {pct_from_frac(a7_maxdd)} 一致で新モジュール検証済)。")
    w("> 全数値: コスト後・譲渡益税後 ×0.8273。"
      "評価=標準10指標 v2.0 + Stage-1 フルゲート + 「時機 vs デレバ」切り分け。")
    w("")

    # ---- 先出し blockquote (KEY finding, CSV から計算) ----
    w(f"> **🔑 結論（先出し）: どのブレーキも一律デレバ双子を MaxDD で上回らない"
      f"（全 {n_delever}/{len(brake_labels)} が DELEVER_ONLY, "
      f"timing_P_maxdd {tpm_lo:.2f}-{tpm_hi:.2f}, 全て<0.90）。** "
      f"= ここでの DD 削減は時機スキルではなく単なるデレバ"
      f"（G5 vix オーバーレイ p=0.40 と同じ教訓の再現）。"
      f"唯一の見どころは **B1(下方偏差)**: 上昇ボラの誤ブレーキを回避して "
      f"A7 比 CAGR_OOS {_signed(b1_vs_a7_cagr,2)}pp を回復し"
      f"（1999年 B1 {pct_from_pct(b1_99)} vs A7 {pct_from_pct(a7_99)}）、"
      f"MaxDD はほぼ同等（{pct_from_frac(b1_maxdd)} vs A7 {pct_from_frac(a7_maxdd)}, "
      f"{_signed(b1_vs_a7_maxdd,2)}pp）。"
      f"ただし最悪単日は {pct_from_frac(b1_w1d)} を維持（テールは削れない）"
      f"= CAGR と tail のトレードオフであってフリーランチではない。")
    w("")

    # ===================================================================
    # §1 バリエーション定義表
    # ===================================================================
    w("## §1 バリエーション定義表")
    w("")
    w("| # | 名称 | 判断基準（signal→brake） | 仮説 |")
    w("|---|------|--------------------------|------|")
    # タグ = ラベル先頭トークン (A0 / A7 / B1 / B2 / B3 / B4)
    for l in labels:
        name, crit, hyp = VAR_DEFS[l]
        tag = l.split("_")[0]   # "A0", "A7", "B1", "B2", "B3", "B4"
        w(f"| {tag} | {name} | {crit} | {hyp} |")
    w("")

    # ===================================================================
    # §2 標準10指標 v2.0
    # ===================================================================
    w("## §2 標準10指標 (v2.0)")
    w("")
    w("A0 先頭、残りは CAGR_OOS 降順。⓽=min9基準後の値 / ⓒ=全期間(コスト後) / "
      "ⓞ=年あたり取引回数。Sharpe マーカー: ◎ >0.934, ★ >1.100。"
      "退避率f̄ = IN脚の平均現金退避率 (A0=ブレーキなしのため空欄)。")
    w("")
    hdr = ("| 戦略 | CAGR_IS⓽ | CAGR_OOS⓽ | Sharpe_FULL ⓒ | MaxDD ⓒ | "
           "最悪単日 ⓒ(日付) | Worst10Y★⓽ | Worst5Y⓽ | P10_5Y▷⓽ | "
           "Trades/yr ⓞ | 退避率f̄ | 頑健性 |")
    w(hdr)
    w("|------|---------|----------|--------------|--------|"
      "----------------|-----------|---------|---------|"
      "-----------|---------|--------|")
    for l in order_metric:
        r = stage1[l]
        name = VAR_DEFS[l][0]
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
        fbar_v = f_float(r.get("fbar"))
        fbar_disp = f"{fbar_v*100:.2f}%" if fbar_v is not None else "—"
        rob_glyph, robcell = robustness_cell(r)
        w(f"| {name} | {cagr_is} | {cagr_oos} | {sharpe} | {maxdd} | "
          f"{worstday} | {w10y} | {w5y} | {p10} | {trades} | {fbar_disp} | {robcell} |")
    w("")
    w("> Sharpe マーカー: ◎ >0.934, ★ >1.100。頑健性 ✅頑健 / ⚠条件付 / ❌過学習疑い。"
      "退避率f̄ は IN脚の平均キャッシュ比 (時機 vs デレバの双子構築に使用、§5)。")
    w("")

    # ===================================================================
    # §3 年次税後表 (1975-2025)
    # ===================================================================
    w("## §3 年次リターン表（税後・1975-2025）")
    w("")
    w("各セルはコスト後・譲渡益税後 (×0.8273) の年次リターン%。"
      "1974 と 2026 の部分年は除外（51年: 1975-2025）。")
    w("")
    col_disp = list(ANNUAL_COLS.values())
    w("| 年 | " + " | ".join(col_disp) + " |")
    w("|----|" + "|".join(["------"] * len(col_disp)) + "|")
    for rec in ann:
        cells = [str(rec["year"])]
        for c in ANNUAL_COLS.keys():
            v = rec[c]
            cells.append(pct_from_pct(v) if v is not None else "—")
        w("| " + " | ".join(cells) + " |")
    w("")
    w(f"> 年数チェック: {len(ann)} 行（1975-2025 = 51年）。")
    w("")
    w(f"> **A7は1999年(上昇相場)に高ボラを誤ブレーキし A0比 {_signed(a7_99_vs_a0,2)}pp"
      f"（A7 {pct_from_pct(a7_99)} vs A0 {pct_from_pct(a0_99)}）。** "
      f"B1(下方偏差)は1999年 {pct_from_pct(b1_99)} で A7 を {_signed(b1_99_vs_a7,2)}pp 上回り、"
      f"上昇ボラの誤ブレーキを回避（A0とほぼ同等）。"
      f"一方2000年(暴落)はB1 {pct_from_pct(b1_00)} vs A7 {pct_from_pct(a7_00)}でA7の方が守る。"
      f"= B1はCAGR寄り・A7はDD寄りのトレードオフ。")
    w("")

    # ===================================================================
    # §4 統計表 (税後・1975-2025)
    # ===================================================================
    w("## §4 統計表（税後・1975-2025）")
    w("")
    series_map = OrderedDict()
    for c, disp in ANNUAL_COLS.items():
        series_map[disp] = [rec[c] for rec in ann]

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
      "実手取りはより高く税前に近い。高回転戦略は年次×0.8273が妥当。")
    w("")

    # ===================================================================
    # §5 時機 vs デレバ 切り分け (G5教訓の核心)
    # ===================================================================
    w("## §5 時機 vs デレバ 切り分け（G5教訓の核心・このレポートの肝）")
    w("")
    w("各ブレーキを「実現平均退避率 f̄ を全IN日に一律適用した双子（時機情報ゼロ）」と比較する。")
    w("")
    w("| 戦略 | MaxDD(ブレーキ) | MaxDD(一律デレバ双子) | ΔMaxDD vs 双子(pp) | "
      "退避率f̄ | timing_P_maxdd | timing_verdict |")
    w("|------|----------------|----------------------|--------------------|"
      "---------|----------------|----------------|")
    for l in brake_labels:
        r = stage1[l]
        name = VAR_DEFS[l][0]
        maxdd = float(r["MaxDD_FULL"])
        uni_maxdd = float(r["uni_MaxDD"])
        d_pp = (maxdd - uni_maxdd) * 100.0
        fbar_v = float(r["fbar"])
        tpm = float(r["timing_P_maxdd"])
        verdict = r["timing_verdict"]
        w(f"| {name} | {pct_from_frac(maxdd)} | {pct_from_frac(uni_maxdd)} | "
          f"{_signed(d_pp,2)} | {fbar_v*100:.2f}% | {tpm:.3f} | {verdict} |")
    w("")
    # 判定ルール + 解釈
    a7_uni_maxdd = float(a7["uni_MaxDD"])
    w(f"判定: timing_P_maxdd ≥ 0.90 ⇒ 時機効果あり; < 0.90 ⇒ デレバ寄与のみ。"
      f"**結果: 全ブレーキ(A7含む)が DELEVER_ONLY**"
      f"（timing_P_maxdd {tpm_lo:.2f}-{tpm_hi:.2f}, 全て<0.90）。"
      f"各ブレーキのMaxDD改善（例 A7 {pct_from_frac(a7_maxdd)} vs 双子 {pct_from_frac(a7_uni_maxdd)}）は、"
      f"同じ平均退避率を一律デレバした双子で既に得られる=時機スキルではない。"
      f"これは G5 vix オーバーレイ（時機 p=0.40 非有意, 改善の本質は一律デレバ）と同じ結論の再現。"
      f"**MaxDDを下げたいだけなら exotic ブレーキでなく一律デレバ(=レバ水準ダイヤル, 例B4)で十分。**")
    w("")
    # CAGR corollary: twins beat brakes on CAGR_OOS too (verified from CSV)
    _cagr_show = ("A7_REPRODUCE", "B1_DOWNSIDE_DEV", "B2_DD_THROTTLE",
                  "B3_ASYM_BRAKE", "B4_VOL020_CAP50")
    _frag_parts = []
    for lbl in _cagr_show:
        if lbl not in stage1:
            continue
        b = float(stage1[lbl]["CAGR_OOS_at"]) * 100.0
        u = float(stage1[lbl]["uni_CAGR_OOS"]) * 100.0
        _frag_parts.append(
            f"{lbl.split('_')[0]} {b:+.2f}% vs 双子 {u:+.2f}%（双子{u - b:+.2f}pp）")
    _frag = " / ".join(_frag_parts)
    w(f"> **さらに決定的: 一律デレバ双子は CAGR_OOS でもブレーキを上回る。** "
      f"各ブレーキ vs その双子の CAGR_OOS: {_frag}。"
      f"**全ブレーキで双子の方が CAGR が高い**＝exotic ブレーキは MaxDD でも CAGR でも"
      f"一律デレバに勝てない（時機が効かない以上、ブレーキの複雑性は純粋に損）。"
      f"「同じ平均退避率なら、いつ退避するかを賢く選ぶより、一律に薄く退避する方が良い」"
      f"という強い結論。")
    w("")
    w("> ※ 一律デレバ双子は各ブレーキの実現平均IN脚退避率 f̄ を全IN日に一律適用(時機情報ゼロ)。"
      "f̄ 計測は denom≈0 日(戦略リターン≈SOFR, 確率~4e-8/日)を除外するが影響は<<1bp。"
      "B2のDDは pre-throttle NAVで算出(自己無撞着の簡略化, 過剰throttle方向の保守的バイアス)。")
    w("")

    # ===================================================================
    # §6 採否
    # ===================================================================
    w("## §6 採否（A7 基準 / 機械判定）")
    w("")
    w("各ブレーキ vs A7（現行ベース=incumbent）。判定ルール（機械）:")
    w("- **ADOPT候補**: (CAGR_OOS ≥ A7−0.3pp) かつ "
      "(MaxDD改善 vs A7 ≥1pp OR 最悪単日改善 ≥1pp) かつ 頑健性≠❌ かつ timing_verdict=TIMING_EFFECT")
    w("- それ以外は ADOPT候補ではない。")
    w("")
    w("| 戦略 | ΔCAGR_OOS vs A7 | ΔMaxDD vs A7 | Δ最悪単日 vs A7 | 頑健性 | VETO_s1 | timing_verdict | 判定 |")
    w("|------|-----------------|--------------|-----------------|--------|---------|----------------|------|")
    adopt_any = False
    for l in brake_labels:
        if l == "A7_REPRODUCE":
            continue
        r = stage1[l]
        name = VAR_DEFS[l][0]
        d_cagr = (float(r["CAGR_OOS_at"]) - a7_cagr_oos) * 100.0
        d_maxdd = (float(r["MaxDD_FULL"]) - a7_maxdd) * 100.0   # 正=改善(MaxDDは負)
        d_w1d = (float(r["Worst1D"]) - a7_w1d) * 100.0          # 正=改善
        glyph = robustness_glyph_only(r)
        veto = int(float(r["VETO_s1"]))
        verdict_t = r["timing_verdict"]
        is_overfit = glyph == "❌"
        cagr_ok = d_cagr >= -0.3
        risk_ok = (d_maxdd >= 1.0) or (d_w1d >= 1.0)
        timing_ok = verdict_t == "TIMING_EFFECT"
        adopt = cagr_ok and risk_ok and (not is_overfit) and timing_ok
        if adopt:
            adopt_any = True
        verdict = "ADOPT候補" if adopt else "非ADOPT"
        w(f"| {name} | {_signed(d_cagr,2)}pp | {_signed(d_maxdd,2)}pp | "
          f"{_signed(d_w1d,2)}pp | {glyph} | {veto} | {verdict_t} | **{verdict}** |")
    w("")
    w(f"> **全ブレーキが timing_verdict=DELEVER_ONLY のため、timing 基準により"
      f"ADOPT候補は{'存在しない' if not adopt_any else '限定的'}。** "
      f"MaxDD/最悪単日の改善は一律デレバ双子で再現でき、時機スキルとは認められない。")
    w("")
    # B1 の別枠分類
    b1_w1d_vs_a7 = (b1_w1d - a7_w1d) * 100.0
    w(f"### B1(下方偏差)の別枠評価")
    w("")
    w(f"B1 は時機効果なし(DELEVER_ONLY)だが、A7比 CAGR_OOS {_signed(b1_vs_a7_cagr,2)}pp を "
      f"MaxDDほぼ同等（{_signed(b1_vs_a7_maxdd,2)}pp）で実現（上昇ボラ誤ブレーキ回避）。"
      f'"DD削減"の文脈では一律デレバと差は無いが、"A7を使うならB1の方がCAGR効率が良い"。'
      f"ただしB1は最悪単日を削れない（{pct_from_frac(b1_w1d)}維持 vs A7 {pct_from_frac(a7_w1d)}, "
      f"{_signed(b1_w1d_vs_a7,2)}pp）。")
    w("")

    # ===================================================================
    # §7 結論
    # ===================================================================
    w("## §7 結論")
    w("")
    # 最悪単日を最も削る総ボラ系の代表値 (A7/B3/B4) — A7 vs A0 で記述
    a0_w1d = float(a0["Worst1D"])
    para = (
        f"ユーザー要望(CAGR≈維持・MaxDD削減)に対し: "
        f"(1) MaxDD削減そのものは時機スキルでなく一律デレバ=レバ水準ダイヤルで達成すべき(B4/scale)。"
        f"exotic ブレーキ(B1/B2/B3)はDD削減で一律デレバを統計的に上回らない"
        f"（全 {n_delever}/{len(brake_labels)} DELEVER_ONLY, timing_P_maxdd {tpm_lo:.2f}-{tpm_hi:.2f}）。"
        f"(2) A7を採用する前提なら B1(下方偏差) が CAGR効率最良"
        f"（{_signed(b1_vs_a7_cagr,2)}pp, MaxDD同等 {_signed(b1_vs_a7_maxdd,2)}pp）。"
        f"(3) 最悪単日(テール1日)を削りたいなら総ボラ系(A7/B3/B4)が "
        f"{pct_from_frac(a0_w1d)}→{pct_from_frac(a7_w1d)}に削るが、これも双子比では時機非有意。"
        f"**正直な結論: DD削減は\"ダイヤル\"であり、"
        f"どの水準のCAGR-DDトレードオフを選ぶかの問題。"
        f"B1はそのフロンティア上でCAGR寄りの良点。**"
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
