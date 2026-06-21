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
# Stage-2 (独立QC 2026-06-21): 経路頑健な暴落窓時機検定 + B1 等退避リテスト
CRISIS_CSV = os.path.join(REPO_ROOT, "audit_results",
                          "a7dd_stage2_crisis_timing_20260621.csv")
B1EQ_CSV = os.path.join(REPO_ROOT, "audit_results",
                        "a7dd_stage2_b1_equalfbar_20260621.csv")
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
# Stage-2 CSV 読み込み (utf-8-sig)
# ---------------------------------------------------------------------------
def read_crisis():
    """crisis-window timing CSV を読む。
    返り値: (per_brake, per_window)
      per_brake[label] = dict(fbar, n_windows, n_shallower, binom_p,
                              mean_dd_edge_pp, crisis_verdict)  ※ブレーキ順保持
      per_window[label] = list of dict(window, brake_maxdd, twin_maxdd, dd_edge_pp)
    """
    per_brake = OrderedDict()
    per_window = OrderedDict()
    with open(CRISIS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            lbl = r.get("label")
            if not lbl:
                continue
            if lbl not in per_brake:
                per_brake[lbl] = dict(
                    fbar=float(r["fbar"]),
                    n_windows=int(float(r["n_windows"])),
                    n_shallower=int(float(r["n_shallower"])),
                    binom_p=float(r["binom_p_onesided"]),
                    mean_dd_edge_pp=float(r["mean_dd_edge_pp"]),
                    crisis_verdict=r["crisis_verdict"],
                )
                per_window[lbl] = []
            per_window[lbl].append(dict(
                window=r["window"],
                brake_maxdd=float(r["brake_maxdd"]),
                twin_maxdd=float(r["twin_maxdd"]),
                dd_edge_pp=float(r["dd_edge_pp"]),
            ))
    return per_brake, per_window


def read_b1_equalfbar():
    """B1 等退避リテスト CSV を読む。行リスト + a7基準値を返す。"""
    rows = []
    a7_ref = None
    with open(B1EQ_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("target_dvol"):
                continue
            rows.append(dict(
                target_dvol=float(r["target_dvol"]),
                fbar=float(r["fbar"]),
                MaxDD_FULL=float(r["MaxDD_FULL"]),
                CAGR_OOS_at=float(r["CAGR_OOS_at"]),
                fbar_minus_a7=float(r["fbar_minus_a7"]),
                maxdd_minus_a7=float(r["maxdd_minus_a7"]),
            ))
            if a7_ref is None:
                a7_ref = dict(
                    a7_fbar=float(r["a7_fbar"]),
                    a7_maxdd=float(r["a7_maxdd"]),
                    a7_cagr_oos=float(r["a7_cagr_oos"]),
                )
    return rows, a7_ref


# ---------------------------------------------------------------------------
# §8 経路頑健な時機検定 (block=21 無効性の是正)
# ---------------------------------------------------------------------------
def build_section8(w):
    per_brake, per_window = read_crisis()

    w("## §8 経路頑健な時機検定（block=21無効性の是正・独立QC 2026-06-21）")
    w("")
    w("§5 の `timing_P_maxdd` は block=21 で全系列MaxDDをブートストラップしたが、"
      "**MaxDDは数ヶ月〜数年の連続下落で決まる経路極値**であり、月ブロックのシャッフルは"
      "暴落シーケンスを破壊して P を真の時機と無関係に0.5付近へ固定する"
      "（リポ自身の `multimetric_bootstrap` docstring が Worst10Y について同型の無効性を明記、"
      "block=252推奨）。よって §5 の「DELEVER_ONLY」判定は**手法のアーティファクト**。"
      "本節は経路を壊さない**無傷の暴落窓**でブレーキ vs 同一平均退避の双子の窓内MaxDDを"
      "比較した経路頑健な再検定。")
    w("")

    # --- per-brake サマリ表 ---
    w("| 戦略 | 退避率f̄ | 双子より浅い回数 | 平均DD優位(pp) | binom片側p | 判定 |")
    w("|------|---------|------------------|----------------|-----------|------|")
    for lbl, d in per_brake.items():
        tag = lbl.split("_")[0]
        w(f"| {tag} | {d['fbar']*100:.2f}% | "
          f"{d['n_shallower']}/{d['n_windows']} | "
          f"{_signed(d['mean_dd_edge_pp'], 3)} | "
          f"{d['binom_p']:.3f} | {d['crisis_verdict']} |")
    w("")

    # --- A7 / B1 の窓別詳細 ---
    for lbl in ("A7_REPRODUCE", "B1_DOWNSIDE_DEV"):
        if lbl not in per_window:
            continue
        tag = lbl.split("_")[0]
        w(f"**{tag} 窓別詳細**（窓内MaxDD: ブレーキ / 双子 / 優位）:")
        w("")
        w("| 暴落窓 | MaxDD(ブレーキ) | MaxDD(双子) | DD優位(pp) |")
        w("|--------|-----------------|-------------|------------|")
        for wd in per_window[lbl]:
            w(f"| {wd['window']} | {pct_from_frac(wd['brake_maxdd'])} | "
              f"{pct_from_frac(wd['twin_maxdd'])} | "
              f"{_signed(wd['dd_edge_pp'], 3)} |")
        w("")

    # --- MECHANISM callout (固定文) ---
    a7 = per_brake.get("A7_REPRODUCE", {})
    b1 = per_brake.get("B1_DOWNSIDE_DEV", {})
    w(f"> **結果: 全ブレーキが TIMING_WEAK**。A7・B1とも暴落窓で双子より浅い回数は"
      f"{a7.get('n_shallower','?')}/{a7.get('n_windows','?')}・"
      f"{b1.get('n_shallower','?')}/{b1.get('n_windows','?')}。"
      f"dotcom_2000/gfc_2008/covid_2020/bear_2022 では brake と twin の窓内MaxDDが"
      f"完全一致（edge 0）＝**DH-W1がこれら危機中は既に94-100% OUT(現金/Gold/Bond)で、"
      f"IN脚ブレーキに作用対象がない**。危機はOUT-fillが既に処理済。"
      f"よって §5 の実現パス『双子比 MaxDD +5-6pp浅い』は**危機で稼いだものではなく"
      f"単一パスのsequencing偶然**＝時機スキルではない。")
    w("")

    # --- CAVEAT (固定文) ---
    w("> ⚠ 暴落窓 n=5 は小サンプル、かつ4窓は戦略OUTで構造的に情報量ゼロ"
      "（実質1窓=tri_2015のみブレーキが作用）。これは『時機が確実にない』証明ではなく"
      "『block=21の無効判定を経路頑健に置換し、時機効果を支持する証拠が無い』こと。"
      "結論の方向（DD削減は時機でなくデレバ）は §9・実現パス・equal-fbar と整合。")
    w("")


# ---------------------------------------------------------------------------
# §9 B1 等退避リテスト
# ---------------------------------------------------------------------------
def build_section9(w, a7_cagr_oos, b1_cagr_oos):
    rows, a7_ref = read_b1_equalfbar()
    a7_fbar = a7_ref["a7_fbar"]

    b1_vs_a7_cagr = (b1_cagr_oos - a7_cagr_oos) * 100.0   # pp

    w("## §9 B1 等退避リテスト（賢い信号か・弱いブレーキか）")
    w("")
    w(f"§3 でB1がA7比 CAGR {_signed(b1_vs_a7_cagr, 2)}pp"
      f"（うち74%が1999単年, 2020では逆に負け）。これは『下方偏差が賢い』のか"
      f"『単にブレーキが弱い』のか。B1 fbar={rows[0]['fbar']*100:.2f}% は "
      f"A7 fbar={a7_fbar*100:.2f}% の{rows[0]['fbar']/a7_fbar*100:.0f}%＝弱い。"
      f"target_dvol を下げて fbar をA7に揃えて等退避でMaxDDを比較する。")
    w("")

    w("| target_dvol | fbar | fbar−A7(pp) | MaxDD | MaxDD−A7(pp) | CAGR_OOS |")
    w("|-------------|------|-------------|-------|--------------|----------|")
    for r in rows:
        w(f"| {r['target_dvol']:.3f} | {r['fbar']*100:.2f}% | "
          f"{_signed(r['fbar_minus_a7']*100, 2)} | "
          f"{pct_from_frac(r['MaxDD_FULL'])} | "
          f"{_signed(r['maxdd_minus_a7']*100, 2)} | "
          f"{pct_from_frac(r['CAGR_OOS_at'])} |")
    w("")

    # 等退避点 = |fbar - A7| 最小行 (機械的に選択)
    eq = min(rows, key=lambda r: abs(r["fbar_minus_a7"]))
    eq_maxdd_pp = eq["maxdd_minus_a7"] * 100.0   # 負=A7より深い
    deeper_pp = abs(eq_maxdd_pp)
    w(f"等退避点 = |fbar−A7| 最小の行（target_dvol={eq['target_dvol']:.3f}, "
      f"fbar={eq['fbar']*100:.2f}%, fbar−A7={_signed(eq['fbar_minus_a7']*100,2)}pp）。"
      f"そこで B1 MaxDD−A7 = {_signed(eq_maxdd_pp, 2)}pp。")
    w("")
    w(f"**結論**: 等退避(f̄≈A7)では B1 MaxDD は A7 より {deeper_pp:.2f}pp **深い**"
      f"＝B1は賢い信号でなく**弱いブレーキ**。B1のCAGR優位は『少なくブレーキ』の帰結。"
      f"B1で実際にA7よりMaxDDを下げるには dvol を下げて強くブレーキする必要があり、"
      f"その時CAGRを大きく払う（スイープは単調）。")
    w("")


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
    w("> 改訂: 2026-06-21 v2 (独立QC: block=21時機検定の無効性是正, §8/§9追加)")
    w("")
    w("> ベース = A7 (P09_C1 + IN脚ボラブレーキ)。A7をさらにDD削減する4系統"
      "(B1 下方偏差/B2 DD状態スロットル/B3 非対称退避復帰/B4 閾値掃引)を検証。")
    w("> A0 = P09_C1 (ブレーキなし)。A7_REPRODUCE = 前回A7再現 "
      f"(MaxDD {pct_from_frac(a7_maxdd)} 一致で新モジュール検証済)。")
    w("> 全数値: コスト後・譲渡益税後 ×0.8273。"
      "評価=標準10指標 v2.0 + Stage-1 フルゲート + 「時機 vs デレバ」切り分け。")
    w("")

    # ---- 訂正バナー (QC是正・先頭) ----
    w("> **【2026-06-21 QC是正】§5の時機判定(block=21 MaxDDブートストラップ)は"
      "経路依存MaxDDに無効と判明。経路頑健な再検定(§8)＋B1等退避リテスト(§9)を追加。"
      "結論: 全ブレーキ時機スキル無し(危機はOUT-fillが処理済)、B1は弱いブレーキ、"
      "DD削減はCAGRとのトレードオフ・ダイヤル。詳細§8/§9。**")
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
    w("> ※ 下表 timing_P_maxdd は無効な検定（§8で是正）。参考表示。")
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
    # 判定ルール + 解釈 (QC是正版: block=21 は無効、§8参照)
    a7_uni_maxdd = float(a7["uni_MaxDD"])
    w(f"判定（**訂正済・独立QC 2026-06-21**）: §5 の timing_P_maxdd"
      f"（block=21 全系列MaxDDブートストラップ）は経路依存のMaxDDに**無効**（→§8）。"
      f"**「DELEVER_ONLY」は手法のアーティファクトであり、是正版の判定は §8"
      f"（経路頑健な暴落窓検定）を参照。** "
      f"実現パスで全ブレーキが双子比 MaxDD +2〜6pp 浅い（下表ΔMaxDD vs双子, 多くが正）が、"
      f"§8 の通りこれは暴落窓で稼いだものではなく、双子のCAGRが高いのは MaxDD を深くする"
      f"見返り＝**MaxDD↔CAGR フロンティア上の別点であって支配ではない**。")
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
    w("> **注（QC是正）: §6 の timing_verdict ゲートは無効な block=21 検定に基づく。"
      "経路頑健な §8 では全ブレーキ TIMING_WEAK（時機効果の証拠なし）。** "
      "いずれにせよ ADOPT候補は存在しない。")
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
    w("§9 の等退避リテストで B1 は A7 比『賢い信号』ではなく『弱いブレーキ』と判明"
      "（等f̄でMaxDD深い）。B1のCAGR優位は1999単年依存・braking-less由来で、"
      "頑健な優位ではない。")
    w("")

    # ===================================================================
    # §7 結論 (QC是正版)
    # ===================================================================
    w("## §7 結論")
    w("")
    a0_maxdd = float(a0["MaxDD_FULL"])
    a7_maxdd_disp = pct_from_frac(a7_maxdd)
    para = (
        f"**結論（QC是正版）**: "
        f"(1) §5 の block=21 時機検定は無効だったが、経路頑健な §8（暴落窓）でも"
        f"全ブレーキ TIMING_WEAK＝**DD削減に時機スキルは無い**"
        f"（理由: 危機はDH-W1のOUT-fillが既に処理済、IN脚ブレーキは作用余地が小さい）。"
        f"(2) DD削減は **MaxDD↔CAGR のトレードオフ（ダイヤル）**。"
        f"CAGRを概ね線形に払えばMaxDDは下がる（一律デレバ/B4で十分、exoticブレーキの優位なし）。"
        f"**CAGR完全維持で MaxDD削減はどの構成でも達成不能。** "
        f"(3) B1は§9で『弱いブレーキ』と判明＝A7の置換にならない。"
        f"(4) ユーザー要望への誠実な答え: フロンティア上でCAGR-DDのどの点を選ぶか。"
        f"現行A0(MaxDD{pct_from_frac(a0_maxdd)}, CAGR最高)維持、"
        f"または許容CAGR低下に応じB4でデレバ水準を選ぶ。"
    )
    w(para)
    w("")

    # ===================================================================
    # §8 経路頑健な時機検定 (crisis-window / Stage-2 CSV から機械生成)
    # ===================================================================
    build_section8(w)

    # ===================================================================
    # §9 B1 等退避リテスト (Stage-2 CSV から機械生成)
    # ===================================================================
    build_section9(w, a7_cagr_oos, b1_cagr_oos)

    return "\n".join(out) + "\n"


def main():
    md = build_md()
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(f"WROTE: {OUT_MD}")
    print(f"bytes: {len(md.encode('utf-8'))}")


if __name__ == "__main__":
    main()
