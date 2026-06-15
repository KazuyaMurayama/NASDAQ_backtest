# -*- coding: utf-8 -*-
"""
scorecard_g5_20260615.py
========================
B3a+G5_vix_hard を既存 6 次元採点器で採点し、
B3a_k365 / P09_C1 / V7_TQQQ と並記する。

【入力根拠】
  combine_phase2_fullgate_20260615.csv  ... G5_vix_hard 行
  combine_g5_defoverlay_20260615.csv    ... G5_vix_hard 行 (Worst5Y / P10_5Y 等)
  scorecard_recompute_20260612.csv      ... B3a/P09/V7 の既存スコア (再利用)

【採点器】
  scorecard_recompute_20260612.py の写像 / 重み / ベト定義を import 再利用。
  B3a+G5_vix_hard を 1 候補として追加採点するのみ。写像変更なし。

【重みモード】
  BAL  ... バランス重視 (標準重み, 飽和あり)
  CAGR ... CAGR 重視   (NONSAT = 飽和なし)

Copyright 2026 Kazuya Oza
"""

import csv
import os
import sys

# ============================================================
# 0. パス設定
# ============================================================
REPO  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT = os.path.join(REPO, "audit_results")
SRC   = os.path.join(REPO, "src", "audit")

# 採点器モジュールを import
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import scorecard_recompute_20260612 as SC

F_PREV = os.path.join(AUDIT, "scorecard_recompute_20260612.csv")
F_OUT  = os.path.join(AUDIT, "scorecard_g5_20260615.csv")

TOTAL_BDAYS = 13204.0


# ============================================================
# 1. G5_vix_hard 入力データ
#    (combine_phase2_fullgate + combine_g5_defoverlay から読取)
# ============================================================

# --- Phase2 fullgate から取得した G5_vix_hard 行 ---
# min9_at=0.207325, CAGR_IS_at=0.222015, Sharpe_OOS=0.927984
# MaxDD_FULL=-0.359318, Worst10Y_star_at=0.137352, P10_5Y_at=0.082592
# Worst5Y_at=-0.000712, Trades_yr=52.948895
# wfa_CI95_lo=0.215375, wfa_WFE=0.989960
# cpcv_p10_at=0.156445, regime_min_at=-0.021034
# boot21_P_min_vs_V7=0.905300, boot21_CI95_min_vs_V7=-2.466144
# excess_days=4329 (B3a と同じ k365 コスト構造)

G5_CANDIDATE = {
    "label":        "B3a+G5_vix_hard",
    "min_at":       0.207325,       # min(IS,OOS)⓽ = 20.73%
    "ci95_lo":      0.215375,       # WFA CI95_lo   = 21.54%
    "wfe":          0.989960,       # WFA WFE       = 0.990
    "cpcv_p10":     0.156445,       # CPCV p10      = 15.64%
    "regime_min":   -0.021034,      # Regime_min    = -2.10%
    "sharpe":       0.927984,       # Sharpe_OOS    = 0.928
    "maxdd":        -0.359318,      # MaxDD         = -35.93%
    "w10y":         0.137352,       # Worst10Y★     = 13.74%
    "p10_5y":       0.082592,       # P10_5Y        = 8.26%
    "w5y":          -0.000712,      # Worst5Y       = -0.07%
    "trades_yr":    52.948895,      # Trades/yr     = 52.9
    # excess_ratio: B3a+G5 は k365 コスト構造を継承 (excess_days=4329)
    "excess_ratio": 4329 / TOTAL_BDAYS,   # ≈ 0.3279  (B3a と同じ)
    # bootstrap 対 V7 (Phase 2 fullgate の boot21 列)
    "boot_p":       0.905300,       # P(min > V7)   = 0.905
    "boot_ci":      -2.466144,      # CI95_lo (pp)  = -2.47pp
}


# ============================================================
# 2. B3a/P09_C1/V7 の既存スコアを CSV から読み込む
# ============================================================

def load_prev_scores():
    rows = {}
    with open(F_PREV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lbl = r["label"]
            # nonsat=False 行のみ (末尾 _nonsat は除外)
            if not lbl.endswith("_nonsat"):
                rows[lbl] = r
    return rows


# ============================================================
# 3. 採点実行
# ============================================================

def score_g5():
    """G5 候補を BAL / CAGR の 2 モードで採点し返す。"""
    # BAL モード (飽和あり, nonsat=False)
    s_bal  = SC.score_candidate(G5_CANDIDATE, nonsat=False)
    # CAGR モード (飽和なし, nonsat=True)
    s_cagr = SC.score_candidate(G5_CANDIDATE, nonsat=True)
    return s_bal, s_cagr


# ============================================================
# 4. 比較テーブル出力
# ============================================================

TARGET_LABELS = ["V7_TQQQ", "P09_C1", "B3a_k365", "B3a+G5_vix_hard"]

DIM_KEYS = ["d1_ret", "d2_rob", "d3_risk", "d4_tail", "d5_cost", "d6_sig"]
DIM_NAMES = [
    "① Ret(22%)", "② Rob(20%)", "③ Risk(14%)",
    "④ Tail(20%)", "⑤ Cost(12%)", "⑥ Sig(12%)",
]


def make_row_from_csv(r, mode_suffix=""):
    """既存 CSV 行を dict に変換 (total/d1…はそのまま)。"""
    return {
        "label":    r["label"] + mode_suffix,
        "veto":     r.get("veto", ""),
        "d1_ret":   float(r["d1_ret"]),
        "d2_rob":   float(r["d2_rob"]),
        "d3_risk":  float(r["d3_risk"]),
        "d4_tail":  float(r["d4_tail"]),
        "d5_cost":  float(r["d5_cost"]),
        "d6_sig":   float(r["d6_sig"]),
        "total":    float(r["total"]),
    }


def print_table(rows, title):
    sep = "=" * 80
    print(sep)
    print(f"  {title}")
    print(sep)
    header = f"  {'候補':<24}" + "".join(f"  {n:>13}" for n in DIM_NAMES) + f"  {'総合':>7}  {'ベト'}"
    print(header)
    print("  " + "-" * 118)
    for r in rows:
        dims = "".join(f"  {r[k]:>13.3f}" for k in DIM_KEYS)
        print(f"  {r['label']:<24}{dims}  {r['total']:>7.3f}  {r['veto']}")
    print()


def print_delta(bal_g5, cagr_g5, bal_b3a, cagr_b3a):
    """B3a -> B3a+G5 の各次元変化を表示。"""
    print("=" * 60)
    print("  B3a → B3a+G5_vix_hard 各次元変化")
    print("=" * 60)
    for k, n in zip(DIM_KEYS, DIM_NAMES):
        delta = bal_g5[k] - bal_b3a[k]
        sign  = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        print(f"  {n:<15}  B3a={bal_b3a[k]:>6.3f}  G5={bal_g5[k]:>6.3f}  "
              f"Δ={delta:>+6.3f}  {sign}")
    dtot_bal  = bal_g5["total"]  - bal_b3a["total"]
    dtot_cagr = cagr_g5["total"] - cagr_b3a["total"]
    print(f"  {'総合(BAL)':<15}  B3a={bal_b3a['total']:>6.3f}  G5={bal_g5['total']:>6.3f}  "
          f"Δ={dtot_bal:>+6.3f}")
    print(f"  {'総合(CAGR)':<15}  B3a={cagr_b3a['total']:>6.3f}  G5={cagr_g5['total']:>6.3f}  "
          f"Δ={dtot_cagr:>+6.3f}")
    print()
    # 期待方向確認
    d3_up  = bal_g5["d3_risk"] > bal_b3a["d3_risk"]
    d6_up  = bal_g5["d6_sig"]  > bal_b3a["d6_sig"]
    d1_dn  = bal_g5["d1_ret"]  < bal_b3a["d1_ret"]
    print("  期待方向確認:")
    print(f"    ③ リスク 上昇? {'YES (期待通り)' if d3_up else 'NO (期待外れ)'}")
    print(f"    ⑥ 有意性 上昇? {'YES (期待通り)' if d6_up else 'NO (期待外れ)'}")
    print(f"    ① リターン 低下? {'YES (期待通り)' if d1_dn else 'NO (期待外れ)'}")
    g5_wins_bal  = bal_g5["total"]  > bal_b3a["total"]
    g5_wins_cagr = cagr_g5["total"] > cagr_b3a["total"]
    print(f"    B3a+G5 総合 > B3a (BAL)?  {'YES' if g5_wins_bal  else 'NO'}")
    print(f"    B3a+G5 総合 > B3a (CAGR)? {'YES' if g5_wins_cagr else 'NO'}")
    print()


# ============================================================
# 5. CSV 出力
# ============================================================

CSV_FIELDS = [
    "label", "mode", "veto", "veto_flags",
    "d1_ret", "d2_rob", "d3_risk", "d4_tail", "d5_cost", "d6_sig",
    "total_bal", "total_cagr",
]


def write_csv(rows_bal, rows_cagr):
    # label -> {bal, cagr} にマージ
    merged = {}
    for r in rows_bal:
        lbl = r["label"].replace("_nonsat","")
        merged.setdefault(lbl, {})["bal"]  = r
    for r in rows_cagr:
        lbl = r["label"].replace("_nonsat","")
        merged.setdefault(lbl, {})["cagr"] = r

    with open(F_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for lbl in TARGET_LABELS:
            b = merged.get(lbl, {}).get("bal",  {})
            c = merged.get(lbl, {}).get("cagr", {})
            writer.writerow({
                "label":    lbl,
                "mode":     "BAL+CAGR",
                "veto":     b.get("veto",       c.get("veto", "")),
                "veto_flags": b.get("veto_flags", c.get("veto_flags", "")),
                "d1_ret":   round(b.get("d1_ret",   0), 3),
                "d2_rob":   round(b.get("d2_rob",   0), 3),
                "d3_risk":  round(b.get("d3_risk",  0), 3),
                "d4_tail":  round(b.get("d4_tail",  0), 3),
                "d5_cost":  round(b.get("d5_cost",  0), 3),
                "d6_sig":   round(b.get("d6_sig",   0), 3),
                "total_bal":  round(b.get("total",  0), 3),
                "total_cagr": round(c.get("total",  0), 3),
            })
    print(f"  CSV output: {F_OUT}")
    print()


# ============================================================
# 6. メイン
# ============================================================

def main():
    prev = load_prev_scores()

    # G5 採点 (BAL / CAGR)
    g5_bal_raw, g5_cagr_raw = score_g5()

    # G5 行を dict 形式に変換
    g5_bal  = {
        "label": "B3a+G5_vix_hard", "veto": g5_bal_raw["veto"],
        "veto_flags": g5_bal_raw["veto_flags"],
        **{k: g5_bal_raw[k] for k in DIM_KEYS},
        "total": g5_bal_raw["total"],
    }
    g5_cagr = {
        "label": "B3a+G5_vix_hard", "veto": g5_cagr_raw["veto"],
        "veto_flags": g5_cagr_raw["veto_flags"],
        **{k: g5_cagr_raw[k] for k in DIM_KEYS},
        "total": g5_cagr_raw["total"],
    }

    # 既存行ロード
    v7_bal    = make_row_from_csv(prev["V7_TQQQ"])
    p09_bal   = make_row_from_csv(prev["P09_C1"])
    b3a_bal   = make_row_from_csv(prev["B3a_k365"])

    # CAGR モード (nonsat=True) の既存行を再計算
    # V7 / P09 / B3a を nonsat で採点
    def rescore_nonsat(cand_dict):
        return SC.score_candidate(cand_dict, nonsat=True)

    # 既存候補の dict を再構築 (build_target_candidates は k365 CSVを読む)
    # -> 手動で入力値を指定する
    _v7_data = {
        "label":"V7_TQQQ","min_at":0.162746,"ci95_lo":0.154828,"wfe":0.97617,
        "cpcv_p10":0.10365,"regime_min":-0.06136,"sharpe":0.87728,"maxdd":-0.34466,
        "w10y":0.1008,"p10_5y":0.0515,"w5y":0.0040,"trades_yr":25.2,
        "excess_ratio":0.0,"boot_p":0.50,"boot_ci":0.0,
    }
    _p09_data = {
        "label":"P09_C1","min_at":0.17587,"ci95_lo":0.17944,"wfe":1.01678,
        "cpcv_p10":0.13555,"regime_min":-0.01565,"sharpe":0.90131,"maxdd":-0.35185,
        "w10y":0.1145,"p10_5y":0.0656,"w5y":-0.0059,"trades_yr":29.2,
        "excess_ratio":0.0,"boot_p":0.797,"boot_ci":-3.663,
    }
    _b3a_data = {
        "label":"B3a_k365","min_at":0.20976,"ci95_lo":0.225175,"wfe":0.98713,
        "cpcv_p10":0.16014,"regime_min":-0.02883,"sharpe":0.90418,"maxdd":-0.38204,
        "w10y":0.14533,"p10_5y":0.08083,"w5y":0.00102,"trades_yr":33.277,
        "excess_ratio":4963/TOTAL_BDAYS,"boot_p":0.8934,"boot_ci":-2.8266,
    }

    def to_score_row(sc_result, lbl):
        return {
            "label": lbl, "veto": sc_result["veto"],
            "veto_flags": sc_result["veto_flags"],
            **{k: sc_result[k] for k in DIM_KEYS},
            "total": sc_result["total"],
        }

    v7_cagr  = to_score_row(rescore_nonsat(_v7_data),  "V7_TQQQ")
    p09_cagr = to_score_row(rescore_nonsat(_p09_data), "P09_C1")
    b3a_cagr = to_score_row(rescore_nonsat(_b3a_data), "B3a_k365")

    # --------------------------------------------------------
    # 入力指標サマリ
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("  B3a+G5_vix_hard 入力指標 (Phase 2 fullgate CSV より)")
    print("=" * 70)
    g = G5_CANDIDATE
    print(f"  min(IS,OOS)_9 : {g['min_at']*100:>7.2f}%   (cf. B3a: {_b3a_data['min_at']*100:.2f}%)")
    print(f"  WFA CI95_lo    : {g['ci95_lo']*100:>7.2f}%   (cf. B3a: {_b3a_data['ci95_lo']*100:.2f}%)")
    print(f"  Sharpe_OOS     : {g['sharpe']:>7.3f}   (cf. B3a: {_b3a_data['sharpe']:.3f})")
    print(f"  MaxDD          : {g['maxdd']*100:>7.2f}%   (cf. B3a: {_b3a_data['maxdd']*100:.2f}%)")
    print(f"  Worst10Y★      : {g['w10y']*100:>7.2f}%   (cf. B3a: {_b3a_data['w10y']*100:.2f}%)")
    print(f"  P10_5Y         : {g['p10_5y']*100:>7.2f}%   (cf. B3a: {_b3a_data['p10_5y']*100:.2f}%)")
    print(f"  Worst5Y        : {g['w5y']*100:>7.3f}%  (cf. B3a: {_b3a_data['w5y']*100:.3f}%)")
    print(f"  Trades/yr      : {g['trades_yr']:>7.1f}   (cf. B3a: {_b3a_data['trades_yr']:.1f})")
    print(f"  excess_ratio   : {g['excess_ratio']:>7.4f}   (cf. B3a: {_b3a_data['excess_ratio']:.4f})")
    print(f"  WFA WFE        : {g['wfe']:>7.4f}   (cf. B3a: {_b3a_data['wfe']:.4f})")
    print(f"  CPCV p10       : {g['cpcv_p10']*100:>7.2f}%   (cf. B3a: {_b3a_data['cpcv_p10']*100:.2f}%)")
    print(f"  Regime_min     : {g['regime_min']*100:>7.2f}%   (cf. B3a: {_b3a_data['regime_min']*100:.2f}%)")
    print(f"  boot P(>V7)    : {g['boot_p']:>7.4f}")
    print(f"  boot CI95_lo   : {g['boot_ci']:>7.2f} pp")
    print()

    # --------------------------------------------------------
    # BAL モード テーブル
    # --------------------------------------------------------
    rows_bal = [v7_bal, p09_bal, b3a_bal, g5_bal]
    print_table(rows_bal, "BAL モード (飽和あり) - 6 次元スコア")

    # --------------------------------------------------------
    # CAGR モード テーブル
    # --------------------------------------------------------
    rows_cagr = [v7_cagr, p09_cagr, b3a_cagr, g5_cagr]
    print_table(rows_cagr, "CAGR モード (飽和なし) - 6 次元スコア")

    # --------------------------------------------------------
    # 差分分析
    # --------------------------------------------------------
    print_delta(g5_bal, g5_cagr, b3a_bal, b3a_cagr)

    # --------------------------------------------------------
    # 結論
    # --------------------------------------------------------
    print("=" * 60)
    print("  総合比較 (4候補)")
    print("=" * 60)
    print(f"  {'候補':<24} {'BAL':>7}  {'CAGR':>7}  {'ベト'}")
    print("  " + "-" * 48)
    all_rows_bal  = {r["label"]: r for r in rows_bal}
    all_rows_cagr = {r["label"]: r for r in rows_cagr}
    for lbl in TARGET_LABELS:
        rb = all_rows_bal.get(lbl,  {})
        rc = all_rows_cagr.get(lbl, {})
        print(f"  {lbl:<24} {rb.get('total',0):>7.3f}  {rc.get('total',0):>7.3f}  {rb.get('veto','')}")
    print()

    # --------------------------------------------------------
    # CSV 出力
    # --------------------------------------------------------
    merged_bal  = {r["label"]: r for r in rows_bal}
    merged_cagr = {r["label"]: r for r in rows_cagr}

    with open(F_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for lbl in TARGET_LABELS:
            b = merged_bal.get(lbl,  {})
            c = merged_cagr.get(lbl, {})
            writer.writerow({
                "label":      lbl,
                "mode":       "BAL+CAGR",
                "veto":       b.get("veto", ""),
                "veto_flags": b.get("veto_flags", ""),
                "d1_ret":     round(b.get("d1_ret",  0), 3),
                "d2_rob":     round(b.get("d2_rob",  0), 3),
                "d3_risk":    round(b.get("d3_risk", 0), 3),
                "d4_tail":    round(b.get("d4_tail", 0), 3),
                "d5_cost":    round(b.get("d5_cost", 0), 3),
                "d6_sig":     round(b.get("d6_sig",  0), 3),
                "total_bal":  round(b.get("total",   0), 3),
                "total_cagr": round(c.get("total",   0), 3),
            })
    print(f"  CSV output: {F_OUT}")
    print()

    # --------------------------------------------------------
    # RETURN_BLOCK
    # --------------------------------------------------------
    g5b = g5_bal
    g5c = g5_cagr
    b3b = b3a_bal
    b3c = b3a_cagr
    import json
    result = {
        "B3a+G5_vix_hard": {
            "d1_ret":  g5b["d1_ret"],  "d2_rob":  g5b["d2_rob"],
            "d3_risk": g5b["d3_risk"], "d4_tail": g5b["d4_tail"],
            "d5_cost": g5b["d5_cost"], "d6_sig":  g5b["d6_sig"],
            "total_bal":  g5b["total"], "total_cagr": g5c["total"],
            "veto": g5b["veto"],
        },
        "B3a_k365": {
            "d1_ret":  b3b["d1_ret"],  "d2_rob":  b3b["d2_rob"],
            "d3_risk": b3b["d3_risk"], "d4_tail": b3b["d4_tail"],
            "d5_cost": b3b["d5_cost"], "d6_sig":  b3b["d6_sig"],
            "total_bal":  b3b["total"], "total_cagr": b3c["total"],
            "veto": b3b["veto"],
        },
        "delta_bal":  round(g5b["total"] - b3b["total"], 3),
        "delta_cagr": round(g5c["total"] - b3c["total"], 3),
        "g5_wins_bal":  g5b["total"] > b3b["total"],
        "g5_wins_cagr": g5c["total"] > b3c["total"],
        "expected_d3_up": g5b["d3_risk"] > b3b["d3_risk"],
        "expected_d6_up": g5b["d6_sig"]  > b3b["d6_sig"],
        "expected_d1_dn": g5b["d1_ret"]  < b3b["d1_ret"],
    }
    print("RETURN_BLOCK:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
