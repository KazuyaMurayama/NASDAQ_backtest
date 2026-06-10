"""
src/audit/compute_wfa_realistic_20260610.py
============================================
修正後 realistic NAV (DELAY=2 + spread修正) を使った全戦略 WFA 再算出スクリプト。

対象:
  7戦略 × realistic(full L×)
    E4, vz065_l5, vz065_l7, DH-W1, V0, V7, P7
  CFD3戦略 × realistic(borrowed) 感度
    E4_borrowed, vz065_l5_borrowed, vz065_l7_borrowed

出力:
  audit_results/wfa_realistic_summary_20260610.csv

公式 realistic 基準 = full L×（borrowedは下限感度）。
IS境界: 2021-05-08 (canonical split)

窓設計に関する注記（2026-06-10 修正）:
  本スクリプトは2つの窓方式を並走する:

  【方式A: 正典窓 (canonical / g1_wfa 準拠)】
    g1_wfa.generate_windows を使用。カレンダー年アンカー方式（各年1月の
    最初の営業日〜12月の最後の営業日）。EVAL_START=1977-01-03(LT2 N=750
    warmup後)・EVAL_END=2026-03-26。short_flag(窓<201日)は除外。
    結果: 約50窓（2026年分が短窓=short_flagのため除外されると49窓相当）。
    → 正典準拠の n_windows/CI95_lo/WFE を提供。列名に _canon サフィックス。

  【方式B: 固定窓 (fixed-252 / 旧来方式)】
    非重複252日固定窓。NAV先頭(warmupスキップなし)から 1974年起点で52窓。
    → 旧来値の継続性・比較のために保持。列名に _fixed サフィックス。

  どちらが正典準拠かを window_method 列に明示。IS/OOS判定・CI95_lo/WFEは
  両方式とも同一の compute_summary_stats を使用。
"""
from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# Patch multitasking
# ---------------------------------------------------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

# repo root & src/ をパスに追加
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_SRC_DIR)
for _p in [_SRC_DIR, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from src.audit.strategy_runners import run_e4, run_vz065, run_dhw1, run_overlay, run_p7
from src.audit.unified_wfa import summarize_wfa, WINDOW_LEN, OOS_START_REF
# 正典窓生成関数を g1_wfa から直接 import（正典ファイルは改変しない）
from src.g1_wfa import generate_windows as _g1_generate_windows

WINDOW = WINDOW_LEN  # 252

# 正典 generate_windows で使う評価期間（g1_wfa と同一）
_EVAL_START_CANON = "1977-01-03"  # LT2 N=750 warmup後
_EVAL_END_CANON   = "2026-03-26"


# ---------------------------------------------------------------------------
# 方式A: 正典窓 (canonical / g1_wfa 準拠) per_window DataFrame 生成
# ---------------------------------------------------------------------------

def build_per_window_df_canonical(nav: pd.Series, label: str = "") -> pd.DataFrame | None:
    """正典窓 (g1_wfa.generate_windows カレンダー年アンカー) で per_window DataFrame を生成。
    EVAL_START=1977-01-03 (LT2 N=750 warmup後), EVAL_END=2026-03-26。
    short_flag=True の窓は除外 (compute_summary_stats 内で除外される)。
    これが正典準拠方式。約50窓 (2026年分が短窓のため実効49窓相当)。
    """
    nav = nav.dropna()
    if len(nav) == 0:
        print(f"  [WFA-canon] {label}: NAV空 → スキップ")
        return None

    # generate_windows は dates: pd.Series (DatetimeIndex相当の値を持つ) を要求。
    # NAV の DatetimeIndex を pd.Series に変換してインデックスベースでスライスできるようにする。
    dates_series = pd.Series(nav.index, name="Date")

    windows = _g1_generate_windows(
        dates_series,
        eval_start=_EVAL_START_CANON,
        eval_end=_EVAL_END_CANON,
        window_days=WINDOW,
        step_days=WINDOW,
    )

    if not windows:
        print(f"  [WFA-canon] {label}: 窓なし → スキップ")
        return None

    nav_arr = nav.values
    records = []
    for w in windows:
        s = w["start_idx"]
        e = w["end_idx"] + 1  # exclusive
        seg = nav_arr[s:e]
        n_days = len(seg)
        if n_days < 5 or seg[0] <= 0:
            continue
        years = n_days / 252.0
        cagr = float((seg[-1] / seg[0]) ** (1.0 / years) - 1)
        daily_ret = np.diff(seg) / seg[:-1]
        std = float(np.std(daily_ret, ddof=1))
        sharpe = float(np.mean(daily_ret) / std * np.sqrt(252)) if std > 1e-10 else np.nan
        records.append(dict(
            start_date=pd.Timestamp(w["start_date"]),
            end_date=pd.Timestamp(w["end_date"]),
            CAGR=cagr,
            Sharpe=sharpe,
            short_flag=w["short_flag"],
            n_days=n_days,
        ))

    if not records:
        print(f"  [WFA-canon] {label}: 0 窓 → スキップ")
        return None

    df = pd.DataFrame(records)
    n_total = len(df)
    n_valid = (df["short_flag"] == False).sum()
    print(f"  [WFA-canon] {label}: 計{n_total}窓, うち有効(short_flag=False)={n_valid}窓 "
          f"(IS境界={OOS_START_REF.date()})")
    return df


# ---------------------------------------------------------------------------
# 方式B: 固定窓 (fixed-252 / 旧来方式) per_window DataFrame 生成
# ---------------------------------------------------------------------------

def build_per_window_df_fixed(nav: pd.Series, label: str = "") -> pd.DataFrame | None:
    """非重複 252 日固定窓の per_window DataFrame を NAV から生成 (旧来方式)。
    NAV先頭から開始（warmupスキップなし）。1974年起点で52窓。
    比較・継続性のために保持するが、正典準拠ではない。
    """
    nav = nav.dropna()
    nav_arr = nav.values
    idx = nav.index
    n = len(nav_arr)

    if n < WINDOW:
        print(f"  [WFA-fixed] {label}: NAV長({n}) < WINDOW({WINDOW}) → スキップ")
        return None

    records = []
    i = 0
    while i + WINDOW <= n:
        seg = nav_arr[i: i + WINDOW]
        start_dt = idx[i]
        end_dt = idx[i + WINDOW - 1]
        years = WINDOW / 252.0
        cagr = float((seg[-1] / seg[0]) ** (1.0 / years) - 1)
        daily_ret = np.diff(seg) / seg[:-1]
        std = float(np.std(daily_ret, ddof=1))
        sharpe = float(np.mean(daily_ret) / std * np.sqrt(252)) if std > 1e-10 else np.nan
        short_flag = WINDOW < 201  # 252 > 201 なので常に False
        records.append(
            dict(
                start_date=pd.Timestamp(start_dt),
                end_date=pd.Timestamp(end_dt),
                CAGR=cagr,
                Sharpe=sharpe,
                short_flag=short_flag,
            )
        )
        i += WINDOW

    if not records:
        print(f"  [WFA-fixed] {label}: 0 窓 → スキップ")
        return None

    df = pd.DataFrame(records)
    print(f"  [WFA-fixed] {label}: {len(df)} 窓 (window={WINDOW}日, IS境界={OOS_START_REF.date()})")
    return df


# ---------------------------------------------------------------------------
# WFA 判定 (EVALUATION_STANDARD §3.13)
# ---------------------------------------------------------------------------

def wfe_verdict(wfe: float) -> str:
    """WFE verdict string (ASCII-safe for CSV)."""
    if np.isnan(wfe):
        return "N/A"
    if wfe <= 1.2:
        return "OK"
    if wfe <= 1.5:
        return "CAUTION"
    return "regime_luck"


def wfa_pass_verdict(ci95_lo: float, t_p: float, wfe: float, n_post: int) -> str:
    """α+β 2基準判定 (g1_wfa.evaluate_criteria と同一ロジック)。"""
    crit_alpha = (not np.isnan(ci95_lo)) and (ci95_lo > 0) and (not np.isnan(t_p)) and (t_p < 0.05)
    if n_post < 3:
        crit_beta = True
    else:
        crit_beta = (not np.isnan(wfe)) and (0.5 <= wfe <= 2.0)
    if crit_alpha and crit_beta:
        return "PASS"
    if crit_alpha:
        return "WARN"
    return "FAIL"


# ---------------------------------------------------------------------------
# 1戦略の WFA を実行して dict を返す
# ---------------------------------------------------------------------------

def _run_wfa_from_df(label: str, per_df: pd.DataFrame | None,
                     basis: str, cfd_notional: str, method: str) -> dict:
    """per_window DataFrame から WFA 結果 dict を返す (内部ヘルパー)。"""
    if per_df is None:
        return dict(strategy=label, basis=basis, cfd_notional=cfd_notional,
                    window_method=method,
                    n_windows=0, CI95_lo=np.nan, WFE=np.nan,
                    t_pvalue=np.nan, mean_CAGR=np.nan,
                    WFE_verdict="N/A", WFA_verdict="N/A", note="")

    try:
        result = summarize_wfa(per_df)
    except Exception as e:
        print(f"  [WFA-{method}] {label}: summarize_wfa エラー: {e}")
        return dict(strategy=label, basis=basis, cfd_notional=cfd_notional,
                    window_method=method,
                    n_windows=len(per_df), CI95_lo=np.nan, WFE=np.nan,
                    t_pvalue=np.nan, mean_CAGR=np.nan,
                    WFE_verdict="N/A", WFA_verdict="N/A", note=str(e))

    ci95_lo = result.get("WFA_CI95_lo", np.nan)
    wfe = result.get("WFA_WFE", np.nan)
    t_p = result.get("t_pvalue", np.nan)
    mean_c = result.get("mean_CAGR", np.nan)
    n_win = result.get("n_windows", 0)
    n_post = result.get("n_windows_postIS", 0)

    verd_wfe = wfe_verdict(wfe)
    verd_wfa = wfa_pass_verdict(ci95_lo, t_p, wfe, n_post)

    print(f"  [{method}] => CI95_lo={ci95_lo*100:+.2f}%  WFE={wfe:.3f}  t_p={t_p:.4f}  "
          f"n={n_win}  WFE:{verd_wfe}  WFA:{verd_wfa}")

    return dict(
        strategy=label, basis=basis, cfd_notional=cfd_notional,
        window_method=method,
        n_windows=n_win, CI95_lo=ci95_lo, WFE=wfe,
        t_pvalue=t_p, mean_CAGR=mean_c,
        WFE_verdict=verd_wfe, WFA_verdict=verd_wfa, note="",
    )


def run_one_wfa(label: str, nav: pd.Series, basis: str, cfd_notional: str = "full") -> list[dict]:
    """1戦略の WFA を両方式(canonical/fixed)で実行して結果 dict のリストを返す。
    戻り値は [canonical結果, fixed結果] の2要素リスト。
    """
    # 方式A: 正典窓 (canonical)
    per_df_canon = build_per_window_df_canonical(nav, label)
    row_canon = _run_wfa_from_df(label, per_df_canon, basis, cfd_notional, "canonical")

    # 方式B: 固定窓 (fixed-252)
    per_df_fixed = build_per_window_df_fixed(nav, label)
    row_fixed = _run_wfa_from_df(label, per_df_fixed, basis, cfd_notional, "fixed252")

    return [row_canon, row_fixed]


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict] = []

    # ========================================================================
    # ヘルパー: 1戦略の両方式 WFA を実行してリストに追加
    # ========================================================================
    def _run(label: str, nav_result: dict, basis: str, cfd_notional: str) -> None:
        for row in run_one_wfa(label, nav_result["nav"], basis, cfd_notional):
            all_rows.append(row)

    # ========================================================================
    # 7戦略 × realistic(full L×)
    # ========================================================================

    print("\n" + "="*60)
    print("[E4] basis=realistic (full)")
    print("="*60)
    _run("E4", run_e4("realistic", cfd_notional="full"), "realistic", "full")

    print("\n" + "="*60)
    print("[vz065_l5] basis=realistic (full)")
    print("="*60)
    _run("vz065_l5", run_vz065(5.0, "realistic", cfd_notional="full"), "realistic", "full")

    print("\n" + "="*60)
    print("[vz065_l7] basis=realistic (full)")
    print("="*60)
    _run("vz065_l7", run_vz065(7.0, "realistic", cfd_notional="full"), "realistic", "full")

    print("\n" + "="*60)
    print("[DH-W1] basis=realistic")
    print("="*60)
    _run("DH-W1", run_dhw1("realistic"), "realistic", "N/A(ETF)")

    print("\n" + "="*60)
    print("[V0] basis=realistic")
    print("="*60)
    _run("V0", run_overlay("V0", "realistic"), "realistic", "N/A(ETF)")

    print("\n" + "="*60)
    print("[V7] basis=realistic")
    print("="*60)
    _run("V7", run_overlay("V7", "realistic"), "realistic", "N/A(ETF)")

    print("\n" + "="*60)
    print("[P7] basis=realistic")
    print("="*60)
    _run("P7", run_p7("realistic"), "realistic", "N/A(投信)")

    # ========================================================================
    # CFD3戦略 × realistic(borrowed) 感度
    # ========================================================================

    print("\n" + "="*60)
    print("[E4] basis=realistic (borrowed 感度)")
    print("="*60)
    _run("E4", run_e4("realistic", cfd_notional="borrowed"), "realistic", "borrowed")

    print("\n" + "="*60)
    print("[vz065_l5] basis=realistic (borrowed 感度)")
    print("="*60)
    _run("vz065_l5", run_vz065(5.0, "realistic", cfd_notional="borrowed"), "realistic", "borrowed")

    print("\n" + "="*60)
    print("[vz065_l7] basis=realistic (borrowed 感度)")
    print("="*60)
    _run("vz065_l7", run_vz065(7.0, "realistic", cfd_notional="borrowed"), "realistic", "borrowed")

    # ========================================================================
    # CSV 保存（両方式を全行出力。window_method 列で識別）
    # ========================================================================
    out_dir = os.path.join(_REPO_ROOT, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wfa_realistic_summary_20260610.csv")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  Saved: {out_path}")

    # ========================================================================
    # サマリ表示（方式別に分けて表示）
    # ========================================================================
    for method_label, method_key in [("方式A: 正典窓 (canonical / g1_wfa準拠)", "canonical"),
                                      ("方式B: 固定252日窓 (fixed252 / 旧来方式)", "fixed252")]:
        rows_m = [r for r in all_rows if r.get("window_method") == method_key]
        print("\n" + "="*105)
        print(f"WFA Summary [{method_label}] (revised realistic NAV)")
        if method_key == "canonical":
            print("  ★ 正典準拠 (g1_wfa.generate_windows, カレンダー年アンカー, EVAL_START=1977-01-03)")
        else:
            print("  ※ 旧来方式 (非重複252日固定窓, NAV先頭起点, 比較参照用)")
        print("="*105)
        hdr = (f"{'strategy':<14} {'basis':<12} {'notional':<12} {'n_win':>5} "
               f"{'CI95_lo':>9} {'WFE':>7} {'t_p':>8} {'mean_CAGR':>10} "
               f"{'WFE_verdict':<14} {'WFA_verdict'}")
        print(hdr)
        print("-"*105)
        for r in rows_m:
            ci = r["CI95_lo"]
            wfe = r["WFE"]
            t_p = r["t_pvalue"]
            mc = r["mean_CAGR"]
            ci_s = f"{ci*100:+.2f}%" if not (isinstance(ci, float) and np.isnan(ci)) else "N/A"
            wfe_s = f"{wfe:.3f}" if not (isinstance(wfe, float) and np.isnan(wfe)) else "N/A"
            tp_s = f"{t_p:.4f}" if not (isinstance(t_p, float) and np.isnan(t_p)) else "N/A"
            mc_s = f"{mc*100:+.2f}%" if not (isinstance(mc, float) and np.isnan(mc)) else "N/A"
            print(f"{r['strategy']:<14} {r['basis']:<12} {r['cfd_notional']:<12} "
                  f"{r['n_windows']:>5} {ci_s:>9} {wfe_s:>7} {tp_s:>8} "
                  f"{mc_s:>10} {r['WFE_verdict']:<14} {r['WFA_verdict']}")

    print("\n[Note] WFE=WalkForwardEfficiency=postIS_mean/IS_mean.")
    print("  WFE>1.2 CAUTION: 小標本OOS偏重 (postIS窓4本) → regime luck疑い。")
    print("    ★ vz065の高WFEはrealisticコストが作ったものではない。")
    print("    scenarioD時点で既にE4=1.221, vz065_l5=1.509と高く、realisticはむしろ")
    print("    僅かに低下(E4: 1.221→1.211, vz065_l5: 1.509→1.477)。")
    print("    WFEの実体はpostIS窓=わずか4本(2022-25テック急騰局面)の小標本regime luck。")
    print("  WFE<1.0: OOS<IS -> cost drag or regime shift.")
    print("  ETF/trust: no CFD overnight -> low cost-drag risk, WFE near 1.0 expected.")
    print("  正典準拠(canonical)方式がWFA判定の公式基準。fixed252は比較参照用。")


if __name__ == "__main__":
    main()
