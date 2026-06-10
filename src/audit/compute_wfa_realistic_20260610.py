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
窓長: 252営業日, 非重複, short_flag = window < 201
評価開始: warmup後の最初の完全窓 (NAV先頭から)
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

WINDOW = WINDOW_LEN  # 252


# ---------------------------------------------------------------------------
# per_window DataFrame 生成 (run_audit._compute_wfa と同一ロジック)
# ---------------------------------------------------------------------------

def build_per_window_df(nav: pd.Series, label: str = "") -> pd.DataFrame | None:
    """非重複 252 日窓の per_window DataFrame を NAV から生成。
    EVAL_STANDARD §3.13: IS境界=2021-05-08, short_flag=window<201 (常にFalse for 252窓)。
    """
    nav = nav.dropna()
    nav_arr = nav.values
    idx = nav.index
    n = len(nav_arr)

    if n < WINDOW:
        print(f"  [WFA] {label}: NAV長({n}) < WINDOW({WINDOW}) → スキップ")
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
        print(f"  [WFA] {label}: 0 窓 → スキップ")
        return None

    df = pd.DataFrame(records)
    print(f"  [WFA] {label}: {len(df)} 窓 (window={WINDOW}日, IS境界={OOS_START_REF.date()})")
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

def run_one_wfa(label: str, nav: pd.Series, basis: str, cfd_notional: str = "full") -> dict:
    """1戦略の WFA を実行して結果 dict を返す。"""
    per_df = build_per_window_df(nav, label)
    if per_df is None:
        return dict(strategy=label, basis=basis, cfd_notional=cfd_notional,
                    n_windows=0, CI95_lo=np.nan, WFE=np.nan,
                    t_pvalue=np.nan, mean_CAGR=np.nan,
                    WFE_verdict="N/A", WFA_verdict="N/A", note="")

    try:
        result = summarize_wfa(per_df)
    except Exception as e:
        print(f"  [WFA] {label}: summarize_wfa エラー: {e}")
        return dict(strategy=label, basis=basis, cfd_notional=cfd_notional,
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

    print(f"  => CI95_lo={ci95_lo*100:+.2f}%  WFE={wfe:.3f}  t_p={t_p:.4f}  n={n_win}  WFE:{verd_wfe}  WFA:{verd_wfa}")

    return dict(
        strategy=label, basis=basis, cfd_notional=cfd_notional,
        n_windows=n_win, CI95_lo=ci95_lo, WFE=wfe,
        t_pvalue=t_p, mean_CAGR=mean_c,
        WFE_verdict=verd_wfe, WFA_verdict=verd_wfa, note="",
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []

    # ========================================================================
    # 7戦略 × realistic(full L×)
    # ========================================================================

    # --- E4 (full) ---
    print("\n" + "="*60)
    print("[E4] basis=realistic (full)")
    print("="*60)
    r_e4 = run_e4("realistic", cfd_notional="full")
    rows.append(run_one_wfa("E4", r_e4["nav"], "realistic", "full"))

    # --- vz065_l5 (full) ---
    print("\n" + "="*60)
    print("[vz065_l5] basis=realistic (full)")
    print("="*60)
    r_l5 = run_vz065(5.0, "realistic", cfd_notional="full")
    rows.append(run_one_wfa("vz065_l5", r_l5["nav"], "realistic", "full"))

    # --- vz065_l7 (full) ---
    print("\n" + "="*60)
    print("[vz065_l7] basis=realistic (full)")
    print("="*60)
    r_l7 = run_vz065(7.0, "realistic", cfd_notional="full")
    rows.append(run_one_wfa("vz065_l7", r_l7["nav"], "realistic", "full"))

    # --- DH-W1 (realistic) ---
    print("\n" + "="*60)
    print("[DH-W1] basis=realistic")
    print("="*60)
    r_dhw1 = run_dhw1("realistic")
    rows.append(run_one_wfa("DH-W1", r_dhw1["nav"], "realistic", "N/A(ETF)"))

    # --- V0 (realistic) ---
    print("\n" + "="*60)
    print("[V0] basis=realistic")
    print("="*60)
    r_v0 = run_overlay("V0", "realistic")
    rows.append(run_one_wfa("V0", r_v0["nav"], "realistic", "N/A(ETF)"))

    # --- V7 (realistic) ---
    print("\n" + "="*60)
    print("[V7] basis=realistic")
    print("="*60)
    r_v7 = run_overlay("V7", "realistic")
    rows.append(run_one_wfa("V7", r_v7["nav"], "realistic", "N/A(ETF)"))

    # --- P7 (realistic) ---
    print("\n" + "="*60)
    print("[P7] basis=realistic")
    print("="*60)
    r_p7 = run_p7("realistic")
    rows.append(run_one_wfa("P7", r_p7["nav"], "realistic", "N/A(投信)"))

    # ========================================================================
    # CFD3戦略 × realistic(borrowed) 感度
    # ========================================================================

    # --- E4 (borrowed) ---
    print("\n" + "="*60)
    print("[E4] basis=realistic (borrowed 感度)")
    print("="*60)
    r_e4_b = run_e4("realistic", cfd_notional="borrowed")
    rows.append(run_one_wfa("E4", r_e4_b["nav"], "realistic", "borrowed"))

    # --- vz065_l5 (borrowed) ---
    print("\n" + "="*60)
    print("[vz065_l5] basis=realistic (borrowed 感度)")
    print("="*60)
    r_l5_b = run_vz065(5.0, "realistic", cfd_notional="borrowed")
    rows.append(run_one_wfa("vz065_l5", r_l5_b["nav"], "realistic", "borrowed"))

    # --- vz065_l7 (borrowed) ---
    print("\n" + "="*60)
    print("[vz065_l7] basis=realistic (borrowed 感度)")
    print("="*60)
    r_l7_b = run_vz065(7.0, "realistic", cfd_notional="borrowed")
    rows.append(run_one_wfa("vz065_l7", r_l7_b["nav"], "realistic", "borrowed"))

    # ========================================================================
    # CSV 保存
    # ========================================================================
    out_dir = os.path.join(_REPO_ROOT, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wfa_realistic_summary_20260610.csv")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  Saved: {out_path}")

    # ========================================================================
    # サマリ表示
    # ========================================================================
    print("\n" + "="*90)
    print("WFA Summary (revised realistic NAV - full Lx / borrowed sensitivity)")
    print("="*90)
    hdr = f"{'strategy':<14} {'basis':<12} {'notional':<12} {'n_win':>5} {'CI95_lo':>9} {'WFE':>7} {'t_p':>8} {'mean_CAGR':>10} {'WFE_verdict':<14} {'WFA_verdict'}"
    print(hdr)
    print("-"*100)
    for r in rows:
        ci = r["CI95_lo"]
        wfe = r["WFE"]
        t_p = r["t_pvalue"]
        mc = r["mean_CAGR"]
        ci_s = f"{ci*100:+.2f}%" if not np.isnan(ci) else "N/A"
        wfe_s = f"{wfe:.3f}" if not np.isnan(wfe) else "N/A"
        tp_s = f"{t_p:.4f}" if not np.isnan(t_p) else "N/A"
        mc_s = f"{mc*100:+.2f}%" if not np.isnan(mc) else "N/A"
        print(f"{r['strategy']:<14} {r['basis']:<12} {r['cfd_notional']:<12} {r['n_windows']:>5} {ci_s:>9} {wfe_s:>7} {tp_s:>8} {mc_s:>10} {r['WFE_verdict']:<14} {r['WFA_verdict']}")

    print("\n[Note] WFE=WalkForwardEfficiency=postIS_mean/IS_mean.")
    print("  WFE>1.2 CAUTION: OOS-biased -> regime luck concern (vz065 high IS cost -> high WFE expected).")
    print("  WFE<1.0: OOS<IS -> cost drag or regime shift (CFD high-cost era concern).")
    print("  ETF/trust: no CFD overnight -> low cost-drag risk, WFE near 1.0 expected.")
    print("  Realistic WFA computed from revised NAV, full Lx basis (borrowed = sensitivity only).")


if __name__ == "__main__":
    main()
