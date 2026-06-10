"""
src/audit/run_audit.py
======================
CLI: E4 パイロット再計算 & コスト基準切替え

使用例:
    python -m src.audit.run_audit --strategy e4 --basis scenarioD --out audit_results/audit_e4_scenarioD.csv
    python -m src.audit.run_audit --strategy e4 --basis realistic  --out audit_results/audit_e4_realistic.csv

    # 両方まとめて実行:
    python -m src.audit.run_audit --strategy e4 --basis all

出力:
    指定 CSV に 10 指標を1行で出力。
    突合テーブルを標準出力に表示。
"""

from __future__ import annotations

import argparse
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

# Repo root をパスに追加（python -m src.audit.run_audit から実行する前提）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import numpy as np

from src.audit.strategy_runners import run_e4

# ---------------------------------------------------------------------------
# 旧基準値 (CURRENT_BEST_STRATEGY.md / e4_regime_klt.py 冒頭コメント, 税引前)
# ---------------------------------------------------------------------------
REF_OLD = {
    "CAGR_OOS":      0.3353,   # +33.53%
    "Sharpe_OOS":    0.891,
    "MaxDD_FULL":   -0.6001,   # -60.01%
    "Worst10Y_star": 0.1867,   # +18.67%
    "Trades_yr":    27.0,
}

# CURRENT_BEST_STRATEGY.md 定額 3.0% 修正後の CAGR_OOS 参照値
REF_FIXED30PCT_CAGR_OOS = 0.200   # +20.0% (定額3.0%コスト修正後)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _save_csv(metrics: dict, basis: str, out_path: str) -> None:
    """10 指標を1行 CSV で保存。NAV キーは除外。"""
    row = {k: v for k, v in metrics.items() if k != "nav" and k != "trades_per_year"}
    row["basis"] = basis
    row["Trades_yr"] = metrics.get("Trades_yr", metrics.get("trades_per_year"))
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Saved: {out_path}")


def _print_comparison(sd_metrics: dict, rl_metrics: dict | None) -> None:
    """突合テーブルを標準出力に表示。"""

    def _fmt(v, pct=True, decimals=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if pct:
            return f"{v*100:+.{decimals}f}%"
        return f"{v:+.{decimals}f}"

    print()
    print("=" * 90)
    print("突合テーブル: E4 再計算 vs 旧基準値 (taxfree/pretax)")
    print("=" * 90)

    headers = ["指標", "旧基準値", "scenarioD 再計算", "差 (pp or ratio)", "評価"]
    col_w = [20, 14, 18, 18, 12]

    def row_str(*cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_w))

    print(row_str(*headers))
    print("  " + "-" * 84)

    checks = [
        ("CAGR_OOS",      True,  ">=0", REF_OLD["CAGR_OOS"]),
        ("Sharpe_OOS",    False, ">=0", REF_OLD["Sharpe_OOS"]),
        ("MaxDD_FULL",    True,  "<=0", REF_OLD["MaxDD_FULL"]),
        ("Worst10Y_star", True,  ">=0", REF_OLD["Worst10Y_star"]),
        ("Trades_yr",     False, "~27", REF_OLD["Trades_yr"]),
    ]

    for key, is_pct, direction, ref_val in checks:
        sd_val = sd_metrics.get(key, np.nan)
        if is_pct:
            ref_str = f"{ref_val*100:+.2f}%"
            sd_str  = _fmt(sd_val, pct=True)
            if not np.isnan(sd_val):
                diff_pp = (sd_val - ref_val) * 100
                diff_str = f"{diff_pp:+.2f}pp"
                ok = abs(diff_pp) < 2.0
                eval_str = "OK" if ok else "WARN"
            else:
                diff_str = "N/A"
                eval_str = "N/A"
        else:
            ref_str = f"{ref_val:.3f}"
            sd_str  = f"{sd_val:.3f}" if not np.isnan(sd_val) else "N/A"
            if not np.isnan(sd_val):
                diff = sd_val - ref_val
                diff_str = f"{diff:+.3f}"
                ok = abs(diff) < 0.05
                eval_str = "OK" if ok else "WARN"
            else:
                diff_str = "N/A"
                eval_str = "N/A"

        print(row_str(key, ref_str, sd_str, diff_str, eval_str))

    print()
    print("=" * 90)
    print("R7 定量化: realistic コスト低下")
    print("=" * 90)

    if rl_metrics is not None:
        sd_cagr  = sd_metrics.get("CAGR_OOS", np.nan)
        rl_cagr  = rl_metrics.get("CAGR_OOS", np.nan)

        if not (np.isnan(sd_cagr) or np.isnan(rl_cagr)):
            drop_pp = (rl_cagr - sd_cagr) * 100
            print(f"  scenarioD CAGR_OOS:  {sd_cagr*100:+.2f}%")
            print(f"  realistic CAGR_OOS:  {rl_cagr*100:+.2f}%")
            print(f"  低下幅 (R7):          {drop_pp:+.2f}pp")
            print(f"  定額3.0%修正値との比較: {REF_FIXED30PCT_CAGR_OOS*100:+.2f}%")
            diff_fixed = (rl_cagr - REF_FIXED30PCT_CAGR_OOS) * 100
            print(f"  realistic vs 定額3.0%: {diff_fixed:+.2f}pp")
        else:
            print("  realistic CAGR_OOS が NaN — 計算エラーの可能性あり")

        print()
        print("realistic 10 indicators:")
        for key in ["CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
                    "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y",
                    "Worst5Y", "Trades_yr"]:
            v = rl_metrics.get(key, np.nan)
            if np.isnan(float(v)):
                print(f"  {key:<20}: N/A")
            elif key == "Trades_yr":
                print(f"  {key:<20}: {v:.2f}")
            elif key == "IS_OOS_gap_pp":
                print(f"  {key:<20}: {v:+.2f}pp")
            elif key == "Sharpe_OOS":
                print(f"  {key:<20}: {v:+.3f}")
            else:
                print(f"  {key:<20}: {v*100:+.2f}%")
    else:
        print("  realistic は計算されませんでした。")

    print("=" * 90)


# ---------------------------------------------------------------------------
# WFA (realistic NAV から非重複 252 日窓)
# ---------------------------------------------------------------------------

def _compute_wfa(nav: pd.Series, label: str = "E4") -> dict | None:
    """非重複 252 日窓の per_window DataFrame を作り summarize_wfa で CI95_lo/WFE を返す。"""
    try:
        from src.audit.unified_wfa import summarize_wfa, WINDOW_LEN
    except Exception as e:
        print(f"  [WFA] unified_wfa import 失敗: {e}")
        return None

    nav_arr = nav.dropna().values
    idx = nav.dropna().index
    n = len(nav_arr)
    window = WINDOW_LEN  # 252

    records = []
    i = 0
    while i + window <= n:
        seg = nav_arr[i:i + window]
        start_dt = idx[i]
        end_dt = idx[i + window - 1]
        years = window / 252
        cagr = float((seg[-1] / seg[0]) ** (1.0 / years) - 1)
        daily_ret = np.diff(seg) / seg[:-1]
        std = np.std(daily_ret, ddof=1)
        sharpe = float(np.mean(daily_ret) / std * np.sqrt(252)) if std > 1e-10 else np.nan
        short_flag = window < 201
        records.append(
            dict(
                start_date=pd.Timestamp(start_dt),
                end_date=pd.Timestamp(end_dt),
                CAGR=cagr,
                Sharpe=sharpe,
                short_flag=short_flag,
            )
        )
        i += window

    if not records:
        print("  [WFA] 窓が 0 個 — スキップ")
        return None

    per_df = pd.DataFrame(records)
    print(f"  [WFA] {label}: {len(per_df)} 窓 (window={window} 日)")
    try:
        result = summarize_wfa(per_df)
        ci95_lo = result.get("WFA_CI95_lo", np.nan)
        wfe = result.get("WFA_WFE", np.nan)
        print(f"  [WFA] CI95_lo={ci95_lo:.4f} ({ci95_lo*100:+.2f}%)  WFE={wfe:.3f}")
        return result
    except Exception as e:
        print(f"  [WFA] summarize_wfa エラー: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E4 パイロット再計算")
    parser.add_argument("--strategy", choices=["e4"], default="e4")
    parser.add_argument(
        "--basis", choices=["scenarioD", "realistic", "all"], default="all"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="出力 CSV パス (basis=all の場合は無視)",
    )
    parser.add_argument(
        "--wfa",
        action="store_true",
        default=True,
        help="WFA を計算する (default: True)",
    )
    parser.add_argument(
        "--no-wfa",
        dest="wfa",
        action="store_false",
        help="WFA をスキップ",
    )
    args = parser.parse_args()

    bases_to_run = ["scenarioD", "realistic"] if args.basis == "all" else [args.basis]

    results: dict[str, dict] = {}

    for basis in bases_to_run:
        print(f"\n{'='*60}")
        print(f"[E4] basis={basis} NAV building...")
        print(f"{'='*60}")
        try:
            result = run_e4(basis)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        results[basis] = result

        # 10 指標をコンソールに表示
        print(f"\n[E4 / {basis}] 10 indicators:")
        for key in ["CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
                    "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y",
                    "Worst5Y", "Trades_yr"]:
            v = result.get(key, float("nan"))
            if np.isnan(float(v)):
                print(f"  {key:<20}: N/A")
            elif key in ("Trades_yr",):
                print(f"  {key:<20}: {v:.2f}")
            elif key in ("IS_OOS_gap_pp",):
                print(f"  {key:<20}: {v:+.2f}pp")
            elif key == "Sharpe_OOS":
                print(f"  {key:<20}: {v:+.3f}")
            else:
                print(f"  {key:<20}: {v*100:+.2f}%")

        # CSV 保存
        if args.basis != "all" and args.out:
            out_path = args.out
        else:
            out_path = os.path.join(
                _REPO_ROOT, "audit_results", f"audit_e4_{basis}.csv"
            )
        _save_csv(result, basis, out_path)

    # 突合テーブル (both results available)
    if "scenarioD" in results:
        _print_comparison(
            results["scenarioD"],
            results.get("realistic"),
        )

    # WFA
    if args.wfa:
        for basis in bases_to_run:
            if basis not in results:
                continue
            print(f"\n[WFA] {basis} — 計算中...")
            nav = results[basis]["nav"]
            wfa_result = _compute_wfa(nav, label=f"E4/{basis}")
            if wfa_result:
                print(f"  WFA summary: {wfa_result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
