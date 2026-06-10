"""
src/audit/run_audit.py
======================
CLI: E4 / vz065_l5 / vz065_l7 パイロット再計算 & コスト基準切替え

使用例:
    python -m src.audit.run_audit --strategy e4 --basis scenarioD --out audit_results/audit_e4_scenarioD.csv
    python -m src.audit.run_audit --strategy e4 --basis realistic  --out audit_results/audit_e4_realistic.csv

    python -m src.audit.run_audit --strategy vz065_l5 --basis scenarioD --out audit_results/audit_vz065_l5_scenarioD.csv
    python -m src.audit.run_audit --strategy vz065_l5 --basis realistic  --out audit_results/audit_vz065_l5_realistic.csv
    python -m src.audit.run_audit --strategy vz065_l7 --basis scenarioD --out audit_results/audit_vz065_l7_scenarioD.csv
    python -m src.audit.run_audit --strategy vz065_l7 --basis realistic  --out audit_results/audit_vz065_l7_realistic.csv

    # 両方まとめて実行:
    python -m src.audit.run_audit --strategy e4 --basis all
    python -m src.audit.run_audit --strategy vz065_l5 --basis all
    python -m src.audit.run_audit --strategy vz065_l7 --basis all

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

from src.audit.strategy_runners import run_e4, run_vz065

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
# vz065 旧基準値 (g27_vz065_lmax_sweep_metrics.csv より, 税引後相当)
# ---------------------------------------------------------------------------
REF_VZ065_L5 = {
    "CAGR_IS":       0.2016,   # +20.16%
    "CAGR_OOS":      0.1893,   # +18.93%
    "Sharpe_OOS":    0.841,
    "MaxDD_FULL":   -0.5672,   # -56.72%
    "Worst10Y_star": 0.1267,   # +12.67%
    "P10_5Y":        0.0875,   # +8.75%
    "Trades_yr":    86.0,
    "WFA_WFE":       1.389,
    "WFA_CI95_lo":   0.1956,   # +19.56%
}

REF_VZ065_L7 = {
    "CAGR_IS":       0.2023,   # +20.23%
    "CAGR_OOS":      0.2149,   # +21.49% (min=+20.23%)
    "Sharpe_OOS":    0.829,
    "MaxDD_FULL":   -0.6595,   # -65.95%
    "Worst10Y_star": 0.0996,   # +9.96%
    "P10_5Y":        np.nan,   # N/A in current table
    "Trades_yr":    105.0,
    "WFA_WFE":       np.nan,   # N/A → to be filled
    "WFA_CI95_lo":   np.nan,   # N/A → to be filled
}


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


def _print_comparison(
    sd_metrics: dict,
    rl_metrics: dict | None,
    ref_old: dict | None = None,
    label: str = "E4",
) -> None:
    """突合テーブルを標準出力に表示。"""

    if ref_old is None:
        ref_old = REF_OLD

    def _fmt(v, pct=True, decimals=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if pct:
            return f"{v*100:+.{decimals}f}%"
        return f"{v:+.{decimals}f}"

    print()
    print("=" * 90)
    print(f"突合テーブル: {label} 再計算 vs 旧基準値 (g27/taxfree)")
    print("=" * 90)

    headers = ["指標", "旧基準値", "scenarioD 再計算", "差 (pp or ratio)", "評価"]
    col_w = [20, 14, 18, 18, 12]

    def row_str(*cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_w))

    print(row_str(*headers))
    print("  " + "-" * 84)

    pct_keys = {"CAGR_OOS", "CAGR_IS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y"}
    for key in ["CAGR_IS", "CAGR_OOS", "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y", "Trades_yr"]:
        ref_val = ref_old.get(key, np.nan)
        if ref_val is None or (isinstance(ref_val, float) and np.isnan(ref_val)):
            ref_str = "N/A"
        elif key in pct_keys:
            ref_str = f"{ref_val*100:+.2f}%"
        else:
            ref_str = f"{ref_val:.3f}"

        sd_val = sd_metrics.get(key, np.nan) if sd_metrics else np.nan
        is_pct = key in pct_keys

        if is_pct:
            sd_str = _fmt(sd_val, pct=True)
            if not np.isnan(sd_val) and not np.isnan(float(ref_val if ref_val is not None else np.nan)):
                diff_pp = (sd_val - ref_val) * 100
                diff_str = f"{diff_pp:+.2f}pp"
                ok = abs(diff_pp) < 3.0
                eval_str = "OK" if ok else "WARN"
            else:
                diff_str = "N/A"; eval_str = "N/A"
        else:
            sd_str = f"{sd_val:.3f}" if not np.isnan(sd_val) else "N/A"
            if not np.isnan(sd_val) and not np.isnan(float(ref_val if ref_val is not None else np.nan)):
                diff = sd_val - ref_val
                diff_str = f"{diff:+.3f}"
                ok = abs(diff) < 10.0
                eval_str = "OK" if ok else "WARN"
            else:
                diff_str = "N/A"; eval_str = "N/A"

        print(row_str(key, ref_str, sd_str, diff_str, eval_str))

    print()
    print("=" * 90)
    print(f"realistic コスト低下 ({label})")
    print("=" * 90)

    if rl_metrics is not None:
        sd_cagr  = sd_metrics.get("CAGR_OOS", np.nan) if sd_metrics else np.nan
        rl_cagr  = rl_metrics.get("CAGR_OOS", np.nan)

        if not (np.isnan(sd_cagr) or np.isnan(rl_cagr)):
            drop_pp = (rl_cagr - sd_cagr) * 100
            print(f"  scenarioD CAGR_OOS:  {sd_cagr*100:+.2f}%")
            print(f"  realistic CAGR_OOS:  {rl_cagr*100:+.2f}%")
            print(f"  低下幅:              {drop_pp:+.2f}pp")
        else:
            print("  realistic CAGR_OOS ga NaN -- keisan error no kanousei ari")

        print()
        print("realistic 10 indicators:")
        for key in ["CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
                    "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y",
                    "Worst5Y", "Trades_yr"]:
            v = rl_metrics.get(key, np.nan)
            try:
                vf = float(v)
            except Exception:
                vf = float("nan")
            if np.isnan(vf):
                print(f"  {key:<20}: N/A")
            elif key == "Trades_yr":
                print(f"  {key:<20}: {vf:.2f}")
            elif key == "IS_OOS_gap_pp":
                print(f"  {key:<20}: {vf:+.2f}pp")
            elif key == "Sharpe_OOS":
                print(f"  {key:<20}: {vf:+.3f}")
            else:
                print(f"  {key:<20}: {vf*100:+.2f}%")
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
        print("  [WFA] 0 windows -- skip")
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

def _print_10metrics(result: dict, label: str) -> None:
    """10 指標をコンソールに表示。"""
    for key in ["CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
                "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y",
                "Worst5Y", "Trades_yr"]:
        v = result.get(key, float("nan"))
        try:
            vf = float(v)
        except Exception:
            vf = float("nan")
        if np.isnan(vf):
            print(f"  {key:<20}: N/A")
        elif key in ("Trades_yr",):
            print(f"  {key:<20}: {vf:.2f}")
        elif key in ("IS_OOS_gap_pp",):
            print(f"  {key:<20}: {vf:+.2f}pp")
        elif key == "Sharpe_OOS":
            print(f"  {key:<20}: {vf:+.3f}")
        else:
            print(f"  {key:<20}: {vf*100:+.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 / vz065 パイロット再計算")
    parser.add_argument(
        "--strategy",
        choices=["e4", "vz065_l5", "vz065_l7"],
        default="e4",
    )
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
    strategy = args.strategy

    # strategy ごとの設定
    if strategy == "e4":
        strat_label = "E4"
        ref_old_val = REF_OLD
        def _run_strategy(basis):
            return run_e4(basis)
        def _csv_name(basis):
            return f"audit_e4_{basis}.csv"
    elif strategy == "vz065_l5":
        strat_label = "vz065_l5"
        ref_old_val = REF_VZ065_L5
        def _run_strategy(basis):
            return run_vz065(5.0, basis)
        def _csv_name(basis):
            return f"audit_vz065_l5_{basis}.csv"
    elif strategy == "vz065_l7":
        strat_label = "vz065_l7"
        ref_old_val = REF_VZ065_L7
        def _run_strategy(basis):
            return run_vz065(7.0, basis)
        def _csv_name(basis):
            return f"audit_vz065_l7_{basis}.csv"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    results: dict[str, dict] = {}

    for basis in bases_to_run:
        print(f"\n{'='*60}")
        print(f"[{strat_label}] basis={basis} NAV building...")
        print(f"{'='*60}")
        try:
            result = _run_strategy(basis)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        results[basis] = result

        # 10 指標をコンソールに表示
        print(f"\n[{strat_label} / {basis}] 10 indicators:")
        _print_10metrics(result, strat_label)

        # CSV 保存
        if args.basis != "all" and args.out:
            out_path = args.out
        else:
            out_path = os.path.join(
                _REPO_ROOT, "audit_results", _csv_name(basis)
            )
        _save_csv(result, basis, out_path)

    # 突合テーブル
    if "scenarioD" in results:
        _print_comparison(
            results["scenarioD"],
            results.get("realistic"),
            ref_old=ref_old_val,
            label=strat_label,
        )
    elif "realistic" in results:
        _print_comparison(
            {},
            results["realistic"],
            ref_old=ref_old_val,
            label=strat_label,
        )

    # WFA
    if args.wfa:
        for basis in bases_to_run:
            if basis not in results:
                continue
            print(f"\n[WFA] {strat_label}/{basis} WFA computing...")
            nav = results[basis]["nav"]
            wfa_result = _compute_wfa(nav, label=f"{strat_label}/{basis}")
            if wfa_result:
                ci95_lo = wfa_result.get("WFA_CI95_lo", np.nan)
                wfe = wfa_result.get("WFA_WFE", np.nan)
                print(f"  => CI95_lo={ci95_lo*100:+.2f}%  WFE={wfe:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
