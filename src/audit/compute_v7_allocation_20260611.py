"""
src/audit/compute_v7_allocation_20260611.py
===========================================
V7 boost overlay (DH-W1 + nasdaq_mom63 M6, mult={1.2,1.1,1.0,1.0}) の
全期間時間平均の保有比率を概算する。

出力:
  1. キャッシュ割合   = 1 - (wn + wg + wb) の時間平均 (OUT日は cash=1)
  2. ナスダック割合   = wn の時間平均 (資本ウェイト)
  3. ゴールド割合     = wg の時間平均
  4. ボンド割合       = wb の時間平均
  併せて、レバ込みの「エクスポージャー（想定元本）」基準も参考表示。
    NASDAQ exposure = wn * lev_raw_masked * mult (V7 overlay 適用後)

注意:
  - wn/wg/wb は simulate_rebalance_A の資本ウェイト × W1 hold mask。
  - V7 overlay は lev(レバレッジ) のみ乗算し、資本ウェイト wn/wg/wb は不変。
    → 資本配分（保有比率）は overlay 有無で同一。exposure のみ変化。
  - IS/OOS/FULL 全期間それぞれで集計。
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.audit.strategy_runners as _sr
from src.audit.strategy_runners import _load_dhw1_shared
_SRC_DIR = _sr._SRC_DIR

IS_END = pd.Timestamp("2021-05-07")
OOS_START = pd.Timestamp("2021-05-08")


def _v7_mult(dates_index: pd.DatetimeIndex) -> np.ndarray:
    """V7 overlay multiplier 系列を再構築 (run_overlay と同一手法)。"""
    mapping = {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}
    macro_path = os.path.join(_SRC_DIR, "..", "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()
    signal_raw = macro["nasdaq_mom63"].dropna()

    from signals.quantize import quantile_cut as _quantile_cut
    from signals.timing import apply_publication_lag as _apply_lag

    sig_q = _quantile_cut(signal_raw, levels=4).dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    sig_aligned = sig_lagged.reindex(dates_index).ffill()
    mult = sig_aligned.map(lambda s: mapping.get(int(s), 1.0) if pd.notna(s) else 1.0).fillna(1.0).values
    return np.clip(np.asarray(mult, dtype=float), 0.0, 3.0)


def main() -> None:
    _load_dhw1_shared()
    s = _sr._DHW1_SHARED
    a = s["assets"]
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))

    wn = np.asarray(s["wn"], dtype=float)   # NASDAQ 資本ウェイト × mask
    wg = np.asarray(s["wg"], dtype=float)   # Gold2x 資本ウェイト × mask
    wb = np.asarray(s["wb"], dtype=float)   # Bond3x 資本ウェイト × mask
    lev_masked = np.asarray(s["lev_raw_masked"], dtype=float)
    mask = np.asarray(s["mask"], dtype=float)

    mult = _v7_mult(dates_dt)

    cash = 1.0 - (wn + wg + wb)
    cash = np.clip(cash, 0.0, 1.0)

    # NASDAQ exposure (想定元本) = wn * lev * mult ; gold/bond は 2x/3x ETF 自体のレバを別途持つが
    # 資本配分(wn/wg/wb)とは別概念。ここでは「資本配分」を主、NASDAQ exposure を参考表示。
    nas_expo = wn * lev_masked * mult

    def agg(m: np.ndarray, label: str) -> None:
        print(f"\n=== {label} (n={m.sum()}) ===")
        print(f"  キャッシュ : {cash[m].mean()*100:6.2f} %")
        print(f"  ナスダック : {wn[m].mean()*100:6.2f} %  (資本ウェイト)")
        print(f"  ゴールド   : {wg[m].mean()*100:6.2f} %")
        print(f"  ボンド     : {wb[m].mean()*100:6.2f} %")
        tot = cash[m].mean() + wn[m].mean() + wg[m].mean() + wb[m].mean()
        print(f"  --- 合計   : {tot*100:6.2f} %")
        print(f"  [参考] NASDAQ想定元本(wn*lev*V7mult) 平均: {nas_expo[m].mean()*100:6.2f} %")
        print(f"  [参考] IN日(mask>=0.5)比率: {(mask[m]>=0.5).mean()*100:6.2f} %  / OUT(cash100%)比率: {(mask[m]<0.5).mean()*100:6.2f} %")

    full = np.ones(len(wn), dtype=bool)
    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    agg(full, "FULL 全期間 (1974-2026)")
    agg(is_mask, "IS (〜2021-05-07)")
    agg(oos_mask, "OOS (2021-05-08〜)")


if __name__ == "__main__":
    main()
