"""
src/audit/unified_wfa.py
========================
全戦略で同一窓設計を強制する薄いラッパ。

公開インターフェース:
  WINDOW_LEN      = 252  (営業日 / 年)
  OOS_START_REF   = pd.Timestamp('2021-05-08')
  summarize_wfa(per_df) -> dict
      src.g1_wfa.compute_summary_stats を呼び出す。
      戻り値は g1_wfa と同じキー群を持つ dict。

設計方針:
  - 統計計算ロジックは g1_wfa.compute_summary_stats に完全委譲。
  - このモジュールは列名の正規化と定数の公開のみを担う。
  - 正典ファイル (src/*.py) は一切改変しない。
"""

from __future__ import annotations

import pandas as pd

# g1_wfa から定数と統計集計関数を import (正典)
from src.g1_wfa import (  # noqa: F401
    compute_summary_stats,
    WINDOW_DAYS as WINDOW_LEN,
    OOS_START_REF as _OOS_START_RAW,
)

# OOS_START_REF を pd.Timestamp として公開
OOS_START_REF: pd.Timestamp = pd.Timestamp(_OOS_START_RAW)

# WINDOW_LEN は g1_wfa.WINDOW_DAYS と同値 (252) — already imported above


# ---------------------------------------------------------------------------
# 列名の正規化マップ
# ---------------------------------------------------------------------------
# per_window CSV の列名が g1_wfa の期待値と異なる場合のリネームルール。
# 現状の g3_wfa_e4_per_window.csv は g1_wfa と同一スキーマのため不要だが、
# 将来の CSV フォーマット変化に備えて拡張ポイントとして定義する。
_COLUMN_ALIAS: dict[str, str] = {
    # 例: 'date_start': 'start_date',
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名を g1_wfa が期待する名前に正規化する (最小限)。"""
    rename = {k: v for k, v in _COLUMN_ALIAS.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)

    # start_date / end_date が文字列のままなら datetime に変換
    for col in ('start_date', 'end_date'):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.copy()
            df[col] = pd.to_datetime(df[col])

    return df


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def summarize_wfa(per_df: pd.DataFrame) -> dict:
    """per_window DataFrame を受け取り、統計サマリ dict を返す。

    Parameters
    ----------
    per_df : pd.DataFrame
        g1_wfa.compute_window_metrics が生成するフォーマットの DataFrame。
        必須列: CAGR, Sharpe, short_flag, start_date
        CAGR は小数形式 (0.33 = +33%)。

    Returns
    -------
    dict
        compute_summary_stats の戻り値と同一キー群。
        主要キー: WFA_CI95_lo, WFA_CI95_hi, WFA_WFE, t_pvalue,
                  n_windows, mean_CAGR, mean_CAGR_IS, mean_CAGR_postIS 等。
    """
    df = _normalize_columns(per_df.copy())
    return compute_summary_stats(df)
