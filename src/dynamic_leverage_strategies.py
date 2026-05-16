"""
Dynamic Leverage Strategies for NASDAQ CFD (1x-7x)
====================================================
P1: SOFR適応型        — SOFRが低いときだけ高レバ
P2: ボラ・ターゲティング型 — 実現ボラに反比例してレバ調整
P3: モメンタム比例型   — 短期トレンド強度に比例してレバ増減
P4: 複合スコア型      — SOFR×ボラ×モメンタムの乗算スコア（本命）
P5: Kelly近似型       — 対数効用最大化の理論解 L* = μ_excess/σ²

全関数の仕様:
  - 戻り値: pd.Series (index = close.index), 値 ∈ [l_min, l_max], step刻み
  - DELAYシフトは含まない (呼び出し側 build_nav_strategy が内部でシフト)
  - min_periods を明示し、初期 warm-up 期間は l_min に固定
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _quantize(series: pd.Series, step: float) -> pd.Series:
    """step刻みに離散化 (取引頻度抑制)"""
    return (series / step).round() * step


# ---------------------------------------------------------------------------
# P1: SOFR適応型
# ---------------------------------------------------------------------------

def compute_L_sofr_adaptive(sofr_daily: pd.Series,
                             sofr_high: float = 0.08,
                             l_min: float = 1.0,
                             l_max: float = 7.0,
                             step: float = 0.5) -> pd.Series:
    """
    SOFRが低いほど高レバ、sofr_high以上で l_min に固定。

    L = l_max - (l_max - l_min) * clip(sofr_ann / sofr_high, 0, 1)

    根拠: NASDAQの超過リターン期待値 ≈ E[r_nas] - SOFR。
    借入コスト (L-1)×SOFR が期待リターンを食い潰す前にデレバ。
    """
    sofr_ann = sofr_daily * TRADING_DAYS
    factor = np.clip(sofr_ann / sofr_high, 0.0, 1.0)
    L_raw = l_max - (l_max - l_min) * factor
    return _quantize(L_raw.clip(l_min, l_max), step)


# ---------------------------------------------------------------------------
# P2: ボラ・ターゲティング型
# ---------------------------------------------------------------------------

def compute_L_vol_target(returns: pd.Series,
                          target_vol: float = 0.60,
                          n: int = 20,
                          l_min: float = 1.0,
                          l_max: float = 7.0,
                          step: float = 0.5) -> pd.Series:
    """
    実現ボラに反比例してレバを調整 (vol targeting)。

    L = clip(TARGET_VOL / sigma_t, l_min, l_max)

    根拠: L×σ を一定化すると損失分布の左裾が安定する。
    VIX急騰前に実現ボラが先行して上昇するため、暴落初期で自動デレバ。
    """
    sigma = returns.rolling(n, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    sigma = sigma.replace(0.0, np.nan)
    L_raw = (target_vol / sigma).clip(l_min, l_max).fillna(l_min)
    return _quantize(L_raw, step)


# ---------------------------------------------------------------------------
# P3: モメンタム比例型
# ---------------------------------------------------------------------------

def compute_L_momentum(close: pd.Series,
                        m: int = 20,
                        k: float = 1.0,
                        l_min: float = 1.0,
                        l_max: float = 7.0,
                        step: float = 0.5) -> pd.Series:
    """
    短期モメンタムのz-scoreをsigmoidで [0,1] にマップしレバ調整。

    f = sigmoid(k * z_mom)
    L = l_min + (l_max - l_min) * f

    根拠: モメンタム効果 (Jegadeesh-Titman) はNASDAQで持続。
    ベア相場初期にモメンタム負転換→自動デレバ、ブル継続中は最大化。
    """
    mom = close / close.shift(m) - 1
    mu_m = mom.rolling(252, min_periods=60).mean()
    std_m = mom.rolling(252, min_periods=60).std().replace(0.0, np.nan)
    z = (mom - mu_m) / std_m
    z_filled = z.fillna(0.0)
    f = 1.0 / (1.0 + np.exp(-k * z_filled))   # sigmoid → (0,1)
    L_raw = l_min + (l_max - l_min) * f
    return _quantize(L_raw.clip(l_min, l_max), step)


# ---------------------------------------------------------------------------
# P4: 複合スコア型（本命）
# ---------------------------------------------------------------------------

def compute_L_composite(close: pd.Series,
                         returns: pd.Series,
                         sofr_daily: pd.Series,
                         sofr_high: float = 0.08,
                         target_vol: float = 0.60,
                         m: int = 20,
                         k: float = 1.0,
                         l_min: float = 1.0,
                         l_max: float = 7.0,
                         step: float = 0.5) -> pd.Series:
    """
    SOFR・ボラ・モメンタムの3因子乗算スコアでレバ決定。

    score = f_sofr × f_vol × f_mom  ∈ [0,1]
    L = l_min + (l_max - l_min) × score

    根拠: 「低金利×低ボラ×上昇トレンド」が同時成立するときのみ
    フルレバを取る。1因子でも悪条件なら自動デレバ（乗算の論理AND効果）。
    """
    # --- f_sofr: SOFRが高いほど0に近づく ---
    sofr_ann = sofr_daily * TRADING_DAYS
    f_sofr = np.clip(1.0 - sofr_ann / sofr_high, 0.0, 1.0)

    # --- f_vol: 実現ボラが高いほど0に近づく ---
    sigma = returns.rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    sigma = sigma.replace(0.0, np.nan).fillna(target_vol)
    f_vol = np.clip(target_vol / sigma, 0.0, 1.0)

    # --- f_mom: モメンタムが正のほど1に近づく ---
    mom = close / close.shift(m) - 1
    mu_m = mom.rolling(252, min_periods=60).mean()
    std_m = mom.rolling(252, min_periods=60).std().replace(0.0, np.nan)
    z = (mom - mu_m) / std_m
    f_mom = 0.5 + 0.5 * np.tanh(k * z.fillna(0.0))

    score = (f_sofr * f_vol * f_mom).clip(0.0, 1.0)
    L_raw = l_min + (l_max - l_min) * score
    return _quantize(L_raw.clip(l_min, l_max), step)


# ---------------------------------------------------------------------------
# P5: Kelly近似型
# ---------------------------------------------------------------------------

def compute_L_kelly(returns: pd.Series,
                    sofr_daily: pd.Series,
                    safety: float = 0.5,
                    mu_window: int = 252,
                    sigma_window: int = 60,
                    l_min: float = 1.0,
                    l_max: float = 7.0,
                    step: float = 0.5) -> pd.Series:
    """
    連続Kelly公式の経験ベイズ近似: L* = safety × μ_excess / σ²

    根拠: 対数効用最大化の解。SAFETY=0.5 はμ推定誤差を考慮した
    ハーフKelly (実務標準)。高金利期はμ_excessが縮小→自動デレバ。

    初期 mu_window 期間はデータ不足のため l_min に固定。
    """
    sofr_ann = sofr_daily * TRADING_DAYS
    mu_ann = returns.rolling(mu_window, min_periods=60).mean() * TRADING_DAYS
    sigma_ann = returns.rolling(sigma_window, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    sigma2 = sigma_ann ** 2
    sigma2 = sigma2.replace(0.0, np.nan)

    mu_excess = mu_ann - sofr_ann
    L_kelly = mu_excess / sigma2
    L_raw = (safety * L_kelly).clip(l_min, l_max).fillna(l_min)
    return _quantize(L_raw, step)


# ---------------------------------------------------------------------------
# S1: A2-Conviction Leverage（確信度スケーリング）
# ---------------------------------------------------------------------------

def compute_L_s1_conviction(raw_a2: pd.Series,
                             returns: pd.Series,
                             alpha: float = 1.0,
                             target_vol: float = 0.60,
                             n: int = 20,
                             l_min: float = 1.0,
                             l_max: float = 7.0,
                             step: float = 0.5) -> pd.Series:
    """
    A2の raw スコア（確信度）をレバレッジに直接変換。

    conviction = raw_A2 ^ alpha        (alpha<1: 中程度でもレバ高め, alpha>1: 高確信のみ)
    vt_mult    = clip(target_vol/σ, 0, 1)  (高ボラ時に上限抑制)
    L = l_min + (l_max - l_min) × conviction × vt_mult

    根拠: A2が「強気確信（0.9）」のときだけフルレバ、「低確信（0.3）」では
    最大でも3〜4倍相当に抑える。ベア相場でA2が低下→自動デレバ。
    """
    sigma = returns.rolling(n, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    sigma = sigma.clip(lower=1e-6)
    vt_mult = np.clip(target_vol / sigma, 0.0, 1.0)

    conviction = raw_a2.fillna(0.0).clip(0.0, 1.0) ** alpha
    L_raw = l_min + (l_max - l_min) * conviction * vt_mult
    return _quantize(L_raw.clip(l_min, l_max), step)


# ---------------------------------------------------------------------------
# S2: VZ-Gated Vol Target（VIXゲート付きP2）
# ---------------------------------------------------------------------------

def compute_L_s2_vz_gated(returns: pd.Series,
                            vz: pd.Series,
                            target_vol: float = 0.60,
                            k_vz: float = 0.30,
                            gate_min: float = 0.35,
                            n: int = 20,
                            l_min: float = 1.0,
                            l_max: float = 7.0,
                            step: float = 0.5) -> pd.Series:
    """
    P2（ボラ・ターゲティング）にVIXプロキシz-scoreの非対称ゲートを前置。

    vz_gate = clip(1 - k_vz × max(vz, 0), gate_min, 1.0)
    L = clip(target_vol / sigma × vz_gate, l_min, l_max)

    根拠: P2は実現ボラが上がってからしか反応しない（遅れ2〜3週）。
    VIXプロキシは実現ボラより1〜5日先行するため、急落初動で先に守れる。
    vz<0（平穏）ではゲート=1.0（P2と同一）、vz>0で線形減衰（非対称）。
    """
    sigma = returns.rolling(n, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    sigma = sigma.clip(lower=1e-6)

    vz_pos = vz.fillna(0.0).clip(lower=0.0)   # 負のVZは無視（非対称）
    vz_gate = (1.0 - k_vz * vz_pos).clip(gate_min, 1.0)

    L_raw = (target_vol / sigma) * vz_gate
    return _quantize(L_raw.clip(l_min, l_max), step)


# ---------------------------------------------------------------------------
# S3: Decomposed A2 Leverage（A2分解再配線）
# ---------------------------------------------------------------------------

def compute_L_s3_decomposed(components: dict,
                              beta_defense: float = 1.0,
                              l_min: float = 1.0,
                              l_max: float = 7.0,
                              step: float = 0.5) -> pd.Series:
    """
    A2の原子コンポーネントを L_t に直接再配線。wn/lev_A 経路とは独立。

    defense     = clip(dd × vm, 0, 1)            — 守備系（ドローダウン×VIX）
    offense_norm = clip(slope × mom / 1.95, 0, 1) — 攻撃系（トレンド×モメンタム）
    vol_adj     = vt                              — ボラ調整
    score       = defense^beta × offense_norm × vol_adj
    L = l_min + (l_max - l_min) × score

    beta_defense < 1: 部分回復でも早めにレバを上げる
    beta_defense > 1: 完全回復を確認してからレバを上げる（保守的）
    """
    dd    = components['dd']
    vt    = components['vt']
    slope = components['slope']
    mom   = components['mom']
    vm    = components['vm']

    defense = (dd * vm).clip(0.0, 1.0)
    defense_curved = defense.clip(0.0, 1.0) ** beta_defense

    offense = (slope * mom).clip(0.0, 1.95)
    offense_norm = (offense / 1.95).clip(0.0, 1.0)

    score = (defense_curved * offense_norm * vt).clip(0.0, 1.0)
    L_raw = l_min + (l_max - l_min) * score
    return _quantize(L_raw.clip(l_min, l_max), step)
