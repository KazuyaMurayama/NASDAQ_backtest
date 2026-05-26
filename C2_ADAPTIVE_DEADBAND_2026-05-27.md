# C2: Adaptive Deadband（ε_t=σ連動）

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 目的

C2: Adaptive Deadband（ε_t=σ連動）- CFAR原理でノイズ耐性向上

F10系のε=0.015固定deadbandを、市場ボラティリティに連動して動的変更。

```
ε_t = ε_0 × (σ_t / σ̄)
  σ_t  = 20日実現ボラ（年率）
  σ̄   = 250日平均ボラ
  clip: vol_ratio ∈ [0.3, 3.0]
```

- **高ボラ時**: εが大きくなり不要な取引を抑制
- **低ボラ時**: εが小さくなりシグナルに素早く反応
- **CFAR（Constant False Alarm Rate）原理**: ボラ環境に関わらず「誤検出率」を一定に保つ

| 項目 | 定義 |
|------|------|
| **Base config** | E4 採用: k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5, LT2-N750, mode B |
| **Tilt 定式** | step-function: clip(tilt × (raw_a2-0.15) × (1-raw_a2), 0, cap_eff), tilt=10.0 |
| **cap_eff** | calm: 0.15, bull-VZ: 0.1, bear-VZ: 0.05 |
| **ε₀ グリッド** | [0.01, 0.015, 0.02] |
| **vol窓** | σ_20d (実現ボラ) / σ̄_250d (平均) |
| **vol_ratio clip** | [0.3, 3.0] |
| **REF** | F10 ε=0.015固定（固定deadband比較） |
| **IS** | 1974-01-02 〜 2021-05-07 |
| **OOS** | 2021-05-08 〜 |

---

## §2 9指標テーブル

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| ε=0.015/fixed | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.31pp |  52 |    —    |    —    |
| ε₀=0.010/adaptive | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.29pp |  52 |    —    |    —    |
| ε₀=0.015/adaptive | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.30pp |  52 |    —    |    —    |
| ε₀=0.020/adaptive | +36.8% | +0.93 ★ | -63.0% | +18.6% | +10.3% | -4.28pp |  51 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

---

## §3 REF（F10 ε=0.015固定）との比較

| config | Tr/yr | Sharpe<br>_OOS | CAGR<br>_OOS | MaxDD | IS-OOS<br>gap | ΔSharpe<br>vs REF | ΔCAGR<br>vs REF | ΔTr/yr<br>vs REF | 判定 |
|:-------|------:|---------------:|-------------:|------:|--------------:|------------------:|----------------:|-----------------:|:-----|
| ε=0.015/fixed |  51.6 | +0.935 | +36.84% | -63.09% | -4.31pp | +0.0000 | +0.00pp | +0.0 | PASS |
| ε₀=0.010/adaptive |  51.8 | +0.934 | +36.83% | -63.07% | -4.29pp | -0.0002 | -0.01pp | +0.2 | PASS |
| ε₀=0.015/adaptive |  51.5 | +0.935 | +36.84% | -63.09% | -4.30pp | -0.0000 | -0.00pp | -0.0 | PASS |
| ε₀=0.020/adaptive |  51.3 | +0.935 | +36.84% | -63.04% | -4.28pp | -0.0000 | -0.00pp | -0.3 | PASS |

---

## §4 採用判断

### 採用条件 (優先順)
| 順位 | 条件 |
|:----:|:-----|
| (1) | Trades/yr ≤ 70（ユーザー許容） |
| (2) | Sharpe_OOS ≥ +0.8915（現行 E4 以上） |
| (3) | IS-OOS gap ≤ +6.0pp |
| (4) | MaxDD > -65.00% (望ましい), > -80% (絶対) |

### 判定: **PASS**

**採用候補: ε₀=0.020/adaptive** → Sharpe=+0.935, CAGR_OOS=+36.84%, MaxDD=-63.04%, Trades/yr=51.3, IS-OOS gap=-4.28pp

### E4 (現行ベスト) との比較
| 指標 | E4 (現行ベスト) | F10 REF (ε=0.015固定) | C2 最良 (ε₀=0.020/adaptive) |
|:-----|----------------:|----------------------:|-------------------------------:|
| CAGR_OOS | +33.53% | +36.84% | +36.84% |
| Sharpe_OOS | +0.891 | +0.935 | +0.935 |
| MaxDD | -60.01% | -63.09% | -63.04% |
| Worst10Y★ | +18.67% | +18.58% | +18.63% |
| IS-OOS gap | -1.81pp | -4.31pp | -4.28pp |
| Trades/yr | 27.1 | 51.6 | 51.3 |
| WFA CI95_lo | +0.265‡ | — | (実施推奨) |
| WFA WFE | +1.131‡ | — | (実施推奨) |

‡ E4 の G3 WFA は CI95_lo=+26.51%, WFE=+1.131（CURRENT_BEST_STRATEGY.md より）。

---

## §5 再現コマンド

```bash
cd "C:\Users\user\Desktop\投資・不動産\nasdaq_backtest"
python -X utf8 src/c2_adaptive_deadband.py
```

出力:
- `c2_adaptive_deadband_results.csv` — 9指標 + verdict
- `C2_ADAPTIVE_DEADBAND_2026-05-27.md` — 本レポート

参照:
- `src/f10_epsilon_deadband.py` — F10 固定ε実装（ベース）
- `src/e4_regime_klt.py` — E4 base 実装
- `CURRENT_BEST_STRATEGY.md` — E4 現行ベスト
- `EVALUATION_STANDARD.md` §3.12 — 9指標標準

---

*生成スクリプト: `src/c2_adaptive_deadband.py`*
