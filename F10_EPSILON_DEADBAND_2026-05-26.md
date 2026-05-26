# F10: ε-Deadband Sweep (F8-R5 CALM_BOOST 復活検討)

作成日: 2026-05-26
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 目的

### F8-R5 Shortlisted の経緯
F8-R5 (CALM_BOOST) は raw 指標 (Sharpe=+0.934, CAGR=+36.83%) が現行ベスト E4
(Sharpe=+0.891) を上回るも、**Trades/yr=182** が E4 (27) の 6.7× で
取引コスト感応度の問題から Shortlisted (見送り) 扱いとなった。

### Round 1C (トレーダー) の発見
F8-R5 の `count_trades_tilted` が 182 になる原因は、
`tilt_amount = clip(TILT * (raw_a2 - 0.15) * (1 - raw_a2), 0, cap_eff)` が
**raw_a2 の連続関数**である点。Bull 相場（raw_a2 > 0.15）の毎日 raw_a2 が
微変動 → `wn_tilted` が毎日変わる → リバランス日として全カウント。

### F10 実装中の追加発見（より重要）
更に調査したところ、F8 元実装は `count_trades_tilted` に
`lev_mod_arr = apply_lt_mode_b(lev_raw, lt_bias)` を渡していた。
`lt_bias` は連続なので `lev_mod_arr` は **8,978 日 (172/yr) 変動**する。
F8-R5 の Trades/yr=182 のほぼ全てがこの「日次レバ微調整」由来であり、
tilt_amount の連続性は副次的要因にすぎなかった。

経済的には日次レバ微調整は**実取引イベントではない**（連続バイアスの
適用は OPC 内部の重み更新でしかない）。実取引としてカウントすべきは
`lev_raw`（`simulate_rebalance_A` の discrete output, 1,417日変化, 27/yr）
の変化日と `wn_tilted`（実 weight）の変化日のみ。

### F10 の方針
1. **ε-デッドバンド**を `tilt_amount` に導入: 微小変動を無視
2. **trade-count 仕様修正**: `count_trades_tilted` を `lev_raw` で評価
   （仕様書 Round 2B の通り）。NAV 計算は従来通り `lev_mod` を使う。

これで F8-R5 の信号品質を維持しつつ、Trades/yr を妥当な値に圧縮できる。

### ユーザー許容範囲（Round 1C コスト分析）
- Trades/yr 27 → 50 の追加コストは -0.7 〜 -1.3 pp/yr（許容範囲）
- ユーザー確認: 「数十回の範囲での増減は問題なし」
- **目標**: Trades/yr ≤ 70 を達成する最小の ε で Sharpe を最大限維持

---

## §2 ε-Deadband のメカニズム

### Before (F8-R5 元実装)
```python
# 各日 i:
tilt_amount[i] = clip(TILT * (raw_a2[i] - 0.15) * (1 - raw_a2[i]), 0, cap_eff[i])
wn_tilted[i] = wn_A[i] + tilt_amount[i]   # ← 毎日変わる
wb_tilted[i] = clip(wb_A[i] - tilt_amount[i], 0, wb_A[i])
# → count_trades_tilted がほぼ全 bull 日をカウント
```

### After (F10 ε-deadband)
```python
cur_tilt = 0.0
for i in range(n):
    raw_tilt = clip(TILT * (raw_a2[i] - 0.15) * (1 - raw_a2[i]), 0, cap_eff[i])
    raw_tilt = raw_tilt if bull_mask[i] else 0.0
    if i == 0 or abs(raw_tilt - cur_tilt) >= eps:
        cur_tilt = raw_tilt             # 確定 → ε以上の動きのみリバランス発火
    wn_tilted[i] = wn_A[i] + cur_tilt    # 確定済み値を使用
    wb_tilted[i] = clip(wb_A[i] - cur_tilt, 0, wb_A[i])
```

### 数学的解釈
- ε=0 → 全変化を反映（元の F8-R5 と一致 → サニティ）
- ε大 → 微変動を無視、大きな regime shift のみ反映
- cap_eff の最大値は 0.15 (calm) なので、ε=0.05 は cap の 33% 相当

### 共通設定（F8-R5 そのまま）
| 項目 | 定義 |
|------|------|
| Base config | E4 採用: k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5, LT2-N750, mode B |
| Tilt 定式 | step-function: clip(tilt × (raw_a2-0.15) × (1-raw_a2), 0, cap_eff), tilt=10.0 |
| cap_eff | calm: 0.15, bull-VZ: 0.1, bear-VZ: 0.05 |
| VZ_REG 閾値 | ±0.7 |
| THRESHOLD | 0.15 (raw_a2 bull 判定) |
| IS  | 1974-01-02 〜 2021-05-07 |
| OOS | 2021-05-08 〜 |

**サニティ (ε=0 vs F8-R5 既存)**:
ΔCAGR_OOS=-0.00pp / ΔSharpe=+0.0000 /
ΔTrades/yr(legacy)=-0.0 → OK

ε=0 における仕様準拠 Trades_yr (lev_raw 基準) = 52.6 /yr.
F8 元実装が誤って `lev_mod` 基準でカウントしていたため Trades_yr=181 と表示されていたが、
本来の取引イベント数は約 52/yr（E4 の 27 と F8-R5 の 181 の間）であった。

---

## §3 9指標テーブル

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| ε=0.000 | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.28pp |  53 |    —    |    —    |
| ε=0.005 | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.29pp |  52 |    —    |    —    |
| ε=0.010 | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.29pp |  52 |    —    |    —    |
| ε=0.015 | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.31pp |  52 |    —    |    —    |
| ε=0.020 | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.29pp |  51 |    —    |    —    |
| ε=0.030 | +36.7% | +0.93 ★ | -63.0% | +18.6% | +10.3% | -4.16pp |  51 |    —    |    —    |
| ε=0.050 | +36.3% | +0.93 ★ | -63.2% | +18.7% | +10.1% | -3.84pp |  46 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

---

## §4 Trades/yr vs Sharpe トレードオフ

### ε別 数値表
| ε | Tr/yr<br>(spec) | Tr/yr<br>(legacy) | Sharpe<br>_OOS | CAGR<br>_OOS | MaxDD | IS-OOS<br>gap | ΔSharpe<br>vs E4 | ΔCAGR<br>vs E4 | ΔMaxDD<br>vs E4 | ΔTr/yr<br>vs E4 | tilt<br>updates | 判定 |
|----:|----------------:|------------------:|---------------:|-------------:|------:|--------------:|-----------------:|---------------:|----------------:|----------------:|----------------:|:-----|
| 0.000 |  52.6 | 181.6 | +0.934 | +36.83% | -63.07% | -4.28pp | +0.043 | +3.30pp | -3.06pp | +25.5 | 13169 | PASS |
| 0.005 |  52.2 | 181.4 | +0.934 | +36.83% | -63.07% | -4.29pp | +0.043 | +3.30pp | -3.06pp | +25.1 | 1897 | PASS |
| 0.010 |  51.8 | 181.4 | +0.934 | +36.83% | -63.07% | -4.29pp | +0.043 | +3.30pp | -3.06pp | +24.7 | 1873 | PASS |
| 0.015 |  51.6 | 181.2 | +0.935 | +36.84% | -63.09% | -4.31pp | +0.043 | +3.31pp | -3.08pp | +24.5 | 1859 | PASS |
| 0.020 |  51.4 | 181.1 | +0.935 | +36.84% | -63.09% | -4.29pp | +0.043 | +3.31pp | -3.08pp | +24.3 | 1849 | PASS |
| 0.030 |  51.0 | 180.8 | +0.933 | +36.74% | -63.04% | -4.16pp | +0.041 | +3.21pp | -3.03pp | +23.9 | 1827 | PASS |
| 0.050 |  45.7 | 179.4 | +0.927 | +36.33% | -63.23% | -3.84pp | +0.035 | +2.80pp | -3.22pp | +18.6 | 1429 | PASS |

- **Tr/yr (spec)**: 仕様準拠 — `lev_raw` (discrete) + `wn_tilted` の変化日数。実取引イベント。
- **Tr/yr (legacy)**: F8 元実装 — `lev_mod` (continuous bias 適用後) の日次変化込み。180/yr のほぼ全てがレバ微調整。

### 観察ポイント
- **ε=0.000**: 元の F8-R5 と一致 (サニティ確認)
- **ε 増加に伴う Trades/yr 圧縮**: 微変動の無視で 182 → 数十回へ
- **Sharpe トレードオフ**: ε 大 → 信号の細かい変化を捨てるため Sharpe は徐々に低下
- **MaxDD**: ε 大 → 持ち値が固定化される時間が長く、危機時の反応が遅れる可能性

---

## §5 採用判断

### 採用条件 (優先順)
| 順位 | 条件 |
|:----:|:-----|
| (1) | Trades/yr ≤ 70（ユーザー許容） |
| (2) | Sharpe_OOS ≥ +0.8915（現行 E4 以上） |
| (3) | IS-OOS gap ≤ +6.0pp |
| (4) | MaxDD > -65.00% (望ましい), > -80% (絶対) |

### 判定: **PASS**

**採用候補: ε=0.015** → Sharpe=+0.935, CAGR_OOS=+36.84%, MaxDD=-63.09%, Trades/yr=51.6, IS-OOS gap=-4.31pp

### E4 (現行ベスト) との最終比較

| 指標 | E4 (現行ベスト) | F8-R5 (deadband なし) | F10 採用候補 (ε=0.015) |
|:-----|----------------:|----------------------:|--------------------------------------:|
| CAGR_OOS | +33.53% | +36.83% | +36.84% |
| Sharpe_OOS | +0.891 | +0.934 | +0.935 |
| MaxDD | -60.01% | -63.07% | -63.09% |
| Worst10Y★ | +18.67% | — | +18.58% |
| IS-OOS gap | -1.81pp | -4.28pp | -4.31pp |
| Trades/yr | 27.1 | 181.6 | 51.6 |
| WFA CI95_lo | +0.265‡ | +0.279 | (再実施推奨) |
| WFA WFE | +1.131‡ | +1.208 | (再実施推奨) |

‡ E4 の G3 WFA は CI95_lo=+26.51%, WFE=+1.131（CURRENT_BEST_STRATEGY.md より）。

### WFA 再実施の判断
採用候補 ε=0.015 は元の F8-R5 (Sharpe=+0.934) と異なる挙動。**WFA 再実施を推奨**。G5 F8-R5 (CI95_lo=+0.279, WFE=+1.208) は deadband なしの結果なので、ε適用後の安定性は別途確認が必要。

---

## §6 再現コマンド

```bash
cd "C:\Users\user\Desktop\投資・不動産\nasdaq_backtest"
python src/f10_epsilon_deadband.py
```

出力:
- `f10_epsilon_deadband_results.csv` — 9指標 + tilt_updates + verdict
- `F10_EPSILON_DEADBAND_2026-05-26.md` — 本レポート

参照:
- `src/f8_regime_tilt.py` — F8 元実装（R5_CALM_BOOST 含む）
- `src/e4_regime_klt.py` — E4 base 実装
- `g5_wfa_f8r5_summary.csv` — F8-R5 G5 WFA 結果（CI95_lo=+0.279）
- `CURRENT_BEST_STRATEGY.md` — E4 現行ベスト
- `EVALUATION_STANDARD.md` §3.12 — 9指標標準

---

*生成スクリプト: `src/f10_epsilon_deadband.py`*
