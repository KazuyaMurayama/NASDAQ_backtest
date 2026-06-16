# CAGR押上げ — レバ拡張 × 早期再エントリー 検証計画（B3a延長）

作成日: 2026-06-16
最終更新日: 2026-06-16

> **目的**: ベスト戦略 [B3a_k365](CURRENT_BEST_STRATEGY.md)（min⓽ +20.98%）の延長線上で **CAGRをさらに押し上げる** 2方向を検証。(#1) uniform leverage を 1.15超へ拡張、(#2) DH-W1 早期再エントリー（OUT期短縮）。combine探索で「CAGRは exotic オーバーレイでは伸びず、レバ水準が本質」と確定したのを受けた、唯一筋の通るCAGR方向。
> 規律: ハードベト（MaxDD<−50%/WFE>1.5/W10Y★<0/Regime_min<−10%）＋WFA正典49窓＋CPCV＋レジーム層別＋multi-metric bootstrap＋選択バイアス割引。**全性能表は標準10指標フル必須**（CAGR_IS⓽/CAGR_OOS⓽/min⓽/IS-OOS gap/Sharpeⓒ/MaxDDⓒ/Worst10Y★⓽/P10_5Y⓽/Worst5Y⓽/Trades/yr）。実行=Sonnet、計画・統合=Opus。

---

## 0. 前提

- **土台**: B3a_k365 = DH-W1（Asymm Hyst, Enter≥0.7/Exit≤0.3）+ V7マップ`{Q0:1.40,Q1:1.40,Q2:1.05,Q3:1.00}`×uniform1.15 + P09充填 + C1 + コスト(≤3x=TQQQ / >3x=くりっく株365 0.25%/yr)。
- **B3a素地（QC独立再現済）**: min⓽+20.98%・MaxDD−38.20%・Sharpe0.904・Worst10Y★+14.53%・Trades33.3。ベト余白＝MaxDD −38.2% vs ベト−50% で**約12pp**。
- **正直な前提**: レバ拡張はCAGRを上げるが **Sharpe（risk-adjusted）は上げない純レバアップ**で、DD/2008防御の余白を食う。bootstrap有意性はB3aと同等。「より多いCAGR」は取れるが「より良いrisk-adjusted CAGR」ではない（combineで確認済）。20%+は既に達成しており、**ベト緩和は禁止**（ベト内で行けるところまで）。

## 1. #1 レバ拡張スイープ（uniform leverage を 1.15超へ）

| 軸 | 水準 |
|---|---|
| uniform scale | **{1.20, 1.25, 1.30, 1.35}**（B3aの1.15を超える未探索域） |
| v7_map | (a) B3a既定 {1.40,1.40,1.05,1.00} (b) 強ブースト {1.60,1.50,1.10,1.00} |
| 固定 | P09充填・C1・k365コスト |

- 各構成で **標準10指標フル＋>3x日比率＋worst暦年**＋ハードベト。
- **MaxDD−50%ベトに当たる scale を特定**（CAGR-DDフロンティアの上端）。
- k365で>3x超過コストが安いので、高レバ構成でも超過分コストは現実的（各構成の>3x日比率も記録）。

## 2. #2 早期再エントリー（OUT期短縮）

| 軸 | 水準 |
|---|---|
| W1 enter 閾値 | **{0.65, 0.60}**（既定0.70より早く再IN） |
| W1 exit 閾値 | 0.30固定（まず enter のみ） |
| 土台 | B3a（scale1.15・既定マップ）に適用 |

- DH-W1 ヒステリシス状態機械を新閾値で**再構築**（mask/lev_raw_masked を作り直す。`g23a_dh_refinement`/`strategy_runners` の DH-W1 構築を参照、lookahead厳禁）。
- OUT比率の変化（現状約47%）・標準10指標フル・ベトを記録。仮説: 市場滞在↑でCAGR↑だがwhipsaw/DD↑。

## 3. 評価ゲート（共通）

1. Stage 0: 標準10指標フル＋4ハードベト（ベト抵触は即失格・緩和禁止）
2. Stage 1: ベト無＆min⓽がB3a超の上位構成に WFA正典49窓（α/β）＋CPCV p10＋レジーム層別＋stress窓
3. Stage 2: multi-metric bootstrap（min/MaxDD/Worst10Y★/Sharpe）対B3a・対V7
4. 選択バイアス: グリッド事前登録（本計画）・min⓽点推定にdeflated併記・採用は応答曲面の滑らかさを確認

## 4. 採用判定

- min⓽が B3a（+20.98%）を**ベト無しで**上回る最良構成を「CAGR強化版」候補に。
- **ベト緩和して20%+を狙うことはしない**（既に20%超）。MaxDDが−50%に当たったらそこが上端。
- Sharpe低下・Trades増は6次元採点⑤⑥③で減点（正直に開示）。

## 5. 成果物

- スクリプト: `src/audit/leverext_scale_20260616.py`(#1) / `leverext_reentry_20260616.py`(#2)
- CSV: `audit_results/leverext_*_20260616.csv`
- 結果: `LEVERUP_EXTENSION_RESULTS_20260616.md`（標準10指標フル表・CAGR-DDフロンティア・正直な留保）

全成果物は push後にURL検証・3列表報告。

---

*管理者: 男座員也（Kazuya Oza）*
