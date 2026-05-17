# Gold・Bond 商品選定とNASDAQ統合戦略：仮説立案・バックテスト計画

作成日: 2026-05-17
最終更新日: 2026-05-17

## 概要

`kazuyamurayama/deep-research` の研究レポート [2026-05-17_gold-bond-3x-leverage-products.md](https://github.com/kazuyamurayama/deep-research/blob/main/outputs/2026-05-17_gold-bond-3x-leverage-products.md) をもとに、SBI証券で日本居住者が実取引可能なGold/Bond商品から有望候補を選定し、現行ベスト戦略DH Dyn 2x3x [A]とCFD動的レバレッジS2との統合シナリオを設計する計画書。

**スコープ**: 計画立案のみ。バックテスト実装は次フェーズ。

---

## 1. 商品選定（有望候補の絞り込み）

### 1.1 Goldスリーブ候補（3本に絞り込み）

#### G1: 1540+信用買い 2x（現行ベースライン、必須）
- **採用理由**: 最低コスト（年率3.24%）、vol dragなし、現行ベースラインとして比較に必須
- **限界**: レバ2xが上限。NASDAQ 3x/CFD 5xと組み合わせると、Gold側のレバレッジ寄与が相対的に小さい
- **コスト式**: `r_g2 = 2*r_gold - 1*sofr - 0.0324/252`

#### G2: SBI金CFD（最大20x、可変レバレッジ）
- **採用理由**:
  - 日本居住者が3x以上を実現できる数少ない手段
  - CFD型のためvol dragなし → 高レバで効くのは線形コストのみ
  - 動的レバレッジ化（S2型）が可能 → NASDAQ S2とのシンクロ設計が組める
- **不採用にする場合のリスク**: コスト非透明（推計4.4〜5.0%/年）、SBI内部スプレッドの実測値が必要
- **コスト式（モデル化用）**: `r_gc = L_g * r_gold - (L_g-1)*(sofr + gold_cfd_spread)/252`
  - 仮定: `gold_cfd_spread = 0.012`（年率1.2%、SBI実測待ち）
  - 合計コスト目安: L_g=3 のとき (3-1)*(0.0356+0.012) = 9.52%/年（線形）

#### G3: TOCOM金先物（限日取引、無制限レバ）
- **採用理由**:
  - ロールコストが現物保管コスト相当（年率1〜3%）と低い
  - レバ無制限（証拠金次第）、CFD型でvol dragなし
  - 国内取引所のため法的明瞭、長期データ利用可能
- **限界**: 追証リスク、清算機関リスク、ロール作業の運用負荷
- **コスト式**: `r_gf = L_g * r_gold - (L_g-1)*sofr/252 - roll_cost/252`
  - 仮定: `roll_cost = 0.02`（年率2%）

#### 不採用
- **UGL**: 2xでvol drag 2.62%、1540+信用買い 2xに対しコスト面で劣る（4.51%+2.62%=7.13% vs 3.24%）
- **WisdomTree 3GOL**: 日本非対応のため除外
- **JNUG**: 金鉱株であり金現物との連動性が不十分

### 1.2 Bondスリーブ候補

#### 採用: TMF維持（理由付き）
- **理由**:
  - SBIで実取引可能な唯一の Bond 3x
  - GMO・IG はバックテストに必要な過去データの一貫性が乏しい
  - 現行戦略との比較が一貫する
- **代替候補（参考検証のみ）**:
  - Bond2x合成（1540類似の低レバ債券）: TMFのコスト負担が重い局面でのアブレーション用
  - TMF軽減版（重み下げ）: wbを縮小し、Gold比率に振る案
- **コスト式**: `r_b3 = 3*r_bond - 2*sofr/252 - 0.0095/252 - vol_drag_b/252`
  - `vol_drag_b = 0.5 * 3 * 2 * sigma_bond^2 ≈ 0.5*3*2*0.125^2 = 0.0469`（年率4.69%）

### 1.3 コストモデル化の統一フォーマット

```python
def sleeve_return(r_underlying, L, sofr, fixed_cost, sigma, is_etf):
    # 借入コスト: (L-1)*sofr
    # 固定コスト: fixed_cost (経費率+スプレッド)
    # vol drag: ETF型のみ
    drag = 0.5*L*(L-1)*sigma**2 if is_etf else 0.0
    return L*r_underlying - (L-1)*sofr/252 - fixed_cost/252 - drag/252
```

| 商品 | L | fixed_cost (年率) | is_etf | sigma仮定 |
|---|---|---|---|---|
| 1540+信用 2x | 2 | 0.0224 | False | 0.16 |
| SBI金CFD 3x | 3 | 0.012 | False | 0.16 |
| SBI金CFD 5x | 5 | 0.012 | False | 0.16 |
| TOCOM金先物 3x | 3 | 0.02 | False | 0.16 |
| TMF 3x | 3 | 0.0095 | True | 0.125 |

---

## 2. 仮説立案（5本）

### H1: Gold3x CFD × TQQQ 3x × TMF 3x（対称3x強化）
- **構造**: NASDAQ=TQQQ(3x固定) / Gold=SBI金CFD 3x / Bond=TMF 3x
- **重み**: 現行ロジック踏襲（wn=0.30〜0.90、wg=wb=(1-wn)/2）
- **期待効果**: ベースライン[A]に対し、Gold側レバを2x→3xに引き上げることで、低wn局面（NASDAQ弱気局面）でのGold寄与を約1.5倍に拡大。Worst5Yの底上げを狙う
- **リスク**: Gold-NASDAQが共倒れする2022年型相場でMaxDDが悪化する可能性。CFDコストの実測誤差

### H2: Gold5x CFD × S2 NASDAQ × TMF 3x（高レバ・対称型）
- **構造**: NASDAQ=S2 CFD(avg 5x) / Gold=SBI金CFD 5x / Bond=TMF 3x
- **重み**: wn=0.30〜0.80（CFD型でリスク高いためwn上限引き下げ）、wg=wb=(1-wn)/2
- **期待効果**: S2版CAGR+32.35%に対し、Gold側のリスクパリティを揃えることでSharpe改善。Worst5Y -4.73%を改善
- **リスク**: MaxDD -62.36%がさらに悪化する可能性。Gold CFD 5xはvol dragなしでも線形コスト=(5-1)*0.0476=19.04%/年で重い

### H3: Gold動的レバレッジ（S2_Gold） × S2 NASDAQ × TMF 3x（二重S2型）
- **構造**: NASDAQ=S2 CFD(target_vol=0.80) / Gold=S2_Goldスリーブ(target_vol=0.30, max 5x) / Bond=TMF 3x
- **重み**: wn動的、wg=wb=(1-wn)/2
- **S2_Goldパラメータ仮置**:
  - `target_vol_gold = 0.30`（Gold単独vol≈16%に対し約2倍）
  - `k_vz_gold = 0.20`（NASDAQよりゲートは緩く）
  - `gate_min_gold = 1.0`（最低1x維持）
  - `gate_max_gold = 5.0`
- **期待効果**: Gold変動率に応じてレバレッジを自動調整、平時は3〜4x、急騰期は5xに伸ばす。固定3xより drag リスク低減
- **リスク**: S2_Goldのパラメータ過剰フィッティング、Gold vz信号の質がNASDAQほど確立されていない

### H4: TOCOM金先物 3x × TQQQ 3x × Bond軽減版（低コスト・Gold傾斜）
- **構造**: NASDAQ=TQQQ(3x) / Gold=TOCOM先物 3x / Bond=TMF 3x but wb軽減
- **重み**: wn=0.30〜0.90、wg=(1-wn)*0.65、wb=(1-wn)*0.35（Gold厚め）
- **期待効果**: TMFのコスト負担(12.76%)を抑え、低コストTOCOM金(年率2%程度のロール)に振り分ける。金利上昇局面に強い
- **リスク**: TMFの株式逆相関が薄まるため、株式急落局面のクッションが弱くなる

### H5: ハイブリッド（CFD Gold + 1540 Gold ブレンド） × TQQQ × TMF
- **構造**: NASDAQ=TQQQ(3x) / Gold=0.5×1540+信用2x + 0.5×SBI金CFD 4x（実効3x相当だがコスト平準化）/ Bond=TMF 3x
- **重み**: 現行と同じwn動的
- **期待効果**: 低コスト1540（年率3.24%）と高レバCFD（線形コスト）をブレンドし、実効レバ3x・実効コスト約4〜5%/年を実現。単一CFD3xより安価
- **リスク**: 実装複雑度。1540信用建玉とCFD両方の証拠金管理が必要

---

## 3. バックテスト検証計画

### 3.1 必要データ

| データ | 期間 | 代替/取得元 |
|---|---|---|
| Gold価格（USD/oz spot） | 2010〜現在 | LBMA PM Fix or `GC=F` (Yahoo Finance) |
| Gold価格（JPY建て、1540連動） | 2010〜現在 | `1540.T` Yahoo（信用買い時のJPY基準） |
| 米国20年超債リターン | 2010〜現在 | `TLT` (Yahoo) → 3x合成でTMF再現 |
| TMF実績 | 2010〜現在 | `TMF` (Yahoo)、ベンチマーク照合用 |
| NASDAQ-100 | 2010〜現在 | `^NDX` または `QQQ`、現行と一致 |
| SOFR/Fed Funds | 2010〜現在 | FRED `SOFR` / `EFFR` |
| TOCOM金先物 | 2010〜現在 | TOCOM公式 or quandl、ロール調整必要 |

### 3.2 コストモデル実装方針（ファイル/関数）

新規ファイル `src/sleeves_extended.py`:
- `gold_cfd_sleeve(r_gold, L, sofr, spread=0.012)`: SBI金CFD（H1, H2用）
- `gold_tocom_sleeve(r_gold, L, sofr, roll_cost=0.02)`: TOCOM先物（H4用）
- `gold_s2_sleeve(r_gold, vz_gold, target_vol=0.30, gate=(1.0,5.0))`: 動的レバ（H3用）
- `gold_hybrid_sleeve(r_gold_etf, r_gold_cfd, w_etf=0.5, ...)`: ハイブリッド（H5用）

既存 `cfd_leverage_backtest.py` の `build_nav_strategy` の修正:
- 引数に `gold_sleeve_fn`, `bond_sleeve_fn` を追加し、関数を差し替え可能に
- 既存挙動を破壊しないよう、デフォルトは現行（Gold2x+Bond3x）

新規スクリプト `src/run_hypotheses_H1_H5.py`:
- 5仮説それぞれを実行し、結果を `H1_H5_SUMMARY_2026-05-17.md` に集約

### 3.3 評価指標と採用基準

| 指標 | 現行ベースライン[A] | S2版ベースライン | 採用ライン |
|---|---|---|---|
| CAGR_OOS | 22.50% | 32.35% | ≥ 22.5%（少なくとも[A]超え） |
| Sharpe_OOS | 0.646 | 0.769 | ≥ 0.65（[A]同等以上） |
| MaxDD | -45.08% | -62.36% | ≥ -55%（S2版より良いこと） |
| Worst5Y | +0.87% | -4.73% | ≥ 0%（必須条件） |
| IS-OOS Gap | — | 5.4pp | ≤ 8pp（オーバーフィット警戒） |

**採用判定ルール**:
1. **必須**: Worst5Y ≥ 0% かつ IS-OOS Gap ≤ 8pp
2. **優先採用**: Sharpe_OOS ≥ 0.769 (S2版超え)
3. **次点採用**: Sharpe_OOS ∈ [0.65, 0.77] かつ MaxDD改善

### 3.4 スクリプト構成

新規:
- `src/sleeves_extended.py`（新規、約200行）
- `src/run_hypotheses_H1_H5.py`（新規、約300行）
- `src/data_loader_gold_bond.py`（新規、Gold/TOCOM/TLTデータ取得）

修正:
- `src/cfd_leverage_backtest.py` に sleeve関数差し替えI/Fを追加（破壊変更なし、引数追加のみ）

結果出力:
- `H1_H5_SUMMARY_2026-05-17.md`（仮説検証レポート）

---

## 4. 実装の注意事項

### 4.1 Gold 3x CFD のモデル化近似
- **SBI金CFD価格**: 公式の長期データは取得困難 → **LBMA PM Fix（USD/oz）をベース価格として使用**し、JPY換算は USD/JPY スポットで日次変換
- **スプレッド推計**: SBI公式で実測したスプレッド（ピップ単位）を年率換算
  - 例: スプレッド30銭 / 価格10000円 × 往復2回 × 250営業日 = 年率1.5%程度
  - 実測がないため、**保守側で1.2%を仮置**し、感度分析で[0.8%, 1.5%, 2.0%]を試す
- **証拠金/オーバーナイト金利**: SOFR + 1.2%を借入レート扱い
- **コード上の数式**:
  ```
  r_gold_cfd_3x = 3*r_gold_usd - 2*sofr_daily - 0.012/252
  ```
  vol dragは加算しない（CFDは日次リバランス型ではない）

### 4.2 TOCOM先物のロールコストモデル化
- **限日取引のロール頻度**: 偶数月限月 → 年6回ロール
- **ロールコスト構造**:
  - コンタンゴ時: 期先プレミアム = 保管コスト+金利-便益利回り
  - 現在の金市場は概ね軽コンタンゴ（年率1〜3%）
- **実装方針**:
  - 簡易版: 一定の年率2%を日次按分（`-0.02/252` を毎日減算）
  - 精密版: NY金先物の期近-期先スプレッド（CME公表）を月次で取得し、TOCOM分は0.7倍程度で近似
- **追証リスクは backtest 上は考慮せず**、別途レバ制限（max 5x）で間接対応

### 4.3 TMF vol dragの正確な実装
- **公式**: `drag = 0.5 * L * (L-1) * sigma^2` （年率ベース）
- **TMFの場合**: L=3, sigma_TMF基礎資産 ≈ 12.5%（20年超債のvol）
  - `drag = 0.5 * 3 * 2 * 0.125^2 = 0.046875 = 4.69%/年`
- **実装の注意**:
  - sigmaは**基礎資産（TLT）のvol**を使う。TMFのvol（≈37%）ではない
  - 動的に推定する場合は60日ローリングsigmaを使用
  - 日次に按分: `vol_drag_daily = 0.0469 / 252`
- **検証**: TMF実績リターンと、TLT 3x合成 - SOFR*2 - 経費率0.95% - vol drag の差を比較し、誤差±0.5%/年以内なら採用

### 4.4 データ入手可能性

| データ | 取得方法 | 代替策 |
|---|---|---|
| SBI金CFD価格 | 過去データ非公開 | **LBMA PM Fix (USD/oz) + USD/JPY** で代替 |
| SBI金CFDスプレッド | 公式非公開 | 実測してconfig化、感度分析で頑健性確認 |
| TOCOM金先物 | TOCOM公式 / Bloomberg | `1540.T`の信用買いコスト構造で近似可（要追加検証） |
| TMF/TLT/QQQ | Yahoo Finance | 問題なし |
| SOFR | FRED API | 2018年以前は **EFFR**（実効FF金利）で代替 |

### 4.5 既存 build_nav_strategy との整合性

現行コードの不変条件（壊さない部分）:
- 入力: NASDAQ収益率系列、ボラ系列、レバ系列
- 出力: 日次ポートフォリオリターン
- wn/wg/wbの計算ロジック（A2 Approach A、閾値0.15）

**追加するインターフェース**（後方互換）:
```python
def build_nav_strategy(
    ...既存引数...,
    gold_sleeve_fn=default_gold_2x,   # 新規、デフォルトは現行Gold2x
    bond_sleeve_fn=default_bond_3x,   # 新規、デフォルトは現行Bond3x
    gold_sleeve_kwargs={},             # 新規
    bond_sleeve_kwargs={},             # 新規
):
```

これにより、H1〜H5それぞれは `gold_sleeve_fn` のみ差し替えて検証可能。既存テストは無変更で通る。

### 4.6 感度分析とロバストネステスト
各仮説について以下のパラメータ感度を取る:
- SOFR: [2.0%, 3.56%, 5.0%]（金利環境変化）
- Gold CFD spread: [0.8%, 1.2%, 1.5%, 2.0%]
- TMF vol drag計算用sigma: [10%, 12.5%, 15%]
- 期間分割: IS/OOS = [2010-2018]/[2019-2026]、および [2010-2020]/[2021-2026] の2通り

これらすべてでSharpe_OOS ≥ 0.65、Worst5Y ≥ 0% を維持できれば採用候補とする。

---

## まとめ

- **Gold候補**: G1=1540+信用2x（ベースライン）、G2=SBI金CFD（3x/5x可変）、G3=TOCOM先物3x の3本に絞る
- **Bond**: TMF 3xを基本維持、H4でwb軽減版を試す
- **仮説5本**: H1〜H3はレバ拡張系、H4は低コスト・Gold傾斜、H5はハイブリッド
- **採用ライン**: Sharpe_OOS ≥ 0.65、Worst5Y ≥ 0%、IS-OOS Gap ≤ 8pp
- **実装**: `sleeves_extended.py` 新規、`build_nav_strategy` に sleeve関数差し替えI/F追加（後方互換）
- **最大の不確実性**: SBI金CFDのスプレッド実測値とTOCOM先物のロールコスト実績 → 感度分析で頑健性を確認

---

## 関連ドキュメント

- [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) — 現行ベスト戦略の正典
- [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md) — S2 CFD動的レバレッジ仕様
- [S2_DH_INTEGRATION_2026-05-17.md](S2_DH_INTEGRATION_2026-05-17.md) — S2 DH統合バックテスト結果
- 参考研究: [kazuyamurayama/deep-research 2026-05-17_gold-bond-3x-leverage-products.md](https://github.com/kazuyamurayama/deep-research/blob/main/outputs/2026-05-17_gold-bond-3x-leverage-products.md)

---

*次フェーズ: H1〜H5 のバックテスト実装 → 採用判定 → CURRENT_BEST_STRATEGY.md 更新*
