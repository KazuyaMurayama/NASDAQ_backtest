# 評価手法アップグレード計画 — 「1本の5年OOSで決めない」評価系への移行

作成日: 2026-06-11
最終更新日: 2026-06-11

> **目的**: 現行の「IS約47年 / OOS約5年の固定1分割 ＋ min(IS,OOS) CAGR」評価を、**ローリング窓（WFA拡張）＋レジーム層別＋bootstrap下限**を主軸とする評価系へ移行し、過学習を抑えつつ汎用性を保ち、"どの局面でも効く"高パフォーマンス戦略を拾えるようにする。
>
> **対象**: v7 で残った主候補 ── **V7-TQQQ（基準）/ P09_TQQQ / LU1 / P7（投信中庸）**、別枠 **LU2**。CFD戦略（E4 等）はアーカイブだが本評価系は環境非依存で適用可能。
>
> **正典との関係**: 本計画は [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) §3 を**置換せず拡張**する提案。確定後に §3.14（新設）として取り込む想定。

---

## 1. 現行評価の問題点（なぜ変えるか）

| # | 問題 | 具体 | 帰結 |
|---|---|---|---|
| 1 | **OOSが短く単一レジーム** | OOS=2021-05〜2026（約5年, n≈1,226営業日）は実質「テック強気＋2022利上げ」の1局面のみ | OOSの良し悪しが「たまたま落ちた局面」に支配される。**P09_TQQQ の対baseline bootstrap が非有意(P=0.80)になった直接原因**。汎用性の証拠として弱い |
| 2 | **固定1分割の選択バイアス** | 同一OOSを反復探索で何度も参照 | 準 look-ahead。OOSが実質IS化し、OOS性能が楽観に歪む |
| 3 | **min(IS,OOS)は保守的だが高分散** | 47年IS vs 5年OOS の単純比較 | OOS側のノイズが採否を決める。点推定1本で「悪い方」を採るだけでは、汎化と偶然の区別がつかない |
| 4 | **レジーム被覆の不明示** | 暦年・IS/OOS区切りのみ | 「2015型(株債金同時安)」「2022型(トリプル安)」「2000/2008型」での頑健性が表に出ない。P09の最悪暦年2015 −18.7% は現行表で埋もれる |
| 5 | **WFAが補助扱い** | WFE/CI95_lo は既に算出済だが採否はmin(IS,OOS)主導 | 既に持っている**窓を跨ぐ汎化情報を主指標に使えていない** |

> **結論**: 現行系は「保守的な点推定1本」。これを「**分布で見る（窓×レジーム×bootstrap）**」評価へ。

---

## 2. 提案する評価フレーム（推奨順）

### 2.1 【主指標へ格上げ】拡張ウォークフォワード（rolling / expanding WFA）
- 既存 `src/g1_wfa.py`（正典窓49窓, WINDOW=252）の **out-of-sample CI95下限** を headline 採否指標に格上げ。min(IS,OOS) は補助に降格。
- 追加: **expanding-origin**（学習窓を伸ばしながら次1年を検証）と **rolling-origin**（固定長窓をスライド）の両方を出し、窓を跨ぐ汎化を直接測る。
- 採否ゲート（β/α は現行 §3.9/§3.10 を踏襲）: WFE∈[0.5,2.0] かつ **CI95_lo>0 かつ t_p<0.05**。

### 2.2 【新設】Purged + Embargo Combinatorial CV（CPCV）
- 全期間を N=8〜10 ブロックに分割 → C(N,k) の組合せで複数のテスト折を構成。各折が**異なるレジーム**を含むようにする。
- **Purge**: テスト窓の前後で特徴量がリークする期間（mom252 等の最長ルックバック=252日）を学習側から除去。
- **Embargo**: テスト窓直後 ~21日を学習から除外（自己相関リーク防止）。
- 出力: 折ごとの標準10指標分布 → **中央値・p10・最悪折** を採否材料に。「1局面依存」を構造的に排除し過学習を検出。

### 2.3 【新設】レジーム層別評価
- 暦年でなく**レジームラベル**で性能を分解:
  - **Bull / Bear / High-Vol**（NASDAQ 200日トレンド＋実現ボラ分位で機械ラベル）
  - **金利局面**（SOFR/長期金利の上昇 vs 低下）
  - **既知ストレス**: 2000, 2008, 2020, 2022, **2015(株債金同時安)** を固定イベント窓として明示
- 各レジームで CAGR / MaxDD / Sharpe を出し「**全レジームで崩れないか**」を可視化。これが「どの局面でも効く」の操作的定義。

### 2.4 【ゲート化】block / stationary bootstrap 有意性
- 既存の paired block bootstrap（`run_p09_tqqq_validate`/`p09_tqqq_ggate`）を**全候補の標準ゲート**に。
- 点推定でなく **対baseline 改善の CI95下限>0** を必須化（P09が落ちたのと同じ基準を全候補に公平適用）。
- 多指標（min CAGR / Sharpe / MaxDD）すべてで P(better)>0.90 を「強昇格」、一部のみ通過を「条件付き」。

### 2.5 補強（任意）
- **DSR / PBO**（Deflated Sharpe・Probability of Backtest Overfitting）: 探索回数を効かせた多重検定補正。`check_overfitting_dsr.py` を全候補へ拡張。
- **完全性クリティック**: 「未検証のレジーム・未実行のモダリティはないか」を最終チェック。

---

## 3. 標準10指標の更新提案（ユーザー依頼の核心）

> 方針: **既存「標準10指標」は10本のまま温存**（後方互換）。下記の追加列は別名 **「拡張指標セット（Extended Set）」** として定義し、"10"のカウントは崩さない。「**分布で見る**」ための列を足し、採否の主従を入れ替える。

### 3.1 列の追加・改訂

| # | 指標 | 現行 | 提案 | 理由 |
|---|---|---|---|---|
| A | **採否 headline** | min(IS,OOS) CAGR⓽ | → **WFA CI95_lo⓽** を headline、min(IS,OOS) は補助 | 窓跨ぎ汎化を主指標化 |
| B | **新規: CPCV_p10 CAGR⓽** | なし | CPCV 折分布の第10パーセンタイル | 「悪い折でもこの水準」= 真のロバスト下限 |
| C | **新規: CPCV_worst_fold⓽** | なし | 最悪折の CAGR / MaxDD | 1局面依存の検出 |
| D | **新規: Regime_min CAGR⓽** | なし | Bull/Bear/HighVol/金利↑↓ の最小レジームCAGR | 全レジーム下限 |
| E | **新規: Stress_MaxDD** | MaxDD(FULL)のみ | 2008/2020/2022/2015 各イベントのDD | 既知ストレス耐性 |
| F | **改訂: bootstrap_CI95_lo(対base)** | P09のみ算出 | **全候補必須**・対baseline改善のCI95下限 | 偶然と汎化の分離をゲート化 |
| G | 既存 Worst10Y★⓽ / P10_5Y▷⓽ | 維持 | 維持（§3.5/§3.6） | カレンダー年10Y・日次5Yp10 は有用 |

### 3.2 採否ルールの改訂案（§3.13 の発展）
- **強昇格(Active候補)**: WFA(α∩β) ∧ CPCV_p10>baseline ∧ Regime_min>0 ∧ bootstrap多指標 P>0.90。
- **条件付き(Shortlist)**: WFA(α∩β) PASS だが bootstrap非有意 or 単一レジーム劣後（**現P09_TQQQはここ**）。
- **棄却**: WFE>2.0(regime luck) or CPCV最悪折で大幅劣後 or Regime_min<0。
- min(IS,OOS) は「保守的参考値」として残すが**単独採否には使わない**。

---

## 4. 実装計画（既存harnessへの増設・ステップバイステップ）

> 既存 `src/audit/` 資産（`unified_metrics.compute_10metrics` / `unified_wfa.summarize_wfa` / `run_p09_tqqq_validate._run_wfa` / block bootstrap）を最大再利用。正典 `src/*.py` は改変しない（audit層に追加）。

### Phase 1 — レジームラベラ＋層別指標（基盤）
- [ ] `src/audit/regime_labeler_20260611.py`: NASDAQ 200日トレンド＋実現ボラ63日分位＋SOFR方向から `regime[t]∈{bull,bear,highvol,rate_up,rate_down}` を生成（DELAY遵守・causal）。
- [ ] `src/audit/regime_stratified_metrics.py`: NAV＋regime → レジーム別 CAGR/MaxDD/Sharpe＋`Regime_min`。
- [ ] 固定イベント窓（2000/2008/2020/2022/2015）の `Stress_MaxDD` 算出。
- [ ] 検証: V7-TQQQ で既知の2022挙動（−0.27%）を再現してラベラを健全性チェック。

### Phase 2 — CPCV スプリッタ
- [ ] `src/audit/cpcv_20260611.py`: N=8ブロック・k=2 テスト折・purge(252)・embargo(21) の組合せ生成器（決定的・seed固定）。
- [ ] 各折で `compute_10metrics` を回し `CPCV_p10` / `CPCV_worst_fold` を集計。
- [ ] 検証: 折数 C(8,2)=28、各折のIS/OOS日数とpurge除去日数をログ出力。

### Phase 3 — bootstrap ゲートの全候補展開
- [ ] `p09_tqqq_ggate` の paired block bootstrap を関数化し、4主候補＋LU2へ適用するドライバ `src/audit/full_gate_20260611.py`。
- [ ] 出力: 候補×{WFA, CPCV, Regime, bootstrap} の統合CSV `audit_results/full_eval_20260611.csv`。

### Phase 4 — 統合レポート＋採否
- [ ] `EVALUATION_UPGRADE_RESULTS_20260611.md`: 拡張標準指標表（§3.1 の追加列込み）＋採否判定。
- [ ] EVALUATION_STANDARD.md に **§3.14 拡張評価系** を追記（確定後）。
- [ ] 独立エージェントQCで critical チェック → CURRENT_BEST 反映。

### 工数・順序
- Phase 1→2→3→4 直列（各Phaseが次の入力）。Phase 1/2 は並列着手可。
- 実行委譲は CLAUDE.md §7 に従い定型実装を `model: sonnet` サブエージェントへ。

---

## 5. 期待される効果と留意点

- **効果**: ①OOS単一局面依存の排除（P09の非有意問題を構造的に解消・公平判定）、②全レジーム下限の可視化で「どの局面でも効く」を定量化、③探索バイアスをCPCV/DSRで抑制 → 過学習を抑えつつ高パフォーマンス候補を拾える。
- **留意**: (a) CPCV/レジーム層別は計算量増（折数×指標）。(b) レジームラベルは設計選択＝**ラベラ自体の頑健性**を sensitivity で確認要。(c) 過去判定（AH/AT/HL 棄却等）は当時ルールで確定済、遡及再判定はしない。(d) 本計画は ETF/投信候補が一次対象だが、E4等CFDアーカイブにも同系を適用すれば復帰判断材料になる。

---

## 6. 確定までにユーザー判断が要る点

1. **採否 headline を WFA CI95_lo に切替**えてよいか（min(IS,OOS) を補助降格）。
2. レジームラベルの定義（トレンド+ボラ+金利で機械ラベル / 手動イベント窓のどちら主体か）。
3. CPCV のブロック数N・テスト折k・embargo日数（既定 N=8,k=2,embargo21 を推奨）。
4. この評価系を **どの候補から** 先に回すか（推奨: 攻め候補 P09_TQQQ / LU1 を最優先＝採否が割れている）。

---

*管理者: 男座員也（Kazuya Oza）*
