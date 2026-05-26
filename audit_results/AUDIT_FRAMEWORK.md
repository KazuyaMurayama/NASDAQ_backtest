# AUDIT_FRAMEWORK.md — NASDAQ Backtest 監査フレームワーク憲法

- バージョン: **v1.0**
- 発行日: 2026-05-26
- 関連: `CLAUDE.md「戦略監査チェックリスト」` / `EVALUATION_STANDARD.md §4`
- 管理者: 男座員也（Kazuya Oza）

---

## §0 本書の位置づけと運用ルール

### §0.1 単一の真実

本書は「戦略を監査するときの手順・スクリプト・判定基準」を一意に固定する正典。
`EVALUATION_STANDARD.md` が「何を評価するか」を定めるのに対し、本書は「どのコードで検証するか」を定める。

### §0.2 「文章での言及」≠「実施済み」の鉄則

検証は必ず以下の**3点セット**で完結する。3点のうち1点でも欠けたら「未実施」：

1. `src/audit/check_*.py` スクリプトを **実行**
2. `audit_results/*.md` レポートを **生成**
3. `audit_results/*.yaml` 機械可読データを **生成**

「コードを読めば自明」「構造的にリークなし」「推定値 +Xpp」は実施扱いにしない。

### §0.3 「旧戦略値の流用」「オフセット推定」の禁止

新戦略バリアント（E4→F→G…）が追加された時点で、Phase 1〜3 の全スクリプトを当該バリアントで再実行する。
「旧戦略 CAGR + 2.37pp オフセット」のような近似推定値を品質レポートの確定値として掲載することを禁止する。

### §0.4 キャッシュのブラックボックス化禁止

`audit_results/_cache/*.pkl` が存在しても「検証済み」と見なさない。
月次で `force_rebuild=True` によるキャッシュ再生成を実施し、前回との差分を確認する。

---

## §1 監査の3フェーズ構造

### §1.1 Phase 1: ロジック正しさ検証（コード単体テスト相当）

**目的**: `build_nav_strategy()` / `build_a2_signal()` / E4 パイプラインが「意図通りに動作している」ことを数値で実証する。

**必須実施条件**: 新バリアント導入・シグナル計算式変更・`build_nav_strategy()` 拡張のいずれかが発生したとき。

**依存関係**: Phase 2/3 は Phase 1 PASS を前提とする。Phase 1 FAIL の場合は Phase 2/3 の結果は無効。

### §1.2 Phase 2: 実現性検証（実コスト・実ブローカー・証拠金）

**目的**: バックテストのコストモデルと現実の取引コストのギャップを定量化し、実際に運用可能かを判定する。

**必須実施条件**: 全バリアントで再実行必須（旧戦略値の流用禁止）。

### §1.3 Phase 3: 過学習検出（統計検定 + WFA + 感度分析）

**目的**: OOS パフォーマンスが偶然でないことを統計的に検証する。

**構成**: Bootstrap（信頼区間）/ Permutation（アルファ源特定）/ WFA（時系列安定性）/ パラメータ感度（過学習耐性）。

### §1.4 Phase 間の依存関係

```
Phase 1 (Logic) → PASS 必須 → Phase 2 (Feasibility)
                              → Phase 3 (Overfitting)
                                  Phase 3 すべてが揃ったら
                                  → QUALITY_REPORT 生成
```

---

## §2 監査スクリプト一覧と責任範囲

### §2.1 Phase 1 スクリプト群

| スクリプト | 実施内容 | 出力ファイル |
|---|---|---|
| `check_logic_delay_and_causal.py` | DELAY=2照合・vz/lt_sig因果性・k_dyn境界・lev_A=0境界 | `LOGIC_CHECK_YYYYMMDD.md` / `logic_check.yaml` |

**新バリアント追加時の対応**: 本スクリプトに新バリアントの Block を追加し、専用 `build_<NEW>_strategy_assets()` で再実行。

### §2.2 Phase 2 スクリプト群

| スクリプト | 実施内容 | 出力ファイル |
|---|---|---|
| `check_realworld_report_corpus.py` | コスト根拠台帳・product_costs.py 整合 | `CHECK_CORPUS_YYYYMMDD.md` / `reports_corpus.yaml` |
| `check_sim_broker_matrix_cagr.py` | 旧戦略・6ブローカーシナリオ | `BROKER_MATRIX_YYYYMMDD.md` / `broker_matrix.yaml` |
| `check_e4_broker_matrix.py` | E4戦略・6ブローカーシナリオ | `E4_BROKER_MATRIX_YYYYMMDD.md` / `e4_broker_matrix.yaml` |
| `check_sim_margin_dynamics.py` | 証拠金維持率・強制ロスカット耐久性 | `MARGIN_DYNAMICS_YYYYMMDD.md` |

**新バリアント追加時**: `check_<NEW>_broker_matrix.py` を作成（`check_e4_broker_matrix.py` をテンプレに `lev_mod_<NEW>` に差し替え）。

### §2.3 Phase 3 スクリプト群

| スクリプト | 実施内容 | 出力ファイル |
|---|---|---|
| `check_overfitting_dsr.py` | DSR/PSR（旧戦略・参考値） | `DSR_YYYYMMDD.md` / `dsr_results.yaml` |
| `check_overfitting_block_bootstrap.py` | Bootstrap CI95（旧戦略） | `BOOTSTRAP_YYYYMMDD.md` / `bootstrap_results.yaml` |
| `check_overfitting_permutation.py` | Permutation a/b/c（旧戦略） | `PERMUTATION_YYYYMMDD.md` / `permutation_results.yaml` |
| `check_overfitting_summary.py` | Phase 3 統合サマリ（旧戦略） | `OVERFITTING_SUMMARY_YYYYMMDD.md` |
| `check_overfitting_e4_bootstrap.py` | Bootstrap CI95（E4戦略） | `E4_BOOTSTRAP_YYYYMMDD.md` / `e4_bootstrap_results.yaml` |
| `check_overfitting_e4_permutation.py` | Permutation a/c/d（E4戦略、d=同時置換） | `E4_PERMUTATION_YYYYMMDD.md` / `e4_permutation_results.yaml` |
| `check_overfitting_e4_summary.py` | E4 Phase 3 統合サマリ | `E4_OVERFITTING_SUMMARY_YYYYMMDD.md` |
| `check_e4_parameter_sensitivity.py` | ±20%感度分析（E4 6パラメータ） | `E4_PARAM_SENSITIVITY_YYYYMMDD.md` / `e4_param_sensitivity.yaml` |

**WFA**: `src/g3_wfa_e4.py`（Phase 3 補助・別管理）。

---

## §3 新戦略バリアント追加プロトコル

E4 → F → G… 系の新バリアントが追加されるたびに以下を機械的に実行する。

### §3.1 命名規約

- バリアント識別子: `E4`, `F8`, `G6` 等（アルファベット + 数字）
- キャッシュファイル: `audit_results/_cache/<lower_name>_nav_cache.pkl`
- 品質レポート: `audit_results/<UPPER_NAME>_QUALITY_REPORT_YYYYMMDD.md`

### §3.2 `_audit_strategy.py` への追加要件

```python
# 必須追加（4点セット）
<NEW>_PARAMS = dict(...)             # パラメータ定数
<NEW>_CACHE_FILE = os.path.join(...)  # キャッシュパス
<NEW>_SHARPE_OOS_EXPECTED = X.XXX    # サニティ用期待値

def build_<new>_strategy_assets(force_rebuild=False) -> dict:
    ...  # キャッシュ込みNAV構築
    # サニティチェック: Sharpe_OOS vs <NEW>_SHARPE_OOS_EXPECTED ±0.005

def build_<new>_strategy_nav_for_scenario(...) -> pd.Series:
    ...  # シナリオ別NAV（ブローカーマトリックス用）

def build_<new>_assets_with_override(param_name, value, base_assets) -> dict:
    ...  # パラメータ感度分析用
```

### §3.3 実行必須スクリプト（バリアントごと）

1. `check_logic_delay_and_causal.py` に新バリアット Block 追加 → 実行
2. `check_<new>_broker_matrix.py` 新規作成 → 実行
3. `check_overfitting_<new>_bootstrap.py` 新規作成 → 実行
4. `check_overfitting_<new>_permutation.py` 新規作成 → 実行（(d) 同時置換必須）
5. `check_overfitting_<new>_summary.py` 新規作成 → 実行
6. `check_<new>_parameter_sensitivity.py` 新規作成 → 実行
7. `src/g_wfa_<new>.py` 新規作成 → 実行（WFA）

### §3.4 品質レポート生成

全スクリプト完了後、`<NEW>_QUALITY_REPORT_YYYYMMDD.md` を生成。**末尾に「実施証跡表」（§5.4）を必須添付**。

### §3.5 CURRENT_BEST_STRATEGY.md 更新

```markdown
### Phase 1〜3 検証ステータス（YYYYMMDD）
- Phase 1 ロジック: ✅ PASS (`audit_results/LOGIC_CHECK_YYYYMMDD.md`)
- Phase 2 ブローカー: ✅ PASS 4/6 (`audit_results/<NEW>_BROKER_MATRIX_YYYYMMDD.md`)
- Phase 3 Bootstrap: ✅ PASS CI95_lo=+X.XXX (`audit_results/<NEW>_BOOTSTRAP_YYYYMMDD.md`)
- Phase 3 Permutation(d): ✅ PASS p=X.XXX (`audit_results/<NEW>_PERMUTATION_YYYYMMDD.md`)
- Phase 3 WFA: ✅ PASS CI95_lo=+XX.X%, WFE=X.XXX (`src/g_wfa_<new>.py`)
- Phase 3 Sensitivity: MIXED X/6 ROBUST (`audit_results/<NEW>_PARAM_SENSITIVITY_YYYYMMDD.md`)
```

---

## §4 監査レポート様式（品質レポートの必須構成）

### §4.1 レポート冒頭ヘッダ（必須）

```markdown
# <戦略名> 品質チェックレポート

- 作成日: YYYY-MM-DD
- EVALUATION_STANDARD: v1.1
- コスト Scenario: D (src/product_costs.py YYYY-MM-DD 基準)
- 期間: IS 1974-01-02〜2021-05-07 / OOS 2021-05-08〜YYYY-MM-DD
- DELAY: 2営業日
- 実行コマンド: python src/audit/check_*.py (各スクリプト)
- 主要コード: src/audit/_audit_strategy.py, src/cfd_leverage_backtest.py
```

### §4.2 必須セクション構成（順序固定）

1. §0 エグゼクティブサマリー（旧/新 9 指標比較 + 3 行結論）
2. §1 戦略の分解（6 レイヤー構造）
3. §2 Q1: パフォーマンスリスクの定量化
4. §3 Q2: 過学習リスクの多角診断
5. §4 Q3: 対策
6. §5 Q4: 改善方針
7. §6 Q5: 汎化性能
8. §7 総合判定と運用推奨（12 観点 RAG ステータス）
9. §8 結論
10. §9 付録（実測データ・YAML パス）
11. **§10 実施証跡表（必須・末尾添付）**

---

## §5 判定基準（Phase 別 PASS 条件）

### §5.1 Phase 1 PASS 基準

| チェック | PASS 条件 |
|---|---|
| DELAY=2 照合 | 全検証日で `abs(手計算 - 公式) ≤ 1e-8` |
| vz/lt_sig 因果性 | 全検証時点で `abs(truncated計算 - full計算) ≤ 1e-8` |
| k_dyn 境界条件 | 7 ケース全 PASS |
| 演算整合性 | `abs(clip(lev_A+lt_bias,0,1) - lev_mod_e4).max() ≤ 1e-10` |
| サニティ | 期待 Sharpe_OOS との差 ≤ 0.005 |

### §5.2 Phase 2 PASS 基準

| チェック | PASS 条件 |
|---|---|
| ブローカーマトリックス | 6 シナリオ中 ≥ 4 が Sharpe≥0.7 ∧ CAGR_OOS≥20% ∧ Worst10Y★≥10% |
| 証拠金ダイナミクス | 5 シナリオ中 ≥ 4 が強制清算 0 日 |
| コスト根拠台帳 | 全レポートが product_costs.py と整合 |

### §5.3 Phase 3 PASS 基準

| チェック | PASS 条件 |
|---|---|
| Bootstrap | CI95_lo > 0 ∧ p < 0.05（L=63 で必須） |
| Permutation (d) 同時置換 | p < 0.05（主判定） |
| WFA | CI95_lo > 0 ∧ 0.5 ≤ WFE ≤ 2.0（α∧β） |
| パラメータ感度 | 主要パラメータ中 ≥ 4/6 が ROBUST（±20%変動幅 CAGR ≤ 2pp ∧ Sharpe ≤ 0.05） |

---

## §6 失敗パターン Anti-Pattern 集（教訓）

### §6.1「コードを読めば自明」で済ませた事例

**事例**: `STRATEGY_QUALITY_REPORT_2026-05-25.md §3.4 Type 4` で  
「✅ 構造的にリークなし」と判定したが、根拠は目視コードレビューのみ。  
**対策**: Phase 1 `check_logic_delay_and_causal.py` を追加（2026-05-26 実施）。

### §6.2「旧戦略値 + オフセット」で済ませた事例

**事例**: `STRATEGY_QUALITY_REPORT_2026-05-25.md §2.4` で  
E4 ブローカー別性能を「旧値 +2.37pp オフセット推定」として掲載。  
**対策**: `check_e4_broker_matrix.py` を追加（2026-05-26 実施）。

### §6.3 レポートの「✅」が「実施済み」に見える問題

**事例**: `§9.3 Layer 3 自己参照性検証` にコードスニペットを掲載 → 実行証跡なし。  
**対策**: `§10 実施証跡表` を必須化、出力ファイル実在確認を義務化（本書 §5 参照）。

### §6.4 キャッシュ存在 → 検証済みと誤認した事例

**事例**: `_cache/best_nav_cache.pkl` 存在で「NAV は検証済み」と暗黙視。  
**対策**: Phase 1 が通らない限り Phase 2/3 の結果を確定判定にしない。

### §6.5 過学習検定だけを「監査」と見なした事例

**事例**: E4 追加時に Phase 3 のみ再実行し Phase 1/2 を旧戦略値で流用。  
**対策**: 本書 §3.3「実行必須スクリプト 7 本」を機械的に全実行する。

---

## §7 監査スケジュール（運用ルーチン）

| タイミング | 必須実行 |
|---|---|
| 新バリアント追加時 | Phase 1〜3 全スクリプト（§3.3 の 7 本） |
| 月次 | Phase 2 ブローカーマトリックス再実行（コスト変動確認） |
| 四半期 | Phase 3 WFA / 感度分析の再実行 |
| 半期 | Phase 1 Logic Check の完全再実行 |
| キャッシュ再生成 | 月次（`force_rebuild=True`）|

---

## §8 改訂履歴

| バージョン | 日付 | 内容 |
|---|---|---|
| v1.0 | 2026-05-26 | 初版作成（漏れ根本分析 + 3フェーズ構造 + Anti-Pattern集） |
