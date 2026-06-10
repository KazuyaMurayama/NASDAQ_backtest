# 次フェーズ実行計画 — (a) realistic WFA再走 / (b) SBI建玉金利の約款確認 / (c) 正典表 修正提案

作成日: 2026-06-10
最終更新日: 2026-06-10

> 前提: DELAY=2修正は round-trip テスト（日次NAV差=0）で正しいと確定済み。realistic確定値は full L× 基準。本計画は報告のみ・正典は(c)で「提案」まで（変更はユーザー承認後）。

---

## (a) realistic版 WFA(CI95_lo / WFE) の再走 — 修正NAV反映

**背景**: realistic の NAV が DELAY=2 + spread修正で変わったため、初版WFA(CI95/WFE)は無効。修正NAVで全戦略を再算出する。

### タスク
1. `src/audit/strategy_runners.py` の各 `run_*('realistic')` が返す**修正後NAV**から、非重複252日窓の per_window DataFrame（列: CAGR小数, Sharpe, short_flag(n_days<201), start_date）を生成。
2. `src/audit/unified_wfa.py::summarize_wfa` で **CI95_lo / WFE / t_p / 窓数** を算出。対象=7戦略×realistic(full)、＋CFD3戦略のborrowed感度。
3. `run_audit.py` に `--wfa` フラグ（realistic NAVのWFA出力）を追加、結果を `audit_results/audit_<s>_realistic_wfa.csv` に保存。
4. EVALUATION_STANDARD §3.13 の WFE>1.5 regime luck 判定を適用（特に vz065 は OOS偏重→WFE高めの可能性）。
5. レポート §2 の「WFA要再走」注記を実値で置換し push。

### 合格条件 / 注意
- 全戦略 52窓で揃うこと（R8整合）。
- realistic は IS期1980年代高金利でコスト重 → WFE<1.0（OOS<IS）になり得る。これは過学習でなく「コスト負け」の兆候として解釈し注記。

### モデル: 実装=Sonnetサブエージェント / 判定=Fable。

---

## (b) SBI CFD 建玉金利が証拠金分を含むか（full L× vs borrowed (L-1)×）の約款確認

**背景**: realistic CFD財務を full L×（建玉全額に金利）と borrowed (L-1)×（借入分のみ）で計算すると CAGR_OOS が約+3.6pp 動く（E4: +21.83% vs +25.43%）。どちらを「公式 realistic」とするかは**SBI CFDの建玉金利（オーバーナイト金利/price adjustment）の課金ベース**で決まる。

### タスク
1. **一次情報の取得**（Web）: SBI証券 くりっく株365 / 店頭CFD の「建玉金利」「オーバーナイト金利」「price adjustment」「金利相当額」の公式説明・約款・取引ルールを取得。`article-extractor` / `WebSearch` / `WebFetch` を使用。確認すべき点:
   - 金利が課されるのは **建玉の想定元本(Notional)全額** か、**証拠金を除いた借入相当分** か。
   - レバレッジ(必要証拠金率)と金利計算の関係。
   - スプレッド/金利水準の実値（(SOFR+3.0%)相当が妥当か）。
2. **判定**: full L× / borrowed (L-1)× のどちらが実態か確定。CFDは一般に建玉全額がエクスポージャーで金利は想定元本ベース → **full L× が有力**だが、一次情報で裏取りする。
3. 結果を `audit_results/SBI_CFD_FINANCING_BASIS_20260610.md` にまとめ（出典URL明記）、realistic の「公式採用ベース」を確定 → 必要なら (a) のWFA・レポート §2 を採用ベースに統一。

### 注意
- 一次情報が取れない/曖昧な場合は「full L× を保守的既定とし、borrowed を下限感度として併記」と明記（断定しない）。
- 投資助言ではなくコスト前提の確定が目的。

### モデル: Web調査=research/extractor系 + Fable判定。

---

## (c) 正典表（CURRENT_BEST_STRATEGY.md v4.5表）の修正提案 — 提案まで（変更はユーザー承認後）

**背景**: 検証で判明した誤り・基準混在を正典に反映する提案。**正典は勝手に変更しない**（報告ルール）。差分提案を別ファイルで提示し、承認後に適用。

### タスク
1. `audit_results/CANONICAL_TABLE_REVISION_PROPOSAL_20260610.md` を作成し、以下の修正案を「現行→提案」形式で提示:
   - **R4**: DH-W1 Trades/yr `68.7` → `17.6`（68.7はNAV符号反転の疑似指標と注記）。
   - **R2/R10**: CAGR列を「⓽税後」と明示し、ⓒ税前指標(Sharpe/MaxDD等)と基準分離。各行に税前提ラベル。
   - **R7**: CFD行(E4/vz065)を realistic(SBI CFD、(b)で確定したfull/borrowed)基準で再掲。E4 realistic +21.83%(full)/+25.43%(borrowed)、リポジトリ+22.4%修正の妥当性を確認済みと注記。
   - **R9**: vz065_l7 の CI95/WFE「N/A」を (a) の実値で補填。
   - **R3**: DH-W1 +13.66%(旧split)併記の整理。
   - **二面性**: ランキングは税引前CFD優位・risk調整/after-tax(NISA)でETF優位、を併記。
2. 反映先候補（CURRENT_BEST_STRATEGY.md / EVALUATION_STANDARD.md §1.1 の0.8273出典明記 / tasks.md）をリスト化。
3. ユーザーに承認可否を確認 → 承認後に正典へ適用し push（CLAUDE.md §9 報告）。

### 注意
- 提案ファイルはGitHubにpushするが、**正典ファイル自体は承認まで変更しない**。
- 数値は本検証の確定値（DELAY=2, full L×）を一次根拠とする。

### モデル: 提案ドラフト=Sonnet / 最終判断=Fable + ユーザー承認。

---

## 実行順序の推奨
1. **(b) を先に**（full/borrowed の公式ベース確定）→ (a) のWFA・(c) の数値を確定ベースで一貫させる。
2. 次に **(a)**（WFA再走で全指標を確定）。
3. 最後に **(c)**（(a)(b)確定値で正典修正提案）→ ユーザー承認 → 適用。

> (b)が曖昧でも (a) は full L× 既定で先行可能。最終 (c) で採用ベースを明示。

---

*管理者: 男座員也（Kazuya Oza）*
