# CURRENT BEST STRATEGY — 単一の真実 (Single Source of Truth)

> **このファイルは「現行のベスト戦略」を一意に特定するための正典です。**
> **「ベスト戦略は？」と問われた時、Claude / 人間ともにまずこのファイルだけを見れば良いように設計されています。**

作成日: 2026-05-11
最終更新日: 2026-06-07 (v4.9: S3 overlay V7 pure_boost を Companion Variant として追加、全 Shortlisted エントリに税後 CAGR ×0.8273 規約適用) / 2026-06-08 (標準10指標・IS/OOS min 表記・ⓒ/⓽コスト前提明確化 全セクション適用) / **2026-06-07 末 (v4.9.1: 税後 CAGR を v2 に修正。v1 は 2026 YTD partial year を完全1年扱いするバグ→OOS aftertax を 1.9〜3.8pp 過小評価していた。`compute_aftertax_cagr_v2_20260607.py` で `years=len()` → `actual_days/365.25` に修正)**

---

## 🆕 v4.5 (2026-06-05) ─ 環境別 Active 候補 + 保守的採用基準導入

### 保守的採用基準 min(IS, OOS) CAGR (Active 昇格判断の新ルール)
v4.5 (2026-06-05) で **min(IS, OOS) CAGR + Worst10Y★ + P10_5Y▷ の 3 軸保守的尺度** を導入。OOS 単独評価は採用判断には使わず、3 軸すべてで baseline を上回ることが Active 候補昇格の必須条件。詳細: [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md)。

### v4.5 推奨 Active 候補（環境別）— 標準10指標

> **コスト前提**: CFD 候補 = D+tax（Scenario D + moderate 0.05% spread + §3-A 税後⓽）/ ETF 候補 = D+tax（0.10% spread）
> **全値⓽税後**（手取り）。§1 Active E4（ⓒ税引前）との**直接比較は不可**（税調整基準が異なる）。
> CAGRは IS/OOS 両記載。**min(IS,OOS)** = 保守的採用基準値（太字）。

| 環境 | 戦略 | **CAGR⓽ IS / OOS（min）** | IS-OOS gap | Sharpeⓒ | MaxDDⓒ | Worst10Y★ⓒ | P10ⓒ 5Y | Tradeⓞ/yr | WFEⓞ | CI95ⓡ_lo | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **CFD (主軸, v4.7)** | **vz=0.65+l5+F10ε** | IS +20.16% / OOS +18.93%（**min +18.93%**） | +1.23pp | 0.841 | −56.72% | ~+12.67% | ~+8.75% | 86 | ✅ 1.389 | N/A | 🟢 CFD Active 候補（防御最優先） |
| ↳ 副候補 (攻め型) | vz=0.65+l7+F10ε | IS +20.23% / OOS +21.49%（**min +20.23%**） | −1.26pp | 0.829 | −65.95% | +9.96% | +4.05% | ~105 | N/A | N/A | 防御弱（MaxDD/Worst10Y/P10 l5 に劣後） |
| **ETF only (NISA等)** | **DH-W1** (Asymm Hysteresis) | IS **+15.79%** / OOS **+14.54%**（**min +14.54%**）†⓽ **v2**<br>_ⓒ canonical: IS+18.10%/OOS+18.91%_ | +1.25pp(⓽ v2) | 0.845 | −34.57% | +10.37% | +4.82% | 68.7 | ✅ 1.023 | +13.61% | 🟡 ETF 環境 Active 候補 |

> ⚠ **コスト・税の比較注意 (税後 ⓽ は 2026-06-07 末 v2 修正済)**:
> - **E4 Active**: ⓒ +33.53% / ⓽公式(aftertax_cagr **v2**) **+24.42%**（calendar OOS, 5.232年正規化, moderate spread + §3-A）。v1 +20.99% は 2026 YTD partial year バグで +3.43pp 過小評価していた
> - **vz=0.65+l5+F10ε**: +18.93% は integrated report §0' ⓽（別経路、別期間）→ E4 ⓽ と直接比較可
> - **DH-W1**: ⓽ **+14.54%** OOS (v2, 課税口座)（aftertax_cagr_v2 calendar 2020-2026, 5.232年正規化）→ NISA 内非課税なら pretax +17.43% そのまま。上の ⓒ +18.91% (canonical daily split) と混同しないこと
> - v1 後で「2026 YTD を完全1年扱い」のバグを修正 (`years = len()` → `actual_days/365.25`)。詳細は SIGNAL_EXPANSION_FINAL_DECISION §3.6 (v2 corrected)

両戦略とも v4.5 STRATEGY_REGISTRY §2 Shortlisted に登録済 (`vz065_l7_F10eps015`、`DH_W1_AsymmHyst`)。**§1 Active への正式昇格 (本ファイル下記「現行ベスト戦略」更新)** は実運用変更を伴うため、ユーザー判断を要する。

### 棄却された v4.x 改善案 (min ルール下で REF を下回る)
- **vz=0.65+l7+F10ε-AH/AT/HL** (v4.4 採用→v4.5 棄却) — OOS 単独では魅力的だが min(IS, OOS) で REF 劣後、WFE>1.5 で regime luck 疑い (STRATEGY_REGISTRY §3 Rejected)
- **DH-T4** (v3 で push→v4 で破棄) — ETF レバ操作違反 (lev_mod 連続 scaling)
- **DH-Z2** (v4 採用→v4.3 で W1 に置換) — Trades 152→18 で W1 が優位

### v4.5 整理対象ファイル
- [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) v4.5 (§7-2 min ルール明文化、§0' 5 戦略 + min 表記、§6-4 AH 棄却根拠)
- [STRATEGY_DH_REFINEMENT_20260603.md](STRATEGY_DH_REFINEMENT_20260603.md) v1.1 (DH-W1 詳細検証)
- [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) (§2 に v4.5 推奨 2 件、§3 に AH/AT/HL 棄却 3 件)

---

## ⚠ 命名規則 (重要・全 Claude 必読)
**vz=0.65+l7+F10ε を「NEW」「NEW CANDIDATE」と呼ぶことは廃止** (2026-06-03 v4.4 以降)。
- ✅ 正: `vz=0.65+l7+F10ε` / `S2_VZGated+LT2_N750+E4(vz=0.65)+F10ε=0.015` / Registry ID `vz065_l7_F10eps015`
- ❌ 廃: `NEW`, `NEW 🟢`, `NEW CANDIDATE`

---

## 現行ベスト戦略 (§1 Active、2026-05-24 確定、v4.5 では未更新)

**戦略名: `S2_VZGated + LT2-N750 + E4 Regime k_lt`（Vol-Zone ゲート CFD + 長期逆張り + ボラレジーム動的 k_lt）**

> **⚠ v4.5 注**: 本 §1 Active は 2026-05-24 確定の E4 RegimeKLT (vz_thr=0.70, F10 なし) で変更なし。**v4.5 で vz=0.65+l7+F10ε を CFD 環境 Active 候補と明確化** したが、§1 正式 Active 昇格は本ファイル下部の「v4.5 環境別 Active 候補」セクションを別途参照、実運用変更を伴うためユーザー承認後に更新。

### 主要指標（標準10指標 / Scenario D ⓒ税引前 / 1974-01-02 〜 2026-03-26）

> **記号**: ⓒ = コスト後・税引前（Scenario D）／⓽ = 税後手取り（§3-A 逐年複利適用）／ⓞ = 原値／ⓡ = WFA 再計算値
> **DH Dyn 2x3x [A] シグナル** (Approach A, 閾値 0.15) の上に CFD Vol-Zone ゲートを適用、LT2-N=750 を vz レジーム条件付き k_lt で重畳。

| 指標 | 値（ⓒ税引前） | 備考 |
|---|---|---|
| **CAGR ⓒ IS / OOS（min）** | IS **+31.72%** / OOS +33.53%（**min: IS +31.72%**） | ⓒ税引前 Scenario D。min = 保守的採用基準値 |
| CAGR ⓽_OOS（税後・§3-A概算, **v2 corrected**） | **+24.42%**（公式 v2）/ 単純推計≈+27.2% | 公式⓽ v2: `aftertax_cagr_v2_20260607.csv` → **+24.42%** (calendar OOS 5.232年正規化 + moderate spread 0.05% + §3-A)。v1 +20.99% は 2026 YTD partial year バグで +3.43pp 過小評価していた (`years=len()` → `actual_days/365.25` に修正)。単純§3-A推計: (33.53−0.66)×0.8273≈+27.2%（moderate spread未加算）。保守: (30.5−0.66)×0.8273≈+24.6% |
| IS-OOS gap ⓒ | **−1.81 pp** | OOS が IS を +1.81pp 上回る（優秀な汎化性） |
| Sharpe ⓒ_OOS | **+0.891** | OOS 期間 Sharpe 比、コスト後・税前 |
| MaxDD ⓒ (FULL) | **−60.01%** | 最大ドローダウン、コスト後・税前 |
| Worst10Y★ ⓒ (FULL) | **+18.67%** | カレンダー年ベース最悪10年ローリング CAGR |
| P10_5Y▷ ⓒ (FULL) | **+9.78%** | 5年 CAGR 分布 P10、コスト後・税前 |
| Trade ⓞ /yr | **約27回**（月2.3回） | 基底 DH Dyn シグナルと同じ。コスト優位性高 |
| WFE ⓞ | **+1.131** | G3 WFA β PASS（0.5 ≤ WFE ≤ 2.0） |
| CI95 ⓡ_lo | **+26.51%** | G3 WFA 50窓、α PASS（t_p=0.0000）、§3-A 税調整後 |

### ベスト戦略選定根拠 (2026-05-24 確定・tilt系棄却後)

| 評価軸 | E4 Regime k_lt **◆ BEST** | F7v3/F8 tilt 系 (Shortlisted) | 判断 |
|---|---|---|---|
| CAGR_OOS | +33.53% | +36.30〜36.83% | tilt系 +2.8〜3.3pp |
| Sharpe_OOS | +0.891 | +0.926〜0.934 | tilt系 +0.035〜0.043 |
| MaxDD | **−60.01%** | −61.96〜63.07% | **E4 優位** |
| Worst10Y★ | **+18.67%** | +18.27〜18.58% | **E4 優位** |
| IS-OOS gap | **−1.81pp** | −4.26〜4.28pp | **E4 優位** (OOS寄り乖離小さい) |
| **Trades/yr** | **約27回** | **約182回** | **E4 圧倒的優位 (1/7コスト)** |
| WFA_CI95_lo | +26.51% | +27.15〜27.92% | tilt系 +0.64〜1.41pp |

> **E4 採用・tilt系棄却理由**:
> tilt 系 (F7v3/F8) は OOS Sharpe を +0.035〜+0.043 改善するが、Trades/yr が 27→182 回（約7倍）に急増。
> 182 回/年の取引コスト（スプレッド・税率 20.315%・CFD スワップ）は CAGR を数〜10% 押し下げる可能性があり、
> OOS 期間 (2021-2026) の NASDAQ 強気相場に対するアウトサンプル偶然性が疑われる。
> 全期間（IS+OOS）での Sharpe 改善幅は軽微で、コストを加味すると実質同等以下と判断。
> IS-OOS gap が E4（−1.81pp）に対し tilt 系（−4.26〜4.28pp）と 2〜2.5 倍拡大している点も汎化性の観点でリスク。
> **G3 WFA (2026-05-24) PASS**: CI95_lo = +26.51%（>0 α PASS, t_p=0.0000）/ WFE = +1.131（0.5–2.0 β PASS）→ **正式 Active 確定**。

---

## Risk-Reduction Overlay Candidate (2026-06-05 Session 4 ADOPT)

> **位置付け**: 本オーバーレイは Session 4 (Phase D) で **DH-W1 (S3 = ETF only) ベース** に対し ADOPT 判定された。**§1 本番 Active (CFD: E4 Regime k_lt) の置換ではない**。
> Session 5 で同オーバーレイの **S2 (D5) / E4 (現行 Active) への転用可否** を audit-grade で検証中。
> 詳細: [data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)

### Adopted Overlay
- **Signal**: `nasdaq_mom63` (macro_features.csv: NASDAQ 63-day momentum, daily lag)
- **Base Strategy**: S3 (DH-W1, Asymm+Hyst, TQQQ/TMF/GLDM ETF only)
- **Injection Method**: M6 (threshold-proxy continuous tilt)
- **Direction**: defensive (high momentum → reduce leverage)
- **Multiplier mapping** (defensive variant): signal_q ∈ {0,1,2,3} → {1.1, 1.0, 0.9, 0.8}
- **Pipeline**: quantile_cut(levels=4) → apply_publication_lag('daily', +1 BD) → lev_raw_mod = lev_raw × mask_W1 × mult

### Audit-Grade Results (Phase D, Session 4)

> ⚠ **注記**: 本表の値は Phase D audit 時点（旧 split）の値。canonical split (2021-05-08) 再計算値は以下と異なる可能性あり。
> canonical overlay 値: Worst10Y★ **+10.75%**（本表 +10.38%）/ P10_5Y **+5.21%**（本表 +5.92%）/ CAGR_OOS **+18.06%**（本表 +18.10%）
> → STRATEGY_REGISTRY §2 / SIGNAL_EXPANSION_FINAL_DECISION_20260607.md §3.1 (v2 canonical) が一次根拠。

| metric | DH-W1 baseline（旧split） | + nasdaq_mom63 overlay（旧split） | diff | judgment |
|---|---:|---:|---:|---|
| CAGR_OOS | +18.96% | +18.10% | −0.86pp | minor degrade |
| Sharpe_OOS | +0.8445 | **+0.8914** | **+0.047** | ✓ improved |
| MaxDD_full | −34.57% | **−28.74%** | **+5.83pp** | ⭐ major improve |
| Worst10Y CAGR | +9.84% | +10.38% | +0.54pp | ✓ improved |
| P10_5Y CAGR | +5.94% | +5.92% | −0.01pp | ≈ neutral |
| IS-OOS gap | −0.88pp | −1.43pp | −0.55pp | ≈ wider abs(gap) |
| Trades/yr | 17.6 | 17.6 | 0.0 | = unchanged |
| WFE | 0.976 | **1.005** | +0.030 | ✓ ≥ 1.0 |
| CI95_lo (CAGR%) | +13.95% | +13.00% | −0.95pp | △ slight retreat |
| **+1 composite** | — | **6 imp / 3 deg** | — | STANDARD_PASS_FULL |

### Phase D Hard Gate (Multi-Metric Block Bootstrap 10,000 + WFA 50-window)

| Gate | Required | Actual | Result |
|---|---|---|---|
| WFE ≥ 1.0 | yes | 1.005 | **PASS** |
| CI95_lo CAGR > 0 | yes | +13.00% | **PASS** |
| Bootstrap P(Sharpe > base) > 0.90 | yes | 0.930 | **PASS** |
| Bootstrap P(MaxDD better) > 0.90 | yes | **0.988** | **PASS (best)** |
| Bootstrap P(CAGR > base) | (info) | 0.295 | (CAGR は trade-off) |

### Status
- **採用判定**: **ADOPT (S3 限定 / strategy-specific)** — DH-W1 (ETF only) ベースの "Risk-Reduction Overlay" として正式記録
- **§1 本番 Active 置換**: **NO** (現行 S2_VZGated + LT2-N750 + E4 Regime k_lt を維持)
- **採用ロジック**: CAGR を −0.86pp 譲って MaxDD を **+5.83pp 改善** + Sharpe +0.047 という、**「収益性を僅かに犠牲にして防御を強く取る」defensive-by-design tilt**。Phase D 4 gate 全 PASS、Bootstrap P(MaxDD better)=0.988 で MaxDD 改善の偶然性はほぼ排除。
- **適用範囲**: **S3 (DH-W1, ETF only) 限定**。S2 (D5) / E4 (Active) への転用は Session 5 で audit-grade で検証済、両方とも **NEEDS_FURTHER_WORK** (WFE<1.0 でハードゲート不通過、ただし P(MaxDD better)>0.94 で MaxDD 改善の方向性は保持)。
- 詳細 audit レポート: [phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)

### Session 5 転用 audit 結果 (2026-06-05)

| baseline | WFE | CI95_lo CAGR | P(CAGR>base) | P(Sharpe>base) | P(MaxDD better) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **S3 (DH-W1)** | **1.005** | **+13.00%** | 0.295 | **0.930** | **0.988** | **ADOPT** ✓ |
| S2 (D5) | 0.963 △ | +22.72% | 0.201 | 0.758 | 0.944 | NEEDS_FURTHER_WORK |
| E4 (現行 Active) | 0.958 △ | +24.41% | 0.355 | 0.858 | 0.964 | NEEDS_FURTHER_WORK |

**結論**: 本 overlay は **S3 (DH-W1, ETF only) 特異**。CFD ベース戦略 (S2, E4) では VZ ゲート + LT2-modeB / Regime k_lt が既に類似の防御機能を担っているため、追加 overlay の限界効用が小さい (MaxDD 改善 +1.2〜1.5pp に減衰)。MaxDD 防御の方向性は全 baseline で一貫 (P>0.94)。

詳細: [session5_transfer_report_20260605.md](data/signals/expansion/session5_transfer_report_20260605.md)

### Companion Variant: V7 pure_boost (2026-06-07 追加, v4.9)

CAGR 死守シナリオの代替オプション。同一 signal (nasdaq_mom63)、同一 method (M6 defensive) ながら mapping を boost-heavy に振った variant。**V0 (defensive, MaxDD改善優先) と V7 (pure_boost, CAGR死守) を並行 Shortlisted 登録、ユーザーリスク選好で選択**。

| 項目 | V0 defensive (既存) | V7 pure_boost (新規) |
|---|---|---|
| Mapping | {q0:1.10, q1:1.00, q2:0.90, q3:0.80} | {q0:1.20, q1:1.10, q2:1.00, q3:1.00} |
| 哲学 | リスク削減 (MaxDD改善優先) | CAGR死守 (MaxDD baseline据置で boost) |
| CAGR_IS (税前 ⓒ, canonical daily split) | +16.69% | **+18.61%** |
| CAGR_OOS (税前 ⓒ, canonical) | +18.06% | **+19.18%** |
| min(CAGR_IS, CAGR_OOS) | +16.69% | **+18.61%** (>18% target達成) |
| MaxDD | **−28.74%** (−5.83pp 改善) | −34.57% (baseline同等) |
| Sharpe_OOS | **+0.892** | +0.841 |
| Worst10Y★ | +10.75% | +11.02% |
| P10_5Y▷ | +5.21% | +5.22% |
| IS-OOS gap | −1.37pp | **−0.57pp** (overlay候補中最良) |
| WFA_CI95_lo (annual) | +12.65% | +14.06% |
| Phase D Bootstrap | ✅ 4/4 PASS (P_MaxDD=0.988) | ⚠ 未実施 (Pending) |

**選択ガイド**:
- **MaxDD を baseline (−34.57%) より縮めたい** → V0 (defensive)
- **CAGR 死守 (>18% target) を最優先、MaxDD は baseline 維持で良い** → V7 (pure_boost)
- 両 variant とも S3 (DH-W1, ETF only) 限定運用、CFD 系には転用不可。

**🔧 税後 CAGR v2 修正 (2026-06-07 末)**: [SIGNAL_EXPANSION_FINAL_DECISION_20260607.md](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) §3.6 (v2 corrected) 参照。
- v1 (`compute_aftertax_cagr_20260607.py`) は `years = len(annual_returns)` で 2026 YTD (Jan-Mar, ~60 営業日) を完全1年として扱うバグを含み、OOS 税後 CAGR を約 2pp 過小評価していた。v2 (`compute_aftertax_cagr_v2_20260607.py`) は **実経過日数 / 365.25** で正規化。
- **NISA 内 (非課税)**: pretax 値そのまま (V7 calendar-year CAGR_OOS = pretax = **+17.69%**)
- **課税口座 ×0.8273 (v2)**: V7 calendar-year CAGR_OOS aftertax = **+14.77%** (v1 +12.76% → +2.01pp 修正)
- IS aftertax: **+16.24%** (v1=v2、IS は 2020-12-31 で完結、partial year なし) / FULL aftertax: **+16.09%** (v1+0.26pp)
- **V7 min CAGR (修正後)**: NISA = min(IS+18.77%, OOS+17.69%) = **+17.69%** (calendar split, 18% target を僅か未達)。canonical daily split (§3.1) では **+18.61%** で 18% target 達成 (差は 4 ヶ月の OOS 起点差による)
- 課税口座 min CAGR (v2): min(IS+16.24%, OOS+14.77%) = **+14.77%**

詳細データ: [S3_OVERLAY_TUNING_REPORT_20260607.md](S3_OVERLAY_TUNING_REPORT_20260607.md) §6.2/**§11 (v2 aftertax)** / [s3_overlay_tuning_20260607.csv](data/signals/expansion/s3_overlay_tuning_20260607.csv) / [aftertax_cagr_v2_20260607.csv](data/signals/expansion/aftertax_cagr_v2_20260607.csv) (v1 は [aftertax_cagr_20260607.csv](data/signals/expansion/aftertax_cagr_20260607.csv) として保存・superseded)

---

## 投信環境 Active 候補 (2026-06-07 追加) ─ DH-W1 キャッシュ・スリーブ 1倍投信置換

> **位置付け**: ETF 環境 Active 候補 **DH-W1** は全営業日の **46.9%(6,171日)をキャッシュ 0% で待機**する。この待機資金を 1 倍投信(ゴールド/米国債20年超)で運用置換した派生系。1 倍投信は **SOFR/スワップなし・信託報酬<0.2%・5営業日ラグ**で実装可能（レバ脚は DELAY=2 / SOFR/スワップ込み、税 20.315%×0.8273）。§1 本番 Active (CFD: E4 Regime k_lt) の置換ではなく、**ETF/投信のみ環境(NISA 等)で DH-W1 を更に押し上げる候補**。

### OUT(キャッシュ)期 6,171日の各1倍資産の素の挙動
| 資産 | 年率リターン | 年率Vol | R/R | 含意 |
|---|---:|---:|---:|---|
| NASDAQ 1x | **−7.63%** ⚠ | 26.3% | −0.29 | キャッシュ期＝risk-off、NASDAQ は大損 → スリーブ不適 |
| Gold 1x | +4.07% | 21.5% | +0.19 | 堅実 |
| Bond 1x (米国債22yr) | **+6.51%** ✅ | **11.9%** | **+0.55** | 低ボラ＋プラスで最良 |

### 4戦略 標準指標（全コスト後・税20.315%後・WFA 50窓）

> ⚠ **DH-W1 baseline 表記注意**: 本表の DH-W1 値 (+13.66%) は **v4.5 §7-2 min(IS,OOS) CAGR = IS 値**（旧 OOS split 基準）。
> 上記 v4.5 推奨表の canonical OOS +18.91% とは **split 日付と計算基準が異なる**。各 Cash Sleeve 戦略の差分（+2.78pp 等）はこの +13.66% ベースの**相対比較値**として正しい。

| 戦略 | CAGR_OOS | Sharpe_OOS | MaxDD | Worst10Y★ | P10_5Y | IS-OOS gap | Trades/yr | WFE | CI95_lo | 性格 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DH-W1 (baseline) | +13.66%† | +0.844 | **−34.57%** | +8.40% | +5.11% | +1.62 | 17.8 | 0.997 | +13.95% | 現状(キャッシュ)† |
| 🟢 P2 GOLD100 | **+16.44%** | **+0.875** | −58.53% | +9.43% | +8.06% | **+0.97** | 18.8 | 1.229 | +16.04% | 攻め (OOS 最高) |
| 🟢 **P7 GOLD75/BOND25** ⭐ | +14.90% | +0.827 | −48.23% | +9.92% | +8.05% | +3.28 | 18.8 | 1.043 | +16.74% | **中庸推奨** |
| 🟢 P5 GOLD50/BOND50 | +13.28% | +0.758 | −35.97% | +10.08% | +8.09% | +5.50 | 18.8 | 0.875 | **+17.23%** | 守り (DD 最良) |

- **P2 GOLD100**: ベース比 +2.78pp の OOS 最高・最汎化(gap +0.97)。代償は MaxDD −58.5%(Gold 単独の宿命)。攻め優先向け。
- **P7 GOLD75/BOND25 ⭐(新規・推奨)**: P2 と P5 の中間最適。OOS +14.90% を確保しつつ Bond 混で MaxDD を −48.23% に緩和。総合バランス最良。
- **P5 GOLD50/BOND50**: MaxDD をベース同等(−36.0%)維持・CI95_lo/P10/Worst10Y 最高。守り最優先向け。gap +5.50pp(汎化やや弱)。

> **昇格保留の理由**: WFA 50窓は CI95_lo>0 (α) ∩ 0.5≤WFE≤2.0 (β) を全 PASS だが、**t_p(permutation 検定)・block bootstrap は未実施**。正式 Active 昇格にはこれらの統計検証が必要（[tasks.md](tasks.md) Pending）。全コスト表・執行ラグ検証は [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md) §4-§6 参照。STRATEGY_REGISTRY §2 に 3 件登録済。

---

## Shortlisted（次善候補 / WFA 完了）— 標準10指標

> **コスト前提**: F8/F7v3 = Scenario D ⓒ（税引前 raw）+ 逐年 ×0.8273 で⓽手取り算出。LT2-N750 = Scenario D ⓒ のみ（⓽未計算）。
> CAGR は IS/OOS 両記載。**min(IS,OOS)** = 保守的採用基準。IS 値が未計算の場合 N/A。

| 戦略 | **CAGR⓽ IS / OOS（min）** | IS-OOS gap | Sharpeⓒ_OOS | MaxDDⓒ(★raw) | Worst10Y★ⓒ | P10ⓒ 5Y | Tradeⓞ/yr | WFEⓞ | CI95ⓡ_lo | 採用留保理由 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F8 R5_CALM_BOOST | N/A / **+30.5%⓽**（min N/A） | N/A | +0.934 | −63.07%(★raw) / ≈−38.1%⓽ | N/A | N/A | 182 | ✅ 1.208 | +27.92% | Trades/yr 過多（E4比7倍）、OOS 偶然性疑い |
| F7v3+E4 A:tilt=2.0 | N/A / **+29.8%⓽**（min N/A） | N/A | +0.926 | −61.96%(★raw) / ≈−36.8%⓽ | N/A | N/A | 183 | ✅ 1.203 | +27.15% | 同上 |
| LT2-N750 固定k=0.5 | ~+31.34%ⓒ / +31.16%ⓒ（**min +31.16%ⓒ**） | ~+0.18pp | +0.858 | −59.45%ⓒ | ~+18.10%ⓒ | N/A | 27 | ✅ 1.145 | +25.7% | E4 に Worst10Y★/IS-OOS gap/CAGR で劣後。WFA PASS 済み fallback 候補 |

> **★ raw = Scenario D コスト後・税引前ⓒ**（旧スプレッド想定）。⓽ 手取り = 年次リターン逐年 ×0.8273（`f8r5_yearly_returns.csv` / `f7v3_yearly_returns.csv` から計算済み）。
> ⚠ F8/F7v3 の IS CAGR⓽ は未計算。min(IS,OOS) の確定には IS 期間の逐年税後計算が必要。

---

## 構成

- **シグナル基盤**: DH Dyn A2 Optimized (DD × AsymEWMA × TrendTV × SlopeMult × MomDecel × VIX_MR), threshold=0.15
- **LT2 オーバーレイ (modeB, regime-conditional k_lt)**:
  - `lt_sig = compute_lt2(close, N=750)` — 750日（≈3年）モメンタム z スコア
  - `k_lt_t = 0.8 if vz_t > +0.7; 0.1 if vz_t < −0.7; 0.5 otherwise` — vz レジーム依存感度
  - `lt_bias_t = (−k_lt_t × lt_sig_t × 0.5).clip(−0.5, +0.5)` — 動的バイアス
  - `lev_mod = clip(lev_A + lt_bias, 0, 1)` — DH Dyn lev に動的バイアス
- **CFD レバレッジ**: `compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, l_max=7.0, step=0.5)`
- **CFD スプレッド**: 低スプレッド想定 (`CFD_SPREAD_LOW`)
- **配分**: `wn·lev_mod·L_s2·r_nas_cfd + wg·r_g2 + wb·r_b3`
  - `wn`, `wg`, `wb`: DH Dyn [A] Approach A と同一 (`simulate_rebalance_A`)
  - `lev_mod`: LT2-modeB 適用後の DH Dyn レバレッジ
  - `L_s2`: VZ ゲート CFD 動的レバレッジ (1x〜7x)
- **Gold 2x**: TER 0.50% (sim proxy) + 1×SOFR
- **Bond 3x (TMF)**: TER 0.91% + 2×SOFR
- **DELAY**: 2営業日
- **実装**: `src/e4_regime_klt.py`（E4 主実装）, `src/b1_s2_lt2.py`, `src/cfd_leverage_backtest.py`, `src/dynamic_leverage_strategies.py`, `src/long_cycle_signal.py`

---

## コスト注意事項

| 補正項目 | 影響 |
|---|---|
| SOFR financing drag (NAS CFD + Gold 1xSOFR) | CFD 軸にも適用 |
| Gold TER ギャップ (proxy 0.50% → UGL 0.95%) | **−10.5 bps/yr** (§16) |
| TMF TER ギャップ (0.91% → 1.06%) | −3.5 bps/yr (§16) |
| スワップスプレッド推定差 (+20.5 bps) | −34 bps/yr 相当 (§16) |
| 合計推定コスト過少計上 | **約 −66 bps/yr** → 現実 CAGR_OOSⓒ ≈ **+30.5%（税引前・補正後）** |
| 売買税ドラッグ (年27回、税率20.315%) | §3-A 逐年適用 → CAGR⓽_OOS ≈ (30.5−0.66)×0.8273 ≈ **+24.6%（最終手取り保守推計）** / 単純推計 **+27.2%** |
| NISA | CFD は原則 NISA 不適用 |

---

## 一次根拠ファイル（GitHub 上の正典）

| ファイル | 役割 | 日付 |
|---|---|---|
| [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md) | E4 sweep レポート（採用根拠・64 config PASS 12） | 2026-05-24 |
| [e4_regime_klt_results.csv](e4_regime_klt_results.csv) | E4 sweep 65行 raw 結果 | 2026-05-24 |
| [src/e4_regime_klt.py](src/e4_regime_klt.py) | E4 Regime k_lt 実装 | 2026-05-24 |
| [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) | G3 WFA レポート（CI95_lo=+26.51% / WFE=+1.131 / PASS） | 2026-05-24 |
| [g3_wfa_e4_summary.csv](g3_wfa_e4_summary.csv) | G3 WFA サマリ | 2026-05-24 |
| [src/g3_wfa_e4.py](src/g3_wfa_e4.py) | G3 WFA 実行スクリプト | 2026-05-24 |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | B1 検証レポート | 2026-05-21 |
| [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py) | DH Dyn シグナル基盤 (Scenario D) | 2026-05-12 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 | 2026-05-12 |

---

## 「ベスト戦略は？」と問われたときの参照プロトコル (Claude 必読)

### 手順

1. **本ファイル (`CURRENT_BEST_STRATEGY.md`) の冒頭ブロックを引用する** — 最優先・最新の真実
2. `tasks.md` の最新 ✅ Completed エントリと突き合わせて整合確認
3. 矛盾があれば必ずユーザーに報告 → 本ファイル更新の提案

### 絶対にやってはいけないこと

- ❌ CSV を Sharpe 降順で並べて「トップ」を答える
- ❌ F7v3/F8 系 (高 Trades/yr) を「Sharpe が高いから」とベストとして提示する
- ❌ MEMORY.md 内の固定記述を一次根拠にする

---

## ⛔ 廃止された旧推奨 (ブラックリスト)

| 旧推奨 | 廃止日 | 廃止理由 |
|---|---|---|
| `F8 R5_CALM_BOOST` Sharpe +0.934, Trades/yr 182 | 2026-05-24 | Trades/yr 182回 (E4比7倍) のコスト負担と OOS 偶然性疑いによりShortlisted降格 |
| `F7v3+E4 A:tilt=2.0` Sharpe +0.926, Trades/yr 183 | 2026-05-24 | 同上。tilt系は全期間でE4との改善幅が軽微、IS-OOS gap拡大 |
| `S2_VZGated + LT2-N750-k0.5-modeB (固定k)` Sharpe 0.858 | 2026-05-24 | E4 が CAGR/Sharpe/Worst10Y★/IS-OOS gap で優位 |
| `S2_VZGated` CAGR_OOS +27.57%, Sharpe_OOS 0.769 | 2026-05-21 | S2+LT2 が全指標で上回ることを確認 |
| `DH Dyn 2x3x [A]` CAGR 22.50%, Sharpe 0.993 | 2026-05-21 | S2_VZGated が上回ることを確認 |
| `Ens2(Asym+Slope)` CAGR 28.58%, Sharpe 1.031 | 2026-04-21 | `DH Dyn 2x3x [A]` に置換 |

---

## 命名規則 (今後の再発防止)

1. **`FINAL_` プレフィックスは使用禁止**
2. **`<TOPIC>_YYYY-MM-DD.md` 形式を使用**
3. **新レポートが旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**
4. **本ファイル (`CURRENT_BEST_STRATEGY.md`) を必ず同時更新**

---

## メタ情報

変更履歴は git log で追跡可能 (`git log --follow CURRENT_BEST_STRATEGY.md`)

### 変更履歴
- **2026-06-07 (DH-W1 Cash-Sleeve 4戦略)**: ETF 環境 Active 候補 DH-W1 の OUT(キャッシュ 46.9%)期を 1 倍投信で運用置換する 4 戦略を **「投信環境 Active 候補」セクション** として §Shortlisted 直前に新設。**P2 GOLD100**(攻め, OOS +16.44% 最高)/**P7 GOLD75/BOND25 ⭐**(中庸推奨, MaxDD −48.23%)/**P5 GOLD50/BOND50**(守り, MaxDD −35.97% 最良)。全 WFA 50窓 α∩β PASS。検証済み: OUT 期資産挙動、全商品コスト表(SOFR/TER/スワップ/売買/税 20.315%)、執行ラグ(レバ脚 DELAY=2 / 投信スリーブ 5BD)。t_p/bootstrap 未実施で正式昇格保留。§1 本番 Active(E4 Regime k_lt)は変更なし。STRATEGY_REGISTRY §2 に 3 件登録。一次根拠: [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md)。
- **2026-06-05 (v4.8: Session 4 + 5)**: `nasdaq_mom63 × S3 × M6 defensive` overlay を **Risk-Reduction Overlay Candidate** として §Shortlisted 直前に新設。Session 4 で S3 (DH-W1, ETF only) に対し Phase D 4 gate 全 PASS → ADOPT。Session 5 で S2 (D5) / E4 (現行 Active) への転用を audit-grade で検証 → 両方 NEEDS_FURTHER_WORK (WFE<1.0、ハードゲート不通過。P(MaxDD better)>0.94 で方向性は一貫)。結論: overlay は **S3 限定 (strategy-specific)**。§1 本番 Active (CFD: E4 Regime k_lt) は変更なし。
- **2026-06-05 (v4.7: CFD Active 候補を l7 → l5 に置換)**: ユーザー判断により vz=0.65+l5+F10ε を CFD 環境 Active 候補に確定。l7 (旧 REF) は副候補に降格。理由: l5 は min CAGR -1.30pp の trade-off と引換に **Worst10Y +12.67% (vs l7 +9.96%、+2.71pp)、P10_5Y +8.75% (vs l7 +4.05%、+4.70pp 大幅改善)、MaxDD -56.72% (vs l7 -65.95%、9.23pp 浅化)、Sharpe +0.841 (vs l7 +0.829)、Trades 86 (vs l7 105、18% 低コスト)** で防御指標が圧倒的優位。
- **2026-06-05 (v4.6: lmax sweep)**: vz=0.65+F10ε の lmax を l5/l5.5/l7 で比較、l5.5/l5 を Shortlisted 追加。
- **2026-06-05 (v4.5: 保守的採用基準 + 環境別 Active 候補導入)**: 本ファイル冒頭に **「v4.5 環境別 Active 候補」セクション** + **「命名規則」セクション** を新設。要点:
  - **min(IS, OOS) CAGR + Worst10Y + P10_5Y の 3 軸保守的尺度** を Active 昇格判断の必須条件として導入 (詳細: STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2)
  - **CFD 環境 Active 候補**: vz=0.65+l7+F10ε (min CAGR=+20.23%、5 戦略中 1 位)
  - **ETF 環境 Active 候補**: DH-W1 (DH Asymm+Hysteresis、ETF 制約下で DH 基線を +4.10pp 上回る唯一の改善)
  - 棄却: vz=0.65+l7+F10ε-AH/AT/HL (OOS のみ評価では魅力的だが min ルールで全敗、WFE>1.5 で regime luck 疑い)
  - 「vz=0.65+l7+F10ε」を「NEW/NEW CANDIDATE」と呼ぶ表記を廃止
  - §1 正式 Active は **未変更** (E4 RegimeKLT を維持)。CFD Active への昇格判断はユーザー承認後
- 2026-05-24 (tilt系棄却・E4 復帰): F7v3+E4 および F8-R5 を Trades/yr 過多（182〜183回/年、E4比7倍）・OOS偶然性疑い・IS-OOS gap拡大（−4.26〜4.28pp）を理由に Shortlisted 降格。E4 Regime k_lt (27回/年) を正式 Active に復帰。棄却判断: コスト加味の実質 CAGR は E4 と同等以下と判断。
- 2026-05-24 (G5 WFA + F8-R5 昇格→即降格): F8-R5 WFA PASS (CI95_lo=+27.92%) を確認するも上記理由で採用見送り。
- 2026-05-24 (G4 WFA + F7v3+E4 昇格→即降格): F7v3+E4 WFA PASS (CI95_lo=+27.15%) を確認するも上記理由で採用見送り。
- 2026-05-24 (G3 WFA): E4 Regime k_lt WFA PASS (CI95_lo=+26.51%, WFE=+1.131)。正式 Active 確定。
- 2026-05-24: E4 Regime k_lt 暫定昇格 (k_lo=0.1, k_hi=0.8, vz_thr=0.7)
- 2026-05-22: B6 N=1500 暫定昇格・即差し戻し
- 2026-05-21: B1 S2+LT2 採用
- 2026-05-12: Scenario D 補正適用
- 2026-05-11: 初版作成

---

*管理者: 男座員也（Kazuya Oza）*
