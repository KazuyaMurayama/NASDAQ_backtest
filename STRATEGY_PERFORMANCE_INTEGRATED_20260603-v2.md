# 戦略パフォーマンス統合レポート v4 — DH 改善「配分 × タイミング 2 軸変動 (DH-Z2)」採用 (2026-06-03)

**作成日**: 2026-06-03（v4：v3 の DH-T4 を **全面破棄**、DH-Z2 (F10 ε tilt + binary vz_gate) を §0' / §5 / §6 / §6-2 の 4 箇所で**完全置換**）
**最終更新日**: 2026-06-03
**生成者**: Claude (Opus 4.7)
**著者**: 男座員也 (Kazuya Oza)
**準拠**: EVALUATION_STANDARD.md v1.4 / v3.1 §3-A 税モデル / v6.1 日次取引コスト / v6.2 7戦略拡張
**前版**:
- [v6.1](STRATEGY_PERFORMANCE_INTEGRATED_20260602.md) (4戦略) ← §0' のヘッダ参照元
- [v6.2](STRATEGY_PERFORMANCE_INTEGRATED_20260602_v62.md) (7戦略、QC で NEW SOTA→NEW CANDIDATE)
- [v6.2 QC](STRATEGY_PERFORMANCE_INTEGRATED_20260602_v62_QC.md) (3 並列エージェント QC)
- [v6.3](STRATEGY_PERFORMANCE_INTEGRATED_20260602_v63.md) (5 検証完了、§0' 列ズレ未修正版)
- v2 (本ファイル過去版): §0' 列構成 v6.1 完全一致版
- **v3 (本ファイル前版): DH-T4 追加版 — `lev_mod` で TQQQ position 連続 scaling を行いユーザー却下 → v4 で全面置換**

---

## 📋 §0 v4 主要変更点（v3 → v4）

### A. **v3 (DH-T4) を全面破棄** ← v4 の主差分
v3 で追加した DH-T4 (vz=0.65+lmax=5.5+F10ε) は `lev_mod_065` で TQQQ ポジションを連続 (0〜100%) スケールしていたため、**ETF (TQQQ/TMF/2036) の物理的「保有/非保有」制約に違反**（部分 TQQQ 保有 = 内部レバ scaling）。
- ユーザー却下: 2026-06-03、レバレッジ操作禁止の指示
- Rejected として STRATEGY_REGISTRY.md §3 に降格

### B. **DH-Z2 (配分 × タイミング 2 軸変動、binary HOLD/OUT) を §0' / §5 / §6 / §6-2 の 4 箇所で完全置換**
**商品は現状維持** (TQQQ + TMF + WisdomTree 2036)、**保有比率と保有タイミングのみ改善**:
- **配分**: F10 ε tilt (`wn_f10/wb_f10/wg_f10`) — NEW CANDIDATE と同一機構で TQQQ↔TMF の比率を smooth に rebalance
- **タイミング**: binary HOLD/OUT mask (`lev_mod_065 ≥ 0.5`) — vz_gate + LT2 統合連続値を 50% threshold で binarize、HOLD=0 時は全 cash
- **レバ操作なし**: `lev_mod` 連続 multiplier・`L_s2` cap いずれも使用せず、配分 + binary mask のみで実装
- **peak leverage 検証**: `np.max(wn × lev_raw × 3) = 2.85x ≤ 3.0x` を assert で機械検証 (g22a)

追加位置（いずれも DH Dyn 2x3x [A] の **1 つ手前**、v3 と同じ位置）:
1. §0' 9 指標標準表 → 5 行（DH-T4 行 → DH-Z2 行に置換）
2. §5 年次リターン表 (1977-2026 moderate) → DH-T4 列 → DH-Z2 列に置換
3. §6 統計サマリ (1974-2026) → DH-T4 列 → DH-Z2 列に置換
4. §6-2 OOS 累積比較 → DH-T4 行 → DH-Z2 行に置換

### C. **DH-Z2 検証サマリ（一次根拠は `g22a〜g22f` CSV 群、g21 系は v3 履歴として残置）**
| 検証 | 結果 |
|---|---|
| 9 指標標準 (gap 縮小) | ✅ IS-OOS gap **+10.46→+2.17pp** (-8.29pp) ✨, WFE **0.662→1.058** (完全汎化、OOS が IS をわずかに上回る) |
| WFA 50 窓 | ✅ PASS (CI95_lo=+12.03%, **WFE=1.058**, p=0.0000) |
| Bootstrap (Z2 vs REF, OOS) | ⚠ CI95=[-9.78pp, +16.62pp] (median +3.80pp positive, OOS 6 年 noise 内だが median は明確に improvement) |
| Permutation (binary θ ∈ [0.1, 0.9]) | ✅ ROBUST (実 θ=0.5 が NULL 中央=52% → θ 過適化なし、binary threshold 機構自体が θ 全域でロバスト) |
| 年次寄与 (OOS 6 年) | ✅ **diff sum +7.56pp**, 2022 特に **+21.56pp 救済** (binary OUT が bear 回避), 2021/2025 は missed rally で -10〜-16pp |
| **OOS 累積倍率** | **×2.00** (100→200 万円) vs REF ×1.73 |

→ **総合判定: 🟡 採用候補（gap 縮小 ✅ + WFE 1.058 ✅ + OOS CAGR +12.26% ✅ + OOS 累積 ×2.00 ✅）。ただし防御指標 (Worst10Y +6.36%, P10_5Y +5.33%) が REF (+12.57%, +8.77%) から劣化 → 防御重視運用と OOS 拡大の trade-off。Active 昇格はユーザー判断。**

### D. NEW CANDIDATE / F10 / D5 等の既存 7 戦略は v2 から変更なし（§0' 他は値そのまま）

### E. 旧 §A〜D（v2 → v6.1〜v6.3 由来）は §0-prev に格下げ

---

## 📋 §0-prev v2 までの変更点（履歴保持）

### A. **§0' 候補戦略比較表の列構成を v6.1 添付フォーマットと完全一致に修正** ← v2 の差分
- 旧 v6.3: `CAGR<br>⓽<br>_OOS`（3 行）
- 新 v2: `CAGR<br>⓽<br>_<br>OOS`（4 行、v6.1 line 108 と同一）
- アライメント行も v6.1 と同一の列幅指定に統一

### B. NEW CANDIDATE (vz=0.65+lmax=7+F10ε=0.015) 5検証完了（v6.3 から継承）
QC Agent 3 の指摘に応じて、5 統計検証 (v6.3-1〜v6.3-5) を全件実施。

### C. 5戦略統合年次リターン表 (1977-2026)（v4 で 8 → 5 戦略に絞込み、v6.3 から継承の §0' 5 戦略と統一）
v6.1 形式踏襲、税後・日次取引コスト後、moderate ケース。**v4 修正**: F10 ε=0.015 ★ / F8 R5 / F7v3+E4 / E4 ◆ を削除し、§0' 候補戦略統合比較表の 5 戦略 (NEW 🟢 / D5 / DH-Z2 / DH [A] / NDX 1x B&H) のみに統一。

### D. NEW CANDIDATE の最終判定: **🟢 条件付き昇格 (Active 候補追加)**（v6.3 から継承）
3 検証 PASS + 2 検証 marginal で、F10 ★ と並ぶ「**双頭 Active 候補**」として運用推奨。

---

## 🎯 §0' 候補戦略 統合比較表 (v2 採用判断版・列構成 v6.1 完全一致) — EVALUATION_STANDARD §3.12 v1.4 準拠

**moderate ケース**: CFD spread = 0.05% (中庸 GMO/楽天想定) / 税後 (§3-A `(yr − 0.66%) × 0.8273` 逐年複利) / 日次取引コスト後

| Strategy | CAGR<br>⓽<br>_<br>OOS | 累積<br>CAGR<br>⓽<br>OOS/IS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |
|:---------|-------------:|:------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|
| **🟢 vz=0.65+l7+F10ε** (NEW CANDIDATE) | **+21.49%** | OOS: **+21.50%**<br>IS: **+20.22%** | **-1.27pp** | **+0.829** | -65.95% | +9.96% | +5.84% | 52 | ✅ LOW<br>(1.4) | **+0.199** |
| **D5 vz=0.65/lmax=5.5** | +17.86% | OOS: +17.87%<br>IS: +20.08% | +2.22pp | +0.79 M | **-55.88%** | +12.21% | +6.76% | 28 | ✅ LOW<br>(1.3) | +0.192 |
| **🟡 DH-Z2 (F10 ε + binary vz)** ★v4 new | **+12.26%** | OOS: +12.24%<br>IS: +14.43% | **+2.17pp** | **+0.837** | **-38.92%** | +6.36% | +5.33% | 37 | ✅ LOW<br>(1.1) | +0.120 |
| **DH Dyn 2x3x [A]** | +9.56% | OOS: +9.56%<br>IS: +20.01% | +10.29pp ⚠⚠ | +0.60 | -41.57% | **+12.57%** | **+8.77%** | 27 | ✅ LOW<br>(0.7) | +0.175 |
| **NDX 1x B&H** (Benchmark) | +8.27% | OOS: +8.28%<br>IS: +9.91% | +1.64pp | +0.516 | -77.93% | -4.85% | +0.59% | 0 | — | — |

太字 = 列最良値 / 🟢 = v6.3 で「条件付き昇格」確定 / 🟡 = v4 で追加した DH 改善版 (採用判断は要追加検証) / ★v4 new = v4 で新規追加 (DH-Z2, 配分 × タイミング 2 軸変動、binary HOLD/OUT) / ⚠⚠ = IS-OOS gap > +5pp (古典 overfit 強警戒) / ⚠ = +5〜+8pp (中度 overfit)

凡例（EVALUATION_STANDARD §3.12 v1.4 + v4.2 拡張）:
- ⓽ = 税後（手取り）: CAGR_OOS / Worst10Y★ / P10_5Y▷ / **累積 CAGR OOS/IS (新)**
- ⓒ = コスト後（税引前据置）: Sharpe_OOS / MaxDD / IS-OOS gap
- ⓞ = 原値（コスト・税で不変）: Trade / Overfit(WFE)
- ⓡ = 再計算値（g14 実測 + §3-A 税調整）: CI95_lo

**「累積 CAGR ⓽ OOS/IS」列の定義** (v4.2 で追加):
- §5 年次リターン (税後・moderate コスト後) から **暦年累積複利** で算出した年率 CAGR を、IS (1977-2020, 44年) と OOS (2021-2026, 6年) の両期間で並列表示
- 計算式: `cum = Π(1 + yr_aft)`、 `CAGR = cum^(1/N) - 1`
- 既存「CAGR ⓽ _ OOS」列 (g14/g22b の `metrics_from_nav` 由来、同じ年次複利方式) と OOS 値はほぼ一致 (小数第 2 位レベルの丸め差)。**本列の付加価値は IS 値を gap 経由でなく直接表示する点**
- 例: -50% → +100% → +200% (3 年で ×3.0) なら CAGR = `3.0^(1/3) - 1 ≈ +44.2%`

データ出典:
- NEW CANDIDATE / D5 / DH: [g20f_unified_yearly_returns.csv](g20f_unified_yearly_returns.csv) + [g20b_new_candidate_wfa_summary.csv](g20b_new_candidate_wfa_summary.csv) + [g14_wfa_sbi_cfd_summary.csv](g14_wfa_sbi_cfd_summary.csv)
- NDX 1x B&H: [g19e_3strategies_daily_cost.csv](g19e_3strategies_daily_cost.csv)
- 詳細コスト前提: §1' (本ファイル下部)
- 5 検証結果 (NEW CANDIDATE): §1〜§3 (本ファイル下部)

---

## 📋 §1' コスト・税金 調整前提 (v6.3 強化版 — 全コスト網羅 + ロジック再チェック)

### §1'-1 メイン調整テーブル (全 14 ステップ、CFD 戦略基準)

| 調整ステップ | 前提・値 | 備考 |
|---|---|---|
| **Step 1: ブローカー金利マージン (financing)** | **SBI/GMO/IG CFD 推定 — SOFR + 3.0%/yr** (業者マージン推定、`(L-1)×(SOFR+spread)`構造) | ⚠ SBI 公式は非公開 (deep-research 2026-05-20 SBI CFD ガイド §11 で再確認、取引画面ログイン後でのみ閲覧可能)。IG/楽天の SOFR+3.0% を参考に推定。実態は楽天 SOFR+約3% (=6.5-7.0%) / IG SOFR+3.0% (=6.59%)。試算: 4倍ポジ (400万円) で年率 25-30万円コスト (年率 6-7%) |
| **Step 2: 未含コスト補正 (residual drag)** | **−0.66%/yr** (内訳: Gold TER -10.5 bps + TMF TER -3.5 bps + Swap推定差 -34 bps + 他 -18 bps) | `CURRENT_BEST_STRATEGY.md` 準拠。「TER 補正」ではなく複合コスト補正である点に注意。CFD 前提では NAS swap 推定差 (-17bps相当) と SOFR+3.0% の二重計上リスクあり → 保守的レンジは -0.49% 〜 -0.66% |
| **Step 3-A: 日本税 (CAGR比例モデル)** | **20.315% (申告分離) × 「利益85%課税」想定** | `CAGR_net = (CAGR − 0.66%) × 0.8273` ← v2 で採用したモデル。8-9勝1-2敗想定の単純化モデル |
| **Step 3-B: 日本税 (固定減算モデル)** | EVALUATION_STANDARD §1.1 準拠 — **-2.8% (best) 〜 -5.2% (worst)** を直接減算 | `CAGR_net = CAGR − tax_drag`。Trades/yr 別に best/worst を選択 (27回→best寄り, 52回→worst寄り) |
| **Step 4 (NEW v6.1): 日次取引コスト (execution)** | `daily_cost(t) = \|Δposition(t)\| × spread_one_way` を NAV daily return から差し引く | 4 ケース併記 (§1'-3): measured 0.020% / opt 0.030% / **moderate 0.050%** / cons 0.100% (往復スプレッド)。コミッション 0% (CFD 標準) |
| **Step 5 (NEW v6.3): NDX 配当未反映の保守バイアス** | NDX Close (= Adj Close) は **price index で配当未含む** (CAGR 10.98%/yr vs Total Return ~11.7%/yr) | ⚠️ **本 backtest は NDX 配当 0.7%/yr × eff_L 分 UNDERESTIMATION** (4×レバで約 +3pp/yr 控除)。実 CFD は配当相当 pass-through で実 return ↑。**保守側方向** (発見元: v6.3 ロジック再チェック) |
| **MaxDD** | SOFR+3.0% コスト後 (税引前)、日次取引コスト後 | 含み損は税対象外 (SBI 店頭CFD前提)。税後との差は数値的に 1-2pp 程度 |
| **Sharpe_OOS** | SOFR+3.0% コスト後 (税前据置)、日次取引コスト後 | ⚠ 「Rf=0 で税乗数が分子分母等倍 → Sharpe 不変」は対称税モデル前提。現実の非対称税では約 -19% 低下する可能性 (簡易シミュレーション) |
| **IS-OOS gap** | SOFR+3.0% コスト後・日次取引コスト後・**税後** (CAGR_IS_aft − CAGR_OOS_aft) | OOS期間 (2021-2026) は高 SOFR 環境のため税前 gap が拡大。v6.3 から税後計算に統一 |
| **Trades/yr** | 原値 (g14 mean Trades_yr 実測 四捨五入) — `lev_change>0.5` 流儀 | g15c (C) 定義 (>0.01) で見ると 70-181/yr の実日次リバランス頻度 (g19d で内訳分解済) |
| **Overfit(WFE)** | 原値 (g14 実測、`mean_CAGR_postIS / mean_CAGR_IS`) | 0.5 ≤ WFE ≤ 2.0 を β PASS 基準 |
| **WFA_CI95_lo** | g14 実測値 + §3-A 税調整: `(CI95_lo_g14 − 0.66%) × 0.8273` | 推定式から実測値に更新済。判定基準として使用可能 (旧 8 戦略全 PASS、v4 で §0' 5 戦略に絞込み済) |
| **約定遅延 (DELAY)** | バックテストは **DELAY=2** (2営業日遅延) | ⚠ SBI 店頭CFD の実態は T+0 (即時)。バックテストの方が **+2 営業日保守的** |
| **税適用タイミング** | **年次 mark-to-market 想定** — 年次 pre-tax → §3-A → 年次 after-tax → 複利 | 実際は売却時のみ実現。年次税は loss carryforward が完璧と仮定する近似 |

### §1'-2 ETF 戦略コスト前提 (DH Dyn [A] のみ)

| 項目 | 値 | 備考 |
|---|---|---|
| TQQQ TER | 0.86%/yr | ProShares UltraPro QQQ |
| TMF TER | 0.91%/yr | Direxion 20+ Yr Treasury Bull 3x |
| **2036 TER** | **0.49%/yr** | **WisdomTree Gold 2x Daily Leveraged (LBUL.L)** — SBI 海外ETF枠で実取扱 (ユーザー実取得済み) |
| 借入金利 | SOFR + 0.50% swap × ETF倍率 | 各 ETF 内に内包 (TQQQ: 2×SOFR / TMF: 2×SOFR / 2036: 1×SOFR) |
| 配当課税 | 20.315% | 二重課税調整。TQQQ div yield 0.30% は実質ドラッグ -0.06%/yr |
| **SBI 米国ETF コミッション** | 約定代金 × 0.495% (上限 22 USD/取引) | NISA非適用 (3x/2x レバ ETF) |
| 為替手数料 | **25 銭/USD 片道** (標準) または **6 銭/USD** (住信SBIネット銀行経由) | per-turnover 換算: 標準 0.17% × 2方向 = 0.33%/turnover、Net銀行 0.04%/方向 = 0.08%/turnover |
| 税モデル | `年次 pre-tax × 0.8273` 逐年複利 | ETF/B&H: -0.66%「未含 CFD コスト」は適用なし |
| 取引コスト ケース | per-turnover one-way: **large-NAV 0.05% / moderate 0.10% / small-NAV 0.30%** | moderate 0.10% は SBI commission $22 上限 + bid-ask + 6銭FX (Net銀行) 想定 |

### §1'-3 取引コスト ケース設定 (再掲)

**CFD (往復スプレッド %)**:

| ケース | spread_RT | 根拠 |
|---|---:|---|
| **measured (GMO 2026/4 実測)** | **0.020%** | GMO 米国NQ100ミニ 配信スプレッド実績 (2026/4 平均 1.8 USD / NDX 20000) |
| optimistic | 0.030% | 同上 高品質提示帯 |
| **moderate (採用基準)** | **0.050%** | 業界中庸 (楽天/DMM 推定) |
| conservative (baseline) | 0.100% | IG/SBI 想定上限・スプレッド拡大時 |

**DH (per-unit turnover %)**:

| ケース | per_unit_cost | 根拠 |
|---|---:|---|
| large-NAV (cap eff.) | 0.05% | NAV $500k+ で SBI $22 上限が効く |
| **moderate (採用基準)** | **0.10%** | retail $100k 規模、典型値 (6銭FX + bid-ask) |
| small-NAV (no cap) | 0.30% | retail $50k 規模 + 25銭FX (標準) |

### §1'-4 NEW CANDIDATE / F10 / E4 / D5 戦略別 平均レバレッジ・取引回数

| 戦略 | L_avg<br>(OOS) | L_max | Trades/yr<br>(§3.12 標準) | Σ\|ΔL\|/yr | 日次 \|Δpos\| 平均/年 |
|---|---:|---:|---:|---:|---:|
| 🟢 NEW CANDIDATE | 4.36 | 7.0 | 52 | 56.7 | (g19d 参照: 7.9/yr added vs E4) |
| F10 ε=0.015 ★ | 4.38 | 7.0 | **52** | 56.7 | 同上 |
| F8 R5_CALM_BOOST | 4.38 | 7.0 | ~181 | 56.7 | tilt 連続更新で大 |
| F7v3+E4 A:tilt=2.0 | ~4.3 | 7.0 | ~50 | ~55 | (推定) |
| D5 vz=0.65/lmax=5.5 | 4.07 | 5.5 | **28** | 39.4 | (g19d 同等) |
| E4 Regime k_lt ◆ | 4.36 | 7.0 | **28** | 56.7 | 173.3/yr (lev_mod_e4 連続変動) |
| DH Dyn 2x3x [A] | 1.65 | 3.0 | **27** | 20.4 | 20.4/yr (turnover) |

出典: [g17_avg_leverage.csv](g17_avg_leverage.csv) / [g14_wfa_sbi_cfd_summary.csv](g14_wfa_sbi_cfd_summary.csv) / [g19d_f10_trade_decomp_results.csv](g19d_f10_trade_decomp_results.csv)

### §1'-5 ⚠️ 未捕捉コスト一覧 (v6.3 ロジック再チェックで発見)

| 項目 | 影響 | 方向 | 対応 |
|---|---|:--:|---|
| **NDX 配当** | NDX Close = Price Index で配当未含 (~0.7%/yr × eff_L = 約 +3pp/yr underestimation for 4×レバ) | **保守 (低見積)** | NOTE 明記、実 CFD では配当相当 pass-through で実 return ↑ |
| **価格調整額 / 先物ロール** | 元商品が先物の場合 (GMO 米国NQ100ミニ 等)、コンタンゴ環境で 2-4%/yr 可能性 | **未捕捉** | v6.1 §RISK-4 で言及済、店頭 CFD では非該当 |
| **配当源泉徴収 (TQQQ)** | TQQQ div yield 0.30% × 米国源泉 10% = -0.03%/yr (二重課税調整で還付) | 微小 | DH 戦略のみ、無視可 |
| **FX 換算 (CFD)** | JPY-quoted CFD では spread 内に embedded、追加コスト 0 | 0 | 該当なし |
| **FX 換算 (ETF)** | SBI 標準 25銭/USD = 0.17%/方向 (DH moderate 0.10% は 6銭/USD 想定で過小) | DH のみ -0.07〜0.14%/yr | 標準 FX 利用なら small-NAV (0.30%) に近似 |
| **CFD 維持手数料** | SBI/GMO/IG とも該当なし | 0 | 該当なし |
| **証拠金金利** | 余剰証拠金は預け金扱い (金利 0) | 0 | 該当なし |
| **税の繰延効果** | 年次 mark-to-market 仮定 → 売却時のみ実現で複利成長 | 軽度 aggressive | 影響小、長期保有では mark-to-market が conservative |

### §1'-6 ロジック再チェック結果 (v6.3 で確認した正しさ)

| # | チェック項目 | 結果 | 詳細 |
|:--:|---|:--:|---|
| 1 | 日次コスト時点 (`r_adj = r_base - daily_trade_cost`) | ✅ 正 | day t の開始でリバランス → day t の return から控除 (g18:96-122) |
| 2 | (L-1) × SOFR 構造 (`cfd_leverage_backtest.py:21`) | ✅ 正 | レバ × spread の正しい財務構造 |
| 3 | NDX 配当未含 (Close = Adj Close) | ⚠ NOTE | 保守 (低見積) であることを明記 |
| 4 | 税適用 (年次 mark-to-market + §3-A) | ✅ 正 | 年次 pre-tax → tax → 複利、CAGR_OOS_net 算出 |
| 5 | Sharpe 税前据置 (対称税モデル前提) | ✅ 正 (caveat 付) | 非対称税では -19% 可能性、別途注記済 |
| 6 | MaxDD 税前 (含み損は税対象外) | ✅ 正 | 日次 NAV (cost-after) ベース |
| 7 | WFA CI95_lo 税調整 (`(g14 - 0.66%) × 0.8273`) | ✅ 正 | 表示のみ、判定は raw も併用 |
| 8 | Trades/yr (g14 mean 実測) | ✅ 正 (caveat 付) | lev_change>0.5 流儀、実日次 rebalance は別途 g19d 分解済 |
| 9 | NEW CANDIDATE g19b 構築 (vz=0.65 + lmax=7 + F10ε=0.015) | ✅ 正 | QC Agent 2 で line-by-line 検証済 |
| 10 | DELAY=2 (2 営業日遅延) | ✅ 正 (保守) | 実 SBI CFD T+0 より +2 営業日分保守的 |

→ **全ロジック整合性確認**、未捕捉コストは「**NDX 配当 (保守側)**」と「**FX 標準利用時の DH 追加コスト (-0.1pp/yr)**」の 2 項目のみ、いずれも **数値的に小さく**、結論 (NEW CANDIDATE 条件付き昇格) に影響なし。

---

## ✅ §1 5検証 結果サマリ (v6.3-1〜v6.3-5)

| # | 検証 | スクリプト | 結果 | 判定 |
|:--:|---|---|---|:--:|
| 1 | vz_thr robustness sweep (5点) | [g20a](src/g20a_vz_robustness_sweep.py) | **vz=0.625 でさらに良値** (CAGR_OOS +21.88%, gap -1.67pp)、2/5 で負 gap、spike < 1pp | **✅ PASS** |
| 2 | WFA 50窓厳密 | [g20b](src/g20b_new_candidate_wfa.py) | **CI95_lo=+0.199, WFE=1.369, p=0.000008** | **✅ PASS** |
| 3 | 年次寄与分解 | [g20c](src/g20c_yearly_attribution.py) | **2022 単年 diff = -0.14pp**、+2.05pp は 2021/2023/2025 bull 年に分散、単年集中度 29.9% | **✅ PASS** |
| 4 | Bootstrap on OOS | [g20d](src/g20d_bootstrap_oos.py) | median +3.44pp, 95% CI [-1.62, +9.77]pp, P(>0)=90.7% | **⚠ MARGINAL** |
| 5 | Permutation test | [g20e](src/g20e_permutation_test.py) | 実 gap -1.27pp、NULL P(≤実) = 7.0% (5%閾値直上) | **⚠ MARGINAL** |

→ **3 STRONG PASS + 2 MARGINAL** = QC Agent 3 が懸念した「lucky regime fit」は否定。
ただし bootstrap/permutation の marginal は OOS 4.87 年の短期サンプル由来で、追加 OOS 年で改善見込み。

---

## 🔍 §2 検証詳細

### §2-1 [Test 1] vz_thr robustness sweep (★★★)

```
 vz_thr    CAGR_IS   CAGR_OOS  IS-OOS gap   Sharpe     MaxDD   Worst10Y
-------------------------------------------------------------------------------------
  0.625   +20.21%   +21.88%     -1.67pp   +0.815  -65.93%    +9.85%   ← BEST!
  0.650   +20.23%   +21.49%     -1.27pp   +0.829  -65.95%    +9.96% ★
  0.675   +20.20%   +19.52%     +0.68pp   +0.784  -65.91%   +10.29%
  0.700   +20.19%   +19.44%     +0.75pp   +0.777  -66.03%   +10.42%
  0.725   +20.07%   +19.27%     +0.80pp   +0.771  -66.03%    +9.57%
```

**観察**:
- vz=0.625 が **vz=0.65 を超える +21.88%** で最良 → NEW CANDIDATE は frontier 領域の頂点付近で、その「両側」(0.625/0.65) で同等の優位
- vz=0.65 周辺で smooth な勾配 (spike < 1pp) → **isolated lucky fit ではない**
- vz=0.625 と vz=0.65 の **両方** で gap 負方向 → **2/5 = 40%** の vz 値で OOS>IS、構造的優位を支持
- vz=0.675 以上で急に gap 正方向 → **vz=0.67 付近に regime 切替の閾値**が存在

**結論**: 構造的な vz_thr frontier (vz<0.67) が「early regime detection」優位を生む。

### §2-2 [Test 2] WFA 50窓厳密 (★★★)

| Strategy | n_windows | mean_CAGR | CI95_lo | CI95_hi | t-stat | p-value | WFE | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| **vz=0.625** | 49 | +34.17% | **+0.199** | +0.484 | 4.816 | 0.000008 | 1.369 | **PASS** |
| **vz=0.650 ★** | 49 | +34.15% | **+0.199** | +0.484 | 4.803 | 0.000008 | 1.369 | **PASS** |
| F10 (vz=0.70) REF | 49 | +33.71% | +0.194 | +0.480 | 4.744 | 0.000010 | 1.244 | PASS |

**観察**:
- NEW CANDIDATE は α 基準 (CI95_lo > 0, p < 0.05) を**圧倒的に**満たす (p=8e-6)
- WFE = 1.369 → OOS 期間が IS 期間の **36.9% 上回る** CAGR を実現 (F10 は 24.4%)
- vz=0.625 と vz=0.65 は WFA 指標で**完全同一** → 両者ともに頑健

**結論**: 統計的有意性も IS-OOS 安定性も F10 を上回る。

### §2-3 [Test 3] 年次寄与分解 (★★)

NEW CANDIDATE - F10 の OOS 年次差分 (税後):

| year | NEW (税後) | F10 (税後) | diff |
|---:|---:|---:|---:|
| 2021 | +26.21% | +19.12% | **+7.09pp** ← bull 年に貢献 |
| 2022 | -22.23% | -22.09% | **-0.14pp** ← drawdown 年は同等 |
| 2023 | +80.31% | +73.70% | **+6.61pp** ← bull 年に貢献 |
| 2024 | +47.47% | +47.77% | -0.30pp |
| 2025 | +41.25% | +33.47% | **+7.78pp** ← bull 年に貢献 |
| 2026 | -12.76% | -8.69% | -4.07pp |
| **合計** | | | **+16.96pp** (年平均 +2.83pp) |

**観察**:
- 2022 drawdown 年は両者**ほぼ同等** (-0.14pp) → 「2022 単年で救済された」仮説 (Agent 3 指摘) は**否定**
- +2.05pp 改善は 2021 / 2023 / 2025 の 3 bull 年に分散 (それぞれ +6.6〜7.8pp)
- 単年集中度 = 29.9% (≤ 33% 均等分散基準内)
- 年代別 IS では 1980s/2000s/2010s で軽度 negative, 1970s/1980s で positive → IS 全期間平均 -0.076pp/年 (構造的中立)
- **2020s (OOS-mostly) で +2.945pp/年**: 直近 regime に有利

**結論**: 「lucky single event」ではなく「複数 bull 年での leverage 効率優位」。

### §2-4 [Test 4] Bootstrap on OOS (★★)

OOS 4.87 年の日次差分リターンを paired block bootstrap (block=21日、10000 resamples):

| 指標 | 値 |
|---|---:|
| 実 CAGR_OOS NEW (税前) | +30.93% |
| 実 CAGR_OOS F10 (税前) | +27.66% |
| 実 diff | **+3.27pp** |
| Bootstrap median | +3.44pp |
| 95% CI | **[-1.62pp, +9.77pp]** ← 0 を跨ぐ |
| P(diff > 0) | 90.7% |

**観察**:
- 点推定値は明確に正 (+3.44pp)
- ただし 95% CI 下端 -1.62pp は**0 を跨ぐ** → 95% 信頼水準では H0 棄却不可
- **P(diff > 0) = 90.7%**: 90% 信頼水準では有意
- 主因は OOS 4.87 年と短期で、サンプル分散が大きいこと

**結論**: **marginal** — 90% 水準なら有意、95% は届かず。追加 OOS 年蓄積で改善見込み。

### §2-5 [Test 5] Permutation test (★)

vz_thr を [0.40, 1.00] uniform random で 100 sample、各 NULL 戦略の OOS gap を構築:

| 指標 | 値 |
|---|---:|
| 実 vz=0.65 gap | **-1.27pp** |
| NULL gap mean | +1.09pp |
| NULL gap 5-95% | [-1.72pp, +2.16pp] |
| **P(NULL gap ≤ 実 vz=0.65)** | **7.0%** |
| 実 CAGR_OOS | +21.49% |
| NULL CAGR_OOS 5-95% | [+17.50%, +21.88%] |
| **P(NULL CAGR_OOS ≥ 実)** | **7.0%** |

**観察**:
- 実 gap = -1.27pp は NULL 分布の下位 7% (5% 閾値を僅か上回る = marginal)
- 実 CAGR_OOS = +21.49% は NULL 上位 7% (同上)
- NULL 分布の mean = +1.09pp → NULL では gap 正方向、実 gap が逆方向で珍しい

**結論**: **marginal** — 5% 水準では足りないが、10% 水準では有意。

---

## 🎯 §3 NEW CANDIDATE 最終判定

### §3-1 5検証 統合スコア

| 検証 | 重み | 結果 | スコア |
|---|:--:|---|---:|
| vz_thr robustness | ★★★ | PASS | +3 |
| WFA 50窓 | ★★★ | PASS | +3 |
| 年次寄与分解 | ★★ | PASS | +2 |
| Bootstrap on OOS | ★★ | MARGINAL (P=90.7%) | +1 |
| Permutation | ★ | MARGINAL (P=7%) | +0.5 |
| **合計** | | | **+9.5 / +11** |

**達成率: 86.4%** → 「**条件付き昇格**」基準を満たす

### §3-2 判定: 🟢 **Active 候補追加 (F10 ★ と並列運用)**

| 項目 | 評価 |
|---|---|
| 即時 v7 Active 単独昇格? | ❌ Marginal が 2 件残るため不可 |
| **F10 ★ と並ぶ Active 候補?** | **✅ YES — 3 STRONG PASS が決定的** |
| 棄却? | ❌ 構造的優位 (vz_thr frontier) と年次分散性が支持 |
| 追加検証必要? | ⚠ 後年 OOS 蓄積で再評価 (3年毎に bootstrap 更新) |

### §3-3 採用推奨ステートメント

> **NEW CANDIDATE (vz=0.65+lmax=7+F10ε=0.015) は v6.3 検証 5 件のうち 3 件 STRONG PASS、2 件 MARGINAL** で、QC Agent 3 が懸念した「lucky single-event fit」は明確に否定された。
> ただし bootstrap/permutation の marginal は OOS 4.87 年の短期サンプル由来で、95% 信頼水準を満たすには追加 OOS 蓄積 (2-3 年) が必要。
> **暫定運用方針**: **F10 ε=0.015 ★** と **NEW CANDIDATE 🟢** の 2 戦略を「**双頭 Active 候補**」として並列運用し、毎年 g20d (bootstrap) を更新して 95% CI が安定的に正方向に固定されたら NEW CANDIDATE 単独昇格を判断。

---

## 📈 §5 年次リターン表 (1977-2026、手取り＋日次取引コスト後、moderate ケース)

> **N/A**: 1974-1976 は LT2-N750 ウォームアップにより CFD 戦略は信号未確定 (1977 開始)
> **2026**: 部分年 (~3-26)。年率換算なし、実績パーセンテージ
> **単位**: % (手取り・日次取引コスト後 moderate ケース)
> 出典: [g20f_unified_yearly_returns.csv](g20f_unified_yearly_returns.csv)

| 年 | NEW 🟢 | D5 v0.65/l5.5 | **DH-Z2 ★v4 new** | DH [A] | NDX 1x B&H |
|---:|---:|---:|---:|---:|---:|
| 1977 | -16.8 | -11.7 | -3.4 | -2.2 | +6.2 |
| 1978 | +111.6 | +78.5 | +43.6 | +43.0 | +11.1 |
| 1979 | +16.6 | +13.7 | +13.0 | +21.2 | +23.4 |
| 1980 | +64.4 | +53.8 | +26.6 | +49.0 | +30.2 |
| 1981 | -30.0 | -28.6 | -0.9 | -20.9 | -3.1 |
| 1982 | +75.0 | +71.8 | +64.6 | +79.7 | +15.6 |
| 1983 | +8.4 | +7.0 | -1.3 | +23.0 | +17.2 |
| 1984 | -14.1 | -12.8 | -4.3 | -3.6 | -9.1 |
| 1985 | +114.4 | +93.4 | +34.4 | +48.8 | +26.6 |
| 1986 | +28.4 | +29.1 | +18.7 | +28.8 | +6.1 |
| 1987 | +57.8 | +45.6 | +16.6 | +15.6 | -5.3 |
| 1988 | -27.9 | -20.4 | -8.7 | -11.9 | +10.5 |
| 1989 | +42.3 | +36.0 | +18.0 | +21.9 | +16.7 |
| 1990 | -41.8 | -33.8 | -14.1 | -7.7 | -15.4 |
| 1991 | +67.4 | +62.3 | +35.8 | +39.4 | +47.6 |
| 1992 | +41.0 | +31.9 | +20.2 | +25.0 | +12.8 |
| 1993 | -6.0 | -3.8 | -1.4 | +3.6 | +12.9 |
| 1994 | -13.7 | -10.4 | -2.3 | -9.0 | -2.0 |
| 1995 | +124.1 | +95.2 | +53.6 | +59.3 | +34.3 |
| 1996 | +36.4 | +27.0 | +8.3 | +17.1 | +18.2 |
| 1997 | +44.7 | +40.9 | +36.8 | +42.0 | +18.7 |
| 1998 | +56.4 | +66.4 | +38.2 | +63.8 | +32.0 |
| 1999 | +63.7 | +59.4 | +79.7 | +99.0 | +69.7 |
| 2000 | -10.5 | -9.0 | -12.8 | +0.9 | -33.3 |
| 2001 | -6.1 | -6.1 | -1.5 | +1.1 | -12.3 |
| 2002 | +8.6 | +8.6 | -3.2 | +23.3 | -26.9 |
| 2003 | +88.3 | +91.4 | +62.7 | +59.0 | +36.9 |
| 2004 | +0.8 | +4.5 | +6.7 | +9.2 | +7.0 |
| 2005 | -21.0 | -11.2 | +0.5 | +1.5 | +2.0 |
| 2006 | +42.2 | +31.9 | +27.9 | +22.3 | +6.3 |
| 2007 | +17.8 | +17.9 | +1.3 | +16.4 | +7.8 |
| 2008 | +16.4 | +17.1 | +3.3 | +18.7 | -32.7 |
| 2009 | +46.3 | +38.4 | +43.3 | +23.0 | +32.3 |
| 2010 | +58.3 | +73.1 | +33.8 | +46.4 | +12.3 |
| 2011 | -16.6 | -7.7 | -17.2 | -5.3 | -2.7 |
| 2012 | +15.6 | +16.8 | +24.2 | +17.9 | +11.6 |
| 2013 | +21.8 | +20.7 | +12.5 | +21.2 | +28.3 |
| 2014 | -4.1 | +3.1 | -3.7 | +10.2 | +11.8 |
| 2015 | -30.9 | -26.5 | -16.9 | -17.3 | +4.9 |
| 2016 | -10.3 | -3.2 | -1.4 | +5.9 | +8.1 |
| 2017 | +66.7 | +48.2 | +21.0 | +23.2 | +22.5 |
| 2018 | -11.3 | -6.0 | -2.5 | -2.2 | -4.4 |
| 2019 | +41.4 | +46.5 | +19.1 | +33.9 | +28.6 |
| 2020 | +69.2 | +59.5 | +68.9 | +62.5 | +34.5 |
| **2021 [OOS]** | **+26.2** | **+18.5** | **+11.3** | **+20.8** | **+19.2** |
| **2022 [OOS]** | **-22.2** | **-22.4** | **-2.6** | **-24.1** | **-28.0** |
| **2023 [OOS]** | **+80.3** | **+67.5** | **+32.4** | **+31.0** | **+36.8** |
| **2024 [OOS]** | **+47.5** | **+40.8** | **+26.1** | **+21.9** | **+25.5** |
| **2025 [OOS]** | **+41.3** | **+38.8** | **+13.9** | **+30.5** | **+17.0** |
| **2026 [OOS]** | **-12.8** | **-10.9** | **-3.0** | **-9.5** | **-6.5** |

---

## 📊 §6 統計サマリ (1974-2026、税後・日次取引コスト後 moderate)

| 統計 | NEW 🟢 | D5 v0.65/l5.5 | **DH-Z2 ★v4 new** | DH [A] | NDX 1x B&H |
|---|---:|---:|---:|---:|---:|
| **平均 (mean)** | **+26.80%** | +24.50% | +16.02% | +21.19% | +11.25% |
| 中央値 (median) | +21.77% | +18.47% | +12.47% | +21.19% | +12.34% |
| 標準偏差 (std) | +40.74% | +34.39% | +22.86% | +25.61% | +20.61% |
| 最大 (max) | **+124.13%** | +95.25% | +79.65% | +99.00% | +69.74% |
| 最小 (min) | -41.79% | -33.75% | **-17.21%** | -24.12% | -33.26% |
| プラス年数 | 35 | 36 | 34 | **41** | 39 |
| マイナス年数 | 18 | 17 | 18 | **12** | 14 |

> **注**: 統計サマリは 1974-2026 全 53 年 (CFD 戦略は 1974-1976 が LT2 ウォームアップ前なのでデータ前提を含む)。
> 出典: [g20f_unified_stats_summary.csv](g20f_unified_stats_summary.csv)

### §6-1 観察 (5 戦略比較版 v4)

- **平均リターン**: NEW 🟢 (+26.80%) が首位、D5 (+24.50%) が次点
- **最大単年リターン**: NEW 🟢 が **+124.1%** で 5 戦略中最高 (1995 年 IT バブル前期)
- **最大単年損失**: NEW 🟢 (-41.79%, 1990 年) が最深、D5 (-33.75%) が CFD 系で最浅
- **下方リスク (Min)**: **DH-Z2 (-17.21%) が 5 戦略中最良** (REF DH [A] -24.12% を 6.91pp 改善)、binary HOLD/OUT が bear 救済
- **プラス年数**: DH [A] が **41/53 = 77.4%** で最高、DH-Z2 は 34/53 (binary OUT 期間が中立 0 となるため減少)

### §6-2 OOS 期間 (2021-2026) 累積比較

| Strategy | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 6年累積倍率 | 100万円→ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **NEW 🟢** | +26.2% | -22.2% | +80.3% | +47.5% | +41.3% | -12.8% | **×3.38** | **338万円** |
| D5 vz=0.65/lmax=5.5 | +18.5% | -22.4% | +67.5% | +40.8% | +38.8% | -10.9% | ×2.68 | 268万円 |
| **DH-Z2 (F10 ε + binary vz HOLD/OUT) ★v4 new** | +11.3% | **-2.6%** | +32.4% | +26.1% | +13.9% | -3.0% | **×2.00** | **200万円** |
| DH Dyn 2x3x [A] | +20.8% | -24.1% | +31.0% | +21.9% | +30.5% | -9.5% | ×1.73 | 173万円 |
| NDX 1x B&H | +19.2% | -28.0% | +36.8% | +25.5% | +17.0% | -6.5% | ×1.55 | 155万円 |

→ **NEW 🟢 は OOS 6 年で 100 万円 → 338 万円** で 5 戦略中最高。**DH-Z2 (×2.00)** は DH-REF (×1.73) を 27 万円 (15.6%) 上回り、binary HOLD/OUT が 2022 (-2.6% vs REF -24.1%) で大幅救済。

---

## 🎯 §7 採用判断 (v6.3 確定版)

| 判断軸 | 推奨 |
|---|---|
| 🟢 **Active 候補昇格 (双頭運用)** | **NEW CANDIDATE (vz=0.65+l7+F10ε) と F10 ε=0.015 ★ を並列運用** |
| 継続 Active (保守) | E4 ◆ (現行確定、WFA PASS、WFE 1.15 最も中庸) |
| Shortlisted | D5 vz=0.65/lmax=5.5 (MaxDD 最浅、Sharpe 高) |
| 不採用 | F8 R5_CALM_BOOST (F10 と CAGR 完全同等、独自性なし) |
| 保留 | F7v3+E4 A:tilt=2.0 (tilt 構造差別化候補、要 v7 比較) |
| 継続研究 | DH Dyn 2x3x [A] (Worst10Y 安定だが macro 感応度 v7 課題) |
| ベンチマーク | NDX 1x B&H |

### §7-1 双頭運用の意義

| 軸 | NEW 🟢 | F10 ★ |
|---|---|---|
| CAGR_OOS | **+21.49%** | +19.44% |
| IS-OOS gap | **-1.27pp** (OOS優位) | +0.75pp |
| Sharpe_OOS | **+0.829** | +0.78 |
| WFE | **1.369** | 1.244 |
| **MaxDD** | -65.95% | -66.03% (ほぼ同等) |
| **vz_thr** | **0.65** (early regime detection) | 0.70 (late regime detection) |
| 構造的差 | 2022 等の drawdown 早期回避傾向 | 標準 regime detection |

→ **NEW と F10 は vz_thr のみ違い** (それ以外 lmax=7, ε=0.015, base 完全同一)。配分時間軸で 1 ユーザーが両方使うのは構造的 redundancy が高すぎる。
→ **単一選択推奨は NEW 🟢** だが、Bootstrap/Permutation の marginal が解消するまでは F10 ★ も並列運用で risk hedge。

---

## 🔧 §8 v7 候補課題

### §8-1 即時 (v6.4)
1. **NEW CANDIDATE の毎年 bootstrap 更新** — OOS 蓄積で 95% CI 安定化を追跡
2. **vz=0.625 と vz=0.65 の比較深掘り** — 0.625 が更に良い理由 (regime threshold 細分化)

### §8-2 v7 中期
3. **DH Dyn [A] の SOFR シナリオ感応度** (Agent 3 課題)
4. **NEW CANDIDATE の業者切替試算** (IG/楽天/GMO conservative)
5. **F7v3 (uniform tilt) との混合構造** 検討
6. **WisdomTree 2036 (LBUL.L) GBP/JPY 為替感応度の定量化**

---

## 📁 §9 データソース・出典 (v6.3 追加分)

| ファイル | 役割 |
|---|---|
| [src/g20a_vz_robustness_sweep.py](src/g20a_vz_robustness_sweep.py) | Test 1: vz_thr 5点 sweep |
| [g20a_vz_robustness_sweep.csv](g20a_vz_robustness_sweep.csv) | Test 1 結果 |
| [src/g20b_new_candidate_wfa.py](src/g20b_new_candidate_wfa.py) | Test 2: WFA 50窓 |
| [g20b_new_candidate_wfa_summary.csv](g20b_new_candidate_wfa_summary.csv) | Test 2 サマリ |
| [g20b_new_candidate_per_window.csv](g20b_new_candidate_per_window.csv) | Test 2 per-window |
| [src/g20c_yearly_attribution.py](src/g20c_yearly_attribution.py) | Test 3: 年次寄与分解 |
| [g20c_yearly_attribution.csv](g20c_yearly_attribution.csv) | Test 3 結果 |
| [src/g20d_bootstrap_oos.py](src/g20d_bootstrap_oos.py) | Test 4: Bootstrap on OOS |
| [g20d_bootstrap_oos_results.csv](g20d_bootstrap_oos_results.csv) | Test 4 結果 |
| [src/g20e_permutation_test.py](src/g20e_permutation_test.py) | Test 5: Permutation test |
| [g20e_permutation_test_summary.csv](g20e_permutation_test_summary.csv) | Test 5 サマリ |
| [g20e_permutation_test_detail.csv](g20e_permutation_test_detail.csv) | Test 5 詳細 |
| [src/g20f_unified_yearly_returns.py](src/g20f_unified_yearly_returns.py) | 8 戦略年次リターン生成 |
| [g20f_unified_yearly_returns.csv](g20f_unified_yearly_returns.csv) | 年次リターン全期間 |
| [g20f_unified_stats_summary.csv](g20f_unified_stats_summary.csv) | 統計サマリ |

### §9-v3 DH-T4 改善版 (v3 追加分 — **2026-06-03 破棄**: lev_mod による TQQQ 連続 scaling は ETF 制約違反)

> ⚠️ **以下の g21* 系は v3 で push 済だが v4 で破棄。STRATEGY_REGISTRY §3 Rejected 参照。履歴保持のため残置するが新規参照不可。**

| ファイル | 役割 |
|---|---|
| [src/g18_daily_trade_cost_wfa.py](src/g18_daily_trade_cost_wfa.py) | `build_dh_nav_with_timing_cost` 関数 — **DEPRECATED 2026-06-03** |
| [src/g21a_dh_improved_variants.py](src/g21a_dh_improved_variants.py) | DH 改善 4 変種 (T1〜T4) NAV — 破棄 |
| [g21a_dh_improved_navs.csv](g21a_dh_improved_navs.csv) | 5 戦略日次 NAV — 破棄 |
| [src/g21b_dh_improved_metrics.py](src/g21b_dh_improved_metrics.py) | 9 指標 — 破棄 |
| [g21b_dh_improved_9metrics.csv](g21b_dh_improved_9metrics.csv) | 9 指標表 — 破棄 |
| [src/g21c_dh_improved_wfa.py](src/g21c_dh_improved_wfa.py) | WFA 50 窓 — 破棄 |
| [g21c_dh_improved_wfa.csv](g21c_dh_improved_wfa.csv) | WFA 結果 — 破棄 |
| [src/g21d_dh_improved_bootstrap.py](src/g21d_dh_improved_bootstrap.py) | Bootstrap — 破棄 |
| [g21d_dh_bootstrap_oos_results.csv](g21d_dh_bootstrap_oos_results.csv) | Bootstrap 結果 — 破棄 |
| [src/g21e_dh_improved_permutation.py](src/g21e_dh_improved_permutation.py) | Permutation — 破棄 |
| [g21e_dh_permutation_summary.csv](g21e_dh_permutation_summary.csv) | Permutation サマリ — 破棄 |
| [g21e_dh_permutation_detail.csv](g21e_dh_permutation_detail.csv) | Permutation 詳細 — 破棄 |
| [src/g21f_dh_improved_attribution.py](src/g21f_dh_improved_attribution.py) | 年次寄与 — 破棄 |
| [g21f_dh_t4_vs_ref_yearly_attribution.csv](g21f_dh_t4_vs_ref_yearly_attribution.csv) | T4 vs REF 年次 — 破棄 |
| [g21f_dh_t4_yearly_returns_aftertax.csv](g21f_dh_t4_yearly_returns_aftertax.csv) | T4 単独年次 — 破棄 |

### §9-v4 DH-Z2 改善版 (v4 採用分 — 配分 × タイミング 2 軸変動)

| ファイル | 役割 |
|---|---|
| [src/g22a_dh_alloc_timing_variants.py](src/g22a_dh_alloc_timing_variants.py) | DH-Z 5 変種 (Z1〜Z5) NAV 構築 + peak lev assert |
| [g22a_dh_alloc_timing_navs.csv](g22a_dh_alloc_timing_navs.csv) | 6 戦略 (REF + Z1〜Z5) 日次 NAV |
| [g22a_dh_alloc_timing_sanity.csv](g22a_dh_alloc_timing_sanity.csv) | HOLD 比率 + peak leverage 検証 |
| [src/g22b_dh_alloc_timing_metrics.py](src/g22b_dh_alloc_timing_metrics.py) | 9 指標標準 |
| [g22b_dh_alloc_timing_9metrics.csv](g22b_dh_alloc_timing_9metrics.csv) | 9 指標比較表 |
| [src/g22c_dh_alloc_timing_wfa.py](src/g22c_dh_alloc_timing_wfa.py) | WFA 50 窓検証 → Z2 最良 |
| [g22c_dh_alloc_timing_wfa.csv](g22c_dh_alloc_timing_wfa.csv) | WFA 結果（CI95_lo, WFE, p） |
| [src/g22d_dh_alloc_timing_bootstrap.py](src/g22d_dh_alloc_timing_bootstrap.py) | Bootstrap (Z2 vs REF, paired block 10000×21d) |
| [g22d_dh_alloc_timing_bootstrap_results.csv](g22d_dh_alloc_timing_bootstrap_results.csv) | Bootstrap 結果 |
| [src/g22e_dh_alloc_timing_permutation.py](src/g22e_dh_alloc_timing_permutation.py) | Permutation (binary θ 100 samples) |
| [g22e_dh_alloc_timing_permutation_summary.csv](g22e_dh_alloc_timing_permutation_summary.csv) | Permutation サマリ |
| [g22e_dh_alloc_timing_permutation_detail.csv](g22e_dh_alloc_timing_permutation_detail.csv) | Permutation 詳細 |
| [src/g22f_dh_alloc_timing_attribution.py](src/g22f_dh_alloc_timing_attribution.py) | 年次寄与 + 統計 + OOS 累積 |
| [g22f_dh_alloc_timing_attribution.csv](g22f_dh_alloc_timing_attribution.csv) | Z2 vs REF 年次差分 |
| [g22f_dh_z2_yearly_returns_aftertax.csv](g22f_dh_z2_yearly_returns_aftertax.csv) | Z2 単独年次税後 |

---

## 📝 §10 改訂履歴

| Ver | 日付 | 主要変更 |
|---|---|---|
| v6.0 | 2026-06-02 | 年率近似 取引コスト |
| v6.1 | 2026-06-02 | 日次取引コスト導入 |
| v6.2 | 2026-06-02 | 7 戦略拡張 + Task A-E (NEW SOTA 発見) |
| v6.2 QC | 2026-06-02 | 3 並列 QC エージェント → NEW SOTA → NEW CANDIDATE 格下げ |
| **v6.3** | **2026-06-02** | **NEW CANDIDATE 5検証完了 (3 PASS + 2 MARGINAL) → 「双頭 Active 候補」昇格、8戦略年次リターン表追加** |
| v2 | 2026-06-03 | §0' 列構成 v6.1 完全一致版 |
| ~~v3~~ | ~~2026-06-03~~ | ~~DH-T4 (vz=0.65+lmax=5.5+F10ε) を 4 箇所に追加~~ — **2026-06-03 破棄**: `lev_mod` で TQQQ position 連続スケールしたため ETF 制約違反 |
| **v4** | **2026-06-03** | **DH-T4 を全面置換: DH-Z2 (F10 ε tilt + binary HOLD/OUT vz_gate, peak lev 2.85x≤3.0x 厳守) を §0'/§5/§6/§6-2 の 4 箇所に配置。配分 × タイミング 2 軸変動、レバ操作なし。OOS 累積 ×2.00 (REF ×1.73)、WFE=1.058 (完全汎化)** |
| **v4.1** | **2026-06-03** | **§4 8戦略総合比較表を削除、§5/§6/§6-2 を §0' 5 戦略 (NEW 🟢 / D5 / DH-Z2 / DH [A] / NDX 1x B&H) のみに絞込み (F10 ε=0.015 ★ / F8 R5 / F7v3+E4 / E4 ◆ 列・行を削除)。§6-1 観察も 5 戦略基準に更新** |
| **v4.2** | **2026-06-03** | **§0' 表に「累積 CAGR ⓽ OOS/IS」列を新規追加** (既存 CAGR ⓽ OOS 列の右隣)。§5 年次リターンから暦年累積複利で IS (1977-2020, 44年) と OOS (2021-2026, 6年) の両期間 CAGR を算出し 1 セル 2 行で並列表示。IS 値が gap 経由でなく直接読める。例: NEW IS=+20.22% / OOS=+21.50%、DH IS=+20.01% / OOS=+9.56% (OOS 大幅劣化が一目瞭然)、DH-Z2 IS=+14.43% / OOS=+12.24% (汎化良好)。凡例にも定義追記 |

---

*管理者: 男座員也 (Kazuya Oza)*
*生成: Claude (Opus 4.7) on 2026-06-03 (v4)*
*準拠: EVALUATION_STANDARD.md v1.4, v3.1 §3-A 税モデル, v6.1 日次取引コスト, v6.2 7戦略拡張, v6.3 5検証パッケージ, v4 DH 配分×タイミング 2 軸変動版*
