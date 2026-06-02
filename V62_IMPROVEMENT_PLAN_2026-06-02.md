# v6.2 改善計画 — 4タスク (A-D) 並列実行 (2026-06-02)

> v6.1 (STRATEGY_PERFORMANCE_INTEGRATED_20260602.md) を起点に、F10 の優位性を更に確証し、過適合警告を晴らすための 4 タスク群を並列実行する。

## 🎯 Goal
v6.1 で示された F10 ε=0.015 優位性 (+19.44% / gap +0.75pp) について:
- **A**: ε 隣接値で頑健性を確認
- **B**: D5 (vz_thr×lmax) と F10 (ε) を組み合わせた 3D 最適点探索
- **C**: DH Dyn [A] +10.29pp 過適合警告の真因究明
- **D**: F10 Trades/yr=52 の内訳分析（オペコスト最小化）

完了時の納品物:
1. 5 つの新規スクリプト (g19a〜g19d)
2. 5 つの結果 CSV
3. v6.2 統合レポート (STRATEGY_PERFORMANCE_INTEGRATED_20260602_v62.md)
4. すべて GitHub に push 済み

---

## 📋 Task A: F10 ε 隣接 sweep + 日次コスト評価

**入力**: `src/f10_epsilon_deadband.py` (既存、ε grid = {0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050})
**追加**: ε = {0.008, 0.012, 0.018, 0.025} 4点を追加 (F10 ε=0.015 周辺を細分化)
**処理**: 各 ε について g18 同等の日次取引コスト (4 spread ケース) を適用、9指標出力

**新規ファイル**:
- `src/g19a_f10_eps_extended.py` — 11点 ε × 4 spread ケース = 44 行
- `g19a_f10_eps_extended_results.csv`

**期待出力**: ε=0.015 の優位性が真に最適か、周辺で更に良い ε が存在するか確認

---

## 📋 Task B: F10 + D5 ハイブリッド 3D sweep

**入力**: `src/d5_vz_lmax_grid.py` (vz × lmax 2D) + `src/f10_epsilon_deadband.py` (ε 1D)
**処理**: 3D grid = vz_thr {0.60, 0.65, 0.70} × lmax {5.0, 5.5, 6.0} × ε {0.010, 0.015, 0.020}
  = 3×3×3 = **27 configs** (現実的な計算量、~30秒/config = ~15分)
**評価**: g18 同等の日次コスト適用、9指標出力

**新規ファイル**:
- `src/g19b_hybrid_sweep.py` — F10 deadband on top of D5 vz×lmax base
- `g19b_hybrid_sweep_results.csv`

**期待出力**: F10 (vz=0.7, lmax=7) と D5 (vz=0.65, lmax=5.5) の中間に SOTA が存在するか

---

## 📋 Task C: DH Dyn [A] 厳密 WFA + signal audit

**入力**: v6.1 で IS-OOS gap +10.29pp、WFE=0.7 ⚠ → 真因究明が必要
**処理**:
1. `src/g14_wfa_sbi_cfd.py` 同等 50-window WFA を DH Dyn 単体に適用
2. signal audit: `build_a2_signal` の IS / OOS 分布比較
3. WFA per-window CAGR 分布を可視化（IS 期間内での variance 確認）

**新規ファイル**:
- `src/g19c_dh_dyn_wfa.py`
- `g19c_dh_dyn_wfa_summary.csv`
- `g19c_dh_dyn_signal_audit.csv`

**期待出力**: 
- True overfit (signal が IS regime に依存) vs Spurious gap (OOS bull の偶発)
- 採用見送り or 部分採用の判断材料

---

## 📋 Task D: F10 Trades/yr=52 内訳分析

**入力**: F10 信号は S2_VZGated (lev_raw 変化) + LT2 (lev_mod 変化) + wn-tilt (wn 変化) の合成
**処理**: 各成分の日次変化回数を独立カウント
1. L_s2 (S2_VZGated レバ) の日次変化
2. wn (Approach A NDX 比率 + F10 tilt) の日次変化
3. lev_mod (LT2 オーバーレイ後) の日次変化
4. 統合 trade count = いずれか変化した日数

**新規ファイル**:
- `src/g19d_f10_trade_decomp.py`
- `g19d_f10_trade_decomp_results.csv`

**期待出力**: F10 の 52 trades/yr のうち、wn-tilt 由来は何 trades/yr か → F10 改良版（tilt頻度抑制版）設計の指針

---

## 📊 Phase 5: v6.2 レポート統合

**新規ファイル**: `STRATEGY_PERFORMANCE_INTEGRATED_20260602_v62.md` (v6.1 を継承し、§A-D の発見を追加)

統合内容:
1. §11 v6.2 で追加した分析 (A-D 要約)
2. §12 新規発見と推奨方針（v7 候補）
3. § 既存表を最新の最良値で更新（必要に応じて）

---

## 📊 Phase 6: Push + Verify

1. `git add` 全新規ファイル
2. `git commit` (per task)
3. `git push origin main`
4. GitHub URL 検証 (各CSV/MD/script)
5. 3列表で成果物報告

---

*管理: Claude (Opus 4.7) / 2026-06-02*
