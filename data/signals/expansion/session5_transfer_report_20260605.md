# Session 5 — nasdaq_mom63 x M6 defensive 転用 audit (S2, E4)

作成日: 2026-06-05
最終更新日: 2026-06-05

## 検証対象
- S3 (DH-W1, ETF only) -> **既に ADOPT 確定** (Session 4)
- **S2 (D5 = vz=0.65, lmax=5)** -> 本セッション転用テスト
- **E4 (現行 §1 Active = S2_VZGated + LT2-N750 + E4 Regime k_lt)** -> 本セッション転用テスト

Overlay: signal = `nasdaq_mom63`, method = M6, direction = defensive, mapping = signal_q {0,1,2,3} -> {1.1, 1.0, 0.9, 0.8}

## 転用結果サマリ

| baseline | WFE | CI95_lo CAGR | P_CAGR | P_Sharpe | P_MaxDD better | Verdict |
|---|---:|---:|---:|---:|---:|---|
| S3 DH-W1 (Session 4 ADOPT) | 1.005 | +13.00% | 0.295 | **0.930** | **0.988** | **ADOPT** |
| S2_D5 | 0.963 | +22.72% | 0.201 | 0.758 | 0.944 | **NEEDS_FURTHER_WORK** |
| E4_Active | 0.958 | +24.41% | 0.355 | 0.858 | 0.964 | **NEEDS_FURTHER_WORK** |

## 各転用先の 9+1 詳細

### S2_D5 (S2 (D5 vz=0.65, lmax=5))

| metric | baseline | candidate | diff |
|---|---:|---:|---:|
| CAGR_OOS_pct | +33.4871 | +31.1247 | -2.3624 |
| IS_OOS_gap_pp | -3.9117 | -3.7349 | +0.1769 |
| Sharpe_OOS | +0.9368 | +0.9552 | +0.0184 |
| MaxDD_full_pct | -51.8185 | -50.6312 | +1.1873 |
| Worst10Y_pct | +18.5687 | +15.2042 | -3.3646 |
| P10_5Y_pct | +14.6325 | +12.0969 | -2.5357 |
| Trades_per_yr | +76.0523 | +90.9679 | +14.9156 |
| WFE_full | +0.9343 | +0.9626 | +0.0283 |
| CI95_lo_window_CAGR_pct | +24.8187 | +22.7181 | -2.1006 |
| +1 composite (n_imp / n_deg) | - | 3 / 6 | - |

### E4_Active (E4 RegimeKLT (current §1 Active))

| metric | baseline | candidate | diff |
|---|---:|---:|---:|
| CAGR_OOS_pct | +33.5301 | +32.1596 | -1.3705 |
| IS_OOS_gap_pp | -1.8127 | -2.5484 | -0.7358 |
| Sharpe_OOS | +0.8793 | +0.9142 | +0.0349 |
| MaxDD_full_pct | -60.0102 | -58.4971 | +1.5131 |
| Worst10Y_pct | +17.5191 | +15.2693 | -2.2498 |
| P10_5Y_pct | +13.4368 | +10.6363 | -2.8005 |
| Trades_per_yr | +75.8225 | +90.8147 | +14.9922 |
| WFE_full | +0.9281 | +0.9581 | +0.0300 |
| CI95_lo_window_CAGR_pct | +26.5093 | +24.4094 | -2.0999 |
| +1 composite (n_imp / n_deg) | - | 4 / 5 | - |

## 解釈

### S2 (D5) 転用結果: **NEEDS_FURTHER_WORK**
- WFE = 0.963 (FAIL <1.0、S3 では 1.005 PASS), CI95_lo = +22.72% (PASS >0)
- Bootstrap: P(CAGR)=0.201 (CAGR は劣化), P(Sharpe)=0.758 (PASS 一歩手前), **P(MaxDD better)=0.944 (PASS >0.90)**
- 9+1 composite: 3 imp / 6 deg — Worst10Y -3.36pp, P10_5Y -2.54pp の保守的指標悪化が目立つ
- **示唆**: MaxDD 改善 (+1.19pp) と Sharpe 改善 (+0.018) はあるが、S3 で観測された MaxDD +5.83pp/Sharpe +0.047 と比較すると効果が大幅に縮小。S2 が既に VZ ゲート + LT2-modeB で「リスク調整済み」のため、nasdaq_mom63 の追加 defensive tilt の限界貢献が小さい。

### E4 (現行 §1 Active) 転用結果: **NEEDS_FURTHER_WORK**
- WFE = 0.958 (FAIL <1.0), CI95_lo = +24.41% (PASS >0)
- Bootstrap: P(CAGR)=0.355, P(Sharpe)=0.858 (gate=0.90 近接), **P(MaxDD better)=0.964 (PASS >0.90)**
- 9+1 composite: 4 imp / 5 deg — Worst10Y -2.25pp, P10_5Y -2.80pp
- **示唆**: S2 と同様に MaxDD/Sharpe は微改善するが、Worst10Y/P10_5Y/CAGR の保守的尺度がほぼ全敗。E4 の Regime k_lt 動的バイアスが既に nasdaq_mom63 と部分的に相関した防御を実装しているため、二重に貼ると過防御で長期 CAGR を圧迫する仮説。

### 戦略基盤特異性 (Cross-Strategy)
- **S3 (DH-W1) で ADOPT 確定** ✓ (Session 4)
- **S2 (D5) / E4 (Active) で NEEDS_FURTHER_WORK** (= 厳格ADOPTは不可)
- 結論: 本 overlay は **S3 (DH-W1, ETF only) 特異 (strategy-specific)**。CFD ベースの S2/E4 では VZ ゲート + LT2-modeB / Regime k_lt が既に類似の防御機能を担っているため、追加 overlay の限界効用が小さく、Worst10Y/P10_5Y を悪化させる副作用が勝つ。
- ただし MaxDD 改善方向はすべての baseline で一貫しており (P>0.94)、**overlay の「方向性」は普遍** だが、「Sharpe/CAGR をネットで改善する能力」は S3 限定。

### MaxDD 改善の Cross-Strategy 一貫性

| baseline | MaxDD diff pp | P(MaxDD better) | Sharpe diff | P(Sharpe better) |
|---|---:|---:|---:|---:|
| S3 (DH-W1) | **+5.83pp** | 0.988 | +0.047 | 0.930 |
| S2 (D5) | +1.19pp | 0.944 | +0.018 | 0.758 |
| E4 (Active) | +1.51pp | 0.964 | +0.035 | 0.858 |

→ MaxDD 防御効果は確実に存在するが、S3 で 5.83pp / S2/E4 で 1.2–1.5pp と **CFD ベースで 1/4 程度に減衰**。

## 推奨アクション

1. **S3 (DH-W1) 限定 ADOPT を維持** (Session 4 確定)。
2. **CURRENT_BEST_STRATEGY.md** の "Risk-Reduction Overlay Candidate" セクションを **「S3 限定」と明記**。S2/E4 では NEEDS_FURTHER_WORK と記録 (REJECT ではないため将来再評価可)。
3. **§1 本番 Active (E4 RegimeKLT) は完全に変更なし**。本 overlay は CFD 環境には現時点で適用しない。
4. ETF only 環境 (NISA 等) における DH-W1 + nasdaq_mom63 overlay は、Session 4 ADOPT に基づき採用可能性あり。実運用判断はユーザーに委ねる。
5. **将来の探索方向**:
   - S2/E4 用に method/direction を変えた探索 (例: M2 defensive、M5 stop_only など)
   - VZ ゲート閾値・k_lt パラメータと nasdaq_mom63 シグナル空間の干渉を検証
   - 別シグナル (nfci_z52w, vix_mom21) の CFD ベース転用
