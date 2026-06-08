# Gold / Bond スタンドアロン・タイミングシグナル タクソノミー

作成日: 2026-06-08
最終更新日: 2026-06-08

> 本書は `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 1.1 の成果物。
> 目的: Gold と Bond の **「保有 vs キャッシュ」** を NASDAQ と同じ体系（9指標＋WFA）で評価するための、検証対象シグナルの台帳。
> 原則: **リポ既存モジュール・取得済みデータに接地**し、各シグナルに仮説（なぜ効くか）を付す（戦略検証プロトコル Step3）。

## 凡例
- **状態**: ✅取得済 / 🟡既存スクリプトに実装あり（要抽出） / 🔵Yahoo/FRED即取得可 / ⬜未実装
- **再利用**: 出発点となる既存モジュール/データ

---

## A. Bond（米国債）シグナル

Bond価格代理: 長期検証=DGS10/DGS30利回り→デュレーション価格換算（1962/1977〜）、実運用整合=IEF/TLT。

| ID | シグナル族 | 仮説（なぜ効くか） | 状態 | 再利用/データ |
|---|---|---|---|---|
| B-TR | デュレーション・モメンタム（価格MA/TS-mom） | 金利トレンドは持続性があり、下降（価格上昇）局面の保有が効率的 | 🟡 | `src/signals/timing.py`, `e2_bond_sweep.py` |
| B-YC1 | イールドカーブ 2s10s | 逆イールド→景気後退期待→長期債価格上昇に先行 | 🔵 | `dgs2_daily.csv`/`dgs10_daily.csv` |
| B-YC2 | イールドカーブ 30s10s / 3M-10Y(T10Y3M) | 同上（区間別）。3M-10Yは古典的後退指標 | 🔵/✅ | FRED T10Y3M、`dgs*` |
| B-TP | タームプレミアム 30y-10y | プレミアム拡大局面は長期債に不利／縮小は有利 | ✅ | `repo_2_..._30y_10y_termpremium.parquet` |
| B-RY | 実質金利（10y - 期待インフレ） | 実質金利低下は債券価格に追い風 | 🟡 | `src/g5_real_yield.py` |
| B-CR | クレジットスプレッド BAA-AAA | スプレッド拡大＝リスクオフ＝質への逃避で国債買い | ✅ | `repo_1_..._baa_minus_aaa_spread.parquet` |
| B-FED1 | Fed funds レジーム（DFF-10y, DFF変化1M） | 利上げ終了/利下げ転換は長期債の強気転換点 | ✅ | `repo_3_dff_change_1m`, `repo_4_dff_minus_10y` |
| B-VOL | 金利ボラ（MOVE）レジーム | 高ボラ期は債券リスク上昇→キャッシュ退避が有利 | 🔵 | Yahoo `^MOVE` |
| B-RG | Bondレジーム（既存ロジック） | 既存の体系を単独hold/cashへ転用 | 🟡 | `src/f5_bond_regime.py`, `bond_variant_sweep.py` |

---

## B. Gold（金）シグナル

Gold価格: LBMA USD（1968〜）。実運用整合=DGP(2x)/GLD。

| ID | シグナル族 | 仮説（なぜ効くか） | 状態 | 再利用/データ |
|---|---|---|---|---|
| G-TR | 価格モメンタム（MA/TS-mom） | 金はトレンド持続が強く、順張り保有が効率的 | 🟡 | `src/signals/timing.py`, `h5_gold_dyn.py` |
| G-RY | 実質金利（負の相関） | 実質金利低下＝金の機会費用低下＝金高 | 🟡 | `src/g5_real_yield.py` |
| G-DXY | ドル指数 DXY（逆相関） | ドル安は金（ドル建て）に追い風 | 🔵 | Yahoo `DX-Y.NYB` |
| G-INF | インフレ（CPI YoYサプライズ） | インフレ加速・サプライズは金需要を押し上げ | ✅ | `repo_7_..._cpi_yoy_surprise.parquet` |
| G-CoT | CFTC CoT 金ネットポジション | 投機筋ポジションの偏りは反転先行のことがある | 🔵 | `src/data_loaders/signals/cftc.py`（GC） |
| G-CuAu | 銅/金レシオ（成長プロキシ） | レシオ低下＝景気減速＝金優位 | 🔵 | Yahoo `HG=F`/`GC=F`（既存 signal 42） |
| G-DYN | 動的Gold配分（既存ロジック） | 既存の動的金フラクションを単独hold/cashへ転用 | 🟡 | `src/h5_gold_dyn.py`, `b9_s2lt2_goldfrac_sweep.py` |

---

## C. 横断（NASDAQ既存族の Gold/Bond 流用）

NASDAQで確立した汎用オーバーレイ族を Gold/Bond にも適用可否評価する。

| 族 | 内容 | Gold | Bond | 再利用 |
|---|---|---|---|---|
| VZ-gating | ボラzスコア・ゲート | ○ | ○ | `src/e3_vzgate_sweep.py`, `c3_yang_zhang_vol.py` |
| LT trend filter | 長期トレンドフィルタ | ○ | ○ | `src/b*_*lt*` 系 |
| E4 Regime k_lt | レジーム連動 | △ | ○ | `src/e4_regime_klt.py` |
| ε-deadband | 取引抑制デッドバンド | ○ | ○ | `src/f10_epsilon_deadband.py` |

---

## D. 評価・検証方針（全シグナル共通）

- 評価指標: **9指標標準**（`src/integration/nine_metric_eval.py`） — CAGR_OOS / IS-OOS gap / Sharpe_OOS / MaxDD / Worst10Y / P10_5Y / Trades_yr / WFE / CI95_lo
- ベースライン: 各資産 B&H と「常時キャッシュ」を必ず併記
- 採用条件: 3軸ベースライン（min(IS,OOS)・Worst10Y・P10_5Y CAGR）同時超過
- ロバスト: `phase_d_wfa` で CI95_lo / WFE ゲート、`phase_d_bootstrap` でブートストラップ
- 過学習警戒: WFE高（>1.5）はレジーム運フラグ、SPA（`src/signals/spa_test.py`）で多重検定補正

## E. 優先度（次の着手）

1. **B-RY / B-CR / B-TP / B-FED1**（Bond, 取得済データ即評価可）
2. **G-RY / G-DXY / G-INF**（Gold, 同上）
3. 既存ロジック転用（B-RG `f5_bond_regime`, G-DYN `h5_gold_dyn`）を単独hold/cash基準で再評価
4. 横断オーバーレイ（VZ/LT/ε）の適用可否
