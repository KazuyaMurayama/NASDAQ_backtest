# 引退後資産サバイバル・シミュレーション 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: 実行は本セッション内インライン（executing-plans 相当）。ステップは checkbox（`- [ ]`）で追跡。

**Goal:** 初期資産3000万円・毎年支出500万円で、P09_C1 レバダイヤル各 scale（1.0/1.4/1.6/1.8/2.0/2.2/2.4）＋NASDAQ 1倍B&H に投資して生活した場合、1975〜2005年の各開始年について「20年間資産がゼロにならずに生き延びられるか」「10年後・20年後の資産額」を算出し、md レポートにまとめる。

**Architecture:** 既存の検証済み NAV ビルダー（`_build_tqqq_base` + `_build_p09_on_base_c1`）を再利用して各 scale の**日次 NAV → 暦年リターン**を生成（ゲート再実行はしない＝軽量）。そのうえで、暦年リターン列に対し「年初に500万円引出 → 残高に当年リターンを掛ける」逐次シミュレーションを start_year × strategy グリッドで回す。同時に P09_C1(scale1.0) の暦年リターンも出力し、既存ダイヤルレポート §2 に列追加する。

**Tech Stack:** Python (numpy/pandas)、リポ内 `src/audit/*` の検証済みビルダー再利用。

---

## モデリングの決定事項（ユーザー依頼の文言に忠実・自分で判断できる細部は確定）

- **税基準**: 暦年リターンは**税後（×0.8273）**を使用（生活＝利益確定の前提で保守側。ダイヤルレポート §2 と同一基準＝混在なし）。
- **引出のタイミング**: ユーザー文言「その年の最初にその生活を開始」に従い、**年初に500万円を引き出してから**残高に当年リターンを適用（withdraw-first, conservative sequence-of-returns 規約）。
- **支出**: 名目固定500万円/年（依頼文言どおり）。インフレ連動はしない（base case）。実質価値の目減りは脚注で言及。
- **破綻判定**: 年初引出後または当年末で残高が **0円以下**になったら破綻（ruin）。破綻年（残高が0以下になった最初の年＝経過年）を記録。破綻後はそれ以降を「破綻（0円）」として扱う。
- **開始年**: 1975〜2005（31 開始年）、各々 20年保有（例: 2005開始→2024末まで＝データ2025内）。
- **戦略**: P09_C1(1.0)・scale1.4/1.6/1.8/2.0/2.2/2.4・NASDAQ 1倍B&H の **8系列**。
- **出力指標（start_year × strategy ごと）**: ①20年生存(Y/N)、②10年後資産額、③20年後資産額、④破綻した場合の破綻年（経過年）。
- **集計**: 戦略ごとに「31開始年のうち生存した数」「20年後資産の中央値/最小/最大」「破綻した開始年の一覧」。

---

## File Structure

- Create: `src/audit/retirement_survival_20260623.py` — サバイバル・シミュレーション本体（NAV→暦年→引出逐次計算）。
- Create: `audit_results/retirement_survival_grid_20260623.csv` — start_year × strategy の生存/10年後/20年後/破綻年グリッド。
- Create: `audit_results/retirement_survival_paths_20260623.csv` — 代表開始年（1975/1995/2000/2005）の年次資産推移（任意・QC用）。
- Create: `audit_results/p09_scale10_annual_20260623.csv` — P09_C1(scale1.0) 暦年税後リターン（既存ダイヤル CSV に無いため）。
- Modify: `P09_C1_SCALE_DIAL_20260623.md` — §2 年次表の左端に「P09_C1(1.0)」列を追加、§2.1 統計に sc1.0 行を追加。
- Create: `RETIREMENT_SURVIVAL_20260623.md` — サバイバルレポート本体（標準フォーマット）。

---

## Task 1: P09_C1(scale1.0) 暦年リターンの算出 ＋ ダイヤルレポート §2 への列追加

**Files:**
- Create: `src/audit/retirement_survival_20260623.py`（暦年生成部）
- Create: `audit_results/p09_scale10_annual_20260623.csv`
- Modify: `P09_C1_SCALE_DIAL_20260623.md`（§2 表・§2.1 統計）

- [ ] **Step 1: NAV ビルダーを呼ぶ共通セットアップを書く**

`src/audit/p09_scale_dial_20260623.py` の `main()` L122-148 と同一の前処理（shared, dates, ret_gold/ret_bond, fund_active, wg/wb, bond_on, sofr_arr）を関数 `build_inputs()` として切り出して再利用する。新規実装はしない（コピーで可、コメントで出典明記）。

- [ ] **Step 2: 各 scale の日次NAV→暦年税後リターンを生成**

```python
SCALES_ALL = [1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
cy_aftertax = {}   # scale -> pd.Series(year -> after-tax annual return, fraction)
for sc in SCALES_ALL:
    cfd = (sc > 1.0)  # scale1.0 は cfd_excess=False（サニティ基準と一致）
    nav_dt, _r, _tpy, _exc = _build_p09_scaled(
        shared, dates_dt, n_years, ret_gold, ret_bond, fund_active, wg, wb,
        bond_on, sofr_arr, lev_scale=sc, cfd_excess=cfd)
    cy = _calendar_year_returns(nav_dt)
    cy = cy[(cy.index >= 1975) & (cy.index <= 2025)]
    cy_aftertax[sc] = cy * AFTER_TAX
```

- [ ] **Step 3: scale1.0 のサニティ確認（既知 min⓽ と整合する年次か）**

scale1.0 の暦年税後を出力し、2008年が +20.34%（OUT年・C1充填でレバ非依存）になることを確認（ダイヤル CSV の sc1.4 2008=+20.34% と一致するはず＝OUT年は scale 不変）。一致しなければ停止。

- [ ] **Step 4: NASDAQ 1倍B&H 暦年税後を生成**

`p09_scale_dial_20260623.py` の `_load_nasdaq_bh()` を再利用（×0.8273 を掛けて after-tax 化）。

- [ ] **Step 5: scale1.0 暦年 CSV を出力**

`audit_results/p09_scale10_annual_20260623.csv`（列: year, sc1.0_aftertax_pct）。

- [ ] **Step 6: ダイヤルレポート §2 表に「P09_C1(1.0)」列を左端に挿入**

`P09_C1_SCALE_DIAL_20260623.md` の §2 表ヘッダ `| 年 | sc1.4 | ...` を `| 年 | P09_C1(1.0) | sc1.4 | ...` に変更し、各年行の先頭に sc1.0 値を挿入。値は Step 2 の cy_aftertax[1.0]（%表記・小数1桁）。

- [ ] **Step 7: §2.1 統計表に sc1.0 行を追加**

平均/中央値/標準偏差/最大/最小/プラス年/マイナス年を sc1.0 暦年税後（1975-2025・ddof=1）で算出し、表の scale1.40 行の上に挿入。

---

## Task 2: サバイバル逐次シミュレーション

**Files:**
- Modify: `src/audit/retirement_survival_20260623.py`
- Create: `audit_results/retirement_survival_grid_20260623.csv`

- [ ] **Step 1: 引出逐次関数を書く**

```python
INIT_ASSET = 30_000_000.0   # 3000万円
ANNUAL_SPEND = 5_000_000.0  # 500万円/年
HORIZON = 20
START_YEARS = list(range(1975, 2006))  # 1975..2005

def simulate(cy_ret_series, start_year, init=INIT_ASSET, spend=ANNUAL_SPEND, horizon=HORIZON):
    """年初引出 -> 当年リターン適用。残高<=0 で破綻。
    returns dict(survived, ruin_year_elapsed, asset_y10, asset_y20, path[list]).
    """
    bal = init
    path = []
    ruin_elapsed = None
    asset_y10 = None
    for k in range(horizon):           # k=0..19 (elapsed years)
        yr = start_year + k
        bal -= spend                   # 年初に支出を引き出す
        if bal <= 0:
            ruin_elapsed = k + 1 if ruin_elapsed is None else ruin_elapsed
            bal = 0.0
            path.append(bal)
            if k + 1 == 10: asset_y10 = 0.0
            continue
        r = float(cy_ret_series.loc[yr])  # after-tax annual return (fraction)
        bal *= (1.0 + r)
        if bal <= 0:
            ruin_elapsed = k + 1 if ruin_elapsed is None else ruin_elapsed
            bal = 0.0
        path.append(bal)
        if k + 1 == 10: asset_y10 = bal
    survived = (ruin_elapsed is None)
    asset_y20 = path[-1]
    return {"survived": survived, "ruin_year_elapsed": ruin_elapsed,
            "asset_y10": asset_y10 if asset_y10 is not None else bal,
            "asset_y20": asset_y20, "path": path}
```

- [ ] **Step 2: start_year × strategy グリッドを回す**

8 系列（cy_aftertax[1.0..2.4] ＋ NDX1x_aftertax）× 31 開始年 = 248 セル。各セルで `simulate()` を実行し、結果を集める。各開始年について必要な暦年（start..start+19）がデータ内にあることを確認（2005開始→2024末・OK）。

- [ ] **Step 3: グリッド CSV 出力**

`audit_results/retirement_survival_grid_20260623.csv`
列: `start_year, strategy, survived, ruin_year_elapsed, asset_y10_yen, asset_y20_yen`。

- [ ] **Step 4: 戦略別サマリを計算**

戦略ごとに: 生存した開始年数/31、20年後資産（生存ケースのみ）の中央値・最小・最大、破綻した開始年リスト、破綻ケースの平均破綻年。

- [ ] **Step 5: 代表パス CSV 出力（QC用）**

開始年 1975/1980/1995/2000/2005 について、8系列の年次資産推移を `audit_results/retirement_survival_paths_20260623.csv` に出力。

---

## Task 3: レポート生成 `RETIREMENT_SURVIVAL_20260623.md`

**Files:**
- Create: `RETIREMENT_SURVIVAL_20260623.md`

- [ ] **Step 1: ヘッダ＋前提（作成日・最終更新日・モデリング仮定）**

H1 直下に `作成日: 2026-06-23` / `最終更新日: 2026-06-23`。前提ブロックに：初期3000万・支出500万/年固定名目・年初引出・税後リターン・1975-2005開始・20年・破綻=残高0以下。

- [ ] **Step 2: §1 戦略別サマリ表**

| 戦略 | 生存開始年/31 | 20年後資産 中央値 | 同 最小 | 同 最大 | 破綻した開始年 |

- [ ] **Step 3: §2 開始年×戦略 生存マトリクス**

行=開始年(1975-2005)、列=8系列、セル=「生存→20年後資産（万円）」or「破綻(N年目)」。

- [ ] **Step 4: §3 10年後・20年後資産の対比表（代表開始年）**

1975/1980/1985/1990/1995/2000/2005 の各戦略 10年後・20年後資産。

- [ ] **Step 5: §4 読み方・結論**

シーケンスリスク（2000/2001/2002 直撃の開始年が最弱）、レバ上げが生存率に与える非単調効果（高 CAGR ↔ 高 DD で破綻確率上昇）を記述。

- [ ] **Step 6: §5 QC**

`analysis-qa-checklist` 準拠。サニティ（scale1.0 暦年が OUT年で scale 不変・既知値整合）、引出規約の妥当性、税基準統一、破綻判定の単調性、データ範囲充足を確認。

- [ ] **Step 7: §6 脚注・前提**

名目固定支出の留保（インフレ実質目減り）、為替（USD名目リターン×円支出のミスマッチ）の留保、後知恵バイアス（過去系列1本＝将来保証でない）の留保。

---

## Self-Review チェック
- [ ] 全 start_year で必要暦年がデータ内（最遅 2005+19=2024 ≤ 2025）→ OK。
- [ ] 税基準が全系列 after-tax で統一（混在トラップ回避）。
- [ ] 破綻判定が単調（残高0到達後は0固定）。
- [ ] scale1.0 暦年のサニティ（OUT年 scale 不変）が PASS。
- [ ] プレースホルダ無し（全コード・全閾値を明記済）。
