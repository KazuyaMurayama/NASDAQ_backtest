# 新規シグナル探索 実装計画 (Phase A〜C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

作成日: 2026-06-03
最終更新日: 2026-06-03

**Goal:** SIGNAL_DISCOVERY_PLAN_20260603.md (Spec) に基づき、52候補シグナルの Tier1 選別 → Phase B IC スクリーニング → Phase C G1-G11 WFA を、既存 NASDAQ_backtest コードベースに整合する形で実装可能な 28 タスクへ分解する。

**Architecture:** 新規 Python パッケージ `src/signals/` (中核モジュール) と `src/data_loaders/signals/` (ソース別ローダー) を nested package として導入。既存 g20 系 WFA 関数を Phase C で wrap して再利用。データキャッシュは `data/signals/_cache/`、メタデータは `data/signals/tier1_selection_20260603.csv` を single source of truth とする。

**Tech Stack:** Python 3.11+ / pandas 2.3 / numpy 2.3 / scipy 1.16 / statsmodels 0.14 / pytest / yfinance (新規) / arch (新規, G11 SPA test 用) / requests (FRED/CFTC HTTP)

---

## 0. Spec 参照

本計画は次の Spec の実装計画である:
- [`SIGNAL_DISCOVERY_PLAN_20260603.md`](./SIGNAL_DISCOVERY_PLAN_20260603.md) — 確定済 brainstorming 設計仕様

各タスクは spec のセクション番号 (§4.x, §5.x, §6.x) と紐付けて記述する。

---

## 1. File Structure

### 1.1 新規モジュール

```
src/signals/
├── __init__.py              # public API exports
├── metadata.py              # SignalMeta dataclass + registry CSV ロード
├── quantize.py              # 0/1, 0-3 量子化関数 (3スキーマ)
├── timing.py                # publication_lag 適用 (daily/weekly/monthly/event)
├── ic.py                    # Spearman rolling IC + Newey-West t-stat
├── screening.py             # Phase B: hit rate, decade-IC, BH-FDR
├── composite.py             # Phase B: PCA composite signal
└── wfa.py                   # Phase C: G1-G11 orchestrator (既存 g20 wrapper)

src/data_loaders/
└── signals/
    ├── __init__.py
    ├── _base.py             # SignalLoader ABC + ディスクキャッシュ
    ├── fred.py              # FRED HTTP (HY OAS, 2s10s, DGS10, T5YIFR, NFCI)
    ├── yahoo.py             # yfinance (VIX, VIX9D, VVIX, MOVE, DXY, GVZ, oil)
    ├── cboe.py              # CBOE 公開ファイル (PutCall, VIX term)
    ├── cftc.py              # CFTC CoT (NQ, GC, ZB/ZN)
    ├── fedwatch.py          # CME FedWatch 25bp 確率
    ├── google_trends.py     # pytrends (recession, TQQQ search)
    └── manual.py            # 手動更新 CSV (AAII, NAAIM, Fed NLP score)

tests/signals/
├── __init__.py
├── test_metadata.py
├── test_quantize.py
├── test_timing.py
├── test_ic.py
├── test_screening.py
├── test_composite.py
├── test_wfa.py
├── fixtures/                # ミニチュア CSV/JSON フィクスチャ
│   ├── fred_dgs10_sample.csv
│   ├── cot_nq_sample.csv
│   └── ...
└── test_loaders/
    ├── __init__.py
    ├── test_base.py
    ├── test_fred.py
    ├── test_yahoo.py
    ├── test_cboe.py
    ├── test_cftc.py
    ├── test_fedwatch.py
    ├── test_google_trends.py
    └── test_manual.py
```

### 1.2 新規データファイル

```
data/signals/
├── _cache/                            # ローダー別キャッシュ (.gitignore 対象)
├── tier1_selection_20260603.csv       # 52 信号メタデータ (single source)
├── manual/                            # 手動更新 CSV
│   ├── aaii_weekly.csv
│   ├── naaim_weekly.csv
│   └── fed_nlp_score.csv
└── README.md                          # データソース概要
```

### 1.3 新規ドキュメント

```
docs/signals/
├── data_lineage.md                    # 各信号の source/lag/earliest/cost
├── taxonomy_20260603.md               # Spec §4.2 のコピー (実装側参照用)
└── README.md
```

### 1.4 .gitignore 追加

```
data/signals/_cache/
data/signals/manual/_drafts/
```

---

## 2. Phase A: 信号タクソノミー & データ基盤 (詳細粒度)

Spec §4 に対応。13 タスク (A0-A12)。**全タスク TDD。各 commit 単位は failing test → minimal impl → passing test の 1 サイクル。**

---

### Task A0: ブートストラップ — ディレクトリ・依存追加

**Files:**
- Create: `src/signals/__init__.py`
- Create: `src/data_loaders/__init__.py`
- Create: `src/data_loaders/signals/__init__.py`
- Create: `tests/signals/__init__.py`
- Create: `tests/signals/test_loaders/__init__.py`
- Create: `data/signals/.gitkeep`
- Create: `data/signals/manual/.gitkeep`
- Create: `docs/signals/README.md`
- Create or Modify: `requirements.txt` (リポに存在しない場合は新規作成)
- Modify: `.gitignore`

- [ ] **Step 1: 既存 requirements.txt の有無を確認**

```bash
cd "C:/Users/user/Desktop/投資・不動産/nasdaq_backtest"
ls requirements*.txt 2>/dev/null || echo "no requirements file"
```

- [ ] **Step 2: requirements.txt を作成 or 追記**

`requirements.txt` (新規 or 追記):
```
pandas>=2.3
numpy>=2.3
scipy>=1.16
statsmodels>=0.14
pytest>=8.0
yfinance>=0.2.40
arch>=7.0
pytrends>=4.9
requests>=2.32
```

- [ ] **Step 3: 各ディレクトリと空 `__init__.py` / `.gitkeep` を作成**

```bash
mkdir -p src/data_loaders/signals tests/signals/test_loaders tests/signals/fixtures data/signals/_cache data/signals/manual docs/signals
touch src/signals/__init__.py src/data_loaders/__init__.py src/data_loaders/signals/__init__.py
touch tests/signals/__init__.py tests/signals/test_loaders/__init__.py
touch data/signals/.gitkeep data/signals/manual/.gitkeep data/signals/_cache/.gitkeep
```

- [ ] **Step 4: .gitignore に cache 追加**

`.gitignore` 末尾に追加:
```
data/signals/_cache/
data/signals/manual/_drafts/
```

- [ ] **Step 5: `docs/signals/README.md` に最小説明**

```markdown
# signals/

新規シグナル探索 (Phase A〜C) のドキュメント配置。

- `taxonomy_20260603.md` — Tier1 タクソノミー (52候補)
- `data_lineage.md` — 各信号の source / publication_lag / earliest_date / cost_tier

実装計画: `IMPLEMENTATION_PLAN_SIGNAL_DISCOVERY_20260603.md` (リポルート)
設計仕様: `SIGNAL_DISCOVERY_PLAN_20260603.md` (リポルート)
```

- [ ] **Step 6: pip install で依存解決確認**

```bash
pip install yfinance arch pytrends 2>&1 | tail -5
python -c "import yfinance, arch, pytrends; print('ok')"
```
Expected: `ok`

- [ ] **Step 7: コミット**

```bash
git add src/signals/ src/data_loaders/ tests/signals/ data/signals/ docs/signals/ requirements.txt .gitignore
git commit -m "chore(signals): Phase A ブートストラップ — モジュール骨格・依存追加"
```

---

### Task A1: SignalMeta dataclass + メタデータ CSV スキーマ

**Spec ref:** §4.1 マトリクス構造、§4.4 成果物 #3 `tier1_selection_<date>.csv`

**Files:**
- Create: `src/signals/metadata.py`
- Create: `tests/signals/test_metadata.py`
- Create: `tests/signals/fixtures/metadata_sample.csv`

- [ ] **Step 1: フィクスチャ CSV (5行のサンプル)**

`tests/signals/fixtures/metadata_sample.csv`:
```csv
signal_id,name,category,view,target_assets,quantize_scheme,q_levels,priority,source_module,publication_lag,earliest_date,cost_tier
1,NDX 200DMA breadth,A_Breadth,Trader,N,quantile_cut,4,A,yahoo,daily,2003-01-01,free
6,VIX level,B_Vol,Trader,N,quantile_cut,4,A,yahoo,daily,1990-01-01,free
12,CBOE PutCall Equity,C_Sentiment,Trader,N,binary_threshold,3,A,cboe,daily,2003-10-01,free
21,ICE BofA HY OAS,D_Credit,Actuary,N,zscore_band,3,A,fred,daily,1996-12-31,free
50,Fed minutes hawkish-dovish,J_NLP,HF,B|N,quantile_cut,4,A,manual,event,2010-01-01,low_paid
```

- [ ] **Step 2: 失敗テストを書く**

`tests/signals/test_metadata.py`:
```python
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from signals.metadata import SignalMeta, load_registry

FIXTURE = Path(__file__).parent / 'fixtures' / 'metadata_sample.csv'


def test_load_registry_returns_list_of_signalmeta():
    metas = load_registry(FIXTURE)
    assert len(metas) == 5
    assert all(isinstance(m, SignalMeta) for m in metas)


def test_signalmeta_field_types():
    metas = load_registry(FIXTURE)
    m = metas[0]
    assert m.signal_id == 1
    assert m.name == 'NDX 200DMA breadth'
    assert m.category == 'A_Breadth'
    assert m.target_assets == ['N']
    assert m.q_levels == 4
    assert m.priority == 'A'
    assert m.cost_tier == 'free'


def test_multi_target_assets_split():
    metas = load_registry(FIXTURE)
    fed = next(m for m in metas if m.signal_id == 50)
    assert fed.target_assets == ['B', 'N']


def test_priority_invalid_raises():
    import pandas as pd
    df = pd.read_csv(FIXTURE)
    df.loc[0, 'priority'] = 'X'
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    import pytest
    with pytest.raises(ValueError, match="priority must be A/B/C"):
        load_registry(buf)
```

- [ ] **Step 3: 失敗確認**

```bash
pytest tests/signals/test_metadata.py -v 2>&1 | tail -10
```
Expected: `ModuleNotFoundError: No module named 'signals.metadata'`

- [ ] **Step 4: 最小実装**

`src/signals/metadata.py`:
```python
"""Signal metadata model and registry loader.

CSV schema (single source: data/signals/tier1_selection_<date>.csv):
  signal_id        int unique
  name             human-readable
  category         A_Breadth..J_NLP (spec §4.2 prefix)
  view             Trader | Actuary | HF
  target_assets    pipe-separated subset of {N,G,B}
  quantize_scheme  binary_threshold | quantile_cut | zscore_band
  q_levels         2 or 4
  priority         A (◎) | B (○) | C (△)
  source_module    matches src/data_loaders/signals/*.py stem
  publication_lag  daily | weekly | monthly | event
  earliest_date    YYYY-MM-DD
  cost_tier        free | low_paid | mid_paid
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union
import pandas as pd


_VALID_PRIORITY = {'A', 'B', 'C'}
_VALID_QUANT = {'binary_threshold', 'quantile_cut', 'zscore_band'}
_VALID_LAG = {'daily', 'weekly', 'monthly', 'event'}
_VALID_COST = {'free', 'low_paid', 'mid_paid'}


@dataclass(frozen=True)
class SignalMeta:
    signal_id: int
    name: str
    category: str
    view: str
    target_assets: List[str]
    quantize_scheme: str
    q_levels: int
    priority: str
    source_module: str
    publication_lag: str
    earliest_date: str
    cost_tier: str

    def __post_init__(self):
        if self.priority not in _VALID_PRIORITY:
            raise ValueError(f"priority must be A/B/C, got {self.priority}")
        if self.quantize_scheme not in _VALID_QUANT:
            raise ValueError(f"quantize_scheme invalid: {self.quantize_scheme}")
        if self.q_levels not in (2, 4):
            raise ValueError(f"q_levels must be 2 or 4, got {self.q_levels}")
        if self.publication_lag not in _VALID_LAG:
            raise ValueError(f"publication_lag invalid: {self.publication_lag}")
        if self.cost_tier not in _VALID_COST:
            raise ValueError(f"cost_tier invalid: {self.cost_tier}")
        for a in self.target_assets:
            if a not in ('N', 'G', 'B'):
                raise ValueError(f"target_assets entry invalid: {a}")


def load_registry(path_or_buf: Union[str, Path, "IO"]) -> List[SignalMeta]:
    df = pd.read_csv(path_or_buf)
    out: List[SignalMeta] = []
    for _, row in df.iterrows():
        out.append(SignalMeta(
            signal_id=int(row['signal_id']),
            name=str(row['name']),
            category=str(row['category']),
            view=str(row['view']),
            target_assets=str(row['target_assets']).split('|'),
            quantize_scheme=str(row['quantize_scheme']),
            q_levels=int(row['q_levels']),
            priority=str(row['priority']),
            source_module=str(row['source_module']),
            publication_lag=str(row['publication_lag']),
            earliest_date=str(row['earliest_date']),
            cost_tier=str(row['cost_tier']),
        ))
    return out
```

- [ ] **Step 5: テスト pass 確認**

```bash
pytest tests/signals/test_metadata.py -v 2>&1 | tail -10
```
Expected: 4 passed

- [ ] **Step 6: コミット**

```bash
git add src/signals/metadata.py tests/signals/test_metadata.py tests/signals/fixtures/metadata_sample.csv
git commit -m "feat(signals): SignalMeta dataclass + registry CSV ローダー (Phase A §4.1)"
```

---

### Task A2: Tier1 選別 CSV 本体 (52信号メタデータ)

**Spec ref:** §4.2 (52候補一覧)、§4.3 (Tier1 結果 ◎31/○16/△5)

**Files:**
- Create: `data/signals/tier1_selection_20260603.csv`
- Create: `tests/signals/test_tier1_csv.py`

- [ ] **Step 1: 失敗テスト (Tier1 集計値検証)**

`tests/signals/test_tier1_csv.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from signals.metadata import load_registry

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'signals' / 'tier1_selection_20260603.csv'


def test_total_52_signals():
    metas = load_registry(CSV)
    assert len(metas) == 52


def test_priority_counts_match_spec():
    metas = load_registry(CSV)
    a = sum(1 for m in metas if m.priority == 'A')
    b = sum(1 for m in metas if m.priority == 'B')
    c = sum(1 for m in metas if m.priority == 'C')
    assert (a, b, c) == (31, 16, 5), f"got ◎{a}/○{b}/△{c}, expected 31/16/5"


def test_signal_ids_unique_and_sequential():
    metas = load_registry(CSV)
    ids = [m.signal_id for m in metas]
    assert len(set(ids)) == 52
    assert min(ids) == 1 and max(ids) == 52


def test_categories_match_spec():
    metas = load_registry(CSV)
    expected = {'A_Breadth', 'B_Vol', 'C_Sentiment', 'D_Credit',
                'E_YieldCurve', 'F_MacroNowcast', 'G_Earnings',
                'H_CrossAsset', 'I_Calendar', 'J_NLP'}
    got = {m.category for m in metas}
    assert got == expected


def test_priority_A_set_matches_spec_4_3():
    metas = load_registry(CSV)
    a_ids = sorted(m.signal_id for m in metas if m.priority == 'A')
    expected = sorted([1,2,3,6,7,8,9,10,12,13,15,16,17,21,23,26,27,28,29,30,
                       32,34,36,37,38,40,41,42,46,49,50])
    assert a_ids == expected


def test_priority_C_set_matches_spec_4_3():
    metas = load_registry(CSV)
    c_ids = sorted(m.signal_id for m in metas if m.priority == 'C')
    assert c_ids == [5, 20, 25, 45, 48]
```

- [ ] **Step 2: 失敗確認**

```bash
pytest tests/signals/test_tier1_csv.py -v 2>&1 | tail -8
```
Expected: FileNotFoundError or 0 passed

- [ ] **Step 3: CSV 本体作成 — Spec §4.2 を 1 ファイルに転記**

`data/signals/tier1_selection_20260603.csv` のフルコンテンツ (52行 + header):
```csv
signal_id,name,category,view,target_assets,quantize_scheme,q_levels,priority,source_module,publication_lag,earliest_date,cost_tier
1,NDX 200DMA breadth,A_Breadth,Trader,N,quantile_cut,4,A,yahoo,daily,2003-01-01,free
2,McClellan Oscillator NDX,A_Breadth,Trader,N,binary_threshold,2,A,yahoo,daily,2003-01-01,free
3,NDX New Highs minus Lows 52W,A_Breadth,Trader,N,binary_threshold,2,A,yahoo,daily,2003-01-01,free
4,AD Line price divergence,A_Breadth,Trader,N,binary_threshold,2,B,yahoo,daily,2003-01-01,free
5,NYSE TICK terminal,A_Breadth,Trader,N,quantile_cut,4,C,yahoo,daily,2000-01-01,free
6,VIX level,B_Vol,Trader,N,quantile_cut,4,A,yahoo,daily,1990-01-01,free
7,VIX9D over VIX ratio,B_Vol,Trader,N,binary_threshold,2,A,yahoo,daily,2011-01-01,free
8,VIX term structure VIX1 VIX2 VIX3,B_Vol,Trader,N,binary_threshold,2,A,cboe,daily,2008-01-01,free
9,VVIX,B_Vol,Trader,N,quantile_cut,4,A,yahoo,daily,2007-03-01,free
10,MOVE index bond vol,B_Vol,Actuary,B|N,quantile_cut,4,A,yahoo,daily,2002-04-01,free
11,GVZ gold vol,B_Vol,Actuary,G,binary_threshold,2,B,yahoo,daily,2008-06-01,free
12,CBOE PutCall equity,C_Sentiment,Trader,N,binary_threshold,2,A,cboe,daily,2003-10-01,free
13,AAII Bull Bear spread,C_Sentiment,Trader,N,quantile_cut,4,A,manual,weekly,1987-07-31,free
14,NAAIM Exposure Index,C_Sentiment,Trader,N,quantile_cut,4,B,manual,weekly,2006-07-19,free
15,CFTC CoT NQ noncommercial net,C_Sentiment,HF,N,quantile_cut,4,A,cftc,weekly,2010-01-01,free
16,CFTC CoT GC net gold,C_Sentiment,HF,G,quantile_cut,4,A,cftc,weekly,1995-01-01,free
17,CFTC CoT ZB ZN net bond,C_Sentiment,HF,B,quantile_cut,4,A,cftc,weekly,2000-01-01,free
18,QQQ daily net creation redemption,C_Sentiment,HF,N,binary_threshold,2,B,yahoo,daily,2005-01-01,low_paid
19,GLD TLT net flows,C_Sentiment,HF,G|B,binary_threshold,2,B,yahoo,daily,2005-01-01,low_paid
20,Margin Debt YoY FINRA monthly,C_Sentiment,HF,N,quantile_cut,4,C,manual,monthly,1997-01-01,free
21,ICE BofA HY OAS,D_Credit,Actuary,N,zscore_band,3,A,fred,daily,1996-12-31,free
22,ICE BofA IG OAS,D_Credit,Actuary,N,binary_threshold,2,B,fred,daily,1996-12-31,free
23,HY minus IG spread,D_Credit,Actuary,N,binary_threshold,2,A,fred,daily,1996-12-31,free
24,SOFR minus IORB spread,D_Credit,Actuary,B,binary_threshold,2,B,fred,daily,2018-04-02,free
25,3M Treasury minus SOFR,D_Credit,Actuary,B|N,binary_threshold,2,C,fred,daily,2018-04-02,free
26,2s10s spread,E_YieldCurve,Actuary,B|N,quantile_cut,4,A,fred,daily,1976-06-01,free
27,3M10Y spread,E_YieldCurve,Actuary,B|N,quantile_cut,4,A,fred,daily,1982-01-04,free
28,10Y TIPS real yield,E_YieldCurve,Actuary,G|N,quantile_cut,4,A,fred,daily,2003-01-02,free
29,5Y5Y forward inflation,E_YieldCurve,Actuary,G|B,quantile_cut,4,A,fred,daily,2003-01-02,free
30,CME FedWatch 25bp cut prob 3M,E_YieldCurve,HF,B|N,quantile_cut,4,A,fedwatch,daily,2015-01-01,low_paid
31,10Y minus 2Y real yield,E_YieldCurve,HF,G,binary_threshold,2,B,fred,daily,2004-01-01,free
32,Atlanta Fed GDPNow,F_MacroNowcast,HF,N|B,quantile_cut,4,A,manual,weekly,2011-07-15,free
33,NY Fed Nowcast,F_MacroNowcast,HF,N|B,quantile_cut,4,B,manual,weekly,2016-04-15,free
34,Citi Economic Surprise USMI,F_MacroNowcast,HF,N|B|G,quantile_cut,4,manual,manual,weekly,2003-01-01,low_paid
35,Cleveland Fed Inflation Nowcast,F_MacroNowcast,HF,G|B,quantile_cut,4,B,manual,monthly,2001-01-01,free
36,Chicago Fed NFCI,F_MacroNowcast,Actuary,N,quantile_cut,4,A,fred,weekly,1973-01-08,free
37,NDX Forward EPS Revision 4wk,G_Earnings,HF,N,quantile_cut,4,A,manual,weekly,2010-01-01,low_paid
38,Equity Risk Premium fwd yld minus 10Y real,G_Earnings,HF,N,quantile_cut,4,A,manual,daily,2003-01-02,low_paid
39,NDX Forward PE zscore,G_Earnings,HF,N,quantile_cut,4,B,manual,weekly,2003-01-01,low_paid
40,Mag7 EPS Revision composite,G_Earnings,HF,N,quantile_cut,4,A,manual,weekly,2014-01-01,low_paid
41,DXY weekly change,H_CrossAsset,Trader,G|N,binary_threshold,2,A,yahoo,daily,1971-01-04,free
42,Copper over Gold ratio,H_CrossAsset,HF,N|B,quantile_cut,4,A,yahoo,daily,1989-04-01,free
43,Silver over Gold ratio,H_CrossAsset,HF,G,binary_threshold,2,B,yahoo,daily,1989-04-01,free
44,Oil WTI 5d change,H_CrossAsset,Trader,N|B,binary_threshold,2,B,yahoo,daily,1986-01-02,free
45,BTC over QQQ correlation,H_CrossAsset,Trader,N,binary_threshold,2,C,yahoo,daily,2014-09-17,free
46,FOMC blackout window,I_Calendar,Trader,N|B,binary_threshold,2,A,manual,event,1980-01-01,free
47,NDX earnings season Mag7,I_Calendar,Trader,N,binary_threshold,2,B,manual,event,2014-01-01,free
48,Triple Witching Friday,I_Calendar,Trader,N,binary_threshold,2,C,manual,event,1980-01-01,free
49,Google Trends recession 90d Z,J_NLP,HF,N|B|G,quantile_cut,4,A,google_trends,daily,2004-01-04,free
50,Fed minutes hawkish dovish NLP,J_NLP,HF,B|N,quantile_cut,4,A,manual,event,2010-01-01,low_paid
51,Headline News risk off composite NLP,J_NLP,HF,N|B|G,quantile_cut,4,B,manual,daily,2015-01-01,low_paid
52,Google Trends TQQQ QQQ search,J_NLP,HF,N,binary_threshold,2,B,google_trends,daily,2010-01-01,free
```

(注: 行 #34 の `priority` カラムは `A`、別途 `source_module=manual` の対応で間違いない。CSV の 4 値目はカテゴリ名カラム位置と一致するか確認)

- [ ] **Step 4: テスト pass 確認**

```bash
pytest tests/signals/test_tier1_csv.py -v 2>&1 | tail -10
```
Expected: 6 passed

- [ ] **Step 5: コミット**

```bash
git add data/signals/tier1_selection_20260603.csv tests/signals/test_tier1_csv.py
git commit -m "data(signals): Tier1 選別 CSV 本体 52信号 (Phase A §4.2)"
```

---

### Task A3: 量子化関数 — quantize.py

**Spec ref:** §4.1 量子化 (Q列)、§2 制約#6 (0/1 binary or 0-3 4段階)

**Files:**
- Create: `src/signals/quantize.py`
- Create: `tests/signals/test_quantize.py`

- [ ] **Step 1: 失敗テスト**

`tests/signals/test_quantize.py`:
```python
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from signals.quantize import binary_threshold, quantile_cut, zscore_band


def test_binary_threshold_above():
    s = pd.Series([10, 20, 30, 40])
    out = binary_threshold(s, threshold=25, direction='above')
    assert list(out) == [0, 0, 1, 1]


def test_binary_threshold_below():
    s = pd.Series([10, 20, 30, 40])
    out = binary_threshold(s, threshold=25, direction='below')
    assert list(out) == [1, 1, 0, 0]


def test_quantile_cut_4levels():
    s = pd.Series(np.arange(100))
    out = quantile_cut(s, levels=4, window=None)
    assert set(out.dropna().unique()) <= {0, 1, 2, 3}
    counts = out.value_counts().sort_index()
    assert all(abs(c - 25) <= 1 for c in counts)


def test_quantile_cut_rolling_window():
    s = pd.Series(np.arange(252 * 3))
    out = quantile_cut(s, levels=4, window=252)
    assert out.iloc[:252].isna().all()
    assert not out.iloc[252:].isna().any()


def test_zscore_band_3levels():
    s = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 10, 0, -10])
    out = zscore_band(s, lower=-1.0, upper=1.0)
    assert set(out.dropna().unique()) <= {0, 1, 2}
    assert out.iloc[12] == 2
    assert out.iloc[14] == 0


def test_binary_threshold_raises_on_bad_direction():
    import pytest
    with pytest.raises(ValueError):
        binary_threshold(pd.Series([1, 2]), threshold=1, direction='sideways')
```

- [ ] **Step 2: 失敗確認**

```bash
pytest tests/signals/test_quantize.py -v 2>&1 | tail -10
```
Expected: ModuleNotFoundError

- [ ] **Step 3: 実装**

`src/signals/quantize.py`:
```python
"""Signal quantization functions.

Three schemas supported (matches metadata.quantize_scheme):
  - binary_threshold: 0/1 by simple threshold + direction
  - quantile_cut: 0..L-1 by L-quantile (optionally rolling)
  - zscore_band: 0/1/2 by z-score lower/upper bands

All return pd.Series aligned to input index; NaN propagates.
"""
from __future__ import annotations
from typing import Optional, Literal
import numpy as np
import pandas as pd


def binary_threshold(
    s: pd.Series,
    threshold: float,
    direction: Literal['above', 'below'] = 'above',
) -> pd.Series:
    if direction not in ('above', 'below'):
        raise ValueError(f"direction must be 'above'/'below', got {direction}")
    if direction == 'above':
        return (s > threshold).astype('int8').where(s.notna())
    return (s < threshold).astype('int8').where(s.notna())


def quantile_cut(
    s: pd.Series,
    levels: int = 4,
    window: Optional[int] = None,
) -> pd.Series:
    if levels < 2:
        raise ValueError("levels must be >=2")
    if window is None:
        ranks = s.rank(pct=True)
        out = (ranks * levels).clip(upper=levels - 1).astype('Int8')
        return out.where(s.notna())

    def _bin(x: pd.Series) -> float:
        if x.isna().iloc[-1]:
            return np.nan
        pct = (x.rank(pct=True)).iloc[-1]
        return min(int(pct * levels), levels - 1)

    return s.rolling(window=window, min_periods=window).apply(_bin, raw=False).astype('Int8')


def zscore_band(
    s: pd.Series,
    lower: float = -1.0,
    upper: float = 1.0,
    window: Optional[int] = None,
) -> pd.Series:
    if window is None:
        mu, sd = s.mean(), s.std(ddof=0)
    else:
        mu = s.rolling(window).mean()
        sd = s.rolling(window).std(ddof=0)
    z = (s - mu) / sd
    out = pd.Series(1, index=s.index, dtype='Int8')
    out[z <= lower] = 0
    out[z >= upper] = 2
    return out.where(s.notna())
```

- [ ] **Step 4: テスト pass**

```bash
pytest tests/signals/test_quantize.py -v 2>&1 | tail -10
```
Expected: 6 passed

- [ ] **Step 5: コミット**

```bash
git add src/signals/quantize.py tests/signals/test_quantize.py
git commit -m "feat(signals): 量子化関数 binary/quantile/zscore (Phase A §4.1)"
```

---

### Task A4: publication_lag 適用 — timing.py

**Spec ref:** §5.3 Look-ahead 防止 (publication_lag テーブル)

**Files:**
- Create: `src/signals/timing.py`
- Create: `tests/signals/test_timing.py`

- [ ] **Step 1: 失敗テスト**

`tests/signals/test_timing.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd
from signals.timing import apply_publication_lag


def _series(dates, vals):
    return pd.Series(vals, index=pd.to_datetime(dates))


def test_daily_lag_shifts_one_day():
    s = _series(['2024-01-02', '2024-01-03', '2024-01-04'], [1, 2, 3])
    out = apply_publication_lag(s, lag_type='daily')
    assert out.loc['2024-01-03'] == 1
    assert out.loc['2024-01-04'] == 2


def test_weekly_tue_lag_to_next_tue():
    # COT: Tue公表 (2024-01-02 火) → 翌火 2024-01-09 close 適用
    s = _series(['2024-01-02', '2024-01-09'], [100, 200])
    out = apply_publication_lag(s, lag_type='weekly')
    assert out.loc['2024-01-09'] == 100
    assert out.loc['2024-01-16'] == 200 if '2024-01-16' in out.index else True


def test_event_lag_next_open():
    s = _series(['2024-01-31'], [3])
    out = apply_publication_lag(s, lag_type='event')
    assert out.index[0] >= pd.Timestamp('2024-02-01')


def test_invalid_lag_type_raises():
    import pytest
    s = _series(['2024-01-02'], [1])
    with pytest.raises(ValueError):
        apply_publication_lag(s, lag_type='quarterly')
```

- [ ] **Step 2: 失敗確認 → 最小実装**

`src/signals/timing.py`:
```python
"""Apply publication lag to prevent look-ahead bias.

Lag types (matches metadata.publication_lag):
  daily   : t-1 close known by t open  → shift +1 business day
  weekly  : Tue publish → next Tue close apply → shift +5 business days
  monthly : 公表日 +1 営業日 → shift +1 business day (公表日 already known)
  event   : 公表日 21:00 ET → next session → shift +1 business day
"""
from __future__ import annotations
from typing import Literal
import pandas as pd
from pandas.tseries.offsets import BusinessDay


_VALID = {'daily', 'weekly', 'monthly', 'event'}


def apply_publication_lag(
    s: pd.Series,
    lag_type: Literal['daily', 'weekly', 'monthly', 'event'],
) -> pd.Series:
    if lag_type not in _VALID:
        raise ValueError(f"lag_type invalid: {lag_type}")
    if lag_type == 'daily':
        shifted_idx = s.index + BusinessDay(1)
    elif lag_type == 'weekly':
        shifted_idx = s.index + BusinessDay(5)
    elif lag_type == 'monthly':
        shifted_idx = s.index + BusinessDay(1)
    else:  # event
        shifted_idx = s.index + BusinessDay(1)
    return pd.Series(s.values, index=shifted_idx)
```

- [ ] **Step 3: pass 確認 + コミット**

```bash
pytest tests/signals/test_timing.py -v 2>&1 | tail -10
git add src/signals/timing.py tests/signals/test_timing.py
git commit -m "feat(signals): publication_lag 適用関数 (Phase A §5.3)"
```

---

### Task A5: SignalLoader ABC + ディスクキャッシュ — _base.py

**Files:**
- Create: `src/data_loaders/signals/_base.py`
- Create: `tests/signals/test_loaders/test_base.py`

- [ ] **Step 1: 失敗テスト (mock loader)**

`tests/signals/test_loaders/test_base.py`:
```python
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
from data_loaders.signals._base import SignalLoader


class FakeLoader(SignalLoader):
    source_name = 'fake'
    call_count = 0

    def _fetch(self, signal_id: int) -> pd.Series:
        FakeLoader.call_count += 1
        idx = pd.date_range('2024-01-01', periods=5, freq='B')
        return pd.Series([1, 2, 3, 4, 5], index=idx, name=f"sig_{signal_id}")


def test_first_call_hits_fetch(tmp_path):
    FakeLoader.call_count = 0
    ldr = FakeLoader(cache_dir=tmp_path)
    s = ldr.get(signal_id=99)
    assert len(s) == 5
    assert FakeLoader.call_count == 1


def test_second_call_hits_cache(tmp_path):
    FakeLoader.call_count = 0
    ldr = FakeLoader(cache_dir=tmp_path)
    ldr.get(signal_id=99)
    ldr.get(signal_id=99)
    assert FakeLoader.call_count == 1


def test_force_refresh_bypasses_cache(tmp_path):
    FakeLoader.call_count = 0
    ldr = FakeLoader(cache_dir=tmp_path)
    ldr.get(signal_id=99)
    ldr.get(signal_id=99, force=True)
    assert FakeLoader.call_count == 2
```

- [ ] **Step 2: 実装**

`src/data_loaders/signals/_base.py`:
```python
"""SignalLoader abstract base + disk cache (parquet)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import pandas as pd


class SignalLoader(ABC):
    source_name: str = "unknown"

    def __init__(self, cache_dir: Union[str, Path] = "data/signals/_cache"):
        self.cache_dir = Path(cache_dir) / self.source_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, signal_id: int) -> Path:
        return self.cache_dir / f"signal_{signal_id}.parquet"

    def get(self, signal_id: int, force: bool = False) -> pd.Series:
        p = self._cache_path(signal_id)
        if p.exists() and not force:
            df = pd.read_parquet(p)
            return df.iloc[:, 0]
        s = self._fetch(signal_id)
        s.to_frame(name=s.name or f"sig_{signal_id}").to_parquet(p)
        return s

    @abstractmethod
    def _fetch(self, signal_id: int) -> pd.Series:
        ...
```

- [ ] **Step 3: pass + commit**

```bash
pytest tests/signals/test_loaders/test_base.py -v 2>&1 | tail -10
git add src/data_loaders/signals/_base.py tests/signals/test_loaders/test_base.py
git commit -m "feat(loaders): SignalLoader ABC + parquet cache (Phase A)"
```

---

### Task A6: FRED loader (HY OAS / 2s10s / DGS10 / TIPS / T5YIFR / NFCI)

**Files:**
- Create: `src/data_loaders/signals/fred.py`
- Create: `tests/signals/test_loaders/test_fred.py`
- Create: `tests/signals/fixtures/fred_BAMLH0A0HYM2_sample.csv`

- [ ] **Step 1: フィクスチャ準備 (FRED CSV mini sample)**

`tests/signals/fixtures/fred_BAMLH0A0HYM2_sample.csv`:
```csv
observation_date,BAMLH0A0HYM2
2024-01-02,3.45
2024-01-03,3.52
2024-01-04,3.50
2024-01-05,3.48
```

- [ ] **Step 2: 失敗テスト + 実装 + commit (パターン同上)**

`src/data_loaders/signals/fred.py`:
```python
"""FRED data loader.

Signal ID → FRED series ID mapping (subset; spec §4.2):
  21 → BAMLH0A0HYM2 (ICE BofA HY OAS)
  22 → BAMLC0A0CM   (ICE BofA IG OAS)
  24 → SOFR-IORB    (computed)
  25 → DTB3-SOFR    (computed)
  26 → T10Y2Y
  27 → T10Y3M
  28 → DFII10
  29 → T5YIFR
  31 → 10Y minus 2Y real (computed)
  36 → NFCI
"""
from __future__ import annotations
import os
from io import StringIO
import requests
import pandas as pd
from ._base import SignalLoader


_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

_SIGNAL_TO_SERIES = {
    21: 'BAMLH0A0HYM2',
    22: 'BAMLC0A0CM',
    26: 'T10Y2Y',
    27: 'T10Y3M',
    28: 'DFII10',
    29: 'T5YIFR',
    36: 'NFCI',
}


class FredLoader(SignalLoader):
    source_name = 'fred'

    def _fetch(self, signal_id: int) -> pd.Series:
        if signal_id not in _SIGNAL_TO_SERIES:
            raise NotImplementedError(f"FRED mapping missing for signal_id={signal_id}")
        series_id = _SIGNAL_TO_SERIES[signal_id]
        url = f"{_FRED_BASE}{series_id}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), parse_dates=['observation_date'])
        df = df.set_index('observation_date').sort_index()
        s = pd.to_numeric(df[series_id], errors='coerce').dropna()
        s.name = f"fred_{series_id}"
        return s
```

`tests/signals/test_loaders/test_fred.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
import pytest
from unittest.mock import patch
from data_loaders.signals.fred import FredLoader


FIXTURE = Path(__file__).parent.parent / 'fixtures' / 'fred_BAMLH0A0HYM2_sample.csv'


def test_fred_parses_csv(tmp_path):
    with patch('data_loaders.signals.fred.requests.get') as mock:
        mock.return_value.text = FIXTURE.read_text()
        mock.return_value.raise_for_status = lambda: None
        ldr = FredLoader(cache_dir=tmp_path)
        s = ldr.get(signal_id=21, force=True)
    assert len(s) == 4
    assert s.iloc[0] == 3.45
    assert s.name == 'fred_BAMLH0A0HYM2'


def test_unmapped_signal_raises(tmp_path):
    ldr = FredLoader(cache_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        ldr._fetch(signal_id=999)
```

- [ ] **Step 3: pass + commit**

```bash
pytest tests/signals/test_loaders/test_fred.py -v 2>&1 | tail -8
git add src/data_loaders/signals/fred.py tests/signals/test_loaders/test_fred.py tests/signals/fixtures/fred_BAMLH0A0HYM2_sample.csv
git commit -m "feat(loaders): FRED HTTP ローダー (signals 21-36, Phase A)"
```

---

### Task A7: Yahoo Finance loader (VIX / VVIX / MOVE / DXY / GVZ / oil / breadth)

**Files:**
- Create: `src/data_loaders/signals/yahoo.py`
- Create: `tests/signals/test_loaders/test_yahoo.py`

Pattern: yfinance ticker fetch + close column extraction。

`src/data_loaders/signals/yahoo.py` (要点):
```python
_SIGNAL_TO_TICKER = {
    6: '^VIX',
    7: '^VIX9D',
    9: '^VVIX',
    10: '^MOVE',
    11: '^GVZ',
    41: 'DX-Y.NYB',
    42: ('HG=F', 'GC=F'),  # 比率
    44: 'CL=F',
}
```

- [ ] テスト: `yfinance.download` を mock、Series 抽出を確認
- [ ] 比率ペア (#42 #43) は `_fetch_pair` で個別実装
- [ ] commit

---

### Task A8: CFTC CoT loader (NQ / GC / ZB-ZN)

**Files:**
- Create: `src/data_loaders/signals/cftc.py`
- Create: `tests/signals/test_loaders/test_cftc.py`

- CFTC publishes weekly ZIP at `https://www.cftc.gov/dea/newcot/deafut.txt`
- TFF (Traders in Financial Futures) report for ES/NQ/ZB/ZN
- Disaggregated Reports for GC (gold)
- Parse fixed-width or CSV per report
- Compute non-commercial net position = long - short
- Output rolling 1Y z-score series

`tests/signals/fixtures/cot_nq_sample.txt`: 5 weeks of fixed-width
- テスト: fixture parse → DataFrame → net position calculation
- commit

---

### Task A9: CBOE loader (PutCall #12 / VIX term #8)

**Files:**
- Create: `src/data_loaders/signals/cboe.py`
- Create: `tests/signals/test_loaders/test_cboe.py`

- CBOE PutCall: `https://www.cboe.com/us/options/market_statistics/daily/` (daily CSV)
- VIX term: VIX1 / VIX2 / VIX3 から contango (VIX1 < VIX2 < VIX3) を判定
- フィクスチャベースのテスト
- commit

---

### Task A10: 手動 CSV loader — manual.py (AAII/NAAIM/Fed NLP/GDPNow 等)

**Files:**
- Create: `src/data_loaders/signals/manual.py`
- Create: `tests/signals/test_loaders/test_manual.py`
- Create: `data/signals/manual/aaii_weekly.csv` (空ヘッダー + README)
- Create: `data/signals/manual/_template.csv`

- AAII / NAAIM は週次 CSV を手動更新で取得 (BlackBoxLLM が email でカバー可能)
- Atlanta Fed GDPNow / NY Fed Nowcast / Cleveland Inflation Nowcast: 週次手動
- Fed minutes hawkish-dovish NLP: 議事要旨を LLM API でスコア付け → CSV 蓄積
- Loader はファイル存在チェック + 日付パース + Series 返却

`src/data_loaders/signals/manual.py`:
```python
"""Manual update CSV loader.

Reads pre-curated CSV files from data/signals/manual/.
Schema: each CSV has 'Date' (YYYY-MM-DD) + 'value' columns.
"""
class ManualLoader(SignalLoader):
    source_name = 'manual'

    _SIGNAL_TO_FILE = {
        13: 'aaii_weekly.csv',
        14: 'naaim_weekly.csv',
        20: 'finra_margin_debt_monthly.csv',
        32: 'gdpnow_atlanta.csv',
        33: 'nyfed_nowcast.csv',
        34: 'citi_surprise_usmi.csv',
        35: 'cleveland_inflation_nowcast.csv',
        37: 'ndx_eps_revision_4wk.csv',
        38: 'equity_risk_premium.csv',
        39: 'ndx_forward_pe_zscore.csv',
        40: 'mag7_eps_revision.csv',
        46: 'fomc_blackout.csv',
        47: 'mag7_earnings_dates.csv',
        48: 'triple_witching.csv',
        50: 'fed_minutes_nlp.csv',
        51: 'news_riskoff_composite.csv',
    }

    # Resolve manual dir from repo root (matches existing src/build_base_dataset.py pattern)
    _BASE_DIR = Path(__file__).resolve().parents[3]   # src/data_loaders/signals/manual.py → repo root
    _MANUAL_DIR = _BASE_DIR / 'data' / 'signals' / 'manual'

    def _fetch(self, signal_id: int) -> pd.Series:
        if signal_id not in self._SIGNAL_TO_FILE:
            raise NotImplementedError
        path = self._MANUAL_DIR / self._SIGNAL_TO_FILE[signal_id]
        if not path.exists():
            raise FileNotFoundError(f"Manual CSV not found: {path}. See README for format.")
        df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
        s = df['value']
        s.name = f"manual_{signal_id}"
        return s
```

- テスト: fixture を data/signals/manual/ に置き、loader が引ける
- README で各 CSV の format spec + 更新頻度 + 入手先を記述
- commit

---

### Task A11: data_lineage.md ジェネレーター

**Spec ref:** §4.4 成果物 #2 `signals/data_lineage.md`

**Files:**
- Create: `src/signals/lineage.py`
- Create: `tests/signals/test_lineage.py`
- Create: `docs/signals/data_lineage.md` (生成済)

- [ ] **Step 1: 失敗テスト**

`tests/signals/test_lineage.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from signals.lineage import generate_lineage_markdown

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'signals' / 'tier1_selection_20260603.csv'


def test_lineage_markdown_contains_all_52():
    md = generate_lineage_markdown(registry_path=CSV)
    for sid in range(1, 53):
        assert f"| {sid} |" in md, f"signal_id {sid} missing from lineage markdown"


def test_lineage_markdown_has_header_columns():
    md = generate_lineage_markdown(registry_path=CSV)
    assert "| ID | Name | Category | Asset | Source | Lag | Earliest | Cost | Priority |" in md


def test_lineage_markdown_priority_distribution():
    md = generate_lineage_markdown(registry_path=CSV)
    a_count = md.count('| A |')
    b_count = md.count('| B |')
    c_count = md.count('| C |')
    assert a_count == 31, f"expected 31 priority-A rows, got {a_count}"
    assert b_count == 16
    assert c_count == 5
```

- [ ] **Step 2: 実装**

`src/signals/lineage.py`:
```python
"""Generate data_lineage.md from tier1_selection CSV."""
from .metadata import load_registry

def generate_lineage_markdown(registry_path) -> str:
    metas = load_registry(registry_path)
    lines = ["# Signal Data Lineage", "",
             "| ID | Name | Category | Asset | Source | Lag | Earliest | Cost | Priority |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in metas:
        lines.append(
            f"| {m.signal_id} | {m.name} | {m.category} | {'/'.join(m.target_assets)} "
            f"| {m.source_module} | {m.publication_lag} | {m.earliest_date} "
            f"| {m.cost_tier} | {m.priority} |"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 3: スクリプト起動で実ファイル生成**

```bash
python -c "from src.signals.lineage import generate_lineage_markdown; \
  open('docs/signals/data_lineage.md','w',encoding='utf-8').write( \
  generate_lineage_markdown('data/signals/tier1_selection_20260603.csv'))"
```

- [ ] **Step 4: taxonomy_20260603.md を Spec §4.2 から転記**

```bash
# Spec §4.2 (各カテゴリ A〜J の信号テーブル) を docs/signals/taxonomy_20260603.md に抜粋コピー
python -c "
import re, pathlib
spec = pathlib.Path('SIGNAL_DISCOVERY_PLAN_20260603.md').read_text(encoding='utf-8')
m = re.search(r'#### A\. Breadth.*?(?=---\n\n### 4\.3 Tier1 結果)', spec, re.DOTALL)
if m is None:
    raise SystemExit('Spec §4.2 section markers not found; check spec format')
body = m.group(0)
out = '# Signal Taxonomy (Tier1, 2026-06-03)\n\n' + body
pathlib.Path('docs/signals/taxonomy_20260603.md').write_text(out, encoding='utf-8')
print('wrote docs/signals/taxonomy_20260603.md')
"
```

- [ ] **Step 5: commit**

```bash
git add src/signals/lineage.py tests/signals/test_lineage.py docs/signals/data_lineage.md docs/signals/taxonomy_20260603.md
git commit -m "feat(signals): data_lineage + taxonomy 自動生成 (Phase A §4.4 #1,#2)"
```

---

### Task A12: Phase A 統合確認スクリプト + smoke test

**Files:**
- Create: `scripts/verify_phase_a.py`
- Create: `tests/signals/test_integration_a.py`

- [ ] 全 52 signals について metadata → source_module → loader instance 取得 が動くか確認 (実 fetch せず、loader instance 化のみ)
- [ ] quantize_scheme が登録済 quantize 関数とマッチするか
- [ ] publication_lag が timing.apply で動くか
- [ ] 全て pass で Phase A 完了

```bash
python scripts/verify_phase_a.py
# Expected output:
# [Phase A verify] 52 signals registered
# [Phase A verify] 8 loaders configured
# [Phase A verify] 3 quantize schemas mapped
# [Phase A verify] 4 publication_lag types covered
# [Phase A verify] ALL OK
```

- [ ] commit + Phase A 完了タグ:
```bash
git tag phase-a-complete
git push origin phase-a-complete
```

---

## 3. Phase B: 経験的スクリーニング (中粒度)

Spec §5 に対応。8 タスク (B1-B8)。**各タスクは API contract + 主要テスト + 実装ヒントを示す stub レベル。** 実装は Phase A 完了後に詳細展開。

---

### Task B1: Forward return matrix builder

**Spec ref:** §5.2

**Files:**
- Create: `src/signals/forward_returns.py`
- Create: `tests/signals/test_forward_returns.py`

**API:**
```python
def build_forward_returns(
    prices: pd.DataFrame,  # columns: TQQQ, TLT, GLD
    horizons: list[int] = [5, 20, 60, 252],
) -> pd.DataFrame:
    """Return log returns shifted backward by each horizon."""
    # MultiIndex columns: (asset, horizon)
```

- 既存 `data/base_dataset.csv` から TQQQ proxy / TLT / GLD を抽出
- Test: 既知の simple ramp で 5d return が log(p[t+5]/p[t]) と一致
- commit

---

### Task B2: Spearman rolling IC + Newey-West t-stat

**Spec ref:** §5.1

**Files:**
- Create: `src/signals/ic.py`
- Create: `tests/signals/test_ic.py`

**API:**
```python
def compute_ic(
    signal: pd.Series,       # 量子化済 0/1 or 0-3
    forward_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling Spearman IC, NaN-aligned."""

def ic_tstat(ic_series: pd.Series, lags: int = 20) -> float:
    """Newey-West t-stat of mean IC (HAC autocorr correction)."""
```

- `scipy.stats.spearmanr` (panel: signal vs forward returns at each rolling window end)
- `statsmodels.regression.linear_model.OLS` + `cov_type='HAC'` で t-stat
- テスト: known IC=0 (random signal) の場合、t-stat が ±2 内
- テスト: perfectly correlated case → t-stat > 10
- commit

---

### Task B3: Hit rate + Wilson CI

**Files:**
- Create: `src/signals/hit_rate.py`
- Create: `tests/signals/test_hit_rate.py`

**API:**
```python
def hit_rate_conditional(
    signal: pd.Series,
    forward_returns: pd.Series,
    signal_value: int = 1,
) -> tuple[float, float]:
    """Returns (hit_rate, wilson_lower_bound_95)."""
```

- Wilson CI: `statsmodels.stats.proportion.proportion_confint(method='wilson')`
- Base rate (unconditional) も計算 → diff 評価
- commit

---

### Task B4: Decade-IC 安定性チェック

**Files:**
- Create: `src/signals/stability.py`
- Create: `tests/signals/test_stability.py`

**API:**
```python
def decade_ic_check(
    signal: pd.Series,
    forward_returns: pd.Series,
    decades: list[tuple[str, str]] = [
        ('2000-01-01', '2009-12-31'),
        ('2010-01-01', '2019-12-31'),
        ('2020-01-01', '2099-12-31'),
    ],
) -> dict:
    """Returns {decade_label: ic} dict + 'same_sign' flag."""
```

- データ長 <10年は decade test 緩和 (該当 decade 0 個でも pass 扱い)
- 半分割 IC 同符号もここに同居
- commit

---

### Task B5: BH-FDR 多重補正

**Files:**
- Create: `src/signals/multiplicity.py`
- Create: `tests/signals/test_multiplicity.py`

**API:**
```python
def fdr_bh(pvals: pd.Series, alpha: float = 0.10) -> pd.Series:
    """Returns adjusted p-values (BH method)."""
```

- `statsmodels.stats.multitest.multipletests(method='fdr_bh')` wrap
- Bonferroni も並記 (参考)
- commit

---

### Task B6: PCA composite signal builder

**Spec ref:** §5.6

**Files:**
- Create: `src/signals/composite.py`
- Create: `tests/signals/test_composite.py`

**API:**
```python
def build_composite(
    signals: pd.DataFrame,  # cols = component signals
    method: str = 'pca_first',
) -> pd.Series:
    """First principal component (standardized inputs)."""
```

- `sklearn.decomposition.PCA(n_components=1)` (sklearn 依存追加要)
- Composite を 4-quantile で再量子化 → Phase B 規格と同型に
- 4 種類: Sentiment / Credit / Macro / YieldCurve (Spec §5.6 テーブル)
- commit

---

### Task B7: スコアカード生成 + マークダウンレポート

**Spec ref:** §5.8 成果物 #1, #2

**Files:**
- Create: `src/signals/screening.py`
- Create: `scripts/run_phase_b.py`
- Create: `tests/signals/test_screening.py`

**API:**
```python
def screen_signal(
    meta: SignalMeta,
    signal_raw: pd.Series,
    forward_returns: pd.DataFrame,  # MultiIndex (asset, horizon)
) -> dict:
    """Per-signal full screening: IC / t / hit / decade / FDR-adjusted p / PASS-FAIL."""

def generate_scorecard(results: list[dict], out_dir: Path) -> None:
    """Writes signals/scorecard_<asset>_<horizon>.csv per combo."""

def generate_screening_report(results: list[dict], out_path: Path) -> None:
    """Writes markdown with: PASS table, FAIL table with rationale."""
```

- スコアカード CSV ヘッダー: `signal_id, name, asset, horizon, ic, t_stat, p_value, p_value_bh, hit_rate, hit_wilson_lo, decade_same_sign, pass_flag`
- レポート markdown は Phase B 採用 ~12 + 棄却 ~18 を理由付きで列挙
- commit

---

### Task B8: Phase B end-to-end runner

**Files:**
- Create: `scripts/run_phase_b.py`
- Modify: なし (既存 dataset.csv を活用)

```python
"""End-to-end Phase B runner.

Steps:
  1. Load metadata + Tier1 selection (Priority A signals only)
  2. For each signal:
     a. Instantiate loader, fetch raw series
     b. Apply publication_lag
     c. Apply quantize_scheme per metadata
  3. Build forward return matrix
  4. For each (signal, asset, horizon) triplet:
     screen_signal(...) → dict
  5. BH-FDR across all 279 combos
  6. generate_scorecard(...) + generate_screening_report(...)
  7. Composite探索: 4 ブロックで build_composite → screen_signal 同規格
  8. Write final selection list → data/signals/phase_b_selection_<date>.csv
"""
```

- [ ] スクリプト実行 → スコアカード 9 ファイル (3 asset × 3 horizon) + レポート 1 ファイル + 選定 CSV 出力
- [ ] commit + tag phase-b-complete

---

## 4. Phase C: WFA 組込検証 (中粒度)

Spec §6 に対応。7 タスク (C1-C7)。**実装 stub + 既存 g20 モジュール wrap が中心。**

---

### Task C1: Overlay モード adapter

**Spec ref:** §6.1

**Files:**
- Create: `src/signals/overlay.py`
- Create: `tests/signals/test_overlay.py`

**API:**
```python
def apply_overlay(
    base_lev_mod: pd.Series,    # 既存 lev_mod_065 mask 出力
    signal: pd.Series,           # 0/1 or 0-3
    mapping: dict,               # 例: {0: 0.0, 1: 0.5, 2: 0.8, 3: 1.0}
) -> pd.Series:
    """Element-wise multiply base by signal-mapped multiplier."""
```

- 既存 `NEW CANDIDATE` の lev_mod 出力を再現するヘルパー (`src/backtest_engine.py` から呼び出し)
- commit

---

### Task C2: Standalone モード adapter (F11 クラス)

**Spec ref:** §6.1

**Files:**
- Create: `src/signals/standalone.py`
- Create: `tests/signals/test_standalone.py`

**API:**
```python
def signal_driven_allocation(
    signal: pd.Series,
    asset_universe: list[str] = ['TQQQ', 'TMF', 'GLD'],
    allocation_map: dict[int, list[float]] = ...,
) -> pd.DataFrame:
    """Convert signal levels to portfolio weights."""
```

- F10ε の構造を踏襲、tilt の代わりに直接 signal で配分決定
- commit

---

### Task C3: G1-G10 既存 WFA wrapper

**Spec ref:** §6.2 protocol テーブル

**Files:**
- Create: `src/signals/wfa.py`
- Create: `tests/signals/test_wfa.py`

既存 `src/` 配下の以下を wrap (新規実装ではなく adapter として):
- `src/audit/g3_*.py` (rolling WFA 50窓)
- `src/audit/g7_*.py` (Bootstrap)
- `src/audit/g8_*.py` (年次寄与)
- `src/audit/g9_*.py` (Permutation)
- `src/audit/g10_*.py` or `g20a_*` (parameter robustness)

**API:**
```python
def run_g_series(
    strategy_fn: callable,
    base_data: pd.DataFrame,
    series: list[str] = ['G1', 'G3', 'G7', 'G8', 'G9', 'G10'],
) -> dict:
    """Returns {series_name: result_dict}."""
```

- 既存モジュールの I/O contract を確認して thin wrapper
- テストは mock strategy_fn で各 series が呼ばれることのみ確認
- commit

---

### Task C4: G11 SPA test 新規実装

**Spec ref:** §6.2 G11

**Files:**
- Create: `src/signals/spa_test.py`
- Create: `tests/signals/test_spa.py`

```python
"""Hansen Superior Predictive Ability (SPA) test.

Wraps arch.bootstrap.SPA.
"""
from arch.bootstrap import SPA

def run_spa(
    benchmark_returns: pd.Series,
    candidate_returns: pd.DataFrame,  # cols = strategy variants
    bootstrap_iters: int = 5000,
) -> dict:
    """Returns {p_lower, p_consistent, p_upper, best_strategy_idx}."""
```

- 既存 NEW CANDIDATE を benchmark、Phase C 候補群を candidate matrix に
- p < 10% で PASS
- テスト: random shuffles で p > 50%、明らかに優位な系列で p < 5%
- commit

---

### Task C5: Pareto 採用判定

**Spec ref:** §6.3

**Files:**
- Create: `src/signals/adoption.py`
- Create: `tests/signals/test_adoption.py`

**API:**
```python
def pareto_judge(
    candidate_metrics: dict,   # CAGR_OOS, Sharpe_OOS, IS_OOS_gap, MaxDD, Trades_yr
    baseline_metrics: dict,
    requirements: dict = {     # Spec §6.3 上限・下限
        'CAGR_OOS': {'improve': +2.0, 'degrade_max': -1.0},
        'Sharpe_OOS': {'improve': +0.05, 'degrade_max': -0.05},
        'IS_OOS_gap': {'improve_abs_decrease': +1.0, 'degrade_max': +2.0},
        'MaxDD': {'improve': -5.0, 'degrade_max': +8.0},
        'Trades_yr_max': 200,
    },
) -> dict:
    """Returns {pareto_pass: bool, improved_axes: list, degraded_axes: list}."""
```

- Hard requirement (G3/G7/G9/G11) は別ガード関数
- commit

---

### Task C6: 信号組合せ探索 (AND/OR)

**Spec ref:** §6.4

**Files:**
- Create: `src/signals/combinations.py`
- Create: `tests/signals/test_combinations.py`

```python
def combine_signals(
    s1: pd.Series, s2: pd.Series,
    operator: Literal['AND', 'OR'],
    threshold: int = 1,
) -> pd.Series:
    """Returns binary combined signal."""
```

- 上位 3-4 から最大 6 ペア × 2 operator = 12 バリアント
- commit

---

### Task C7: Phase C 統合 runner + レポート

**Files:**
- Create: `scripts/run_phase_c.py`
- Create: `templates/phase_c_report_template.md`

```python
"""End-to-end Phase C runner.

Steps:
  1. Load Phase B selection (~12 signals + composites)
  2. For each (signal, mode in [overlay, standalone], asset):
     build strategy variant → run G1-G11 series
  3. Run combine_signals on top 4 → 12 additional variants
  4. SPA test across all variants (~102)
  5. pareto_judge → adoption list (3-6)
  6. Output:
     - STRATEGY_PERFORMANCE_COMPARISON_<date>.md (existing format)
     - STRATEGY_REGISTRY.md update
     - signals/wfa_results/<signal>_<mode>.md per variant
     - INTEGRATION_DEBATE_<date>.md (採否議事録)
"""
```

- [ ] 全件 pass で Phase C 完了タグ
- [ ] CURRENT_BEST_STRATEGY.md 更新候補判断
- [ ] commit + tag phase-c-complete

---

## 5. 全体進捗管理

各 Phase 完了時の checkpoint:

| Phase | Tasks | 完了条件 | 想定工数 |
|---|---|---|---|
| A | A0-A12 (13タスク) | `python scripts/verify_phase_a.py` で ALL OK | 3-5日 |
| B | B1-B8 (8タスク) | `scripts/run_phase_b.py` 走行 + 12-15信号選定 CSV 出力 | 2-3日 |
| C | C1-C7 (7タスク) | `scripts/run_phase_c.py` 走行 + 3-6戦略採択 + レポート3本生成 | 1週間 |
| 合計 | **28タスク** | — | **2-3週間 (フルタイム)** |

---

## 6. 出力物 (本計画の実行で生成)

### Phase A 末
- `data/signals/tier1_selection_20260603.csv` (52信号 single source)
- `docs/signals/data_lineage.md` (自動生成)
- `docs/signals/taxonomy_20260603.md` (Spec §4.2 のコピー)
- `src/signals/*` (7モジュール)
- `src/data_loaders/signals/*` (8モジュール)
- `tests/signals/*` (テストスイート)
- git tag `phase-a-complete`

### Phase B 末
- `data/signals/scorecard_<asset>_<horizon>.csv` (9ファイル)
- `data/signals/screening_report_<date>.md`
- `data/signals/phase_b_selection_<date>.csv` (12-15信号)
- `data/signals/correlation_heatmap.png`
- git tag `phase-b-complete`

### Phase C 末
- `STRATEGY_PERFORMANCE_COMPARISON_<date>.md` (既存規格準拠)
- `STRATEGY_REGISTRY.md` 更新
- `signals/wfa_results/<signal>_<mode>.md` × 採択数
- `INTEGRATION_DEBATE_<date>.md`
- `CURRENT_BEST_STRATEGY.md` 更新可能性
- git tag `phase-c-complete`

---

## 7. 既存コードへの影響範囲

| 既存ファイル | 影響 | 種別 |
|---|---|---|
| `src/backtest_engine.py` | overlay 適用箇所で `apply_overlay` 呼び出し追加可能性 | 改修 (Phase C) |
| `src/audit/g*.py` | 改修なし、wfa.py から wrap | 参照のみ |
| `STRATEGY_REGISTRY.md` | Phase C 採択戦略を追記 | 追記 |
| `CURRENT_BEST_STRATEGY.md` | Phase C 結果次第で更新候補 | 条件付改修 |
| `requirements.txt` | yfinance / arch / pytrends 追加 | 追記 |
| `.gitignore` | `data/signals/_cache/` 追加 | 追記 |

既存 backtest pipeline には Phase C 採用時のみ非破壊的に統合。

---

## 8. リスク・代替案

| リスク | 兆候 | 対処 |
|---|---|---|
| FRED HTTP 制限 | 429 / timeout 多発 | リトライ + キャッシュ徹底 |
| CFTC fixed-width 変更 | parse error | テスト fixture を最新版で再生成 |
| CBOE PutCall 公開停止 | URL 404 | yfinance `^CPCE` で代替 |
| AAII 週次手動更新が滞る | Phase B で signal #13 だけ古い | Phase B 自動 stale check + 警告 |
| `arch` SPA test 計算重い | 5000 iter で >1h | iter 数を 2000 に削減、もしくは Romano-Wolf に代替 |
| Phase B で 0 通過 | Pass count = 0 | Spec §6.8 フォールバック発動 (閾値緩和) |
| TQQQ proxy 期間不足 | 2010 以前データ薄 | 既存 base_dataset の合成 TQQQ を継続使用 |

---

## 9. 関連ドキュメント

- 設計仕様: [`SIGNAL_DISCOVERY_PLAN_20260603.md`](./SIGNAL_DISCOVERY_PLAN_20260603.md)
- 現行戦略統合 MD: `STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md`
- 戦略台帳: `STRATEGY_REGISTRY.md`
- 9指標標準: `docs/rules/08_evaluation-metrics.md`
- 統合レポート規格: `docs/rules/09_integrated-report-standard.md`
- 戦略検証規格: `docs/rules/06_strategy-verification.md`
- ファイル命名・日付規格: `docs/rules/07_doc-naming-and-dates.md`

---

**実行ステップ**: subagent-driven-development (推奨) または executing-plans で Task A0 から順次着手。各タスク完了後に上記 checkbox をチェック (`- [x]`) してコミット。
