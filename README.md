# NASDAQ Backtest — 3倍レバレッジ投資戦略バックテストシステム

> 3倍レバレッジNASDAQ（TQQQ等）を対象とした投資戦略のバックテストシステムです。

## 📋 概要

3倍レバレッジNASDAQ（TQQQ等）を対象とした投資戦略のバックテストシステムです。ドローダウン制御・ボラティリティスパイク検知・季節性フィルターなど複数シグナルを組み合わせた戦略を検証します。

## ✨ 主な機能

- 3倍レバレッジNASDAQ戦略の高精度バックテスト
- ドローダウン制御（DD）アルゴリズム実装
- ボラティリティスパイク検知フィルター
- Sharpe比・最大ドローダウン・Worst5Y CAGR評価
- 複数戦略の並列比較・ランキング出力

## 🛠️ 技術スタック

| カテゴリ | 技術・ライブラリ |
|----------|----------------|
| 言語 | Python 3.10+ |
| データ処理 | pandas, numpy |
| 可視化 | matplotlib, plotly |
| バックテストエンジン | カスタム実装（src/backtest_engine.py） |

## 🚀 セットアップ

### 前提条件

- Python 3.9 以上
- APIキー（Claude / OpenAI 等）を `.env` ファイルに設定

### インストール

```bash
git clone https://github.com/KazuyaMurayama/NASDAQ_backtest.git
cd NASDAQ_backtest
pip install -r requirements.txt
```

### 環境設定

```bash
cp .env.example .env
# .env ファイルに必要なAPIキーを設定
```

## 💻 使い方

```bash
python src/run_r4_backtest.py
```

## 👨‍💻 開発者情報

**男座員也（Kazuya Oza / おざ かずや）**

| | |
|---|---|
| GitHub | [@KazuyaMurayama](https://github.com/KazuyaMurayama) |
| 専門領域 | データサイエンス・生成AIコンサルタント |
| 主要スキル | Python, LightGBM, LangChain, RAG, Streamlit, React, TypeScript |
| 事業 | AIコンサルティング（月単価目標300万円）/ SaaS開発 / 定量投資 |

## 📄 ライセンス

© 2025 男座員也（Kazuya Oza）. All rights reserved.

---

> このリポジトリは **男座員也（Kazuya Oza）** が開発・管理しています。
> 命名・ドキュメント等での表記は必ず **男座員也** または **Kazuya Oza** を使用してください。
