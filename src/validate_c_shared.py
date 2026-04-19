"""
方向性C バリデーション: 共有ユーティリティ
- A2シグナル・NAV・資産NAVの事前計算（キャッシュ）
- Python ループを排除した numpy ベクトル化ポートフォリオビルダー
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

# まるごとレバレッジ リスクパリティウェイト（逆ボラ加重, 30yr bond想定）
_vols30 = [0.075, 0.16, 0.19, 0.21]
_inv30  = [1/v for v in _vols30]
_s30    = sum(_inv30)
RP_30YR = [v/_s30 for v in _inv30]   # Bond45%/Gold21%/SP18%/REIT16%

# 10yr bond 想定版（実際のまるごとレバレッジは T-Note 10yr 使用の可能性）
_vols10 = [0.050, 0.16, 0.19, 0.21]
_inv10  = [1/v for v in _vols10]
_s10    = sum(_inv10)
RP_10YR = [v/_s10 for v in _vols10]  # Bond57%/Gold20%/SP14%/REIT12%


# ─── 高速データ読み込み（pandas reindex で Python loop 排除）────────────────

def load_price_series(csv_path, dates_dt):
    """CSV の日付インデックス Series を NASDAQ 日付軸に reindex（前値埋め）"""
    df = pd.read_csv(csv_path, parse_dates=['Date']).set_index('Date')['Close']
    df = df.reindex(dates_dt, method='ffill')
    return df.values.astype(float)


def load_ratio_series(csv_path, dates_dt):
    """価格 → 日次比率（1+r）配列。データなし期間は 1.0（変動ゼロ）"""
    px = load_price_series(csv_path, dates_dt)
    ratio = np.ones(len(px))
    mask = (px[:-1] > 0)
    ratio[1:][mask] = px[1:][mask] / px[:-1][mask]
    return ratio


def build_lev_nav_vec(ratio_1x, lev, cost_yr, n):
    """日次比率配列から レバ倍 × 経費率 の NAV を vectorized で構築"""
    daily_cost = cost_yr / 252
    r1x = ratio_1x - 1.0
    r_lev = r1x * lev - daily_cost
    nav = np.ones(n)
    np.cumprod(1 + r_lev, out=nav)         # cumprod は inplace 非対応なので
    nav = np.concatenate([[1.0], np.cumprod(1 + r_lev[1:])])
    return nav


# ─── A2 シグナル（逐次 AsymEWMA のみ loop, 残りは vectorized）───────────────

def calc_asym_ewma_vec(returns_arr, su=30, sd=10):
    """NumPy で AsymEWMA を計算（Python loop だが 13k 行でも ~0.15s）"""
    n = len(returns_arr)
    var = np.empty(n)
    var[0] = np.nanvar(returns_arr[:20]) if n > 20 else 1e-4
    a_up = 2.0 / (su + 1); a_dn = 2.0 / (sd + 1)
    for i in range(1, n):
        a = a_dn if returns_arr[i] < 0 else a_up
        var[i] = (1 - a) * var[i-1] + a * returns_arr[i] ** 2
    return np.sqrt(var * 252)


def build_a2_signals(df):
    """A2 Optimized シグナルを計算。全部返す。"""
    close = df['Close']; dates = df['Date']
    returns = close.pct_change()
    ret_arr = returns.fillna(0).values

    dd = calc_dd_signal(close, 0.82, 0.92).values
    av = calc_asym_ewma_vec(ret_arr, 30, 10)

    close_v = close.values
    ma150 = pd.Series(close_v).rolling(150).mean().values
    ratio_c = np.where(ma150 > 0, close_v / ma150, 1.0)
    ttv = np.clip(0.10 + 0.20 * (ratio_c - 0.85) / 0.30, 0.10, 0.30)
    ttv = np.where(np.isnan(ma150), 0.20, ttv)
    vt = np.clip(ttv / np.where(av > 0, av, 1e-6), 0, 1.0)

    ma200 = pd.Series(close_v).rolling(200).mean().values
    sl_s = pd.Series(ma200).pct_change()
    sm = sl_s.rolling(60).mean().values
    ss = sl_s.rolling(60).std().replace(0, 1e-4).values
    z = (sl_s.values - sm) / ss
    slope = np.clip(0.9 + 0.35 * np.nan_to_num(z), 0.3, 1.5)

    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3).values

    vp = calc_vix_proxy(returns).values
    vp_s = pd.Series(vp)
    vma = vp_s.rolling(252).mean().values
    vs  = vp_s.rolling(252).std().values
    vs  = np.where(vs < 1e-3, 1e-3, vs)
    vz  = (vp - vma) / vs
    vm  = np.clip(1.0 - 0.25 * np.nan_to_num(vz), 0.5, 1.15)

    raw = np.clip(dd * vt * slope * mom * vm, 0, 1.0)
    raw = np.nan_to_num(raw)
    return raw, vz, returns.fillna(0).values


def build_a2_nav(close_s, raw, ret_arr):
    """A2 NASDAQ NAV（delay=2, cost=0.86%, 3x leverage）"""
    lev_s = rebalance_threshold(pd.Series(raw, index=close_s.index), 0.20)
    lev_shifted = lev_s.shift(2).fillna(0).values
    lr = ret_arr * 3.0
    dc = 0.0086 / 252
    sr = lev_shifted * (lr - dc)
    return np.cumprod(1 + sr)


# ─── まるごとレバレッジ合成 NAV（vectorized）──────────────────────────────────

def build_marugoto_nav_vec(dates_dt, bond_1x_arr, gold_1x_arr, weights=None,
                            bond_csv=None, use_10yr_bond=False):
    """
    まるごとレバレッジ合成 NAV（完全 vectorized）。
    bond_csv: None → prepare_bond_data の 30yr 推定、
              'ief' → IEF 7-10yr bond proxy
    weights: None → RP_30YR or RP_10YR（use_10yr_bond に応じて）
    """
    n = len(dates_dt)

    if weights is None:
        weights = RP_10YR if use_10yr_bond else RP_30YR
    w_b, w_g, w_s, w_r = weights

    # Bond ratio
    if bond_csv == 'ief':
        bond_ratio = load_ratio_series(os.path.join(DATA_DIR, 'ief_daily.csv'), dates_dt)
    else:
        bond_ratio = np.ones(n)
        bond_ratio[1:] = np.where(bond_1x_arr[:-1] > 0,
                                   bond_1x_arr[1:] / bond_1x_arr[:-1], 1.0)

    # Gold ratio
    gold_ratio = np.ones(n)
    gold_ratio[1:] = np.where(gold_1x_arr[:-1] > 0,
                               gold_1x_arr[1:] / gold_1x_arr[:-1], 1.0)

    # S&P500 ratio
    sp_ratio = load_ratio_series(os.path.join(DATA_DIR, 'sp500_daily.csv'), dates_dt)

    # REIT ratio (VNQ 2004+, S&P500 proxy pre-2004)
    vnq_ratio = load_ratio_series(os.path.join(DATA_DIR, 'vnq_daily.csv'), dates_dt)
    vnq_df    = pd.read_csv(os.path.join(DATA_DIR, 'vnq_daily.csv'), parse_dates=['Date'])
    vnq_start = vnq_df['Date'].min()
    use_sp = pd.Series(dates_dt) < vnq_start
    reit_ratio = np.where(use_sp.values, sp_ratio, vnq_ratio)

    # 3x blended daily return
    r_bond = bond_ratio - 1.0
    r_gold = gold_ratio - 1.0
    r_sp   = sp_ratio   - 1.0
    r_reit = reit_ratio - 1.0
    r_fund = 3.0 * (w_b*r_bond + w_g*r_gold + w_s*r_sp + w_r*r_reit)
    r_net  = r_fund - 0.004675 / 252

    nav = np.concatenate([[1.0], np.cumprod(1 + r_net[1:])])
    return nav


# ─── 4資産ポートフォリオ（vectorized ドリフトチェック）──────────────────────

def build_portfolio_vec(nasdaq_nav, gold2x_nav, maru_nav, bond3x_nav,
                         wn, wg, wm, wb,
                         drift_thr=0.10, cal_freq=63):
    """
    4資産ポートフォリオ NAV。
    rebalance: drift > drift_thr OR i % cal_freq == 0
    完全 Python loop だが 13k イテレーションで ~0.3s（許容範囲）。
    """
    n = len(nasdaq_nav)
    rn = np.zeros(n); rg = np.zeros(n); rm = np.zeros(n); rb = np.zeros(n)

    safe_div = lambda a, b: np.where(b > 0, a/b - 1.0, 0.0)
    rn[1:] = safe_div(nasdaq_nav[1:], nasdaq_nav[:-1])
    rg[1:] = safe_div(gold2x_nav[1:], gold2x_nav[:-1])
    rm[1:] = safe_div(maru_nav[1:],   maru_nav[:-1])
    rb[1:] = safe_div(bond3x_nav[1:], bond3x_nav[:-1])

    pnav = np.ones(n)
    cn, cg, cm, cb = wn[0], wg[0], wm[0], wb[0]

    for i in range(1, n):
        ret = cn*rn[i] + cg*rg[i] + cm*rm[i] + cb*rb[i]
        pnav[i] = pnav[i-1] * (1 + ret)
        total = (cn*(1+rn[i]) + cg*(1+rg[i]) + cm*(1+rm[i]) + cb*(1+rb[i]))
        if total > 0:
            cn /= total; cg /= total; cm /= total; cb /= total
            cn *= (1+rn[i]); cg *= (1+rg[i]); cm *= (1+rm[i]); cb *= (1+rb[i])
        if (abs(cn-wn[i])+abs(cg-wg[i])+abs(cm-wm[i])+abs(cb-wb[i]) > drift_thr
                or i % cal_freq == 0):
            cn, cg, cm, cb = wn[i], wg[i], wm[i], wb[i]

    return pnav


def make_weights(raw, vz, maru_ratio, n):
    """Dyn 2x3x ウェイト（B0.55/L0.25/V0.1/G0.5）+ まるごとレバレッジ混合"""
    vz_clip = np.clip(vz, 0, None)
    vz_clip = np.nan_to_num(vz_clip)
    wn = np.clip(0.55 + 0.25*raw - 0.10*vz_clip, 0.30, 0.90)
    gold_slot = (1 - wn) * 0.50
    wg = gold_slot * (1 - maru_ratio)
    wm = gold_slot * maru_ratio
    wb = (1 - wn) * 0.50
    return wn, wg, wm, wb


def calc_metrics_arr(nav, dates_dt, label=""):
    """NAV 配列 → 指標 dict"""
    nav_s = pd.Series(nav, index=dates_dt)
    ret   = nav_s.pct_change().dropna()
    years = len(nav_s) / 252
    cagr  = float(nav_s.iloc[-1] ** (1/years) - 1)
    maxdd = float((nav_s / nav_s.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    worst5y = float(((nav_s / nav_s.shift(252*5)) ** (1/5) - 1).min())

    oos = ret[ret.index >= '2021-01-01']
    oos_sh = float(oos.mean()/oos.std()*np.sqrt(252)) if len(oos)>10 and oos.std()>0 else np.nan

    return {'label': label, 'cagr': cagr, 'maxdd': maxdd,
            'sharpe': sharpe, 'worst5y': worst5y, 'oos_sharpe': oos_sh}


# ─── 共有データのロード（一度だけ）──────────────────────────────────────────

def load_all():
    df       = load_data(DATA_PATH)
    close    = df['Close']
    dates    = df['Date']
    dates_dt = pd.to_datetime(dates.values)
    n        = len(df)

    raw, vz, ret_arr = build_a2_signals(df)
    nav_a2           = build_a2_nav(close, raw, ret_arr)

    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)

    # Gold 2x NAV
    g_ratio = np.ones(n)
    g_ratio[1:] = np.where(gold_1x[:-1] > 0, gold_1x[1:]/gold_1x[:-1], 1.0)
    g2 = np.concatenate([[1.0], np.cumprod(1 + (g_ratio[1:]-1)*2 - 0.005/252)])

    # TMF 3x NAV
    b_ratio = np.ones(n)
    b_ratio[1:] = np.where(bond_1x[:-1] > 0, bond_1x[1:]/bond_1x[:-1], 1.0)
    b3 = np.concatenate([[1.0], np.cumprod(1 + (b_ratio[1:]-1)*3 - 0.0091/252)])

    # まるごとレバレッジ合成 NAV（ベースケース: 30yr bond）
    maru = build_marugoto_nav_vec(dates_dt, bond_1x, gold_1x, weights=RP_30YR)

    return {
        'df': df, 'close': close, 'dates': dates, 'dates_dt': dates_dt, 'n': n,
        'raw': raw, 'vz': vz, 'ret_arr': ret_arr,
        'nav_a2': nav_a2, 'gold_1x': gold_1x, 'bond_1x': bond_1x,
        'g2': g2, 'b3': b3, 'maru': maru,
    }
