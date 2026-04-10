"""
T3: OOS (2021-2026) 分析 ─ IS で最良の置換比率が OOS でも有効か
T4: 10年ごとの一貫性確認 ─ 特定の時代にのみ効果が偏っていないか

判定基準:
  PASS (T3): OOS 期間で Gold→Maru 25% が Gold 100% ベースラインを上回る
  PASS (T4): 6 デケードのうち 4 以上で Sharpe 改善
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from validate_c_shared import (
    load_all, build_portfolio_vec, make_weights, BASE_DIR
)

PASS = 0; FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ PASS: {name}")
        PASS += 1
    else:
        print(f"  ❌ FAIL: {name}  [{detail}]")
        FAIL += 1


def sub_metrics(nav, dates_dt, start, end, label=""):
    """特定期間のみの指標"""
    mask = (dates_dt >= pd.Timestamp(start)) & (dates_dt < pd.Timestamp(end))
    sub = nav[mask]
    if len(sub) < 30:
        return None
    sub_s = pd.Series(sub / sub[0])
    ret   = sub_s.pct_change().dropna()
    years = len(sub) / 252
    cagr  = float(sub_s.iloc[-1]**(1/years)-1) if years > 0.1 else np.nan
    mdd   = float((sub_s/sub_s.cummax()-1).min())
    sharpe = float(ret.mean()/ret.std()*np.sqrt(252)) if ret.std()>0 else np.nan
    return {'label': label, 'start': start, 'end': end,
            'cagr': cagr, 'mdd': mdd, 'sharpe': sharpe, 'years': years}


def run():
    print("=" * 85)
    print("T3 + T4: OOS 検証 & 10年ごと一貫性")
    print("=" * 85)

    d = load_all()
    dates_dt = d['dates_dt']; n = d['n']
    nav_a2 = d['nav_a2']; g2 = d['g2']; b3 = d['b3']; maru = d['maru']
    raw = d['raw']; vz = d['vz']

    years = n / 252

    # ── ベースライン（Gold 100%）と計画案（Gold→Maru 50%）を全期間計算 ──────
    ratios_to_test = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
    navs = {}
    for r in ratios_to_test:
        wn, wg, wm, wb = make_weights(raw, vz, r, n)
        navs[r] = build_portfolio_vec(nav_a2, g2, maru, b3, wn, wg, wm, wb)

    base_nav  = navs[0.0]
    base_cagr = float(pd.Series(base_nav).iloc[-1]**(1/years)-1)

    # ── T3: OOS 2021-2026 ──────────────────────────────────────────────────────
    print("\n【T3】OOS 分析（2021-2026）")
    print(f"  IS 期間: 1974-2020 / OOS 期間: 2021-2026")
    print(f"  {'比率':<12} {'IS CAGR':>10} {'OOS CAGR':>10} {'IS Sharpe':>10} {'OOS Sharpe':>10}")
    print(f"  {'-'*55}")

    oos_results = []
    for r in ratios_to_test:
        nav = navs[r]
        is_m  = sub_metrics(nav, dates_dt, '1974-01-01', '2021-01-01', f"IS r={r}")
        oos_m = sub_metrics(nav, dates_dt, '2021-01-01', '2027-01-01', f"OOS r={r}")
        if is_m and oos_m:
            print(f"  r={r*100:4.0f}%  {is_m['cagr']*100:>10.2f}%  "
                  f"{oos_m['cagr']*100:>10.2f}%  "
                  f"{is_m['sharpe']:>10.4f}  {oos_m['sharpe']:>10.4f}")
            oos_results.append({'ratio': r, 'is': is_m, 'oos': oos_m})

    # OOS での最良比率は何か
    oos_best_sharpe = max(oos_results, key=lambda x: x['oos']['sharpe'])
    oos_best_cagr   = max(oos_results, key=lambda x: x['oos']['cagr'])
    print(f"\n  OOS Best Sharpe: r={oos_best_sharpe['ratio']*100:.0f}% "
          f"(Sharpe={oos_best_sharpe['oos']['sharpe']:.4f})")
    print(f"  OOS Best CAGR:   r={oos_best_cagr['ratio']*100:.0f}% "
          f"(CAGR={oos_best_cagr['oos']['cagr']*100:.2f}%)")

    # ベースライン OOS 指標
    base_oos = [x for x in oos_results if x['ratio'] == 0.0][0]['oos']

    # 計画案（25% 置換）が OOS でもプラス改善
    plan_oos = [x for x in oos_results if x['ratio'] == 0.25][0]
    check("T3-1: 計画案(25%) が OOS Sharpe でベースラインを上回る",
          plan_oos['oos']['sharpe'] >= base_oos['sharpe'],
          f"plan={plan_oos['oos']['sharpe']:.4f}, base={base_oos['sharpe']:.4f}")
    check("T3-2: 計画案(25%) が OOS CAGR でベースラインを上回る",
          plan_oos['oos']['cagr'] >= base_oos['cagr'],
          f"plan={plan_oos['oos']['cagr']*100:.2f}%, base={base_oos['cagr']*100:.2f}%")

    # IS 最良比率（50%）は OOS で最悪でないか
    plan_50_oos = [x for x in oos_results if x['ratio'] == 0.50][0]
    all_oos_sharpes = sorted([x['oos']['sharpe'] for x in oos_results])
    rank_50 = all_oos_sharpes.index(plan_50_oos['oos']['sharpe']) + 1
    check(f"T3-3: IS 最良(50%)の OOS Sharpe ランク ≥ 中位（rank {rank_50}/{len(oos_results)}）",
          rank_50 >= len(oos_results) // 2,
          f"rank={rank_50}/{len(oos_results)}")

    # OOS で 0% が最良でない（過学習の最悪ケース回避）
    check("T3-4: OOS 最良 Sharpe の比率が 0%（ゼロ置換）でない",
          oos_best_sharpe['ratio'] > 0.0,
          f"OOS best ratio={oos_best_sharpe['ratio']*100:.0f}%")

    # ── T4: 10年ごと一貫性 ──────────────────────────────────────────────────
    print("\n【T4】10年ごとの Sharpe 改善（Gold100% vs Gold→Maru 50%）")
    decades = [
        ('1974', '1980'), ('1980', '1990'), ('1990', '2000'),
        ('2000', '2010'), ('2010', '2020'), ('2020', '2027'),
    ]
    nav_50  = navs[0.50]

    improved_count = 0; decade_count = 0
    worst_sharpe_drop = 0
    print(f"  {'期間':<15} {'Base Sharpe':>12} {'50% Sharpe':>12} {'差':>8} {'判定':>6}")
    print(f"  {'-'*58}")
    for (s, e) in decades:
        bm = sub_metrics(base_nav, dates_dt, f'{s}-01-01', f'{e}-01-01')
        mm = sub_metrics(nav_50,   dates_dt, f'{s}-01-01', f'{e}-01-01')
        if bm and mm:
            diff = mm['sharpe'] - bm['sharpe']
            improved = diff >= 0
            if improved:
                improved_count += 1
            else:
                worst_sharpe_drop = min(worst_sharpe_drop, diff)
            decade_count += 1
            mark = "✓" if improved else "✗"
            print(f"  {s}s  ({bm['years']:.0f}y)  {bm['sharpe']:>12.4f}  "
                  f"{mm['sharpe']:>12.4f}  {diff:>+8.4f}  {mark}")

    print(f"\n  10年ごと改善: {improved_count}/{decade_count} デケード")
    check(f"T4-1: 改善デケード ≥ 4/6",
          improved_count >= 4, f"{improved_count}/{decade_count}")
    check("T4-2: 最悪デケードの Sharpe 低下 ≤ 0.15（壊滅的な失敗なし）",
          worst_sharpe_drop >= -0.15, f"worst drop={worst_sharpe_drop:.4f}")

    # 1970s の検証（インフレ局面でまるごとが Gold より悪くても許容範囲か）
    m70_base = sub_metrics(base_nav, dates_dt, '1974-01-01', '1980-01-01')
    m70_50   = sub_metrics(nav_50,   dates_dt, '1974-01-01', '1980-01-01')
    if m70_base and m70_50:
        diff_70 = m70_50['cagr'] - m70_base['cagr']
        print(f"\n  1970s CAGR 差（50% 置換 vs Gold 100%）: {diff_70*100:+.2f}pp")
        check("T4-3: 1970年代インフレ局面でも CAGR 差が -5pp 以内（壊滅なし）",
              diff_70 >= -0.05, f"diff={diff_70*100:.2f}pp")

    print(f"\n{'='*85}")
    print(f"T3+T4 COMPLETE: {PASS} PASSED, {FAIL} FAILED")
    print(f"{'='*85}")


if __name__ == '__main__':
    run()
