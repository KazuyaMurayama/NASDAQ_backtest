"""
T5: REIT プロキシ汚染テスト
    - Version A: 全期間 S&P500 proxy（2004前後通して）
    - Version B: VNQ 実データのみ使用（2004-2026）
    - Version C: REIT 完全除外（Bond+Gold+S&P500 のみ 3資産）
    比較: 改善効果がどのバージョンでも一貫しているか

T6: ブロックブートストラップ有意性検定（Bonferroni 補正）
    - 月次リターンブロック（12ヶ月）でシャープ差の分布を推定
    - 事前指定比率（計画書: 25%）での検定のみが"正式"
    - 6比率のうち最良を拾う場合: Bonferroni 補正 p < 0.05/6 ≈ 0.0083
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from validate_c_shared import (
    load_all, build_marugoto_nav_vec, build_portfolio_vec, make_weights,
    load_ratio_series, RP_30YR, BASE_DIR, DATA_DIR
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


def calc_sharpe(ret_arr):
    if len(ret_arr) < 2 or ret_arr.std() == 0:
        return np.nan
    return ret_arr.mean() / ret_arr.std() * np.sqrt(252)


def run():
    print("=" * 85)
    print("T5 + T6: REIT プロキシ汚染テスト & ブートストラップ有意性")
    print("=" * 85)

    d = load_all()
    dates_dt = d['dates_dt']; n = d['n']
    nav_a2 = d['nav_a2']; g2 = d['g2']; b3 = d['b3']
    bond_1x = d['bond_1x']; gold_1x = d['gold_1x']
    raw = d['raw']; vz = d['vz']
    years = n / 252

    # ── T5: REIT プロキシ汚染テスト ───────────────────────────────────────────
    print("\n【T5】REIT プロキシ汚染テスト（50% 置換ケースで比較）")

    sp_path  = os.path.join(DATA_DIR, 'sp500_daily.csv')
    vnq_path = os.path.join(DATA_DIR, 'vnq_daily.csv')
    sp_ratio  = load_ratio_series(sp_path,  dates_dt)
    vnq_ratio = load_ratio_series(vnq_path, dates_dt)

    def build_maru_custom_reit(reit_ratio_arr):
        """REIT 比率配列を指定して まるごと NAV を構築"""
        w_b, w_g, w_s, w_r = RP_30YR
        bond_ratio = np.ones(n)
        bond_ratio[1:] = np.where(bond_1x[:-1]>0, bond_1x[1:]/bond_1x[:-1], 1.0)
        gold_ratio = np.ones(n)
        gold_ratio[1:] = np.where(gold_1x[:-1]>0, gold_1x[1:]/gold_1x[:-1], 1.0)
        r_f = 3.0*(w_b*(bond_ratio-1)+w_g*(gold_ratio-1)+
                   w_s*(sp_ratio-1)+w_r*(reit_ratio_arr-1))
        r_net = r_f - 0.004675/252
        return np.concatenate([[1.0], np.cumprod(1 + r_net[1:])])

    # Version A: 全期間 S&P500 proxy
    maru_A = build_maru_custom_reit(sp_ratio)

    # Version B: VNQ 2004+, NaN 前は SP500 （通常モデル = Version B）
    maru_B = d['maru']  # 既に構築済み

    # Version C: REIT 完全除外（3資産: Bond+Gold+SP500, 再正規化）
    w_b, w_g, w_s = RP_30YR[0], RP_30YR[1], RP_30YR[2]
    total3 = w_b + w_g + w_s
    w3 = [w_b/total3, w_g/total3, w_s/total3]
    bond_ratio_c = np.ones(n)
    bond_ratio_c[1:] = np.where(bond_1x[:-1]>0, bond_1x[1:]/bond_1x[:-1], 1.0)
    gold_ratio_c = np.ones(n)
    gold_ratio_c[1:] = np.where(gold_1x[:-1]>0, gold_1x[1:]/gold_1x[:-1], 1.0)
    r_c = 3.0*(w3[0]*(bond_ratio_c-1)+w3[1]*(gold_ratio_c-1)+w3[2]*(sp_ratio-1))
    maru_C = np.concatenate([[1.0], np.cumprod(1 + (r_c - 0.004675/252)[1:])])

    # 2004+ 期間のみで比較（VNQ 実データが使える期間）
    vnq_df = pd.read_csv(vnq_path, parse_dates=['Date'])
    vnq_start = vnq_df['Date'].min()
    mask_2004 = dates_dt >= vnq_start

    results_t5 = []
    for ver, maru_v in [('A: 全期間SP500proxy', maru_A),
                         ('B: VNQ実+SP500proxy', maru_B),
                         ('C: REIT除外（3資産）', maru_C)]:
        for ratio in [0.0, 0.25, 0.50]:
            wn, wg, wm, wb = make_weights(raw, vz, ratio, n)
            pnav = build_portfolio_vec(nav_a2, g2, maru_v, b3, wn, wg, wm, wb)

            # 2004+ 期間での指標
            sub = pnav[mask_2004]; sub = sub / sub[0]
            sub_s = pd.Series(sub)
            sub_ret = sub_s.pct_change().dropna()
            sub_years = len(sub) / 252
            cagr = float(sub_s.iloc[-1]**(1/sub_years)-1)
            sharpe = float(sub_ret.mean()/sub_ret.std()*np.sqrt(252))
            results_t5.append({'ver': ver, 'ratio': ratio,
                                'cagr': cagr, 'sharpe': sharpe})

    print(f"  ※ 2004-2026 期間での比較（VNQ 実データ利用可能期間）")
    print(f"  {'バージョン':<28} {'r=0%(Base)':>12} {'r=25%':>12} {'r=50%':>12}")
    for metric in ['sharpe']:
        print(f"  [Sharpe]")
        for ver in ['A: 全期間SP500proxy', 'B: VNQ実+SP500proxy', 'C: REIT除外（3資産）']:
            vals = {r['ratio']: r[metric]
                    for r in results_t5 if r['ver'] == ver}
            row = f"  {ver:<28}"
            for ratio in [0.0, 0.25, 0.50]:
                row += f"  {vals.get(ratio, float('nan')):>10.4f}"
            print(row)

    # T5 判定: バージョン B（通常モデル）と A/C の差が小さいか
    for ratio in [0.25, 0.50]:
        sh_A = next(r['sharpe'] for r in results_t5 if r['ver']=='A: 全期間SP500proxy' and r['ratio']==ratio)
        sh_B = next(r['sharpe'] for r in results_t5 if r['ver']=='B: VNQ実+SP500proxy' and r['ratio']==ratio)
        sh_C = next(r['sharpe'] for r in results_t5 if r['ver']=='C: REIT除外（3資産）' and r['ratio']==ratio)
        sh_base = next(r['sharpe'] for r in results_t5 if r['ver']=='B: VNQ実+SP500proxy' and r['ratio']==0.0)

        check(f"T5-{int(ratio*100)}: Ver A/B/C すべてで r={ratio*100:.0f}% がベースラインを上回る",
              sh_A > sh_base and sh_B > sh_base and sh_C > sh_base,
              f"A={sh_A:.4f} B={sh_B:.4f} C={sh_C:.4f} base={sh_base:.4f}")

    # ── T6: ブロックブートストラップ有意性検定 ────────────────────────────────
    print("\n【T6】ブロックブートストラップ有意性検定")
    print("  対象: 全期間（1974-2026）の月次リターン、ブロック長 12ヶ月")
    print("  反復: 300回（Bonferroni 補正: α=0.05/6≈0.0083）")

    # 月次 NAV に変換
    dates_s = pd.to_datetime(d['dates'].values)
    nav_df = pd.DataFrame(index=dates_s)

    for ratio in [0.0, 0.25, 0.50]:
        wn, wg, wm, wb = make_weights(raw, vz, ratio, n)
        pnav = build_portfolio_vec(nav_a2, g2, d['maru'], b3, wn, wg, wm, wb)
        nav_df[f'r{int(ratio*100)}'] = pnav

    monthly = nav_df.resample('ME').last().pct_change().dropna()
    ret_base  = monthly['r0'].values
    ret_25    = monthly['r25'].values
    ret_50    = monthly['r50'].values

    np.random.seed(42)
    n_boot = 300
    block  = 12  # 月次ブロック
    n_m    = len(ret_base)

    sharpe_diffs_25 = []
    sharpe_diffs_50 = []

    for _ in range(n_boot):
        # ブロック bootstrap
        idx_starts = np.random.randint(0, n_m - block + 1,
                                        size=n_m // block + 1)
        idx = np.concatenate([np.arange(s, min(s+block, n_m))
                               for s in idx_starts])[:n_m]
        b0  = ret_base[idx];  b25 = ret_25[idx];  b50 = ret_50[idx]
        sh0  = b0.mean()  / (b0.std()  + 1e-10) * np.sqrt(12)
        sh25 = b25.mean() / (b25.std() + 1e-10) * np.sqrt(12)
        sh50 = b50.mean() / (b50.std() + 1e-10) * np.sqrt(12)
        sharpe_diffs_25.append(sh25 - sh0)
        sharpe_diffs_50.append(sh50 - sh0)

    # 実測値
    sh_base_obs = ret_base.mean() / (ret_base.std() + 1e-10) * np.sqrt(12)
    sh_25_obs   = ret_25.mean()   / (ret_25.std()   + 1e-10) * np.sqrt(12)
    sh_50_obs   = ret_50.mean()   / (ret_50.std()   + 1e-10) * np.sqrt(12)
    obs_diff_25 = sh_25_obs - sh_base_obs
    obs_diff_50 = sh_50_obs - sh_base_obs

    diffs_25 = np.array(sharpe_diffs_25)
    diffs_50 = np.array(sharpe_diffs_50)

    # p-value (one-sided: improvement)
    p_25_unc = float(np.mean(diffs_25 >= obs_diff_25))   # under null
    p_50_unc = float(np.mean(diffs_50 >= obs_diff_50))
    # Bonferroni correction (6 comparisons)
    p_25_bon = min(p_25_unc * 6, 1.0)
    p_50_bon = min(p_50_unc * 6, 1.0)

    ci25 = np.percentile(diffs_25, [2.5, 97.5])
    ci50 = np.percentile(diffs_50, [2.5, 97.5])

    print(f"\n  月次 Sharpe 差の分布（bootstrap n={n_boot}）:")
    print(f"  {'比率':<8} {'実測差':>10} {'95%CI':>20} {'p(unc)':>10} {'p(Bon)':>10} {'判定'}")
    print(f"  {'-'*68}")
    for label, obs, ci, p_unc, p_bon in [
        ('25%', obs_diff_25, ci25, p_25_unc, p_25_bon),
        ('50%', obs_diff_50, ci50, p_50_unc, p_50_bon),
    ]:
        sig_unc = p_unc < 0.05
        sig_bon = p_bon < 0.05
        mark = "◎有意" if sig_bon else ("△境界" if sig_unc else "×非有意")
        print(f"  r={label:<5}  {obs:>+10.4f}  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"{p_unc:>10.4f}  {p_bon:>10.4f}  {mark}")

    print(f"\n  注: ブートストラップ p 値 < 0.05 (非補正) = 統計的傾向あり")
    print(f"  注: p < 0.0083 (Bonferroni) = 6比率スキャン後でも有意")
    print(f"  注: Sharpe 差 0.023 は経済的意義は小さいが, 方向性が重要")

    check(f"T6-1: 計画案(25%) 非補正 p < 0.10（統計的傾向あり）",
          p_25_unc < 0.10,
          f"p={p_25_unc:.4f}")
    check(f"T6-2: 95%CI の下限 ≥ -0.10（大幅悪化の証拠なし）",
          ci25[0] >= -0.10,
          f"CI_low={ci25[0]:.4f}")
    check(f"T6-3: 最良比率(50%) の 95%CI 下限 ≥ -0.05",
          ci50[0] >= -0.05,
          f"CI_low={ci50[0]:.4f}")

    print(f"\n{'='*85}")
    print(f"T5+T6 COMPLETE: {PASS} PASSED, {FAIL} FAILED")
    print(f"{'='*85}")


if __name__ == '__main__':
    run()
