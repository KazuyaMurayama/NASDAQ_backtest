"""Build the 71-signal untested inventory CSV for Session 1 of SIGNAL_EXPANSION_PLAN_20260605.

Output: data/signals/expansion/untested_signal_inventory_20260605.csv
Columns: signal_id, name, category, source_module, data_source_url_or_path,
         data_status, earliest_date, latest_date, n_obs, cost_tier, priority, notes
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'signals' / 'expansion' / 'untested_signal_inventory_20260605.csv'
REPO_SUMMARY = ROOT / 'data' / 'signals' / 'expansion' / 'repo_derivation_summary_20260605.csv'


def repo_meta():
    """Read REPO derivation summary if present so we can fill earliest/latest/n_obs."""
    if not REPO_SUMMARY.exists():
        return {}
    df = pd.read_csv(REPO_SUMMARY)
    return {row['signal']: row for _, row in df.iterrows()}


def main():
    rows = []

    # -------- 2.1 Tier1 ◎ (25 signals, priority=A) --------
    tier1_a = [
        ('1',  'NDX 200DMA breadth %',          'Breadth',    'A6 yahoo loader (planned)',  'Yahoo / Polygon (computed from constituents)', 'needs_implementation', 'free',     'A', 'requires NDX constituent close data'),
        ('2',  'McClellan Oscillator NDX',      'Breadth',    'new module (planned)',       'StockCharts equivalent — computed from advancers/decliners', 'needs_implementation', 'free', 'A', 'requires daily advancers/decliners feed'),
        ('3',  'NDX New Hi-Lo 52W',             'Breadth',    'new module (planned)',       'Polygon Starter ($29/mo) constituents',  'needs_paid_api',       'low_paid', 'A', 'Polygon Starter or per-ticker yfinance loop'),
        ('7',  'VIX9D / VIX ratio',             'Vol',        'A7 yahoo loader (planned)',  '^VIX9D, ^VIX via yfinance',              'available_yahoo',      'free',     'A', 'short-term stress'),
        ('8',  'VIX Term Structure VIX1/2/3',   'Vol',        'A9 cboe loader (planned)',   'CBOE futures settlement files',          'available_cboe',       'free',     'A', 'CBOE settlement CSV daily'),
        ('9',  'VVIX',                          'Vol',        'A7 yahoo loader (planned)',  '^VVIX via yfinance',                     'available_yahoo',      'free',     'A', 'vol-of-vol'),
        ('10', 'MOVE Index',                    'Vol',        'A7 yahoo loader (planned)',  '^MOVE via yfinance (Yahoo intermittent)','available_yahoo',     'free',     'A', 'Yahoo sometimes blocks ^MOVE; backup ICE direct'),
        ('12', 'CBOE Put/Call (Equity)',        'Sentiment',  'A9 cboe loader (planned)',   'CBOE public data',                       'available_cboe',       'free',     'A', 'updated daily by CBOE'),
        ('13', 'AAII Bull-Bear spread',         'Sentiment',  'A10 manual loader (planned)','AAII members survey weekly CSV',         'needs_manual_csv',     'free',     'A', 'weekly Thursday release; manual download'),
        ('15', 'CFTC CoT NQ Non-Comm Net',      'Sentiment',  'A8 cftc loader (planned)',   'CFTC TFF weekly CSV',                    'available_cftc',       'free',     'A', 'released Fridays'),
        ('16', 'CFTC CoT GC Net',               'Sentiment',  'A8 cftc loader (planned)',   'CFTC TFF weekly CSV',                    'available_cftc',       'free',     'A', 'released Fridays'),
        ('17', 'CFTC CoT ZB/ZN Net',            'Sentiment',  'A8 cftc loader (planned)',   'CFTC TFF weekly CSV',                    'available_cftc',       'free',     'A', 'released Fridays'),
        ('27', '3M-10Y spread',                 'YieldCurve', 'A6 fred loader (planned)',   'FRED T10Y3M',                            'available_fred',       'free',     'A', 'classical recession indicator'),
        ('29', '5Y5Y BEI',                      'YieldCurve', 'A6 fred loader (planned)',   'FRED T5YIFR',                            'available_fred',       'free',     'A', '5Y5Y forward inflation expectation'),
        ('30', 'CME FedWatch 25bp cut prob 3M', 'YieldCurve', 'new module (planned)',       'CME public FedWatch (HTML scrape)',      'needs_implementation', 'free',     'A', 'scrape required; static snapshot available'),
        ('32', 'Atlanta Fed GDPNow',            'Macro',      'A10 manual loader (planned)','Atlanta Fed GDPNow CSV',                 'needs_manual_csv',     'free',     'A', 'public, updated 4-6x per quarter'),
        ('34', 'Citi Economic Surprise USMI',   'Macro',      'A10 manual loader (planned)','Citi USESI (proprietary scrape)',        'needs_paid_api',       'mid_paid', 'A', 'Bloomberg/Citi license; partial public mirrors exist'),
        ('36', 'Chicago Fed NFCI',              'Macro',      'A6 fred loader (planned)',   'FRED NFCI',                              'available_fred',       'free',     'A', 'already partially in nfci_weekly.csv'),
        ('37', 'NDX Forward EPS Rev 4wk',       'Earnings',   'A10 manual loader (planned)','EODHD or Finnhub',                       'needs_paid_api',       'low_paid', 'A', 'subscription needed'),
        ('38', 'Equity Risk Premium',           'Earnings',   'A10 manual loader (planned)','computed: Fwd EPS Yld - 10Y real',       'needs_implementation', 'low_paid', 'A', 'requires fwd EPS data'),
        ('40', 'Mag-7 EPS Revision composite',  'Earnings',   'A10 manual loader (planned)','Finnhub per-ticker',                     'needs_paid_api',       'low_paid', 'A', 'Finnhub free tier may cover'),
        ('42', 'Copper/Gold ratio',             'Cross-Asset','A7 yahoo loader (planned)',  'HG=F / GC=F via yfinance',               'available_yahoo',      'free',     'A', 'classical growth proxy'),
        ('46', 'FOMC blackout window',          'Calendar',   'A10 manual loader (planned)','Fed FOMC schedule (manual)',             'needs_manual_csv',     'free',     'A', 'computable from FOMC meeting dates'),
        ('49', 'Google Trends "recession" 90d Z','NLP/Search','new module (planned)',       'pytrends',                               'needs_implementation', 'free',     'A', 'rate-limited; needs caching'),
        ('50', 'Fed minutes hawkish-dovish NLP','NLP',        'A10 manual loader (planned)','FOMC minutes + LLM',                     'needs_paid_api',       'mid_paid', 'A', 'requires LLM API'),
    ]
    for sid, name, cat, mod, src, status, cost, prio, notes in tier1_a:
        rows.append({'signal_id': f'T1A-{sid}', 'name': name, 'category': cat,
                     'source_module': mod, 'data_source_url_or_path': src,
                     'data_status': status, 'earliest_date': '', 'latest_date': '',
                     'n_obs': '', 'cost_tier': cost, 'priority': prio, 'notes': notes})

    # -------- 2.2 Tier1 ○ (16 signals, priority=B) --------
    tier1_b = [
        ('4',  'A/D Line price divergence',     'Breadth',    'A6 yahoo loader (planned)',  'Yahoo (computed)',                'needs_implementation', 'free',     'B', 'computed from advancers/decliners'),
        ('11', 'GVZ (Gold Vol)',                'Vol',        'A7 yahoo loader (planned)',  '^GVZ via yfinance',               'available_yahoo',      'free',     'B', ''),
        ('14', 'NAAIM Exposure Index',          'Sentiment',  'A10 manual loader (planned)','NAAIM weekly CSV',                'needs_manual_csv',     'free',     'B', 'weekly Wednesday release'),
        ('18', 'QQQ creation/redemption',       'Flow',       'A10 manual loader (planned)','ICI / Finnhub',                   'needs_paid_api',       'low_paid', 'B', ''),
        ('19', 'GLD/TLT flows',                 'Flow',       'A10 manual loader (planned)','ICI / Finnhub',                   'needs_paid_api',       'low_paid', 'B', ''),
        ('22', 'ICE BofA IG OAS',               'Credit',     'A6 fred loader (planned)',   'FRED BAMLC0A0CM',                 'available_fred',       'free',     'B', ''),
        ('24', 'SOFR-IORB spread',              'Liquidity',  'A6 fred loader (planned)',   'NY Fed / FRED (SOFR, IORB)',      'available_fred',       'free',     'B', ''),
        ('31', '10Y-2Y real yield diff',        'YieldCurve', 'A6 fred loader (planned)',   'FRED DFII10 - DFII2',             'available_fred',       'free',     'B', ''),
        ('33', 'NY Fed Nowcast',                'Macro',      'A10 manual loader (planned)','NY Fed weekly publication',       'needs_manual_csv',     'free',     'B', 'currently paused; check status'),
        ('35', 'Cleveland Fed Inflation Nowcast','Macro',     'A10 manual loader (planned)','Cleveland Fed monthly',           'needs_manual_csv',     'free',     'B', ''),
        ('39', 'NDX Forward PE z-score',        'Earnings',   'A10 manual loader (planned)','EODHD / Finnhub',                 'needs_paid_api',       'low_paid', 'B', ''),
        ('43', 'Silver/Gold ratio',             'Cross-Asset','A7 yahoo loader (planned)',  'SI=F / GC=F via yfinance',        'available_yahoo',      'free',     'B', ''),
        ('44', 'Oil (WTI) 5d change',           'Cross-Asset','A7 yahoo loader (planned)',  'CL=F via yfinance',               'available_yahoo',      'free',     'B', ''),
        ('47', 'Mag-7 earnings season flag',    'Calendar',   'A10 manual loader (planned)','Earnings calendar',               'needs_manual_csv',     'free',     'B', 'computable from earnings dates'),
        ('51', 'Headline News risk-off NLP',    'NLP',        'A10 manual loader (planned)','News API + LLM',                  'needs_paid_api',       'mid_paid', 'B', ''),
        ('52', 'Google Trends TQQQ/QQQ',        'NLP/Search', 'new module (planned)',       'pytrends',                        'needs_implementation', 'free',     'B', ''),
    ]
    for sid, name, cat, mod, src, status, cost, prio, notes in tier1_b:
        rows.append({'signal_id': f'T1B-{sid}', 'name': name, 'category': cat,
                     'source_module': mod, 'data_source_url_or_path': src,
                     'data_status': status, 'earliest_date': '', 'latest_date': '',
                     'n_obs': '', 'cost_tier': cost, 'priority': prio, 'notes': notes})

    # -------- 2.3 NEW (25 signals, priority=N) --------
    new_signals = [
        ('NEW-1',  'SPX 25-delta put skew',                'Options-Implied', 'A9 cboe loader (planned)',   'CBOE SKEW or computed',         'available_cboe',       'free',     'N', 'tail risk'),
        ('NEW-2',  'GEX (S&P 500 Gamma Exposure)',         'Options',         'new module (planned)',       'SqueezeMetrics scrape',         'needs_implementation', 'mid_paid', 'N', 'paid but scrapable mirror'),
        ('NEW-3',  'DIX (Dark Index)',                     'Options',         'new module (planned)',       'SqueezeMetrics scrape',         'needs_implementation', 'mid_paid', 'N', ''),
        ('NEW-4',  '0DTE option flow ratio',               'Options',         'A9 cboe loader (planned)',   'CBOE aggregate',                'available_cboe',       'free',     'N', ''),
        ('NEW-5',  'VVIX/VIX ratio',                       'Options',         'A7 yahoo loader (planned)',  'derived from Yahoo ^VVIX, ^VIX', 'available_yahoo',     'free',     'N', 'vol risk premium'),
        ('NEW-6',  'VIX9D/VIX ratio',                      'Options',         'A7 yahoo loader (planned)',  'derived from Yahoo ^VIX9D, ^VIX','available_yahoo',     'free',     'N', 'short-term stress'),
        ('NEW-7',  'TED-equivalent (DTB3-SOFR)',           'Cross-Asset',     'A6 fred loader (planned)',   'FRED DTB3, SOFR (computed)',    'available_fred',       'free',     'N', 'bank funding stress'),
        ('NEW-8',  'USD/JPY 3M XCB',                       'Cross-Asset',     'new module (planned)',       'Bloomberg/Polygon',             'needs_paid_api',       'mid_paid', 'N', ''),
        ('NEW-9',  'HYG/SHY ratio',                        'Cross-Asset',     'A7 yahoo loader (planned)',  'HYG, SHY via yfinance',         'available_yahoo',      'free',     'N', 'risk-on/off'),
        ('NEW-10', 'Bond-Stock corr rolling 60d',          'Cross-Asset',     'computed locally',           'derived from repo prices',      'available_repo',       'free',     'N', 'compute via existing CSVs'),
        ('NEW-11', 'AAA-BAA spread',                       'Cross-Asset',     'A6 fred loader (planned)',   'FRED AAA, BAA (also REPO-1)',   'available_fred',       'free',     'N', 'duplicates REPO-1'),
        ('NEW-12', 'Single-A OAS',                         'Cross-Asset',     'A6 fred loader (planned)',   'FRED BAMLC1A0C13Y',             'available_fred',       'free',     'N', ''),
        ('NEW-13', 'Fed Balance Sheet weekly delta',       'Policy',          'A6 fred loader (planned)',   'FRED WALCL',                    'available_fred',       'free',     'N', ''),
        ('NEW-14', 'Treasury General Account balance',     'Policy',          'A6 fred loader (planned)',   'FRED WTREGEN',                  'available_fred',       'free',     'N', ''),
        ('NEW-15', 'Reverse Repo balance',                 'Policy',          'A6 fred loader (planned)',   'FRED RRPONTSYD',                'available_fred',       'free',     'N', ''),
        ('NEW-16', 'Net Treasury issuance schedule',       'Policy',          'A10 manual loader (planned)','Treasury QRA',                  'needs_manual_csv',     'free',     'N', ''),
        ('NEW-17', 'Fed Funds Rate change delta',          'Policy',          'A6 fred loader (planned)',   'FRED DFF (also REPO-3)',        'available_fred',       'free',     'N', 'duplicates REPO-3'),
        ('NEW-18', '30Y-10Y term premium',                 'YieldCurve',      'computed locally',           'repo dgs30/dgs10 (also REPO-2)', 'available_repo',      'free',     'N', 'duplicates REPO-2'),
        ('NEW-19', 'TIPS 5Y-10Y slope',                    'YieldCurve',      'A6 fred loader (planned)',   'FRED DFII5, DFII10',            'available_fred',       'free',     'N', ''),
        ('NEW-20', 'Yield curve curvature',                'YieldCurve',      'A6 fred loader (planned)',   'FRED DGS2, DGS5, DGS10',        'available_fred',       'free',     'N', '5Y - (2Y+10Y)/2'),
        ('NEW-21', 'Bloomberg ECO US Surprise',            'Macro Surprise',  'new module (planned)',       'Bloomberg ECO',                 'needs_paid_api',       'mid_paid', 'N', ''),
        ('NEW-22', 'ISM Manufacturing PMI delta',          'Macro',           'A6 fred loader (planned)',   'FRED NAPM',                     'available_fred',       'free',     'N', ''),
        ('NEW-23', 'Initial Jobless Claims weekly delta',  'Macro',           'A6 fred loader (planned)',   'FRED ICSA',                     'available_fred',       'free',     'N', ''),
        ('NEW-24', 'Wikipedia "bear market" page views',   'Behavioral',      'new module (planned)',       'Wikimedia REST API',            'needs_implementation', 'free',     'N', 'free public API'),
        ('NEW-25', 'RVOL vs IV gap (60d - VIX)',           'Volatility',      'computed locally',           'repo prices + vix',             'available_repo',       'free',     'N', 'compute via existing CSVs'),
    ]
    for sid, name, cat, mod, src, status, cost, prio, notes in new_signals:
        rows.append({'signal_id': sid, 'name': name, 'category': cat,
                     'source_module': mod, 'data_source_url_or_path': src,
                     'data_status': status, 'earliest_date': '', 'latest_date': '',
                     'n_obs': '', 'cost_tier': cost, 'priority': prio, 'notes': notes})

    # -------- 2.4 REPO-derived (10 signals, priority=R) --------
    rmeta = repo_meta()
    def rmeta_lookup(key, field, default=''):
        if key in rmeta:
            v = rmeta[key].get(field, default)
            return '' if pd.isna(v) else v
        return default

    repo_signals = [
        ('REPO-1',  'AAA-BAA spread (credit quality)',         'Cross-Asset', 'scripts/derive_repo_signals.py', 'data/aaa_monthly.csv + data/baa_monthly.csv',           'available_repo',       'free', 'R', 'materialized'),
        ('REPO-2',  '30Y-10Y term premium',                    'YieldCurve',  'scripts/derive_repo_signals.py', 'data/dgs30_daily.csv + data/dgs10_daily.csv',          'available_repo',       'free', 'R', 'materialized'),
        ('REPO-3',  'Fed Funds Δ (1mo)',                        'Policy',     'scripts/derive_repo_signals.py', 'data/dff_daily.csv',                                   'available_repo',       'free', 'R', 'materialized'),
        ('REPO-4',  'Fed Funds vs 10Y spread',                 'YieldCurve',  'scripts/derive_repo_signals.py', 'data/dff_daily.csv + data/dgs10_daily.csv',            'available_repo',       'free', 'R', 'materialized'),
        ('REPO-5',  'DGP (DB Gold 2x ETN) 21d log-return',     'Cross-Asset', 'scripts/derive_repo_signals.py', 'data/dgp_daily.csv',                                   'available_repo',       'free', 'R', 'identified: 2x gold ETN via DXY beta=-1.98'),
        ('REPO-6',  'DRN (Direxion 3x REIT) 21d log-return',   'Cross-Asset', 'scripts/derive_repo_signals.py', 'data/drn_daily.csv',                                   'available_repo',       'free', 'R', 'identified: 3x VNQ-equivalent, beta=2.93'),
        ('REPO-7',  'CPI YoY surprise vs 12mo trailing avg',   'Macro',       'scripts/derive_repo_signals.py', 'data/cpiaucsl_monthly.csv',                            'available_repo',       'free', 'R', 'materialized'),
        ('REPO-8',  'ML predictions sign',                     'ML',          'scripts/derive_repo_signals.py', 'data/ml_oos_predictions.csv',                          'available_repo',       'free', 'R', 'materialized (pred col sign)'),
        ('REPO-9',  'ML features PC1',                         'ML',          'scripts/derive_repo_signals.py', 'data/ml_features.csv',                                 'available_repo',       'free', 'R', 'materialized (StandardScaler+PCA n=1, 120 features)'),
        ('REPO-10', 'Macro features (existing engineered)',    'Macro',       'pre-existing in repo',           'data/macro_features.csv',                              'available_repo',       'free', 'R', '50 cols 1974-2026; treat as feature pool, pick best by IC in Session 5'),
    ]
    repo_id_to_key = {
        'REPO-1': 'repo_1', 'REPO-2': 'repo_2', 'REPO-3': 'repo_3', 'REPO-4': 'repo_4',
        'REPO-5': 'repo_5', 'REPO-6': 'repo_6', 'REPO-7': 'repo_7', 'REPO-8': 'repo_8',
        'REPO-9': 'repo_9',
    }
    for sid, name, cat, mod, src, status, cost, prio, notes in repo_signals:
        ed_start = rmeta_lookup(repo_id_to_key.get(sid, ''), 'date_start', '')
        ed_end = rmeta_lookup(repo_id_to_key.get(sid, ''), 'date_end', '')
        n_obs = rmeta_lookup(repo_id_to_key.get(sid, ''), 'n_obs', '')
        rows.append({'signal_id': sid, 'name': name, 'category': cat,
                     'source_module': mod, 'data_source_url_or_path': src,
                     'data_status': status, 'earliest_date': ed_start, 'latest_date': ed_end,
                     'n_obs': n_obs, 'cost_tier': cost, 'priority': prio, 'notes': notes})

    df = pd.DataFrame(rows, columns=['signal_id', 'name', 'category', 'source_module',
                                     'data_source_url_or_path', 'data_status',
                                     'earliest_date', 'latest_date', 'n_obs',
                                     'cost_tier', 'priority', 'notes'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} signals to {OUT.relative_to(ROOT)}")
    print("\nBreakdown by data_status:")
    print(df['data_status'].value_counts().to_string())
    print("\nBreakdown by priority:")
    print(df['priority'].value_counts().to_string())
    print("\nBreakdown by cost_tier:")
    print(df['cost_tier'].value_counts().to_string())


if __name__ == '__main__':
    main()
