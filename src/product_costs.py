"""
Product Cost Constants for NASDAQ Backtest Simulation
======================================================
Single source of truth for per-product cost parameters.

All costs are expressed as ANNUAL rates (decimal, not %).
Convert to daily inside simulation: daily_rate = annual_rate / 252

Empirical basis:
- TQQQ financing: OLS regression beta_SOFR = -2.1306 (approx 2xSOFR)
- TMF financing: consistent with 3x leverage swap structure
- Gold 2x financing: 1xSOFR (2x futures uses 1x notional financing)

Last updated: 2026-05-12 (Scenario D corrected baseline)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductCost:
    """Immutable cost parameters for a single leveraged product."""
    ticker: str
    name: str
    leverage: float
    ter: float                  # Total Expense Ratio (annual)
    sofr_multiplier: float      # Financing = sofr_multiplier x SOFR
    swap_spread: float          # Annual spread on leverage swap
    dividend_yield: float       # Gross annual dividend yield
    div_tax_rate: float         # Effective tax on dividends (Japan resident)
    nisa_eligible: bool         # NISA eligibility (3x leverage = False)
    notes: str = ""

    @property
    def annual_drag_ex_sofr(self) -> float:
        """Total annual cost drag EXCLUDING SOFR (since SOFR is time-varying).
        Use for quick estimates only; simulation must apply SOFR daily."""
        after_tax_div_drag = self.dividend_yield * self.div_tax_rate
        return self.ter + self.swap_spread + after_tax_div_drag


# ---------------------------------------------------------------------------
# TQQQ -- ProShares UltraPro QQQ (3x NDX)
# ---------------------------------------------------------------------------
TQQQ = ProductCost(
    ticker="TQQQ",
    name="ProShares UltraPro QQQ",
    leverage=3.0,
    ter=0.0086,                 # 0.86%/yr
    sofr_multiplier=2.0,        # empirical: OLS beta_SOFR = -2.1306
    swap_spread=0.0050,         # 0.50%/yr
    dividend_yield=0.003,       # ~0.3%/yr (mostly reinvested)
    div_tax_rate=0.20315,
    nisa_eligible=False,
    notes="SOFR proxy in sim: DTB3 (FRED 3M T-bill). Beta_SOFR=-2.13 (OLS verified).",
)

# ---------------------------------------------------------------------------
# TMF -- Direxion Daily 20+ Year Treasury Bull 3x
# ---------------------------------------------------------------------------
TMF = ProductCost(
    ticker="TMF",
    name="Direxion Daily 20+ Year Treasury Bull 3x",
    leverage=3.0,
    ter=0.0091,                 # 0.91%/yr historical; current 1.06% but sim uses 0.91%
    sofr_multiplier=2.0,
    swap_spread=0.0050,
    dividend_yield=0.035,       # ~3.5%/yr monthly distributions
    div_tax_rate=0.20315,
    nisa_eligible=False,
    notes="Duration ~17yr effective; sim uses dgs30 + time-varying Dmod. TER 0.91% historical.",
)

# ---------------------------------------------------------------------------
# Gold 2x -- UGL (ProShares Ultra Gold, practical JP alternative)
# Simulation proxy was WisdomTree 2036 2x Gold ETP (LSE) -- NOT available at JP brokers
# ---------------------------------------------------------------------------
GOLD2X = ProductCost(
    ticker="UGL",
    name="ProShares Ultra Gold (2x)",
    leverage=2.0,
    ter=0.0095,                 # UGL 0.95%/yr (sim proxy WisdomTree 2036: 0.49%)
    sofr_multiplier=1.0,        # 2x futures: 1x notional SOFR financing
    swap_spread=0.0050,
    dividend_yield=0.0,
    div_tax_rate=0.20315,
    nisa_eligible=False,
    notes=(
        "WisdomTree 2036 (LSE) NOT available at JP retail brokers (SBI/Rakuten). "
        "UGL is practical product. TER gap: 0.95% (UGL) vs 0.49% (sim proxy). "
        "Sim previously omitted SOFR financing -- corrected in v2 of corrected_strategy_backtest.py."
    ),
)

# ---------------------------------------------------------------------------
# Gold 2x -- WisdomTree 2036 (LSE ETP) -- USER-CHOSEN gold 2x product
# Cost basis: analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md §5-1
# ---------------------------------------------------------------------------
GOLD2X_2036 = ProductCost(
    ticker="2036",
    name="WisdomTree Gold 2x Daily Leveraged (LSE 2036)",
    leverage=2.0,
    ter=0.0050,                 # 0.50%/yr (disclosed ~0.49%)
    sofr_multiplier=1.0,        # 2x leverage -> 1x notional financing
    swap_spread=0.0050,         # 0.50%/yr
    dividend_yield=0.0,
    div_tax_rate=0.20315,
    nisa_eligible=False,
    notes="User-chosen gold 2x. LSE-listed ETP; verify JP-broker availability "
          "(product_costs previously flagged it as not at SBI/Rakuten retail).",
)

# ---------------------------------------------------------------------------
# 1x mutual funds / ETFs (no leverage, no SOFR) -- researched SBI products
# Cost basis: analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md §5-2
# Execution lag T+5 business days (vs T+2 for leveraged ETFs).
# ---------------------------------------------------------------------------
NASDAQ1X = ProductCost(
    ticker="SBI-NASDAQ100",
    name="SBI NASDAQ100 Index Fund (1x)",
    leverage=1.0, ter=0.001958, sofr_multiplier=0.0, swap_spread=0.0,
    dividend_yield=0.0, div_tax_rate=0.20315, nisa_eligible=True,
    notes="Researched 信託報酬 0.1958% (alt: Nissay NASDAQ100 0.2035%). Lag T+5.",
)
GOLD1X = ProductCost(
    ticker="SBI-GOLD",
    name="SBI iShares Gold Fund (sakutto-jun-kin, 1x, unhedged)",
    leverage=1.0, ter=0.001838, sofr_multiplier=0.0, swap_spread=0.0,
    dividend_yield=0.0, div_tax_rate=0.20315, nisa_eligible=True,
    notes="Researched 信託報酬 0.1838%. LBMA gold. Lag T+5.",
)
BOND1X = ProductCost(
    ticker="2255",
    name="iShares 20+ Year US Treasury ETF (2255, 1x, unhedged)",
    leverage=1.0, ter=0.00154, sofr_multiplier=0.0, swap_spread=0.0,
    dividend_yield=0.0, div_tax_rate=0.20315, nisa_eligible=True,
    notes="Researched 信託報酬 0.154%. Lag T+5.",
)

# Execution-lag constants (business days): leveraged ETF vs 1x fund.
EXEC_LAG_ETF = 2     # T+2
EXEC_LAG_FUND_1X = 5  # T+5
PER_TRADE_COST = 0.0010  # 0.10% one-way (moderate), applied x turnover

# ---------------------------------------------------------------------------
# >3x leverage via exchange CFD (くりっく株365 NASDAQ-100) — margin-reservation
# cost assumption. MANDATORY for any strategy whose effective leverage exceeds
# 3x (i.e. beyond a 3x ETF), because the >3x EXCESS notional must be held on a
# margin CFD that requires posted collateral. See EVALUATION_STANDARD §1.5.
# Basis: PRODUCT_COST_COMPARISON_2026-06-10.md §9 / MARGIN_CAPACITY_STRESS_RESULTS_20260617.md
# ---------------------------------------------------------------------------
K365_FINANCING_SPREAD = 0.0075   # k365 金利相当 ≈ SOFR + 0.75pp (exchange CFD, 業者マージン極小)
K365_EXCESS_EXTRA = K365_FINANCING_SPREAD - TQQQ.swap_spread  # 0.0025: extra over TQQQ swap on (L-3)+ notional
K365_MARGIN_RATE_MIN = 0.0424    # 取引所最小証拠金 ¥9,320/枚 ÷ 想定元本¥220K/枚 ≈ 4.24% (危険・評価非採用)
K365_MARGIN_RATE_STD = 0.08      # 評価標準: 最小の約2倍。全期間52年で強制清算0回 (M1-M3/M5b)
K365_MARGIN_RATE_CONSERV = 0.12  # 保守: 1987 Black Monday(-11.35%)も耐える
K365_MARGIN_ROLL_COST = 0.0020   # 四半期ロールの bid-ask 概算 ~0.2%/yr on notional (推定・DATA GAP)
K365_MARGIN_NONEARNING = True    # 取り置き証拠金は無利息 (機会損失 = margin x SOFR)
# Realistic margin-funded drag (S2 isolated account + roll cost, m=8%): ~-0.7~-0.9pp
# for high-lever configs (sc1.35 -0.9 / B3a -0.7 / P09 -0.1pp). High-lever advantage preserved.

# ---------------------------------------------------------------------------
# Japan tax constants
# ---------------------------------------------------------------------------
JP_CAPITAL_GAINS_TAX = 0.20315   # 所得税15% + 復興税0.315% + 住民税5%
JP_DIVIDEND_TAX_EFFECTIVE = 0.20315  # via niju-kazei-chosei-seido (treaty credit)

# Rebalancing tax drag estimate at ~27 trades/yr
# After-tax CAGR reduction vs pre-tax: -2.8% (conservative) to -5.2% (worst-case)
REBALANCING_TRADES_PER_YEAR = 27
REBALANCING_TAX_DRAG_LOW  = 0.028   # best-case (partial realization)
REBALANCING_TAX_DRAG_HIGH = 0.052   # worst-case (all gains realized each rebalance)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PRODUCTS = {"TQQQ": TQQQ, "TMF": TMF, "GOLD2X": GOLD2X}

# User-chosen leverage set (no CFD; NASDAQ/Bond max 3x, Gold max 2x) + 1x funds.
LEVERAGED_SET = {"NASDAQ": TQQQ, "Gold": GOLD2X_2036, "Bond": TMF}
FUND_1X_SET = {"NASDAQ": NASDAQ1X, "Gold": GOLD1X, "Bond": BOND1X}


def daily_financing_cost(product: ProductCost, sofr_annual: float) -> float:
    """Daily financing cost (decimal) for given SOFR annual rate (decimal).

    daily = (sofr_multiplier x SOFR + swap_spread) / 252
    """
    return (product.sofr_multiplier * sofr_annual + product.swap_spread) / 252.0


def daily_ter(product: ProductCost) -> float:
    """Daily TER deduction (decimal)."""
    return product.ter / 252.0


if __name__ == "__main__":
    print("Product Cost Constants Summary")
    print("=" * 70)
    for key, p in PRODUCTS.items():
        print(f"\n{key} ({p.ticker}) -- {p.name}")
        print(f"  Leverage : {p.leverage}x")
        print(f"  TER      : {p.ter*100:.2f}%/yr")
        print(f"  Financing: {p.sofr_multiplier}xSOFR + {p.swap_spread*100:.2f}% swap")
        print(f"  Div yield: {p.dividend_yield*100:.2f}%/yr")
        print(f"  NISA     : {'Yes' if p.nisa_eligible else 'No (3x leverage prohibited)'}")
        if p.notes:
            print(f"  Notes    : {p.notes}")
    print(f"\nJP Tax: {JP_CAPITAL_GAINS_TAX*100:.3f}%")
    print(f"Rebalancing drag: -{REBALANCING_TAX_DRAG_LOW*100:.1f}% to -{REBALANCING_TAX_DRAG_HIGH*100:.1f}% CAGR @ {REBALANCING_TRADES_PER_YEAR} trades/yr")
