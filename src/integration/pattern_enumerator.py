"""Enumerate all (signal_id, strategy, method, direction) patterns per Tier.

Plan reference: SIGNAL_INTEGRATION_PLAN_20260604.md §3 Tier table.

Tier 1: M1+M2 × 6 signals × 3 strategies × 2 directions      = 72 patterns
Tier 2: M3 (S3 only) × 6 × 2  +  M4 × 6 × 3 × 1  +  M5 × 6 × 3 × 2
                                                              = 12 + 18 + 36 = 66 patterns
Tier 3: AND/OR top-4 pairs (C(4,2)=6) × 2 ops × 3 strategies × M1_defensive
                                                              = 36 patterns
Tier 4: PCA composite (2 blocks) × 3 strategies × M2_defensive = 6 patterns

Total ~ 180 patterns.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union


# Phase B 通過 signal IDs (per plan §2.2)
PHASE_B_PASS_SIGNAL_IDS: List[int] = [6, 21, 23, 26, 28, 41]

# Top-4 IC signals for AND/OR combinations (per plan §3 Tier 3)
TOP4_IC_SIGNAL_IDS: List[int] = [21, 6, 41, 23]

# Phase B available composite blocks (per plan §3 Tier 4)
PCA_COMPOSITES: List[str] = ['sentiment', 'credit_stress']

# Baseline strategies (per plan §2.1)
BASE_STRATEGIES: List[str] = ['S1', 'S2', 'S3']


@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    tier: int
    signal_id: Union[int, str]   # int for single signals; str for combos/composites
    strategy: str                # 'S1' | 'S2' | 'S3'
    method: str                  # 'M1' | 'M2' | 'M3' | 'M4' | 'M5' | 'M1_combo' | 'M2_composite'
    direction: str               # 'defensive' | 'procyclical' | 'risk_off' | 'reverse' | 'vol_adj' | 'stop_only' | 'filter_entry'


# ------------------------------------------------------------------
# Tier enumerators
# ------------------------------------------------------------------

def enumerate_tier1() -> List[Pattern]:
    """M1 (lev mask) + M2 (lev tilt) × all signals × all strategies × 2 directions."""
    out: List[Pattern] = []
    for sid in PHASE_B_PASS_SIGNAL_IDS:
        for strat in BASE_STRATEGIES:
            for method in ['M1', 'M2']:
                for direction in ['defensive', 'procyclical']:
                    pid = f"T1_{method}_{sid}_{strat}_{direction}"
                    out.append(Pattern(pid, 1, sid, strat, method, direction))
    return out


def enumerate_tier2() -> List[Pattern]:
    """M3 (S3 only, 2 dirs) + M4 (all strats, 1 dir) + M5 (all strats, 2 dirs)."""
    out: List[Pattern] = []
    # M3: S3 only, 2 directions
    for sid in PHASE_B_PASS_SIGNAL_IDS:
        for direction in ['risk_off', 'reverse']:
            pid = f"T2_M3_{sid}_S3_{direction}"
            out.append(Pattern(pid, 2, sid, 'S3', 'M3', direction))
    # M4: all strategies, 1 direction
    for sid in PHASE_B_PASS_SIGNAL_IDS:
        for strat in BASE_STRATEGIES:
            pid = f"T2_M4_{sid}_{strat}_vol_adj"
            out.append(Pattern(pid, 2, sid, strat, 'M4', 'vol_adj'))
    # M5: all strategies, 2 directions
    for sid in PHASE_B_PASS_SIGNAL_IDS:
        for strat in BASE_STRATEGIES:
            for direction in ['stop_only', 'filter_entry']:
                pid = f"T2_M5_{sid}_{strat}_{direction}"
                out.append(Pattern(pid, 2, sid, strat, 'M5', direction))
    return out


def enumerate_tier3() -> List[Pattern]:
    """AND/OR combinations of top-4 IC signals × 3 strategies × M1 (defensive)."""
    pairs = [
        (TOP4_IC_SIGNAL_IDS[i], TOP4_IC_SIGNAL_IDS[j])
        for i in range(len(TOP4_IC_SIGNAL_IDS))
        for j in range(i + 1, len(TOP4_IC_SIGNAL_IDS))
    ]
    out: List[Pattern] = []
    for s1, s2 in pairs:
        for op in ['AND', 'OR']:
            combo_id = f"{s1}_{op}_{s2}"
            for strat in BASE_STRATEGIES:
                pid = f"T3_M1_{combo_id}_{strat}_defensive"
                out.append(Pattern(pid, 3, combo_id, strat, 'M1_combo', 'defensive'))
    return out


def enumerate_tier4() -> List[Pattern]:
    """PCA composite blocks × 3 strategies × M2 (defensive)."""
    out: List[Pattern] = []
    for comp in PCA_COMPOSITES:
        for strat in BASE_STRATEGIES:
            pid = f"T4_M2_{comp}_{strat}_defensive"
            out.append(Pattern(pid, 4, comp, strat, 'M2_composite', 'defensive'))
    return out


def enumerate_all() -> List[Pattern]:
    return enumerate_tier1() + enumerate_tier2() + enumerate_tier3() + enumerate_tier4()


def tier_counts() -> dict:
    """Return per-tier and total pattern counts (for plan/spec verification)."""
    return {
        'tier1': len(enumerate_tier1()),
        'tier2': len(enumerate_tier2()),
        'tier3': len(enumerate_tier3()),
        'tier4': len(enumerate_tier4()),
        'total': len(enumerate_all()),
    }


if __name__ == '__main__':
    import json
    print(json.dumps(tier_counts(), indent=2))
