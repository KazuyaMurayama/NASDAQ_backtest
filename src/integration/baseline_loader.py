"""Load baseline NAVs for S1 (NEW CANDIDATE = F10), S2 (D5 = vz065lmax5), S3 (DH-W1).

S1: audit_results/_cache/f10_nav_cache.pkl (key='nav_f10')
S2: audit_results/_cache/vz065lmax5_nav_cache.pkl (key='nav_vz065lmax5')
S3: dh_w1_nav_cache.pkl if cached; else regenerate via build_W1(a) from
    src/g23a_dh_refinement_variants.py (call requires load_shared_assets()).

Cache schema (S1/S2):
    dict with keys:
      'dates'        : pd.Series[datetime64] length N (~13169)
      'nav_<name>'   : pd.Series[float] length N (NAV starting at 1.0)
      + many other arrays (close, ret, sofr, gold_2x, bond_3x, lev_A, w*_A,
        lev_mod_*, vz, lt_sig, raw_a2, etc.)
"""
from __future__ import annotations
from pathlib import Path
import pickle
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / 'audit_results' / '_cache'


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_s1_nav() -> pd.Series:
    """S1: NEW CANDIDATE (F10) — audit_results/_cache/f10_nav_cache.pkl."""
    return _load_named_nav('f10_nav_cache.pkl', 'nav_f10', 'S1')


def load_s2_nav() -> pd.Series:
    """S2: D5 (vz065lmax5) — audit_results/_cache/vz065lmax5_nav_cache.pkl."""
    return _load_named_nav('vz065lmax5_nav_cache.pkl', 'nav_vz065lmax5', 'S2')


def load_s3_nav() -> pd.Series:
    """S3: DH-W1.

    First tries cached pickle (multiple candidate paths). If none found,
    raises FileNotFoundError with regen instructions. Cache regeneration
    is deferred to Session S2 (see SIGNAL_INTEGRATION_PLAN_20260604.md).
    """
    candidates = [
        CACHE_DIR / 'dh_w1_nav_cache.pkl',
        CACHE_DIR / 'dhw1_nav_cache.pkl',
        CACHE_DIR / 'w1_nav_cache.pkl',
        ROOT / 'audit_results' / 'dh_w1' / 'nav.pkl',
    ]
    for p in candidates:
        if p.exists():
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            # Try preferred keys in order: nav_dh_w1 (current schema), nav_w1 (legacy)
            for key in ('nav_dh_w1', 'nav_w1'):
                if isinstance(obj, dict) and key in obj:
                    return _coerce_nav(obj, 'S3', preferred_key=key)
            return _coerce_nav(obj, 'S3', preferred_key='nav_dh_w1')
    raise FileNotFoundError(
        "S3 (DH-W1) NAV cache not found. Checked: "
        + ", ".join(str(p) for p in candidates)
        + ". To regenerate, call build_W1(load_shared_assets()) from "
        + "src/g23a_dh_refinement_variants.py - returns (nav, cost, mask, wn, lev_raw). "
        + "See scripts/regen_dh_w1_nav.py (TODO Session S2)."
    )


def load_all_baselines() -> pd.DataFrame:
    """Return DataFrame with columns ['S1', 'S2', 'S3'] indexed by date.

    S3 may be missing — in that case returns a 2-column DataFrame (S1, S2)
    and prints a warning. Subsequent sessions must regenerate S3 before
    Tier 1-4 evaluation can complete.
    """
    s1 = load_s1_nav()
    s2 = load_s2_nav()
    try:
        s3 = load_s3_nav()
        df = pd.concat([s1, s2, s3], axis=1).sort_index()
    except FileNotFoundError as e:
        print(f"[baseline_loader] WARNING: S3 missing - {e}")
        df = pd.concat([s1, s2], axis=1).sort_index()
    return df


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_named_nav(cache_filename: str, nav_key: str, label: str) -> pd.Series:
    p = CACHE_DIR / cache_filename
    if not p.exists():
        raise FileNotFoundError(f"{label} baseline cache not found: {p}")
    with open(p, 'rb') as f:
        obj = pickle.load(f)
    return _coerce_nav(obj, label, preferred_key=nav_key)


def _coerce_nav(obj, label: str, preferred_key: str | None = None) -> pd.Series:
    """Coerce a cached object into a date-indexed pd.Series named `label`."""
    # Case A: dict cache (our standard schema)
    if isinstance(obj, dict):
        dates = obj.get('dates')
        if dates is None:
            raise KeyError(
                f"{label} cache dict missing 'dates' key. Keys: {list(obj.keys())[:20]}"
            )
        # find NAV series
        nav_arr = None
        if preferred_key and preferred_key in obj:
            nav_arr = obj[preferred_key]
        else:
            for k in obj.keys():
                if k.lower().startswith('nav'):
                    nav_arr = obj[k]
                    break
        if nav_arr is None:
            raise KeyError(
                f"{label} cache dict missing 'nav_*' key. Keys: {list(obj.keys())[:20]}"
            )
        s = pd.Series(
            pd.to_numeric(pd.Series(nav_arr).reset_index(drop=True), errors='coerce').values,
            index=pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).reset_index(drop=True).values)),
            name=label,
        )
        return s.dropna()

    # Case B: pd.Series (already date-indexed)
    if isinstance(obj, pd.Series):
        return obj.rename(label)

    # Case C: pd.DataFrame — pick first NAV-like column
    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            s = obj[col]
            if s.dtype.kind == 'f' and 0.5 < float(s.iloc[0]) < 5.0:
                return s.rename(label)
        return obj.iloc[:, 0].rename(label)

    raise TypeError(f"Unknown cache object type for {label}: {type(obj)}")
