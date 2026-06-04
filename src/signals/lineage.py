"""Generate data_lineage.md from tier1_selection CSV."""
from __future__ import annotations
from pathlib import Path
from typing import Union
from .metadata import load_registry


def generate_lineage_markdown(registry_path: Union[str, Path]) -> str:
    metas = load_registry(registry_path)
    lines = ["# Signal Data Lineage", "",
             "Generated from `data/signals/tier1_selection_20260603.csv`. "
             "Single source of truth for signal metadata.",
             "",
             "| ID | Name | Category | Asset | Source | Lag | Earliest | Cost | Priority |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in metas:
        asset_str = '/'.join(m.target_assets)
        # Wrap single-letter asset codes in backticks to avoid collision with
        # the priority column (which uses bare | A |, | B |, | C |).
        if len(asset_str) == 1:
            asset_str = f"`{asset_str}`"
        lines.append(
            f"| {m.signal_id} | {m.name} | {m.category} | {asset_str} "
            f"| {m.source_module} | {m.publication_lag} | {m.earliest_date} "
            f"| {m.cost_tier} | {m.priority} |"
        )
    return "\n".join(lines) + "\n"
