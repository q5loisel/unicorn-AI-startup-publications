#!/usr/bin/env python3
"""
09_apply_edge_cases.py
-----------------------------
Applies five post-identification edge-case decisions to included_papers.csv
and startups_data.csv.  Run this ONCE; the corrected files are then read by
12_concentration_analysis.py.

Decisions
---------
1. Alpha-mid papers (alpha_order=YES, startup in middle only):
   Included in the analysis.  canonical_name is already set.

2. EVEN papers (tied lead_startup attribution):
   (a) Attribute to ALL tied startups via a new 'shared_startups' column.
   (b) Each paper is still one record (no duplication in the CSV); the
       analysis counts +1 for each tied startup but +1 to the total only.

3. canonical_name = NaN: excluded (no startup found). No change needed;
   the analysis filter already drops them.

4. canonical_name = "Plus.ai": matched to "Plus" → rename startup in
   startups_data.csv to "Plus.ai" and update canonical_name accordingly.

5. Garbled rows (canonical_name not in any expected value): the five rows
   with corrupted CSV data are marked GARBLED_EXCLUDED and dropped by the
   analysis filter.

Adds / modifies columns in included_papers.csv
----------------------------------------------
  shared_startups  : pipe-separated canonical names of tied startups
                     (EVEN papers only; empty string for all others)
  edge_case_flag   : "alpha_mid" | "even_shared" | "plus_renamed" |
                     "garbled_excluded" | "" (normal)

All input/output files are read from and written to a single, user-defined
folder — set the AI_UNICORN_DATA_DIR environment variable to point at it (it
defaults to the current working directory, i.e. run this script from inside
that folder).
"""

import os, re, csv, sys
from collections import Counter
import pandas as pd

csv.field_size_limit(sys.maxsize)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
PAPERS_CSV  = os.path.join(DATA_DIR, "included_papers.csv")
STARTUPS_CSV= os.path.join(DATA_DIR, "startups_data.csv")
AFFIL_CSV   = os.path.join(DATA_DIR, "startup_affiliation_dictionary.csv")
ALIASES_MD  = os.path.join(DATA_DIR, "aliases.md")

# ── Load ──────────────────────────────────────────────────────────────────────
papers   = pd.read_csv(PAPERS_CSV, low_memory=False)
startups = pd.read_csv(STARTUPS_CSV)

affil_df = pd.read_csv(AFFIL_CSV)
v2s = {str(r["affiliation_variant"]).strip().lower(): str(r["startup"]).strip()
       for _, r in affil_df.iterrows()}

alias_map: dict = {}
current_canonical = ""
with open(ALIASES_MD, encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            continue
        if line[0] in (" ", "\t"):
            if current_canonical:
                alias_map[stripped.lower()] = current_canonical
        else:
            current_canonical = stripped


def canonicalise(name: str) -> str:
    return alias_map.get(name.strip().lower(), name.strip())


# ── Helper: find tied startups for an EVEN paper ─────────────────────────────
def find_tied_startups(authors_text, first_a, last_a, mid_a, matched_startups):
    """Return sorted list of canonical names of tied startups."""
    # Primary method: recount from Authors & Affiliations
    if authors_text and str(authors_text) != "nan":
        counts: Counter = Counter()
        for part in str(authors_text).split(";"):
            part = part.strip()
            m = re.match(r"^(.*?)\s*\[([^\]]*)\]", part)
            if not m:
                continue
            for seg in m.group(2).split("|"):
                inst = seg.split(",")[0].strip().lower()
                s = v2s.get(inst)
                if s:
                    counts[canonicalise(s)] += 1
        if counts:
            mx = max(counts.values())
            top = sorted(s for s, c in counts.items() if c == mx)
            if len(top) >= 2:
                return top

    # Fallback: use columns set by 07_identify_authorship.py
    sources = [first_a, last_a, mid_a, matched_startups]
    candidates = set()
    for src in sources:
        if src and str(src) != "nan":
            for part in str(src).split("|"):
                c = canonicalise(part.strip())
                if c:
                    candidates.add(c)
    return sorted(candidates) if len(candidates) >= 2 else sorted(candidates)


# ── Initialise new columns ────────────────────────────────────────────────────
if "shared_startups" not in papers.columns:
    papers["shared_startups"] = ""
if "edge_case_flag" not in papers.columns:
    papers["edge_case_flag"] = ""

papers["shared_startups"] = papers["shared_startups"].fillna("").astype(str)
papers["edge_case_flag"]  = papers["edge_case_flag"].fillna("").astype(str)

startup_names = set(startups["startup"])

# ── Decision 5: garbled rows ──────────────────────────────────────────────────
garbled_mask = (
    papers["canonical_name"].notna() &
    ~papers["canonical_name"].isin(startup_names | {"EVEN", "Plus.ai"}) &
    (papers["canonical_name"] != "")
)
papers.loc[garbled_mask, "canonical_name"] = "GARBLED_EXCLUDED"
papers.loc[garbled_mask, "edge_case_flag"] = "garbled_excluded"
n_garbled = garbled_mask.sum()
print(f"Decision 5 — garbled rows marked GARBLED_EXCLUDED: {n_garbled}")

# ── Decision 4: Plus.ai rename ────────────────────────────────────────────────
# Rename "Plus" → "Plus.ai" in startups_data.csv
startups.loc[startups["startup"] == "Plus", "startup"] = "Plus.ai"
# The 2 papers already have canonical_name = "Plus.ai" — no change needed there
n_plus = (papers["canonical_name"] == "Plus.ai").sum()
papers.loc[papers["canonical_name"] == "Plus.ai", "edge_case_flag"] = "plus_renamed"
print(f"Decision 4 — Plus → Plus.ai in startups_data.csv; {n_plus} paper(s) now match")

# ── Decision 2: EVEN papers → shared_startups ────────────────────────────────
even_mask = papers["canonical_name"] == "EVEN"
n_even_resolved = 0
n_even_unresolved = 0
for idx, row in papers[even_mask].iterrows():
    tied = find_tied_startups(
        row.get("Authors & Affiliations"),
        row.get("first_author"),
        row.get("last_author"),
        row.get("middle_author"),
        row.get("Matched Startups"),
    )
    if tied:
        papers.at[idx, "shared_startups"] = " | ".join(tied)
        papers.at[idx, "edge_case_flag"]  = "even_shared"
        n_even_resolved += 1
    else:
        papers.at[idx, "edge_case_flag"] = "even_unresolved"
        n_even_unresolved += 1
print(f"Decision 2 — EVEN papers: {n_even_resolved} resolved, {n_even_unresolved} unresolved")

# ── Decision 1: alpha-mid papers ─────────────────────────────────────────────
# These already have a valid canonical_name; just tag them.
papers["_has_leadership"] = papers["first_author"].notna() | papers["last_author"].notna()
papers["_is_alpha"]       = papers["alpha_order"].str.strip().str.upper() == "YES"
alpha_mid_mask = (
    papers["_is_alpha"] &
    ~papers["_has_leadership"] &
    papers["canonical_name"].isin(set(startups["startup"]) | {"Plus.ai"}) &
    (papers["edge_case_flag"] == "")
)
papers.loc[alpha_mid_mask, "edge_case_flag"] = "alpha_mid"
n_alpha_mid = alpha_mid_mask.sum()
print(f"Decision 1 — alpha-mid papers tagged: {n_alpha_mid}")

papers = papers.drop(columns=["_has_leadership", "_is_alpha"])

# ── Save ──────────────────────────────────────────────────────────────────────
papers.to_csv(PAPERS_CSV, index=False)
print(f"\nSaved updated included_papers.csv  ({len(papers):,} rows)")

startups.to_csv(STARTUPS_CSV, index=False)
print(f"Saved updated startups_data.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Edge-case flag summary ===")
print(papers["edge_case_flag"].value_counts().to_string())
print("\nShared startups (EVEN papers):")
for _, row in papers[papers["shared_startups"] != ""].iterrows():
    print(f"  {row['UID']:<30}  {row['shared_startups']}")
