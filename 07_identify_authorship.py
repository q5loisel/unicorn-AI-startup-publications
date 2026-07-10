"""
Round 1 identification for included_papers.csv.

Six new columns are added:
  I1  alpha_order    : YES / NO / N/A  – authors sorted alphabetically by last name
  I2  first_author   : startup(s) of the first author (pipe-sep if multiple)
      middle_author  : unique startup(s) across all middle authors (pipe-sep)
      last_author    : startup(s) of the last author (pipe-sep if multiple)
  I3  lead_startup   : startup with the most affiliated authors in the paper;
                       EVEN if two or more startups are tied
  I4  canonical_name : canonical/current name after alias resolution
                       (from aliases.md; same as lead_startup when no alias found)

Output: included_papers.csv (updated in place)

All input/output files are read from and written to a single, user-defined
folder — set the AI_UNICORN_DATA_DIR environment variable to point at it (it
defaults to the current working directory, i.e. run this script from inside
that folder).
"""

import os
import re
import time
from collections import Counter

import pandas as pd

DATA_DIR = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())

FILE_IN    = os.path.join(DATA_DIR, "included_papers.csv")
FILE_OUT   = os.path.join(DATA_DIR, "included_papers.csv")
FILE_AFFIL = os.path.join(DATA_DIR, "startup_affiliation_dictionary.csv")
FILE_ALIAS = os.path.join(DATA_DIR, "aliases.md")


# =============================================================================
# Helpers
# =============================================================================

def build_affil_index(filepath: str) -> dict:
    """variant_lower → canonical startup name."""
    aff_df = pd.read_csv(filepath)
    return {
        str(row["affiliation_variant"]).strip().lower(): str(row["startup"]).strip()
        for _, row in aff_df.iterrows()
    }


def load_aliases(filepath: str) -> dict:
    """
    Parse aliases.md with indented format:
        CanonicalName
            AliasName1
            AliasName2

        CanonicalName2
            AliasName3

    Non-indented non-blank lines set the current canonical name.
    Indented lines are aliases that map → that canonical name.
    """
    alias_map: dict = {}
    current_canonical: str = ""
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if line[0] in (" ", "\t"):
                    # Indented → alias for current canonical
                    if current_canonical:
                        alias_map[stripped.lower()] = current_canonical
                else:
                    # New canonical name
                    current_canonical = stripped
    except FileNotFoundError:
        pass
    return alias_map


def parse_authors(text) -> list:
    """
    Parse 'Authors & Affiliations' into a list of (name, [inst1, inst2, …]).
    Each inst is the text before the first comma in its address segment
    (i.e. the institution identifier, matching the affiliation dictionary).
    Authors are separated by ';', affiliations within [...] by '|'.
    """
    if pd.isna(text) or not str(text).strip():
        return []

    authors = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\s*\[([^\]]*)\]\s*$", part)
        if m:
            name   = m.group(1).strip()
            insts  = [seg.split(",")[0].strip()
                      for seg in m.group(2).split("|") if seg.strip()]
        else:
            name  = part.strip()
            insts = []
        if name:
            authors.append((name, insts))
    return authors


def insts_to_startups(insts: list, variant_to_startup: dict) -> list:
    """Map institution names to startup names (unique, order-preserving)."""
    seen = []
    for inst in insts:
        s = variant_to_startup.get(inst.lower())
        if s and s not in seen:
            seen.append(s)
    return seen


def get_last_name(author_name: str) -> str:
    return author_name.split(",")[0].strip().lower()


# =============================================================================
# Per-row identification
# =============================================================================

def identify(row, variant_to_startup: dict, alias_map: dict) -> tuple:
    authors = parse_authors(row.get("Authors & Affiliations", ""))
    n = len(authors)

    # ------------------------------------------------------------------
    # I1 – Alphabetical order
    # ------------------------------------------------------------------
    if n < 2:
        alpha = "N/A"
    else:
        last_names = [get_last_name(name) for name, _ in authors]
        alpha = "YES" if last_names == sorted(last_names) else "NO"

    # ------------------------------------------------------------------
    # I2 – First / middle / last author startup affiliation
    # ------------------------------------------------------------------
    def position_startups(idx: int) -> str:
        _, insts = authors[idx]
        return " | ".join(insts_to_startups(insts, variant_to_startup))

    if n == 0:
        first_s = middle_s = last_s = ""
    elif n == 1:
        first_s  = position_startups(0)
        middle_s = ""
        last_s   = first_s
    elif n == 2:
        first_s  = position_startups(0)
        middle_s = ""
        last_s   = position_startups(1)
    else:
        first_s = position_startups(0)
        last_s  = position_startups(n - 1)
        # Unique startups across all middle authors, preserving first occurrence
        mid_startups: list = []
        for i in range(1, n - 1):
            for s in insts_to_startups(authors[i][1], variant_to_startup):
                if s not in mid_startups:
                    mid_startups.append(s)
        middle_s = " | ".join(mid_startups)

    # ------------------------------------------------------------------
    # I3 – Lead startup (most affiliated authors across the full author list)
    # ------------------------------------------------------------------
    counts: Counter = Counter()
    for _, insts in authors:
        for s in insts_to_startups(insts, variant_to_startup):
            counts[s] += 1

    if not counts:
        lead = ""
    else:
        max_count = max(counts.values())
        top = [s for s, c in counts.items() if c == max_count]
        lead = top[0] if len(top) == 1 else "EVEN"

    # ------------------------------------------------------------------
    # I4 – Canonical name via alias table
    # ------------------------------------------------------------------
    canonical = alias_map.get(lead.lower(), lead) if lead else ""

    return alpha, first_s, middle_s, last_s, lead, canonical


# =============================================================================
# Main
# =============================================================================

t0 = time.time()
print("Loading data …")
df = pd.read_csv(FILE_IN, low_memory=False)
print(f"  {len(df):,} records")

variant_to_startup = build_affil_index(FILE_AFFIL)
alias_map          = load_aliases(FILE_ALIAS)
print(f"  {len(variant_to_startup):,} affiliation variants loaded")
print(f"  {len(alias_map):,} aliases loaded")

print("\nRunning identifications …")
results = df.apply(lambda r: identify(r, variant_to_startup, alias_map), axis=1)

df["alpha_order"]    = results.apply(lambda r: r[0])
df["first_author"]   = results.apply(lambda r: r[1])
df["middle_author"]  = results.apply(lambda r: r[2])
df["last_author"]    = results.apply(lambda r: r[3])
df["lead_startup"]   = results.apply(lambda r: r[4])
df["canonical_name"] = results.apply(lambda r: r[5])

# =============================================================================
# Summary
# =============================================================================
print(f"""
=== I1 – Alphabetical order ===
{df['alpha_order'].value_counts().to_string()}

=== I2 – Author position affiliations ===
  Papers with startup first author   : {(df['first_author']  != '').sum():,}
  Papers with startup middle author  : {(df['middle_author'] != '').sum():,}
  Papers with startup last author    : {(df['last_author']   != '').sum():,}

=== I3 – Lead startup (top 20) ===
{df[df['lead_startup'] != '']['lead_startup'].value_counts().head(20).to_string()}

=== I4 – Aliases resolved ===
  Papers remapped to canonical name  : {(df['canonical_name'] != df['lead_startup']).sum():,}
""")

# =============================================================================
# Save
# =============================================================================
print(f"Saving to {FILE_OUT} …")
df.to_csv(FILE_OUT, index=False)
print(f"Done in {time.time()-t0:.1f}s")
