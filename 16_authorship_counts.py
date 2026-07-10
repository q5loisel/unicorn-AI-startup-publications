#!/usr/bin/env python3
"""
16_authorship_counts.py
------------------------------
Counts, over included_papers.csv, of three authorship-attribution
situations relevant to interpreting lead_startup/canonical_name:

  (1) Leadership position : a startup-affiliated author is first OR last
                             author (first_author / last_author non-empty)
  (2) Alphabetical order   : alpha_order == "YES" (author order carries no
                             positional-seniority signal)
  (3) EVEN case            : lead_startup == "EVEN" (two or more startups
                             tied on affiliated-author count, no unique lead)

Input
-----
  included_papers.csv

Note
----
5 of the 3,410 rows are catastrophically corrupted by a CSV parsing failure
upstream (see 09_apply_edge_cases.py, Decision 5) — these are
flagged with canonical_name == "GARBLED_EXCLUDED" and are excluded here
before counting, since their first_author/last_author/alpha_order fields
contain address fragments, not real values.
"""

import os

import pandas as pd

# Input is read from a single, user-defined folder — set the
# AI_UNICORN_DATA_DIR environment variable to point at it (it defaults to the
# current working directory, i.e. run this script from inside that folder).
DATA_DIR   = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
PAPERS_CSV = os.path.join(DATA_DIR, "included_papers.csv")

df = pd.read_csv(PAPERS_CSV, low_memory=False)
total_raw = len(df)

clean = df[df["canonical_name"] != "GARBLED_EXCLUDED"].copy()
n_excluded = total_raw - len(clean)

fa = clean["first_author"].notna() & (clean["first_author"].astype(str).str.strip() != "")
la = clean["last_author"].notna()  & (clean["last_author"].astype(str).str.strip()  != "")
leadership = fa | la
alpha_yes  = clean["alpha_order"] == "YES"
even       = clean["lead_startup"] == "EVEN"

print(f"Total rows in included_papers.csv       : {total_raw:,}")
print(f"Excluded (garbled rows, Decision 5)      : {n_excluded:,}")
print(f"Usable rows                              : {len(clean):,}")
print()
print(f"(1) Leadership position (first or last)  : {int(leadership.sum()):,}")
print(f"      - first author only                : {int((fa & ~la).sum()):,}")
print(f"      - last author only                  : {int((la & ~fa).sum()):,}")
print(f"      - both first and last               : {int((fa & la).sum()):,}")
print(f"(2) Alphabetical order (alpha_order=YES) : {int(alpha_yes.sum()):,}")
print(f"(3) EVEN case (lead_startup=EVEN)        : {int(even.sum()):,}")
print()
print("Overlap (categories are not mutually exclusive):")
print(f"  Leadership & alphabetical : {int((leadership & alpha_yes).sum()):,}")
print(f"  Leadership & EVEN         : {int((leadership & even).sum()):,}")
print(f"  Alphabetical & EVEN       : {int((alpha_yes & even).sum()):,}")
