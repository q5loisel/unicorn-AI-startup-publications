#!/usr/bin/env python3
"""
10_update_startups.py
-----------------------
Updates startups_data.csv to reflect the edge-case attribution decisions
applied by 09_apply_edge_cases.py.

Changes relative to the original 08_aggregate_startups.py run:
  - EVEN papers: added to each tied startup's counts (both lead & mid)
  - Plus.ai papers: now correctly attributed (startup renamed Plus → Plus.ai)
  - Alpha-mid papers: already in startups_data.csv (classified as lead
    by 08_aggregate_startups.py); no change needed
  - Garbled rows: excluded from counts
  - NaN rows: already absent from counts

Updates applied to startups_data.csv
-------------------------------------
  pub_lead_n/cit/hc/nhc  : add EVEN lead published + Plus.ai published
  pub_mid_n/cit/hc/nhc   : add EVEN mid  published
  pre_lead_n/cit/hc/nhc  : add EVEN lead preprint
  pre_mid_n/cit/hc/nhc   : add EVEN mid  preprint
  total_citations         : recomputed
  has_any_output          : recomputed
  has_highly_cited        : recomputed
  mean_citations_per_paper: recomputed
  median_citations_per_paper: recomputed

All input/output files are read from and written to a single, user-defined
folder — set the AI_UNICORN_DATA_DIR environment variable to point at it (it
defaults to the current working directory, i.e. run this script from inside
that folder).
"""

import os, csv, sys
from collections import defaultdict
import pandas as pd

csv.field_size_limit(sys.maxsize)

DATA_DIR     = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
PAPERS_CSV   = os.path.join(DATA_DIR, "included_papers.csv")
STARTUPS_CSV = os.path.join(DATA_DIR, "startups_data.csv")
HC_THRESHOLD = 200

# ── Load ──────────────────────────────────────────────────────────────────────
papers   = pd.read_csv(PAPERS_CSV, low_memory=False)
startups = pd.read_csv(STARTUPS_CSV)

# ── Paper preprocessing ───────────────────────────────────────────────────────
papers["cit"]          = pd.to_numeric(papers["Total Times Cited"], errors="coerce").fillna(0).astype(int)
papers["is_preprint"]  = papers["Document Type"].str.strip().str.lower() == "preprint"
papers["hc_200"]       = papers["cit"] >= HC_THRESHOLD
papers["edge_case_flag"]  = papers["edge_case_flag"].fillna("").astype(str)
papers["shared_startups"] = papers["shared_startups"].fillna("").astype(str)
papers["has_leadership"]  = papers["first_author"].notna() | papers["last_author"].notna()
papers["is_alpha"]        = papers["alpha_order"].str.strip().str.upper() == "YES"

startup_names = set(startups["startup"])

# ── Build delta rows: EVEN and Plus.ai papers (not yet in startups_data) ──────
# Alpha-mid already in startups_data (counted as lead by populate_startups_data)
# Garbled excluded; NaN not attributed.

new_papers = papers[papers["edge_case_flag"].isin(["even_shared", "plus_renamed"])].copy()
print(f"New papers to attribute: {len(new_papers)} "
      f"({(new_papers['edge_case_flag']=='even_shared').sum()} EVEN, "
      f"{(new_papers['edge_case_flag']=='plus_renamed').sum()} Plus.ai)")

# Expand EVEN papers per tied startup
rows_to_add = []
for _, row in new_papers.iterrows():
    if row["edge_case_flag"] == "even_shared":
        for startup in row["shared_startups"].split(" | "):
            startup = startup.strip()
            if startup in startup_names:
                rows_to_add.append({
                    "startup":      startup,
                    "is_preprint":  row["is_preprint"],
                    "hc_200":       row["hc_200"],
                    "cit":          row["cit"],
                    "is_lead":      row["has_leadership"] or row["is_alpha"],
                })
    elif row["edge_case_flag"] == "plus_renamed":
        # Plus.ai papers: canonical_name = "Plus.ai" (lead papers, both first+last)
        if "Plus.ai" in startup_names:
            rows_to_add.append({
                "startup":     "Plus.ai",
                "is_preprint": row["is_preprint"],
                "hc_200":      row["hc_200"],
                "cit":         row["cit"],
                "is_lead":     row["has_leadership"] or row["is_alpha"],
            })

delta = pd.DataFrame(rows_to_add)
print(f"Attribution rows: {len(delta)}")

# ── Aggregate delta by startup ────────────────────────────────────────────────
def agg_delta(sub: pd.DataFrame, is_preprint: bool, is_lead: bool) -> dict:
    mask = (sub["is_preprint"] == is_preprint) & (sub["is_lead"] == is_lead)
    s = sub[mask]
    return {
        "n":   len(s),
        "cit": int(s["cit"].sum()),
        "hc":  int(s["hc_200"].sum()),
        "nhc": int((~s["hc_200"]).sum()),
    }

delta_by_startup: dict = {}
for startup, sub in delta.groupby("startup"):
    delta_by_startup[startup] = {
        "pub_lead": agg_delta(sub, False, True),
        "pub_mid":  agg_delta(sub, False, False),
        "pre_lead": agg_delta(sub, True,  True),
        "pre_mid":  agg_delta(sub, True,  False),
    }

# ── Apply delta to startups_data ──────────────────────────────────────────────
breakdown_cols = [
    "pub_lead_n","pub_lead_cit","pub_lead_hc","pub_lead_nhc",
    "pub_mid_n", "pub_mid_cit", "pub_mid_hc", "pub_mid_nhc",
    "pre_lead_n","pre_lead_cit","pre_lead_hc","pre_lead_nhc",
    "pre_mid_n", "pre_mid_cit", "pre_mid_hc", "pre_mid_nhc",
]
for col in breakdown_cols:
    startups[col] = pd.to_numeric(startups[col], errors="coerce").fillna(0).astype(int)

updated = 0
for idx, row in startups.iterrows():
    s = row["startup"]
    if s not in delta_by_startup:
        continue
    d = delta_by_startup[s]
    startups.at[idx, "pub_lead_n"]   += d["pub_lead"]["n"]
    startups.at[idx, "pub_lead_cit"] += d["pub_lead"]["cit"]
    startups.at[idx, "pub_lead_hc"]  += d["pub_lead"]["hc"]
    startups.at[idx, "pub_lead_nhc"] += d["pub_lead"]["nhc"]
    startups.at[idx, "pub_mid_n"]    += d["pub_mid"]["n"]
    startups.at[idx, "pub_mid_cit"]  += d["pub_mid"]["cit"]
    startups.at[idx, "pub_mid_hc"]   += d["pub_mid"]["hc"]
    startups.at[idx, "pub_mid_nhc"]  += d["pub_mid"]["nhc"]
    startups.at[idx, "pre_lead_n"]   += d["pre_lead"]["n"]
    startups.at[idx, "pre_lead_cit"] += d["pre_lead"]["cit"]
    startups.at[idx, "pre_lead_hc"]  += d["pre_lead"]["hc"]
    startups.at[idx, "pre_lead_nhc"] += d["pre_lead"]["nhc"]
    startups.at[idx, "pre_mid_n"]    += d["pre_mid"]["n"]
    startups.at[idx, "pre_mid_cit"]  += d["pre_mid"]["cit"]
    startups.at[idx, "pre_mid_hc"]   += d["pre_mid"]["hc"]
    startups.at[idx, "pre_mid_nhc"]  += d["pre_mid"]["nhc"]
    updated += 1
    print(f"  Updated {s}: +{sum(d[k]['n'] for k in d)} papers")

print(f"\nStartups updated: {updated}")

# ── Recompute aggregate columns ───────────────────────────────────────────────
startups["total_citations"] = (
    startups["pub_lead_cit"] + startups["pub_mid_cit"] +
    startups["pre_lead_cit"] + startups["pre_mid_cit"]
)
n_papers = (
    startups["pub_lead_n"] + startups["pub_mid_n"] +
    startups["pre_lead_n"] + startups["pre_mid_n"]
)
n_hc = (
    startups["pub_lead_hc"] + startups["pub_mid_hc"] +
    startups["pre_lead_hc"] + startups["pre_mid_hc"]
)
startups["has_any_output"]   = (n_papers > 0).map({True: "yes", False: "no"})
startups["has_highly_cited"] = (n_hc > 0).map({True: "yes", False: "no"})
startups["mean_citations_per_paper"] = (
    startups["total_citations"] / n_papers.replace(0, float("nan"))
).round(2)

# Median: per-paper citation distribution (approximate from 2×2 data)
# We use the original paper records for accuracy
median_per_startup = (
    papers[
        papers["edge_case_flag"].isin(["", "alpha_mid", "even_shared", "plus_renamed"]) &
        papers["canonical_name"].isin(startup_names)
    ]
    .groupby("canonical_name")["cit"]
    .median()
    .rename("median_citations_per_paper")
)
# Also add EVEN papers (expanded)
if not delta.empty:
    even_median = delta.groupby("startup")["cit"].median().rename("median_citations_per_paper")
    median_per_startup = median_per_startup.add(even_median, fill_value=0) / 2
    # Simpler: just recompute from all attributed papers
    all_attr = pd.concat([
        papers[
            papers["edge_case_flag"].isin(["", "alpha_mid", "plus_renamed"]) &
            papers["canonical_name"].isin(startup_names)
        ][["canonical_name", "cit"]].rename(columns={"canonical_name": "startup"}),
        delta[["startup", "cit"]],
    ], ignore_index=True)
    median_per_startup = (
        all_attr.groupby("startup")["cit"].median()
        .rename("median_citations_per_paper")
    )

startups = startups.drop(columns=["median_citations_per_paper"], errors="ignore")
startups = startups.merge(
    median_per_startup.reset_index(), on="startup", how="left"
)
startups["median_citations_per_paper"] = startups["median_citations_per_paper"].round(1)

# ── Save ──────────────────────────────────────────────────────────────────────
startups.to_csv(STARTUPS_CSV, index=False)
print(f"\nSaved {STARTUPS_CSV}  ({len(startups)} rows)")

# ── Verification ──────────────────────────────────────────────────────────────
print("\n=== Verification: startups affected ===")
for s in sorted(delta_by_startup):
    row = startups[startups["startup"] == s].iloc[0]
    print(f"  {s:<32}  total_cit={int(row['total_citations']):>8,}  "
          f"has_output={row['has_any_output']}  has_hc={row['has_highly_cited']}")
