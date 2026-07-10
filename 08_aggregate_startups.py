"""
Populate startups_data.csv from included_papers.csv.

Papers are split along two dimensions for each startup:

  TYPE   : published (peer-reviewed) vs. preprint
  ROLE   : lead (startup has first, last, or alpha-order author)
            vs. collaborative (startup has only middle authors, non-alpha)

For each of the 4 cells (type × role), four metrics are produced:
  _n   : number of papers
  _cit : total citations
  _hc  : highly cited papers  (≥ HIGHLY_CITED_THRESHOLD citations)
  _nhc : not highly cited papers

Column naming:
  pub_lead_n / pub_lead_cit / pub_lead_hc / pub_lead_nhc
  pub_mid_n  / pub_mid_cit  / pub_mid_hc  / pub_mid_nhc
  pre_lead_n / pre_lead_cit / pre_lead_hc / pre_lead_nhc
  pre_mid_n  / pre_mid_cit  / pre_mid_hc  / pre_mid_nhc

Summary columns kept:
  has_any_output, has_highly_cited, total_citations,
  mean_citations_per_paper, median_citations_per_paper

Old n_papers_* / n_highly_cited_* / n_papers_middle_only columns are replaced.

Attribution: canonical_name column (alias-resolved lead startup).
Highly cited threshold: >= HIGHLY_CITED_THRESHOLD citations.

All input/output files are read from and written to a single, user-defined
folder — set the AI_UNICORN_DATA_DIR environment variable to point at it (it
defaults to the current working directory, i.e. run this script from inside
that folder).
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())

FILE_PAPERS   = os.path.join(DATA_DIR, "included_papers.csv")
FILE_STARTUPS = os.path.join(DATA_DIR, "startups_data.csv")
FILE_OUT      = os.path.join(DATA_DIR, "startups_data.csv")

HIGHLY_CITED_THRESHOLD = 200

PREPRINT_TYPES = {"preprint"}


# =============================================================================
# Load & prep papers
# =============================================================================

print("Loading papers …")
papers = pd.read_csv(FILE_PAPERS, low_memory=False)
print(f"  {len(papers):,} records")

papers["_cit"] = (
    pd.to_numeric(papers["Total Times Cited"], errors="coerce")
    .fillna(0).astype(int)
)
papers["_is_preprint"] = (
    papers["Document Type"].fillna("").str.strip().str.lower()
    .isin(PREPRINT_TYPES)
)
papers["_is_hc"] = papers["_cit"] >= HIGHLY_CITED_THRESHOLD

# Normalise alpha_order — treat garbled values as NO
papers["_alpha"] = papers["alpha_order"].fillna("NO").str.strip() == "YES"

# Update is_highly_cited in included_papers.csv with new threshold
papers_out = papers.drop(
    columns=["_cit", "_is_preprint", "_is_hc", "_alpha"], errors="ignore"
)
papers_out["is_highly_cited"] = (
    pd.to_numeric(papers_out["Total Times Cited"], errors="coerce")
    .fillna(0).ge(HIGHLY_CITED_THRESHOLD)
    .map({True: "yes", False: "no"})
)
papers_out.to_csv(FILE_PAPERS, index=False)
print(f"  Updated is_highly_cited (>= {HIGHLY_CITED_THRESHOLD}) in {FILE_PAPERS}")
del papers_out

print(f"  Highly cited (>= {HIGHLY_CITED_THRESHOLD}): "
      f"{papers['_is_hc'].sum():,}  ({100*papers['_is_hc'].mean():.1f}%)")


# =============================================================================
# Helpers
# =============================================================================

def startup_in(field_val, startup: str) -> bool:
    if pd.isna(field_val) or not str(field_val).strip():
        return False
    return any(v.strip() == startup for v in str(field_val).split("|"))


def classify_role(row, startup: str) -> str:
    """
    'lead'   → startup has first author, OR last author, OR paper is alpha-ordered
    'mid'    → startup has only middle author(s), non-alpha paper
    """
    has_first = startup_in(row.get("first_author"),  startup)
    has_last  = startup_in(row.get("last_author"),   startup)
    is_alpha  = bool(row.get("_alpha", False))

    if has_first or has_last or is_alpha:
        return "lead"
    return "mid"


# =============================================================================
# Aggregate per startup
# =============================================================================

print("\nLoading startups …")
startups = pd.read_csv(FILE_STARTUPS)
print(f"  {len(startups):,} startups")

ZERO_ROW = dict(
    pub_lead_n=0, pub_lead_cit=0, pub_lead_hc=0, pub_lead_nhc=0,
    pub_mid_n=0,  pub_mid_cit=0,  pub_mid_hc=0,  pub_mid_nhc=0,
    pre_lead_n=0, pre_lead_cit=0, pre_lead_hc=0, pre_lead_nhc=0,
    pre_mid_n=0,  pre_mid_cit=0,  pre_mid_hc=0,  pre_mid_nhc=0,
    has_any_output="no", has_highly_cited="no",
    total_citations=0,
    mean_citations_per_paper=np.nan,
    median_citations_per_paper=np.nan,
)

results = {}
n_matched = 0

for startup in startups["startup"]:
    subset = papers[papers["canonical_name"] == startup].copy()

    if len(subset) == 0:
        results[startup] = dict(ZERO_ROW)
        continue

    n_matched += 1

    # Classify each paper
    subset["_role"] = subset.apply(lambda r: classify_role(r, startup), axis=1)

    def agg(mask):
        s = subset[mask]
        n   = len(s)
        cit = int(s["_cit"].sum())
        hc  = int(s["_is_hc"].sum())
        return n, cit, hc, n - hc

    pub  = ~subset["_is_preprint"]
    pre  =  subset["_is_preprint"]
    lead =  subset["_role"] == "lead"
    mid  =  subset["_role"] == "mid"

    pl = agg(pub & lead);  pm = agg(pub & mid)
    rl = agg(pre & lead);  rm = agg(pre & mid)

    total_cit = int(subset["_cit"].sum())

    results[startup] = dict(
        pub_lead_n=pl[0], pub_lead_cit=pl[1], pub_lead_hc=pl[2], pub_lead_nhc=pl[3],
        pub_mid_n=pm[0],  pub_mid_cit=pm[1],  pub_mid_hc=pm[2],  pub_mid_nhc=pm[3],
        pre_lead_n=rl[0], pre_lead_cit=rl[1], pre_lead_hc=rl[2], pre_lead_nhc=rl[3],
        pre_mid_n=rm[0],  pre_mid_cit=rm[1],  pre_mid_hc=rm[2],  pre_mid_nhc=rm[3],
        has_any_output="yes",
        has_highly_cited="yes" if subset["_is_hc"].any() else "no",
        total_citations=total_cit,
        mean_citations_per_paper=round(float(subset["_cit"].mean()), 2),
        median_citations_per_paper=round(float(subset["_cit"].median()), 2),
    )

print(f"  Startups with ≥1 paper: {n_matched} / {len(startups)}")


# =============================================================================
# Write into startups_data.csv
# =============================================================================

# Drop old columns that are being replaced
DROP_COLS = [
    "n_papers_total", "n_papers_published", "n_papers_preprint",
    "n_highly_cited_total", "n_highly_cited_published", "n_highly_cited_preprint",
    "n_papers_middle_only", "cit_middle_only",
]
startups = startups.drop(columns=DROP_COLS, errors="ignore")

new_cols = [
    "pub_lead_n", "pub_lead_cit", "pub_lead_hc", "pub_lead_nhc",
    "pub_mid_n",  "pub_mid_cit",  "pub_mid_hc",  "pub_mid_nhc",
    "pre_lead_n", "pre_lead_cit", "pre_lead_hc", "pre_lead_nhc",
    "pre_mid_n",  "pre_mid_cit",  "pre_mid_hc",  "pre_mid_nhc",
    "has_any_output", "has_highly_cited",
    "total_citations", "mean_citations_per_paper", "median_citations_per_paper",
]
for col in new_cols:
    startups[col] = startups["startup"].map({k: v[col] for k, v in results.items()})

print(f"\nSaving to {FILE_OUT} …")
startups.to_csv(FILE_OUT, index=False)
print("Done.")


# =============================================================================
# Summary
# =============================================================================
s = startups[startups["has_any_output"] == "yes"].copy()
for c in ["pub_lead_n","pub_mid_n","pre_lead_n","pre_mid_n"]:
    s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0).astype(int)

print(f"""
=== Summary (startups with ≥1 paper: {len(s)}) ===
                          n papers    citations    HC (≥{HIGHLY_CITED_THRESHOLD})
Published  – lead   :  {int(s['pub_lead_n'].sum()):>8,}
Published  – middle :  {int(s['pub_mid_n'].sum()):>8,}
Preprint   – lead   :  {int(s['pre_lead_n'].sum()):>8,}
Preprint   – middle :  {int(s['pre_mid_n'].sum()):>8,}

Top 10 by total papers:
{s.assign(total_n=s['pub_lead_n']+s['pub_mid_n']+s['pre_lead_n']+s['pre_mid_n'])
  .nlargest(10,'total_n')
  [['startup','pub_lead_n','pub_mid_n','pre_lead_n','pre_mid_n','total_citations']]
  .to_string(index=False)}
""")
