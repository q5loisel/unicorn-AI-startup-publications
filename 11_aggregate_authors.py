"""
Author-level analysis for included_papers.csv.

For every author who has at least one startup affiliation, one aggregated
row is produced in authors_data.csv:

  author_name        : full name as written in the source data
  affiliation_address: most frequent full address string across papers
  startup            : startup(s) they are affiliated with (pipe-sep)
  affil_type         : startup-only | startup+academia |
                       startup+others | all three

  n_first   / cit_first   : papers / citations as FIRST author
  n_last    / cit_last    : papers / citations as LAST author
  n_middle  / cit_middle  : papers / citations as MIDDLE author
  n_alpha   / cit_alpha   : papers / citations on ALPHABETICALLY-ORDERED papers

  n_lead    : unique papers as lead (first OR last, not double-counted)
  cit_lead  : total citations as lead

Output: authors_data.csv

All input/output files are read from and written to a single, user-defined
folder — set the AI_UNICORN_DATA_DIR environment variable to point at it (it
defaults to the current working directory, i.e. run this script from inside
that folder).
"""

import os
import re
from collections import Counter, defaultdict

import pandas as pd

DATA_DIR   = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
FILE_IN    = os.path.join(DATA_DIR, "included_papers.csv")
FILE_OUT   = os.path.join(DATA_DIR, "authors_data.csv")
FILE_AFFIL = os.path.join(DATA_DIR, "startup_affiliation_dictionary.csv")


# =============================================================================
# Helpers
# =============================================================================

def build_affil_index(filepath: str) -> dict:
    aff_df = pd.read_csv(filepath)
    return {
        str(row["affiliation_variant"]).strip().lower(): str(row["startup"]).strip()
        for _, row in aff_df.iterrows()
    }


# Placeholder / null institution strings to skip entirely
_NULL_INSTS = {"no affiliation listed", "no affiliations listed", "none", "nan", ""}

# Sub-string tokens that strongly suggest an academic institution
_ACADEMIC_TOKENS = {
    # Full words / strong prefixes
    "univ", "universit", "instit", "institution", "college",
    "school", "sch", "acad", "polytech",
    "hosp", "hospital", "clin", "clinic",
    "ctr", "center", "centre",
    "lab", "laborat",
    "faculty", "dept",
    "research", "sciences", "medicine", "medical",
    "natl", "national", "federal", "govern",
    "technol",                          # Institute of Technology forms
    "ecole", "hochsch", "facult",       # French / German academic terms
}

# Well-known university/institution abbreviations (matched as exact tokens)
_ACADEMIC_ABBREVS = {
    "mit", "ucl", "eth", "epfl",
    "ucla", "ucsf", "usc", "ubc", "uva", "uw",
    "nyu", "cmu", "lse",
    "kit", "tum", "lmu", "rwth", "kth", "dtu", "tue",
    "nus", "ntu", "nthu", "hkust", "cuhk",
    "iit", "iitb", "iitm",
    "kaist", "postech", "kaust", "supsi",
    "inria", "cnrs", "cea",             # French research orgs
    "uga", "unc", "ncsu",
}


def classify_inst(inst_name: str, v2s: dict):
    """
    Return 'startup', 'academic', 'other', or None (skip / placeholder).

    Rules:
      1. Empty / placeholder strings → None (caller should ignore)
      2. In startup affiliation dict → 'startup'
      3. Contains academic keyword token or known abbreviation → 'academic'
      4. Everything else → 'other'
    """
    if not inst_name:
        return None
    lower = inst_name.lower().strip()
    if lower in _NULL_INSTS:
        return None
    if lower in v2s:
        return "startup"
    tokens = set(re.split(r"[\s,./\-()&]+", lower))
    if tokens & _ACADEMIC_TOKENS:
        return "academic"
    if tokens & _ACADEMIC_ABBREVS:
        return "academic"
    return "other"


def parse_authors(text) -> list:
    """
    Parse 'Authors & Affiliations' into:
        [(name, [inst1, inst2, …], full_bracket_address), …]
    Authors separated by ';', affiliations within [...] by '|'.
    inst is the first-comma part of each affiliation segment.
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
            name  = m.group(1).strip()
            brk   = m.group(2)
            insts = [seg.split(",")[0].strip() for seg in brk.split("|") if seg.strip()]
            addr  = brk.strip()
        else:
            name, insts, addr = part.strip(), [], ""
        if name:
            authors.append((name, insts, addr))
    return authors


# =============================================================================
# Main
# =============================================================================

print("Loading data …")
df = pd.read_csv(FILE_IN, low_memory=False)
# Exclude garbled rows (unrecoverable CSV artifacts)
if "edge_case_flag" in df.columns:
    n_before = len(df)
    df = df[df["edge_case_flag"] != "garbled_excluded"].copy().reset_index(drop=True)
    print(f"  Excluded {n_before - len(df)} garbled row(s)")
v2s = build_affil_index(FILE_AFFIL)
print(f"  {len(df):,} papers  |  {len(v2s):,} affiliation variants")

# Per-author accumulator
records = defaultdict(lambda: {
    "insts":      Counter(),   # institution name (first-comma part) → occurrence count
    "startups":   set(),
    "inst_types": set(),       # "startup", "academic", "other"
    "inst_class": {},          # inst name → its classification (for address building)
    "n_first":  0, "cit_first":  0,
    "n_last":   0, "cit_last":   0,
    "n_middle": 0, "cit_middle": 0,
    "n_alpha":  0, "cit_alpha":  0,
    "lead":     {},            # uid → citations  (deduplicates first+last)
})

print("Processing papers …")
for _, row in df.iterrows():
    uid     = str(row.get("UID", ""))
    try:
        raw_cit = row.get("Total Times Cited", 0)
        cit = 0 if pd.isna(raw_cit) else int(float(raw_cit))
    except (ValueError, TypeError):
        cit = 0
    alpha   = str(row.get("alpha_order", "")).strip() == "YES"
    authors = parse_authors(row.get("Authors & Affiliations", ""))
    n       = len(authors)

    for pos, (name, insts, addr) in enumerate(authors):
        # Keep only authors with at least one startup affiliation
        author_startups = [v2s[i.lower()] for i in insts if i.lower() in v2s]
        if not author_startups:
            continue

        r = records[name]

        # Startup(s) and institution-type classification
        r["startups"].update(author_startups)
        for inst in insts:
            itype = classify_inst(inst, v2s)
            if itype is not None:                   # skip placeholders / nulls
                r["inst_types"].add(itype)
                r["insts"][inst] += 1               # track occurrence count
                r["inst_class"][inst] = itype       # store its classification

        # Position counters
        is_first = (pos == 0)
        is_last  = (pos == n - 1)
        is_mid   = not is_first and not is_last

        if is_first:
            r["n_first"]  += 1
            r["cit_first"] += cit
        if is_last:
            r["n_last"]   += 1
            r["cit_last"]  += cit
        if is_mid:
            r["n_middle"]  += 1
            r["cit_middle"] += cit

        # Lead (first OR last, deduplicated by UID)
        if is_first or is_last:
            r["lead"][uid] = cit   # same UID overwrites → no double-count

        # Alpha: paper is alphabetically ordered (position-independent)
        if alpha:
            r["n_alpha"]  += 1
            r["cit_alpha"] += cit

print(f"  {len(records):,} unique startup-affiliated authors found")

# =============================================================================
# Build output DataFrame
# =============================================================================
rows = []
for name, r in records.items():
    # Build affiliation_address: all unique institutions sorted by type
    # (startup first, then academic, then other), ties broken by frequency
    TYPE_ORDER = {"startup": 0, "academic": 1, "other": 2}
    sorted_insts = sorted(
        r["insts"].keys(),
        key=lambda i: (TYPE_ORDER.get(r["inst_class"].get(i, "other"), 2),
                       -r["insts"][i])
    )
    top_addr     = " | ".join(sorted_insts)
    startups_str = " | ".join(sorted(r["startups"]))

    types = r["inst_types"]
    has_s = "startup"  in types
    has_a = "academic" in types
    has_o = "other"    in types

    # Rule 1: multiple startups with nothing else → startup-only
    # Rule 2: academic presence always wins over "other"
    #         (eliminates the "all three" category)
    if has_s and has_a:
        atype = "startup+academia"       # academic takes priority
    elif has_s and has_o:
        atype = "startup+others"
    else:
        atype = "startup-only"           # startup(s) only, nothing else

    n_lead   = len(r["lead"])
    cit_lead = sum(r["lead"].values())

    rows.append({
        "author_name":         name,
        "affiliation_address": top_addr,
        "startup":             startups_str,
        "affil_type":          atype,
        "n_first":    r["n_first"],
        "cit_first":  r["cit_first"],
        "n_last":     r["n_last"],
        "cit_last":   r["cit_last"],
        "n_middle":   r["n_middle"],
        "cit_middle": r["cit_middle"],
        "n_alpha":    r["n_alpha"],
        "cit_alpha":  r["cit_alpha"],
        "n_lead":     n_lead,
        "cit_lead":   cit_lead,
    })

out = (
    pd.DataFrame(rows)
    .sort_values("n_lead", ascending=False)
    .reset_index(drop=True)
)

# =============================================================================
# Summary
# =============================================================================
print(f"""
=== Summary ===
Total startup-affiliated authors    : {len(out):,}

Affiliation type distribution:
{out['affil_type'].value_counts().to_string()}

Top 15 authors by lead papers:
{out[['author_name','startup','n_lead','cit_lead','n_first','n_last']].head(15).to_string(index=False)}
""")

# =============================================================================
# Save
# =============================================================================
print(f"Saving to {FILE_OUT} …")
out.to_csv(FILE_OUT, index=False)
print(f"Done — {len(out):,} rows, {len(out.columns)} columns")
