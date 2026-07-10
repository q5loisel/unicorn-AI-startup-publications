"""
Filter treated_data.csv down to included_papers.csv.

05_clean_papers.py does not drop any rows: it keeps all 144,185 merged
records and marks each of the five filters IN/EX in its own column, so every
exclusion decision stays auditable. This script performs the actual
filtering step described in the manuscript's Data preparation section: it
keeps only the rows where ALL FIVE filter columns are "IN", then drops the
per-filter helper/reason columns that are no longer needed once the
IN/EX decision has been applied.

Reads:  treated_data.csv        (144,185 rows, manuscript figure)
Writes: included_papers.csv     (3,410 rows, manuscript figure)

Note on provenance (see README Section 6, item 3): an earlier, undocumented
copy of included_papers.csv in the working repository had 3,420 rows — 10
more than the manuscript's 3,410. Comparing that file against treated_data.csv
shows the extra 10 rows do not appear anywhere in merged_papers.csv or
treated_data.csv at all: they are fragments of address/author text sitting in
the UID column (e.g. "15 Cotswold Rd", "Wurzburg", "MA"), the signature of a
handful of source records whose fields were corrupted by unescaped commas
during CSV export and got split across extra rows. Rebuilding
included_papers.csv directly from treated_data.csv, as this script does,
reproduces the manuscript's 3,410 exactly and naturally excludes those
corrupted rows since they never actually pass all five filters here.

All input/output files are read from and written to a single, user-defined
folder — set the AI_UNICORN_DATA_DIR environment variable to point at it (it
defaults to the current working directory, i.e. run this script from inside
that folder).
"""

import os

import pandas as pd

DATA_DIR = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())

FILE_IN  = os.path.join(DATA_DIR, "treated_data.csv")
FILE_OUT = os.path.join(DATA_DIR, "included_papers.csv")

FILTER_COLS = [
    "filter_keywords",
    "filter_doctype",
    "filter_date",
    "filter_dedup",
    "filter_affil",
]

# Per-filter helper/reason columns produced by 05_clean_papers.py that are
# only needed to audit the filtering decision, not for downstream analysis.
HELPER_COLS_TO_DROP = FILTER_COLS + [
    "filter_date_reason",
    "duplicate_of",
    "duplicate_reason",
    "affil_matched_startup",
]

print("Loading treated_data.csv …")
df = pd.read_csv(FILE_IN, low_memory=False)
print(f"  {len(df):,} records")

missing = [c for c in FILTER_COLS if c not in df.columns]
if missing:
    raise ValueError(
        f"treated_data.csv is missing expected filter column(s): {missing}. "
        "Run 05_clean_papers.py first."
    )

all_in = (df[FILTER_COLS] == "IN").all(axis=1)
included = (
    df[all_in]
    .drop(columns=HELPER_COLS_TO_DROP, errors="ignore")
    .reset_index(drop=True)
)

print(f"Records passing all 5 filters: {len(included):,} / {len(df):,}")

included.to_csv(FILE_OUT, index=False)
print(f"Saved to {FILE_OUT}")
