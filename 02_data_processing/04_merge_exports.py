"""
Merge the three Stage-1 raw exports into a single file for Stage 2.

This is the "Combining the three raw exports" step described in the
manuscript's Data preparation section: the Web of Science Core Collection,
INSPEC, and Preprint Citation Index exports are concatenated (no filtering
or deduplication here — deduplication is Filter 4 in 05_clean_papers.py).

Reads (from AI_UNICORN_DATA_DIR):
  wos_papers.csv        (Web of Science Core Collection;  manuscript n=116,233)
  inspec_papers.csv     (INSPEC;                          manuscript n=22,619)
  preprints_papers.csv  (Preprint Citation Index;          manuscript n=5,333)

Writes:
  merged_papers.csv (manuscript n=144,185)
"""

import os

import pandas as pd

DATA_DIR = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())

FILE_WOS       = os.path.join(DATA_DIR, "wos_papers.csv")
FILE_INSPEC    = os.path.join(DATA_DIR, "inspec_papers.csv")
FILE_PREPRINTS = os.path.join(DATA_DIR, "preprints_papers.csv")
FILE_OUT       = os.path.join(DATA_DIR, "merged_papers.csv")

SOURCES = [
    ("Web of Science Core Collection", FILE_WOS),
    ("INSPEC",                         FILE_INSPEC),
    ("Preprint Citation Index",        FILE_PREPRINTS),
]

print("Loading Stage-1 exports …")
frames = []
for label, path in SOURCES:
    df = pd.read_csv(path, low_memory=False)
    print(f"  {label:<32} {len(df):>8,} records   ({path})")
    frames.append(df)

# All three Stage-1 scripts write the same 13-column schema, but the
# preprints_papers.csv file actually used for the manuscript carries 3 extra
# columns ("Dictionary match", "Alphabetical order", "First last author
# match") left over from an earlier, exploratory identification pass that
# predates 07_identify_authorship.py. They are not read by any script in
# this package and are superseded by the filter_affil/affil_matched_startup
# and alpha_order columns computed later in the pipeline. Rather than drop
# or silently realign them, this merge is column-union (like the original,
# undocumented merge must have been, since merged_papers.csv carries these
# same 3 extra columns) so the merge is transparent about what's there.
ref_cols = set(frames[0].columns)
for (label, path), df in zip(SOURCES, frames):
    extra = set(df.columns) - ref_cols
    missing = ref_cols - set(df.columns)
    if extra:
        print(f"  [!] {label} has extra column(s) not in {SOURCES[0][0]}: "
              f"{sorted(extra)} — carried through as-is, not used downstream")
    if missing:
        print(f"  [!] {label} is missing column(s) present in {SOURCES[0][0]}: "
              f"{sorted(missing)} — will be NaN after merge")

merged = pd.concat(frames, ignore_index=True, sort=False)
print(f"\nMerged total: {len(merged):,} records "
      f"({' + '.join(str(len(f)) for f in frames)})")

merged.to_csv(FILE_OUT, index=False)
print(f"Saved to {FILE_OUT}")
