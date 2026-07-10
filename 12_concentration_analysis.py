#!/usr/bin/env python3
"""
12_concentration_analysis.py
------------
Combined firm-level and author-level analysis.
Scope: lead papers — startup-affiliated author in first OR last position.

SECTION 1 — Firm-level
  PANEL A  Distribution of publications per firm
  PANEL B  Lorenz concentration by firm
  PANEL C  Top 10 firms by citations
  PANEL D  Lorenz concentration by author

SECTION 2 — Author-level
  A. Overview
  B. Papers-per-author distribution
  C. Concentration metrics
  D. Affiliation type breakdown
  E. Top contributing authors
  F. Per-startup summary

Input files
-----------
  startups_data.csv   – firm metadata
  included_papers.csv – paper-level records
  authors_data.csv    – author-level aggregation
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
# All input files are read from a single, user-defined folder — set the
# AI_UNICORN_DATA_DIR environment variable to point at it (it defaults to the
# current working directory, i.e. run this script from inside that folder).
DATA_DIR     = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
STARTUPS_CSV = os.path.join(DATA_DIR, "startups_data.csv")
PAPERS_CSV   = os.path.join(DATA_DIR, "included_papers.csv")
AUTHORS_CSV  = os.path.join(DATA_DIR, "authors_data.csv")

HC_THRESHOLD = 200

# ── Helpers ───────────────────────────────────────────────────────────────────
def require_columns(data: pd.DataFrame, columns: list, source: str) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    total = x.sum()
    if total == 0:
        return 0.0
    n, cum = len(x), np.cumsum(x)
    return (n + 1 - 2.0 * cum.sum() / total) / n


def hhi(x) -> float:
    x = np.asarray(x, dtype=float)
    total = x.sum()
    return 0.0 if total == 0 else float(((x / total) ** 2).sum())


def lorenz_points(series: pd.Series, n_deciles: int = 10) -> list:
    x = np.sort(np.asarray(series, dtype=float))
    total = x.sum()
    if total == 0:
        return [(i / n_deciles, 0.0) for i in range(n_deciles + 1)]
    pts = [(0.0, 0.0)]
    for i in range(1, n_deciles + 1):
        idx = min(int(np.floor(len(x) * i / n_deciles)), len(x))
        pts.append((i / n_deciles, float(x[:idx].sum() / total)))
    return pts


def top_share(series: pd.Series, pct: float) -> float:
    """Share held by the top `pct` fraction of observations (fraction-based)."""
    k = max(1, int(len(series) * pct))
    total = series.sum()
    return float(series.nlargest(k).sum() / total) if total > 0 else 0.0


def top_n_for_share(series: pd.Series, pct: float) -> int:
    return max(1, int(len(series) * pct))


def top_k_share(series: pd.Series, k: int) -> float:
    """Share held by the top k observations (count-based)."""
    total = series.sum()
    return float(series.nlargest(k).sum() / total) if total > 0 else 0.0


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def sub(title: str) -> None:
    print(f"\n  ── {title} ──")


def pct(n, total) -> str:
    return f"{100 * n / total:.1f} %" if total else "—"


def fmt(n) -> str:
    return f"{n:,}"


def _parse_first_last_names(text):
    """Return (first_author_name, last_author_name) from Authors & Affiliations."""
    if not text or pd.isna(text):
        return None, None
    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    if not parts:
        return None, None
    m0 = re.match(r"^(.*?)\s*\[", parts[0])
    first = m0.group(1).strip() if m0 else parts[0].strip()
    m1 = re.match(r"^(.*?)\s*\[", parts[-1])
    last  = m1.group(1).strip() if m1 else parts[-1].strip()
    return first or None, last or None


def _parse_author_names(text) -> list:
    """Return author names, in source order, from Authors & Affiliations."""
    if not text or pd.isna(text):
        return []
    names = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\s*\[", part)
        name = m.group(1).strip() if m else part.strip()
        if name:
            names.append(name)
    return names


def _split_pipe(text) -> set:
    if not text or pd.isna(text):
        return set()
    return {p.strip() for p in str(text).split(" | ") if p.strip()}


# ── Load ──────────────────────────────────────────────────────────────────────
df      = pd.read_csv(STARTUPS_CSV)
papers  = pd.read_csv(PAPERS_CSV, low_memory=False)
authors = pd.read_csv(AUTHORS_CSV)

N = len(df)

require_columns(df,     ["startup"],                                   os.path.basename(STARTUPS_CSV))
require_columns(papers, ["UID", "canonical_name", "Total Times Cited",
                          "Document Type", "first_author", "last_author",
                          "alpha_order", "is_highly_cited",
                          "shared_startups", "edge_case_flag"],        os.path.basename(PAPERS_CSV))

if df["startup"].duplicated().any():
    raise ValueError(f"Duplicate startup names: {df.loc[df['startup'].duplicated(), 'startup'].head(5).tolist()}")

# ── Paper-level preprocessing ─────────────────────────────────────────────────
papers["cit"]           = pd.to_numeric(papers["Total Times Cited"], errors="coerce").fillna(0).astype(int)
papers["is_preprint"]   = papers["Document Type"].str.strip().str.lower() == "preprint"
papers["hc_200"]        = papers["cit"] >= HC_THRESHOLD
papers["has_leadership"]= papers["first_author"].notna() | papers["last_author"].notna()
papers["is_alphabetical"]= papers["alpha_order"].str.strip().str.upper() == "YES"
papers["edge_case_flag"]  = papers["edge_case_flag"].fillna("").astype(str)
papers["shared_startups"] = papers["shared_startups"].fillna("").astype(str)

stored = papers["is_highly_cited"].fillna("no").str.strip().str.lower()
if not stored.eq(np.where(papers["hc_200"], "yes", "no")).all():
    n_bad = int((~stored.eq(np.where(papers["hc_200"], "yes", "no"))).sum())
    raise ValueError(f"is_highly_cited mismatches >={HC_THRESHOLD} citations for {n_bad} row(s).")

# ── Paper scope ───────────────────────────────────────────────────────────────
# Included:
#   - Lead papers: canonical_name in startup_names AND startup in first/last position
#   - Alpha-mid papers: canonical_name in startup_names AND alpha_order=YES
#     (startup in middle only; included because alphabetical listing treats all
#     co-authors as equal contributors)
#   - EVEN papers: two startups tied; canonical_name="EVEN"; attributed to each
#     tied startup via shared_startups (counted +1 per startup, +1 total)
# Excluded:
#   - NaN canonical_name (no startup found)
#   - GARBLED_EXCLUDED (corrupted rows, unrecoverable)
startup_names = set(df["startup"])  # "Plus" renamed to "Plus.ai" by correction script

papers = papers[
    (
        papers["canonical_name"].isin(startup_names) &
        (papers["has_leadership"] | (papers["edge_case_flag"] == "alpha_mid"))
    ) |
    (papers["edge_case_flag"] == "even_shared")
].copy()

# ── Expand EVEN papers for per-startup attribution ────────────────────────────
# EVEN papers are expanded so each tied startup gets +1 paper/citation.
# Grand-total counts (N_PAPERS, total citations) use the unexpanded set
# so EVEN papers are counted only once.
_normal = papers[papers["edge_case_flag"] != "even_shared"].copy()
_even   = papers[papers["edge_case_flag"] == "even_shared"].copy()
_even_rows = []
for _, _r in _even.iterrows():
    for _s in _r["shared_startups"].split(" | "):
        _row = _r.copy()
        _row["canonical_name"] = _s.strip()
        _even_rows.append(_row)
_even_exp = (pd.DataFrame(_even_rows) if _even_rows
             else pd.DataFrame(columns=_normal.columns))
# papers_for_firms: used for per-startup groupby (EVEN counted per tied startup)
papers_for_firms = pd.concat([_normal, _even_exp], ignore_index=True)


# ── Firm-level stats recomputed from papers ───────────────────────────────────
# Uses papers_for_firms so EVEN papers are attributed to each tied startup.
_pf = papers_for_firms[papers_for_firms["canonical_name"].isin(startup_names)]
per_startup = _pf.groupby("canonical_name").agg(
    n_papers_total      =("UID",         "size"),
    n_papers_published  =("is_preprint", lambda s: int((~s).sum())),
    n_papers_preprint   =("is_preprint", "sum"),
    n_highly_cited_total=("hc_200",      "sum"),
    total_citations     =("cit",         "sum"),
)
per_startup["n_highly_cited_published"] = (
    _pf[_pf["hc_200"] & ~_pf["is_preprint"]]
    .groupby("canonical_name").size().reindex(per_startup.index, fill_value=0)
)
per_startup["n_highly_cited_preprint"] = (
    _pf[_pf["hc_200"] & _pf["is_preprint"]]
    .groupby("canonical_name").size().reindex(per_startup.index, fill_value=0)
)
per_startup = per_startup.fillna(0).astype(int).reset_index().rename(columns={"canonical_name": "startup"})

count_cols = [
    "n_papers_total", "n_papers_published", "n_papers_preprint",
    "n_highly_cited_total", "n_highly_cited_published", "n_highly_cited_preprint",
    "total_citations",
]
df = df.drop(columns=[c for c in count_cols if c in df.columns], errors="ignore")
df = df.merge(per_startup, on="startup", how="left")
df[count_cols] = df[count_cols].fillna(0).astype(int)

# ── Author-level shared computations ─────────────────────────────────────────
# Active lead authors (n_lead > 0)
authors_lead = authors[authors["n_lead"] > 0].copy().reset_index(drop=True)
N_AUTH_ALL   = len(authors)
N_AUTH_LEAD  = len(authors_lead)

# HC lead papers per author (shared between Panel D and Section 2)
hc_count: Counter = Counter()
for _, row in papers[papers["hc_200"]].iterrows():
    fn, ln   = _parse_first_last_names(row.get("Authors & Affiliations"))
    has_first = pd.notna(row["first_author"]) and str(row["first_author"]).strip()
    has_last  = pd.notna(row["last_author"])  and str(row["last_author"]).strip()
    if has_first and fn:
        hc_count[fn] += 1
    if has_last and ln and ln != fn:
        hc_count[ln] += 1

hc_auth = (
    pd.DataFrame(list(hc_count.items()), columns=["author_name", "n_hc"])
    .merge(authors[["author_name", "startup", "affil_type",
                    "n_lead", "cit_lead", "affiliation_address"]],
           on="author_name", how="left")
    .sort_values("n_hc", ascending=False)
    .reset_index(drop=True)
)
N_AUTH_HC      = len(hc_auth)
all_ppa        = authors_lead["n_lead"].astype(int)
hc_ppa         = hc_auth["n_hc"].astype(int)
TOTAL_LEAD_OBS = int(all_ppa.sum())
TOTAL_HC_OBS   = int(hc_ppa.sum())


# Author-publication observations for concentration by author.
# Scope: non-alpha first/last startup authors; all startup-affiliated authors on
# alphabetical papers; and all tied-startup authors on EVEN/shared papers.
author_meta = authors.set_index("author_name")[["startup", "affil_type"]]
alpha_author_names = set(authors.loc[authors["n_alpha"] > 0, "author_name"])
obs_rows = []

for _, row in papers.iterrows():
    uid = str(row["UID"])
    names = _parse_author_names(row.get("Authors & Affiliations"))
    if not names:
        continue

    cit = int(row["cit"])
    is_hc = bool(row["hc_200"])
    selected = set()

    if row["edge_case_flag"] == "even_shared":
        tied_startups = _split_pipe(row["shared_startups"])
        for name in names:
            if name not in author_meta.index:
                continue
            author_startups = _split_pipe(author_meta.at[name, "startup"])
            if author_startups & tied_startups:
                selected.add(name)
    elif row["is_alphabetical"]:
        selected.update(name for name in names if name in alpha_author_names)
    else:
        fn = names[0]
        ln = names[-1]
        has_first = pd.notna(row["first_author"]) and str(row["first_author"]).strip()
        has_last  = pd.notna(row["last_author"])  and str(row["last_author"]).strip()
        if has_first and fn in author_meta.index:
            selected.add(fn)
        if has_last and ln in author_meta.index:
            selected.add(ln)

    for name in selected:
        obs_rows.append({
            "UID": uid,
            "author_name": name,
            "affil_type": author_meta.at[name, "affil_type"],
            "cit": cit,
            "hc_200": is_hc,
        })

author_pub_obs = (
    pd.DataFrame(obs_rows)
    .drop_duplicates(["UID", "author_name"])
    if obs_rows else
    pd.DataFrame(columns=["UID", "author_name", "affil_type", "cit", "hc_200"])
)
author_pub_counts = (
    author_pub_obs.groupby("author_name")
    .agg(n_author_pub=("UID", "size"), cit_author_pub=("cit", "sum"))
    .reset_index()
)
authors_pub = (
    author_pub_counts
    .merge(authors[["author_name", "startup", "affil_type",
                    "n_lead", "cit_lead", "affiliation_address"]],
           on="author_name", how="left")
    .sort_values("n_author_pub", ascending=False)
    .reset_index(drop=True)
)
hc_pub_counts = (
    author_pub_obs[author_pub_obs["hc_200"]]
    .groupby("author_name")
    .agg(n_hc_author_pub=("UID", "size"), cit_hc_author_pub=("cit", "sum"))
    .reset_index()
)
hc_pub_auth = (
    hc_pub_counts
    .merge(authors[["author_name", "startup", "affil_type",
                    "n_lead", "cit_lead", "affiliation_address"]],
           on="author_name", how="left")
    .sort_values("n_hc_author_pub", ascending=False)
    .reset_index(drop=True)
)
all_author_pub_ppa = authors_pub["n_author_pub"].astype(int)
hc_author_pub_ppa = (
    authors_pub[["author_name"]]
    .merge(hc_pub_counts[["author_name", "n_hc_author_pub"]],
           on="author_name", how="left")
    ["n_hc_author_pub"].fillna(0).astype(int)
)
TOTAL_AUTHOR_PUB_OBS = int(all_author_pub_ppa.sum())
TOTAL_HC_AUTHOR_PUB_OBS = int(hc_author_pub_ppa.sum())
N_AUTH_AUTHOR_PUB = len(authors_pub)
N_AUTH_HC_AUTHOR_PUB = len(hc_pub_auth)

category_pub_obs = author_pub_obs.drop_duplicates(["UID", "affil_type"])
hc_category_pub_obs = category_pub_obs[category_pub_obs["hc_200"]]


# ── Dataset-level summary (for results text) ─────────────────────────────────
N_PAPERS      = len(papers)
N_COUNTRIES   = df["Country"].nunique()
N_PUB         = int((~papers["is_preprint"]).sum())
N_PRE         = int(papers["is_preprint"].sum())
N_HC          = int(papers["hc_200"].sum())
N_WITH_OUTPUT = int((df["n_papers_total"] > 0).sum())
N_NO_OUTPUT   = int((df["n_papers_total"] == 0).sum())
# Firms with lead preprints but no lead peer-reviewed papers (lead scope only)
for _c in ["pub_lead_n", "pre_lead_n"]:
    df[_c] = pd.to_numeric(df[_c], errors="coerce").fillna(0).astype(int)
N_PRE_ONLY = int(
    ((df["pub_lead_n"] == 0) & (df["pre_lead_n"] > 0)).sum()
)

banner("DATASET OVERVIEW")
print(f"  Sample:    {N} AI unicorn startups across {N_COUNTRIES} countries")
print(f"  Outputs:   {N_PAPERS:,} qualifying papers  "
      f"({N_PUB:,} peer-reviewed, {N_PRE:,} preprints)")
print(f"  HC:        {N_HC:,} papers ({100*N_HC/N_PAPERS:.1f} %) received ≥{HC_THRESHOLD} citations")
print(f"  Firms with ≥1 qualifying output:      {N_WITH_OUTPUT:,}  ({100*N_WITH_OUTPUT/N:.1f} %)")
print(f"  Firms with no qualifying output:      {N_NO_OUTPUT:,}  ({100*N_NO_OUTPUT/N:.1f} %)")
print(f"  Firms with lead preprints only (no lead peer-reviewed): {N_PRE_ONLY}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FIRM-LEVEL ANALYSES
# ═════════════════════════════════════════════════════════════════════════════

# ── PANEL A ───────────────────────────────────────────────────────────────────
banner("PANEL A  –  Distribution of publications per firm")

mask_pub   = df["n_papers_total"] > 0
n_pub      = int(mask_pub.sum())
pub_all    = df["n_papers_total"]
pub_firms  = df.loc[mask_pub, "n_papers_total"]
peer_firms = df.loc[mask_pub, "n_papers_published"]
pre_firms  = df.loc[mask_pub, "n_papers_preprint"]

sub("Summary statistics")
W1, W2, W3 = 30, 22, 28
print(f"  {'Metric':<{W1}}  {'All firms (N=' + str(N) + ')':<{W2}}  "
      f"{'Publishers only (n=' + str(n_pub) + ')':<{W3}}")
print("  " + "─" * (W1 + W2 + W3 + 4))
for m, v1, v2 in [
    ("Mean",              f"{pub_all.mean():.2f}",        f"{pub_firms.mean():.2f}"),
    ("Median",            f"{pub_all.median():.1f}",       f"{pub_firms.median():.1f}"),
    ("SD",                "—",                             f"{pub_firms.std(ddof=1):.2f}"),
    ("IQR (Q1–Q3)",       "—",
     f"{pub_firms.quantile(.25):.0f}–{pub_firms.quantile(.75):.0f}"),
    ("Maximum",           f"{pub_all.max()}",              f"{pub_firms.max()}"),
    ("Mean peer-reviewed","—",                             f"{peer_firms.mean():.2f}"),
    ("Median peer-reviewed","—",                           f"{peer_firms.median():.1f}"),
    ("Mean preprints",    "—",                             f"{pre_firms.mean():.2f}"),
    ("Median preprints",  "—",                             f"{pre_firms.median():.1f}"),
]:
    print(f"  {m:<{W1}}  {v1:<{W2}}  {v2:<{W3}}")

sub("Output category breakdown")
no_output   = int((df["n_papers_total"] == 0).sum())
has_hc_lead = int((df["n_highly_cited_total"] >= 1).sum())
lead_no_hc  = int(((df["n_papers_total"] > 0) & (df["n_highly_cited_total"] == 0)).sum())
print(f"  {'Category':<40}  {'n':>5}  {'%':>8}")
print("  " + "─" * 58)
for lbl, cnt in [("No output (0 papers)",       no_output),
                 ("Has ≥1 HC paper",             has_hc_lead),
                 ("Papers, no highly-cited",     lead_no_hc)]:
    print(f"  {lbl:<40}  {cnt:>5}  {100 * cnt / N:>7.1f} %")

sub("Bin-level counts")
bin_edges  = [-1, 0, 2, 5, 10, 15, 20, 25, 30, int(pub_all.max()) + 1]
bin_labels = ["0", "1–2", "3–5", "6–10", "11–15", "16–20", "21–25", "26–30", "31+"]
cuts = pd.cut(pub_all, bins=bin_edges, labels=bin_labels, right=True)
print(f"  {'Bin':<10}  {'n':>5}  {'% of N':>8}")
print("  " + "─" * 28)
for lbl in bin_labels:
    cnt = int((cuts == lbl).sum())
    print(f"  {lbl:<10}  {cnt:>5}  {100 * cnt / N:>7.1f} %")


# ── PANEL B ───────────────────────────────────────────────────────────────────
banner("PANEL B  –  Lorenz concentration by firm (publications and citations)")

g_pubs = gini(df["n_papers_total"])
g_cits = gini(df["total_citations"])

sub("Summary statistics")
W4, W5, W6 = 36, 16, 16
print(f"  {'Metric':<{W4}}  {'Publications':>{W5}}  {'Citations':>{W6}}")
print("  " + "─" * (W4 + W5 + W6 + 4))
for m, v1, v2 in [
    ("Total (all firms)",
     f"{df['n_papers_total'].sum():,}",       f"{df['total_citations'].sum():,}"),
    ("Mean per firm",
     f"{df['n_papers_total'].mean():.1f}",    f"{df['total_citations'].mean():.1f}"),
    ("Median per firm",
     f"{df['n_papers_total'].median():.1f}",  f"{df['total_citations'].median():.1f}"),
    ("Maximum (single firm)",
     f"{df['n_papers_total'].max():,}",       f"{df['total_citations'].max():,}"),
    ("Gini coefficient",
     f"{g_pubs:.2f}",                         f"{g_cits:.2f}"),
    (f"Top 5 %  (n={top_n_for_share(df['n_papers_total'], 0.05)}) → share",
     f"{100 * top_share(df['n_papers_total'],   0.05):.1f} %",
     f"{100 * top_share(df['total_citations'],   0.05):.1f} %"),
    (f"Top 10 % (n={top_n_for_share(df['n_papers_total'], 0.10)}) → share",
     f"{100 * top_share(df['n_papers_total'],   0.10):.1f} %",
     f"{100 * top_share(df['total_citations'],   0.10):.1f} %"),
    (f"Top 20 % (n={top_n_for_share(df['n_papers_total'], 0.20)}) → share",
     f"{100 * top_share(df['n_papers_total'],   0.20):.1f} %",
     f"{100 * top_share(df['total_citations'],   0.20):.1f} %"),
]:
    print(f"  {m:<{W4}}  {v1:>{W5}}  {v2:>{W6}}")

sub("Lorenz curve data points — publications (by firm decile)")
pts = lorenz_points(df["n_papers_total"])
print(f"  {'Decile':<8}  {'Cum. firms %':>12}  {'Cum. pubs %':>12}")
print("  " + "─" * 36)
for pop_pct, share_pct in pts:
    print(f"  {pop_pct*100:>6.0f}%    {pop_pct*100:>10.1f}%  {share_pct*100:>11.1f}%")

sub("Lorenz curve data points — citations (by firm decile)")
pts_c = lorenz_points(df["total_citations"])
print(f"  {'Decile':<8}  {'Cum. firms %':>12}  {'Cum. cit %':>12}")
print("  " + "─" * 36)
for pop_pct, share_pct in pts_c:
    print(f"  {pop_pct*100:>6.0f}%    {pop_pct*100:>10.1f}%  {share_pct*100:>11.1f}%")


# ── PANEL C ───────────────────────────────────────────────────────────────────
banner("PANEL C  –  Top 10 firms by cumulated citations")

cit_split = (
    papers_for_firms[papers_for_firms["canonical_name"].isin(startup_names)]
    .groupby(["canonical_name", "is_preprint"])["cit"]
    .sum().unstack(fill_value=0)
)
cit_split.columns.name = None
for col in (False, True):
    if col not in cit_split.columns:
        cit_split[col] = 0
cit_split = cit_split.rename(columns={False: "peer", True: "preprint_cit"})
cit_split["total"] = cit_split["peer"] + cit_split["preprint_cit"]

top10 = cit_split.nlargest(10, "total").reset_index()
# Grand total: count EVEN papers once (unexpanded)
grand_total = int(papers["cit"].sum())
top10_total = int(top10["total"].sum())

sub("Rankings")
print(f"  {'Rk':<3}  {'Firm':<26}  {'Total cit':>10}  "
      f"{'Peer-rev (%)':>14}  {'Preprint (%)':>13}  {'Share':>6}")
print("  " + "─" * 80)
for rank, row in enumerate(top10.itertuples(), start=1):
    tot  = int(row.total)
    peer = int(row.peer)
    pre  = int(row.preprint_cit)
    pp   = 100 * peer / tot if tot else 0.0
    rp   = 100 * pre  / tot if tot else 0.0
    sh   = 100 * tot  / grand_total
    print(f"  {rank:<3}  {row.canonical_name[:26]:<26}  {tot:>10,}  "
          f"{peer:>6,} ({pp:>4.1f}%)  {pre:>5,} ({rp:>4.1f}%)  {sh:>5.1f}%")
print("  " + "─" * 80)
print(f"  {'Top 10 total':<32}  {top10_total:>10,}  "
      f"{'':>14}  {'':>13}  {100 * top10_total / grand_total:>5.1f}%")


# ── PANEL D ───────────────────────────────────────────────────────────────────
banner("PANEL D  –  Lorenz concentration by author")
print(f"\n  Unit: startup-affiliated authors with ≥1 lead paper (n={N_AUTH_LEAD:,})")
print(f"  Papers with startup in BOTH first and last positions count once per author.")

g_all = gini(all_ppa.values)
g_hc  = gini(hc_ppa.values)

sub("Overview")
print(f"  {'Metric':<40}  {'All lead authors':>16}  {'HC lead authors':>14}")
print("  " + "─" * 74)
for label, v_all, v_hc in [
    ("Unique authors",                fmt(N_AUTH_LEAD),    fmt(N_AUTH_HC)),
    ("Author–paper observations",     fmt(TOTAL_LEAD_OBS), fmt(TOTAL_HC_OBS)),
    ("Mean papers per author",        f"{all_ppa.mean():.2f}", f"{hc_ppa.mean():.2f}"),
    ("Median papers per author",      str(int(all_ppa.median())), str(int(hc_ppa.median()))),
    ("Max papers per author",         str(int(all_ppa.max())),    str(int(hc_ppa.max()))),
    ("Gini coefficient",              f"{g_all:.3f}", f"{g_hc:.3f}"),
    (f"Top 1 % (n={max(1,int(N_AUTH_LEAD*0.01))}) share",
     f"{100*top_share(all_ppa,0.01):.1f} %", f"{100*top_share(hc_ppa,0.01):.1f} %"),
    (f"Top 5 % (n={max(1,int(N_AUTH_LEAD*0.05))}) share",
     f"{100*top_share(all_ppa,0.05):.1f} %", f"{100*top_share(hc_ppa,0.05):.1f} %"),
    (f"Top 10 % (n={max(1,int(N_AUTH_LEAD*0.10))}) share",
     f"{100*top_share(all_ppa,0.10):.1f} %", f"{100*top_share(hc_ppa,0.10):.1f} %"),
    (f"Top 20 % (n={max(1,int(N_AUTH_LEAD*0.20))}) share",
     f"{100*top_share(all_ppa,0.20):.1f} %", f"{100*top_share(hc_ppa,0.20):.1f} %"),
]:
    print(f"  {label:<40}  {v_all:>16}  {v_hc:>14}")

sub("Lorenz curve data points — by author decile")
pts_all = lorenz_points(all_ppa)
pts_hc  = lorenz_points(hc_ppa)
print(f"  {'Decile':>6}  {'Cum. authors %':>14}  "
      f"{'Cum. output % (all)':>20}  {'Cum. output % (HC)':>18}")
print("  " + "─" * 66)
for (pop_pct, s_all), (_, s_hc) in zip(pts_all, pts_hc):
    print(f"  {pop_pct*100:>5.0f}%    {pop_pct*100:>12.1f}%  "
          f"{s_all*100:>18.1f}%  {s_hc*100:>17.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AUTHOR-LEVEL ANALYSES
# ═════════════════════════════════════════════════════════════════════════════

banner("SECTION 2  –  Author-level analyses")
print(f"\n  Scope: authors with ≥1 lead paper (n_lead > 0)")
print(f"  Total startup-affiliated authors in authors_data.csv: {N_AUTH_ALL:,}")

# ── A. Overview ───────────────────────────────────────────────────────────────
sub("A. Overview")
print(f"  {'Metric':<44}  {'All lead authors':>16}  {'HC lead authors':>15}")
print("  " + "─" * 80)
for label, v_all, v_hc in [
    ("Unique authors",               fmt(N_AUTH_LEAD),  fmt(N_AUTH_HC)),
    ("Author–paper observations",    fmt(TOTAL_LEAD_OBS), fmt(TOTAL_HC_OBS)),
    ("Authors with exactly 1 paper", fmt(int((all_ppa==1).sum())), fmt(int((hc_ppa==1).sum()))),
    ("Authors with > 1 paper",       fmt(int((all_ppa>1).sum())),  fmt(int((hc_ppa>1).sum()))),
    ("Distinct startups represented",
     fmt(authors_lead["startup"].str.split(" | ", regex=False).explode().nunique()),
     fmt(hc_auth["startup"].dropna().str.split(" | ", regex=False).explode().nunique())),
]:
    print(f"  {label:<44}  {v_all:>16}  {v_hc:>15}")

# ── B. Papers-per-author distribution ────────────────────────────────────────
sub("B. Papers-per-author distribution (n_lead)")
auth_bins = [("1",    lambda p: p == 1), ("2",    lambda p: p == 2),
             ("3–5",  lambda p: (p>=3)&(p<=5)), ("6–10", lambda p: (p>=6)&(p<=10)),
             ("11+",  lambda p: p >= 11)]
print(f"  {'Papers':>8}  {'All authors':>12}  {'% all':>6}  {'HC authors':>11}  {'% HC':>5}")
print("  " + "─" * 50)
for lbl, fn in auth_bins:
    na = int(fn(all_ppa).sum())
    nh = int(fn(hc_ppa).sum())
    print(f"  {lbl:>8}  {fmt(na):>12}  {pct(na, N_AUTH_LEAD):>6}  {fmt(nh):>11}  {pct(nh, N_AUTH_HC):>5}")
print("  " + "─" * 50)
for stat, key in [("Mean","mean"),("Median","median"),("Max","max")]:
    va = f"{getattr(all_ppa,key)():.2f}" if key=="mean" else str(int(getattr(all_ppa,key)()))
    vh = f"{getattr(hc_ppa, key)():.2f}" if key=="mean" else str(int(getattr(hc_ppa, key)()))
    print(f"  {stat:>8}  {va:>12}  {'':>6}  {vh:>11}")

# ── C. Concentration metrics ──────────────────────────────────────────────────
sub("C. Concentration metrics")
g_author_pub = gini(all_author_pub_ppa.values)
g_hc_author_pub = gini(hc_author_pub_ppa.values)
h_author_pub = hhi(all_author_pub_ppa.values)
h_hc_author_pub = hhi(hc_author_pub_ppa.values)

print("  Scope: unique author-publication observations")
print("         (lead authors, alphabetical startup authors, and EVEN/shared authors).")
print("         HC counts are zero-filled over the same author set.")
print(f"  {'Metric':<36}  {'All author pubs':>16}  {'HC author pubs':>15}")
print("  " + "─" * 72)
for label, v_all, v_hc in [
    ("Gini coefficient (n_author_pub)", f"{g_author_pub:.3f}", f"{g_hc_author_pub:.3f}"),
    ("HHI (0–1 scale)",                 f"{h_author_pub:.4f}", f"{h_hc_author_pub:.4f}"),
    ("Top-1 author share",              f"{100*top_k_share(all_author_pub_ppa,1):.1f} %",  f"{100*top_k_share(hc_author_pub_ppa,1):.1f} %"),
    ("Top-3 authors share",             f"{100*top_k_share(all_author_pub_ppa,3):.1f} %",  f"{100*top_k_share(hc_author_pub_ppa,3):.1f} %"),
    ("Top-5 authors share",             f"{100*top_k_share(all_author_pub_ppa,5):.1f} %",  f"{100*top_k_share(hc_author_pub_ppa,5):.1f} %"),
    ("Top-10 authors share",            f"{100*top_k_share(all_author_pub_ppa,10):.1f} %", f"{100*top_k_share(hc_author_pub_ppa,10):.1f} %"),
]:
    print(f"  {label:<36}  {v_all:>16}  {v_hc:>15}")
cit_all = authors_pub["cit_author_pub"]
print()
print(f"  Citation-based (cit_author_pub, all author-publication authors):")
print(f"    Gini: {gini(cit_all.values):.3f}   "
      f"Top-1: {100*top_k_share(cit_all,1):.1f} %   "
      f"Top-5: {100*top_k_share(cit_all,5):.1f} %")

# ── D. Affiliation type ───────────────────────────────────────────────────────
sub("D. Affiliation type breakdown")
type_map = {
    "startup-only":    "Startup-only",
    "startup+academia":"Startup + academic institution",
    "startup+others":  "Startup + other (non-academic)",
}
obs_by_type    = (category_pub_obs.groupby("affil_type")["UID"].size()
                  .reindex(list(type_map), fill_value=0))
hc_obs_by_type = (hc_category_pub_obs.groupby("affil_type")["UID"].size()
                  .reindex(list(type_map), fill_value=0))

print("  Paper-category observations count a paper once in every represented")
print("  author affiliation category, including alphabetical and EVEN/shared papers.")
print(f"  {'Affiliation type':<38}  {'Authors':>8}  {'%':>6}  "
      f"{'Pub-cat':>9}  {'% obs':>6}  {'HC auth':>8}  {'% HC auth':>9}  {'HC cat':>7}  {'% HC obs':>8}")
print("  " + "─" * 110)
for key, label in type_map.items():
    na  = int((authors_pub["affil_type"] == key).sum())
    nh  = int((hc_pub_auth["affil_type"] == key).sum()) if "affil_type" in hc_pub_auth else 0
    oa  = int(obs_by_type[key])
    oh  = int(hc_obs_by_type[key])
    print(f"  {label:<38}  {fmt(na):>8}  {pct(na, N_AUTH_AUTHOR_PUB):>6}  "
          f"{fmt(oa):>9}  {pct(oa, len(category_pub_obs)):>6}  {nh:>8}  {pct(nh, N_AUTH_HC_AUTHOR_PUB):>9}  "
          f"{oh:>7}  {pct(oh, len(hc_category_pub_obs)):>8}")

# ── E. Top contributing authors ────────────────────────────────────────────────
sub("E.1  Top 15 authors by lead papers (n_lead)")
print(f"  {'Rk':<4}  {'Author':<28}  {'n_lead':>6}  {'cit_lead':>9}  {'Startup':<22}  Affil.")
print("  " + "─" * 90)
for rank, row in enumerate(authors_lead.head(15).itertuples(), 1):
    s = str(row.startup)[:22] if pd.notna(row.startup) else "—"
    print(f"  {rank:<4}  {row.author_name:<28}  {row.n_lead:>6}  {row.cit_lead:>9,}  {s:<22}  {row.affil_type}")

sub("E.2  Top 15 authors by lead citations (cit_lead)")
top_cit = authors_lead.sort_values("cit_lead", ascending=False).head(15)
print(f"  {'Rk':<4}  {'Author':<28}  {'cit_lead':>9}  {'n_lead':>6}  {'Startup':<22}  Affil.")
print("  " + "─" * 90)
for rank, row in enumerate(top_cit.itertuples(), 1):
    s = str(row.startup)[:22] if pd.notna(row.startup) else "—"
    print(f"  {rank:<4}  {row.author_name:<28}  {row.cit_lead:>9,}  {row.n_lead:>6}  {s:<22}  {row.affil_type}")

sub("E.3  Top 15 HC lead authors (by n_hc) and repeat HC author summary")
print(f"  {'Rk':<4}  {'Author':<28}  {'n_hc':>5}  {'n_lead':>6}  {'cit_lead':>9}  {'Startup':<22}  Affil.")
print("  " + "─" * 95)
for rank, row in enumerate(hc_auth.head(15).itertuples(), 1):
    s      = str(row.startup)[:22]  if pd.notna(row.startup)  else "—"
    n_lead = int(row.n_lead)   if pd.notna(row.n_lead)   else 0
    c_lead = int(row.cit_lead) if pd.notna(row.cit_lead) else 0
    at     = str(row.affil_type) if pd.notna(row.affil_type) else "—"
    print(f"  {rank:<4}  {row.author_name:<28}  {row.n_hc:>5}  {n_lead:>6}  {c_lead:>9,}  {s:<22}  {at}")

# Repeat HC authors (appeared on ≥2 HC papers)
repeat_hc       = hc_auth[hc_auth["n_hc"] >= 2].copy()
n_repeat_hc     = len(repeat_hc)
n_repeat_acad   = int((repeat_hc["affil_type"] == "startup+academia").sum())
repeat_hc_obs   = int(repeat_hc["n_hc"].sum())
print(f"\n  Repeat HC authors (n_hc ≥ 2):  {n_repeat_hc}")
print(f"    With academic co-affiliation: {n_repeat_acad}")
print(f"    HC author-paper obs. covered: {repeat_hc_obs} / {TOTAL_HC_OBS} "
      f"({100*repeat_hc_obs/TOTAL_HC_OBS:.1f} %)")

# ── F. Per-startup summary ────────────────────────────────────────────────────
sub("F. Per-startup summary (top 20 by lead papers)")
startup_stats = (
    authors_lead
    .assign(_s=authors_lead["startup"].str.split(" | ", regex=False))
    .explode("_s")
    .groupby("_s")
    .agg(n_authors  =("author_name","nunique"),
         n_lead_pap =("n_lead",     "sum"),
         total_cit  =("cit_lead",   "sum"),
         pct_acad   =("affil_type", lambda s: 100*(s=="startup+academia").sum()/len(s)))
    .reset_index().rename(columns={"_s":"startup"})
    .sort_values("n_lead_pap", ascending=False)
)
hc_ss = (
    hc_auth.dropna(subset=["startup"])
    .assign(_s=hc_auth["startup"].str.split(" | ", regex=False))
    .explode("_s")
    .groupby("_s")
    .agg(n_hc_auth=("author_name","nunique"), n_hc_pap=("n_hc","sum"))
    .reset_index().rename(columns={"_s":"startup"})
)
startup_stats = startup_stats.merge(hc_ss, on="startup", how="left")
startup_stats[["n_hc_auth","n_hc_pap"]] = startup_stats[["n_hc_auth","n_hc_pap"]].fillna(0).astype(int)

print(f"  {'Rk':<4}  {'Startup':<28}  {'Auth':>5}  {'Lead':>5}  "
      f"{'Cit':>10}  {'HC auth':>7}  {'HC pap':>6}  {'%Acad':>6}")
print("  " + "─" * 84)
for rank, row in enumerate(startup_stats.head(20).itertuples(), 1):
    print(f"  {rank:<4}  {row.startup[:28]:<28}  {row.n_authors:>5}  {row.n_lead_pap:>5}  "
          f"{row.total_cit:>10,}  {row.n_hc_auth:>7}  {row.n_hc_pap:>6}  {row.pct_acad:>5.0f}%")


print(f"\n{'─' * 72}")
print(f"  Sources:  {os.path.basename(STARTUPS_CSV)}")
print(f"            {os.path.basename(PAPERS_CSV)}")
print(f"            {os.path.basename(AUTHORS_CSV)}")
print(f"{'─' * 72}\n")
