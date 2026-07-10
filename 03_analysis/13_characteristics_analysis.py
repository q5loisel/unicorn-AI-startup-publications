#!/usr/bin/env python3
"""
13_characteristics_analysis.py
----------------
Descriptive and inferential statistics for Analyses 1–9:

  1.  Country-level distribution of scientific activity
  2.  Scientific output by valuation tier
  3.  Funding raised vs scientific publishing
  4.  Startup age and scientific productivity
  5.  Country × valuation hybrid analysis
  6.  Startup leadership/collaboration trends vs overall AI publications
  7.  Preprints vs peer-reviewed outputs across financial magnitude
  8.  Concentration analysis of scientific impact
  9.  Publication participation typology

Input files
-----------
  startups_data.csv      – firm metadata and financial variables (N = 317)
  included_papers.csv    – paper-level records used to reconstruct firm outputs
  all_ai_publication.xlsx – annual AI publication counts used as an external baseline

Usage
-----
  python 13_characteristics_analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, chi2_contingency

# ── Paths ─────────────────────────────────────────────────────────────────────
# All input files are read from a single, user-defined folder — set the
# AI_UNICORN_DATA_DIR environment variable to point at it (it defaults to the
# current working directory, i.e. run this script from inside that folder).
DATA_DIR     = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
STARTUPS_CSV = os.path.join(DATA_DIR, "startups_data.csv")
PAPERS_CSV   = os.path.join(DATA_DIR, "included_papers.csv")
ALL_AI_XLSX  = os.path.join(DATA_DIR, "all_ai_publication.xlsx")

REFERENCE_YEAR = 2025   # fixed data cutoff year for age calculations
HC_CITATION_THRESHOLD = 200
TREND_YEARS = list(range(2016, 2026))

FIRM_COUNT_COLS = [
    "n_papers_total", "n_papers_published", "n_papers_preprint",
    "n_highly_cited_total", "total_citations",
]

REQUIRED_STARTUP_COLS = [
    "startup", "Country", "Year Founded", "valuation_usd_m",
    "amount_raised_usd_m", "has_any_output", "has_highly_cited",
]

REQUIRED_PAPER_COLS = [
    "UID", "Year", "canonical_name", "Total Times Cited", "Document Type",
    "first_author", "middle_author", "last_author", "alpha_order",
    "is_highly_cited", "shared_startups", "edge_case_flag",
]

# ── Load ──────────────────────────────────────────────────────────────────────
df     = pd.read_csv(STARTUPS_CSV)
papers = pd.read_csv(PAPERS_CSV)

N = len(df)

# ── Helpers ───────────────────────────────────────────────────────────────────
def require_columns(data: pd.DataFrame, columns: list[str], source: str) -> None:
    """Fail fast if an input file is missing a column used by the analysis."""
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")


def require_nonblank(series: pd.Series, col_name: str) -> None:
    """Reject blank identifiers/categories used for grouping or joining."""
    blank = series.isna() | series.astype(str).str.strip().eq("")
    if blank.any():
        raise ValueError(f"{col_name} contains {int(blank.sum())} blank value(s).")


def normalize_yes_no(series: pd.Series, col_name: str) -> pd.Series:
    """Normalize yes/no fields and reject unexpected category labels."""
    values = series.fillna("").astype(str).str.strip().str.lower()
    unexpected = sorted(set(values) - {"yes", "no", ""})
    if unexpected:
        raise ValueError(
            f"{col_name} contains unexpected value(s): {unexpected}. "
            "Expected yes/no or blank."
        )
    return values


def coerce_nonnegative_int(series: pd.Series, col_name: str) -> pd.Series:
    """Coerce required count/citation fields, rejecting missing or invalid data."""
    values = pd.to_numeric(series, errors="coerce")
    bad = values.isna() | (values < 0) | (values % 1 != 0)
    if bad.any():
        examples = series.loc[bad].head(5).tolist()
        raise ValueError(
            f"{col_name} must contain non-negative integers. "
            f"Invalid example value(s): {examples}"
        )
    return values.astype(int)


def coerce_nonnegative_float(series: pd.Series, col_name: str, allow_missing: bool) -> pd.Series:
    """Coerce required monetary fields, rejecting negative values."""
    values = pd.to_numeric(series, errors="coerce")
    bad = values.lt(0)
    if not allow_missing:
        bad = bad | values.isna()
    if bad.any():
        examples = series.loc[bad].head(5).tolist()
        raise ValueError(
            f"{col_name} must contain non-negative numeric values. "
            f"Invalid example value(s): {examples}"
        )
    return values


def banner(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def sub(title: str) -> None:
    print(f"\n  ── {title} ──")


def gini(x: np.ndarray) -> float:
    """Gini coefficient for a non-negative array (zeros included)."""
    x = np.sort(np.asarray(x, dtype=float))
    total = x.sum()
    if total == 0:
        return 0.0
    n   = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2.0 * cum.sum() / total) / n


def top_share(series: pd.Series, pct: float) -> float:
    """Fraction of total held by the top `pct` proportion of observations.
    Uses floor (int) to count firms, consistent with 12_concentration_analysis.py."""
    k     = max(1, int(len(series) * pct))
    total = series.sum()
    return float(series.nlargest(k).sum() / total) if total > 0 else 0.0


def lorenz_points(series: pd.Series, n_deciles: int = 10) -> list[tuple[float, float]]:
    """Return (cumulative population %, cumulative share %) for each decile."""
    x = np.sort(np.asarray(series, dtype=float))
    total = x.sum()
    if total == 0:
        return [(i / n_deciles, 0.0) for i in range(n_deciles + 1)]
    pts = [(0.0, 0.0)]
    for i in range(1, n_deciles + 1):
        idx = min(int(np.floor(len(x) * i / n_deciles)), len(x))
        pts.append((i / n_deciles, float(x[:idx].sum() / total)))
    return pts


def describe_series(s: pd.Series, label: str = "") -> None:
    """Print a compact five-number summary."""
    prefix = f"  {label:30s}" if label else "  "
    print(
        f"{prefix}  n={len(s):4d}  "
        f"median={s.median():.1f}  "
        f"mean={s.mean():.1f}  "
        f"IQR={s.quantile(.25):.0f}–{s.quantile(.75):.0f}  "
        f"max={s.max():.0f}"
    )


def build_startup_ai_trends() -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Return the Figure 2C trend table and 2016→2025 growth coefficients."""
    all_ai = pd.read_excel(ALL_AI_XLSX)
    require_columns(all_ai, ["Publication Years", "Count"], os.path.basename(ALL_AI_XLSX))
    all_ai = all_ai.rename(columns={"Publication Years": "year", "Count": "all_ai"})
    all_ai["year"] = coerce_nonnegative_int(all_ai["year"], "Publication Years")
    all_ai["all_ai"] = coerce_nonnegative_int(all_ai["all_ai"], "Count")
    all_ai = all_ai[all_ai["year"].isin(TREND_YEARS)][["year", "all_ai"]]

    trend_source = papers_startup.copy()
    trend_source["is_alphabetical"] = (
        trend_source["alpha_order"].fillna("").astype(str).str.strip().str.upper()
        == "YES"
    )
    trend_source["is_lead_role"] = (
        trend_source["first_author"].notna() |
        trend_source["last_author"].notna() |
        trend_source["is_alphabetical"]
    )
    trend_source["is_middle_role"] = trend_source["middle_author"].notna()

    rows = []
    for yr in TREND_YEARS:
        sub_y = trend_source[trend_source["paper_year"] == yr]
        rows.append({
            "year": yr,
            "lead_startup_papers": int(
                sub_y.loc[sub_y["is_lead_role"], "UID"].nunique()
            ),
            "collaborative_only_startup_papers": int(
                sub_y.loc[
                    sub_y["is_middle_role"] & ~sub_y["is_lead_role"],
                    "UID"
                ].nunique()
            ),
        })

    trends = pd.DataFrame(rows).merge(all_ai, on="year", how="left")
    required = ["lead_startup_papers", "collaborative_only_startup_papers", "all_ai"]
    if trends[required].isna().any().any():
        raise ValueError("Startup/AI trend data contain missing values.")
    trends["startup_involved_papers"] = (
        trends["lead_startup_papers"] +
        trends["collaborative_only_startup_papers"]
    )
    trends["startup_share_of_all_ai_pct"] = (
        trends["startup_involved_papers"] / trends["all_ai"] * 100
    )

    baseline = trends.loc[trends["year"] == TREND_YEARS[0]].iloc[0]
    final = trends.loc[trends["year"] == TREND_YEARS[-1]].iloc[0]
    growth = {}
    cagr = {}
    n_years = TREND_YEARS[-1] - TREND_YEARS[0]
    for col in required + ["startup_involved_papers"]:
        if baseline[col] <= 0:
            raise ValueError(f"Trend baseline for {col} must be positive.")
        growth[col] = float(final[col] / baseline[col])
        cagr[col] = float(growth[col] ** (1 / n_years) - 1)
        trends[f"{col}_index_2016"] = trends[col] / baseline[col]

    return trends, growth, cagr


def assign_region(country: str) -> str:
    """Regional grouping used in Figure 2D."""
    europe = {
        "United Kingdom", "Germany", "France", "Sweden", "Ireland",
        "The Netherlands", "Finland", "Portugal", "Switzerland",
        "Belgium", "Norway", "Estonia",
    }
    if country == "United States":
        return "United States"
    if country == "China":
        return "China"
    if country in europe:
        return "Europe"
    return "Others"


def validate_inputs() -> None:
    """Validate schema and reconstruct firm-level aggregates from paper records."""
    global df, papers, papers_for_firms, papers_startup

    require_columns(df, REQUIRED_STARTUP_COLS, os.path.basename(STARTUPS_CSV))
    require_columns(papers, REQUIRED_PAPER_COLS, os.path.basename(PAPERS_CSV))
    require_nonblank(df["startup"], "startup")
    require_nonblank(df["Country"], "Country")
    papers["edge_case_flag"] = papers["edge_case_flag"].fillna("").astype(str)
    papers = papers[papers["edge_case_flag"] != "garbled_excluded"].copy()
    blank_uid = papers["UID"].isna() | papers["UID"].astype(str).str.strip().eq("")
    blank_startup_marker = (
        papers["canonical_name"].isna() |
        papers["canonical_name"].astype(str).str.strip().eq("")
    )
    papers = papers[~(blank_uid & blank_startup_marker)].copy()

    if df["startup"].duplicated().any():
        dupes = df.loc[df["startup"].duplicated(), "startup"].head(5).tolist()
        raise ValueError(f"Duplicate startup names in firm file: {dupes}")

    df["valuation_usd_m"] = coerce_nonnegative_float(
        df["valuation_usd_m"], "valuation_usd_m", allow_missing=False
    )
    df["amount_raised_usd_m"] = coerce_nonnegative_float(
        df["amount_raised_usd_m"], "amount_raised_usd_m", allow_missing=True
    )
    normalize_yes_no(df["has_any_output"], "has_any_output")
    normalize_yes_no(df["has_highly_cited"], "has_highly_cited")

    startup_names = set(df["startup"])
    papers["has_leadership"] = (
        papers["first_author"].notna() | papers["last_author"].notna()
    )
    papers["shared_startups"] = papers["shared_startups"].fillna("").astype(str)

    # Broad startup-involved paper universe used by Figure 2C:
    # includes both lead and middle-author participation, plus EVEN/shared papers.
    papers_startup = papers[
        papers["canonical_name"].isin(startup_names) |
        (papers["edge_case_flag"] == "even_shared")
    ].copy()
    require_nonblank(papers_startup["UID"], "UID")

    papers_startup["paper_year"] = coerce_nonnegative_int(
        papers_startup["Year"], "Year"
    )
    papers_startup["cit"] = coerce_nonnegative_int(
        papers_startup["Total Times Cited"], "Total Times Cited"
    )
    papers_startup["is_highly_cited_norm"] = normalize_yes_no(
        papers_startup["is_highly_cited"], "is_highly_cited"
    )
    papers_startup["is_preprint"] = (
        papers_startup["Document Type"].fillna("").astype(str).str.strip().str.lower()
        == "preprint"
    )
    papers_startup["is_hc"] = papers_startup["cit"] >= HC_CITATION_THRESHOLD
    if not papers_startup["is_highly_cited_norm"].eq(
        np.where(papers_startup["is_hc"], "yes", "no")
    ).all():
        raise ValueError(
            "Paper-level is_highly_cited does not match "
            f">={HC_CITATION_THRESHOLD} citations."
        )

    # Firm-output counts mirror 12_concentration_analysis.py:
    #   - standard lead papers with a startup in first/last position
    #   - alpha-mid papers where alphabetical order makes position uninformative
    #   - EVEN/shared papers attributed to every tied startup
    papers = papers_startup[
        (
            papers_startup["canonical_name"].isin(startup_names) &
            (
                papers_startup["has_leadership"] |
                (papers_startup["edge_case_flag"] == "alpha_mid")
            )
        ) |
        (papers_startup["edge_case_flag"] == "even_shared")
    ].copy()

    normal = papers[papers["edge_case_flag"] != "even_shared"].copy()
    even = papers[papers["edge_case_flag"] == "even_shared"].copy()
    even_rows = []
    for _, row in even.iterrows():
        tied = [s.strip() for s in row["shared_startups"].split(" | ") if s.strip()]
        for startup in tied:
            row_expanded = row.copy()
            row_expanded["canonical_name"] = startup
            even_rows.append(row_expanded)
    even_exp = (
        pd.DataFrame(even_rows) if even_rows
        else pd.DataFrame(columns=normal.columns)
    )
    papers_for_firms = pd.concat([normal, even_exp], ignore_index=True)
    unknown = sorted(set(papers_for_firms["canonical_name"]) - startup_names)
    if unknown:
        raise ValueError(
            "Expanded paper records contain startup(s) absent from firm file: "
            f"{unknown[:5]}"
        )

    paper_counts = (
        papers_for_firms.groupby("canonical_name")
        .agg(
            n_papers_total=("UID", "size"),
            n_papers_published=("is_preprint", lambda s: int((~s).sum())),
            n_papers_preprint=("is_preprint", "sum"),
            n_highly_cited_total=("is_hc", "sum"),
            total_citations=("cit", "sum"),
        )
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"canonical_name": "startup"})
    )
    df = df.drop(columns=[c for c in FIRM_COUNT_COLS if c in df.columns], errors="ignore")
    df = df.merge(paper_counts, on="startup", how="left")
    df[FIRM_COUNT_COLS] = df[FIRM_COUNT_COLS].fillna(0).astype(int)
    df["has_any_output_norm"] = np.where(df["n_papers_total"] > 0, "yes", "no")
    df["has_highly_cited_norm"] = np.where(
        df["n_highly_cited_total"] > 0, "yes", "no"
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────
validate_inputs()

# Normalise year_founded: strip whitespace, coerce to numeric, drop unparseable
df["year_founded_raw"] = df["Year Founded"].astype(str).str.strip()
df["year_founded"] = pd.to_numeric(df["year_founded_raw"], errors="coerce")
df["age"] = REFERENCE_YEAR - df["year_founded"]
future_years = df.loc[df["year_founded"] > REFERENCE_YEAR, ["startup", "Year Founded"]]
if not future_years.empty:
    raise ValueError(
        "Founding year is after the fixed reference year. "
        f"First example: {future_years.iloc[0].to_dict()}"
    )

# Binary flags on firm level
df["has_pub"]     = df["n_papers_total"] > 0
df["has_hc"]      = df["n_highly_cited_total"] > 0
df["has_preprint"] = df["n_papers_preprint"] > 0
df["region"] = df["Country"].map(assign_region)

# Valuation tiers (all firms are unicorns ≥ $1 B)
_val = df["valuation_usd_m"]
val_bins   = [-np.inf, 1000.0, 5000.0, 10000.0, np.inf]
val_labels = ["$1B (exactly)", ">$1B–$5B", ">$5B–$10B", ">$10B"]
df["val_tier"] = pd.cut(_val, bins=val_bins, labels=val_labels, right=True)
if (df["valuation_usd_m"] < 1000.0).any():
    raise ValueError("valuation_usd_m contains value(s) below the unicorn threshold.")

# Funding tertiles (excludes 2 firms with missing funding)
_fund = df["amount_raised_usd_m"].dropna()
t33, t67 = _fund.quantile(1/3), _fund.quantile(2/3)
df["fund_tier"] = pd.cut(
    df["amount_raised_usd_m"],
    bins=[-np.inf, t33, t67, np.inf],
    labels=["Low (bottom third)", "Medium (middle third)", "High (top third)"],
    right=True,
)

# Founding cohort
df["cohort"] = pd.cut(
    df["year_founded"],
    bins=[1997, 2009, 2014, 2019, 2025],
    labels=["Pre-2010", "2010–2014", "2015–2019", "2020–2025"],
    right=True,
)

trend_table, trend_growth, trend_cagr = build_startup_ai_trends()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  Country-level distribution of scientific activity
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 1  –  Country-level distribution of scientific activity")

# Aggregate per country
country_stats = (
    df.groupby("Country", dropna=False)
    .agg(
        n_firms=("startup", "count"),
        n_publishing=("has_pub", "sum"),
        n_hc=("has_hc", "sum"),
        med_pubs=("n_papers_total", "median"),
        med_cits=("total_citations", "median"),
        sum_total_cits=("total_citations", "sum"),
    )
    .reset_index()
)
country_stats["pct_pub"]     = 100 * country_stats["n_publishing"] / country_stats["n_firms"]
country_stats["pct_hc"]      = 100 * country_stats["n_hc"]        / country_stats["n_firms"]

# HC paper shares by country (expanded paper-firm attributions, join startup → country)
country_map = df.set_index("startup")["Country"].to_dict()
papers_for_firms["Country"] = papers_for_firms["canonical_name"].map(country_map)
hc_by_country = (
    papers_for_firms[papers_for_firms["is_hc"]]
    .groupby("Country").size().rename("n_hc_papers")
)
total_by_country = papers_for_firms.groupby("Country").size().rename("n_papers")
hc_share = (hc_by_country / total_by_country * 100).rename("hc_paper_share_pct").reset_index()
country_stats = country_stats.merge(hc_share, on="Country", how="left")
country_stats["hc_paper_share_pct"] = country_stats["hc_paper_share_pct"].fillna(0)

country_stats = country_stats.sort_values("n_firms", ascending=False)

sub("All countries (sorted by number of firms)")
print(f"  {'Country':<24}  {'Firms':>5}  {'Pub':>5}  {'%Pub':>6}  "
      f"{'Med pubs':>8}  {'Med cits':>8}  {'HC firms':>8}  "
      f"{'%HC firms':>9}  {'HC paper%':>9}")
print("  " + "─" * 95)
for _, row in country_stats.iterrows():
    print(
        f"  {str(row['Country']):<24}  {int(row['n_firms']):>5}  "
        f"{int(row['n_publishing']):>5}  {row['pct_pub']:>5.0f}%  "
        f"{row['med_pubs']:>8.1f}  {row['med_cits']:>8.0f}  "
        f"{int(row['n_hc']):>8}  {row['pct_hc']:>8.0f}%  "
        f"{row['hc_paper_share_pct']:>8.1f}%"
    )

sub("Key regions (≥3 firms): Kruskal-Wallis on n_papers_total")
major = country_stats[country_stats["n_firms"] >= 3]["Country"].tolist()
groups = [df.loc[df["Country"] == c, "n_papers_total"].values for c in major]
if len(groups) >= 2:
    H, p_kw = kruskal(*groups)
    print(f"  Countries with ≥3 firms: {', '.join(major)}")
    print(f"  Kruskal-Wallis H = {H:.2f},  p = {p_kw:.3e}  (k = {len(groups)} groups)")
else:
    print("  Insufficient groups for Kruskal-Wallis test.")

sub("US vs. non-US comparison (Mann-Whitney U)")
us_pubs   = df.loc[df["Country"] == "United States", "n_papers_total"]
non_us    = df.loc[df["Country"] != "United States", "n_papers_total"]
U_us, p_us = mannwhitneyu(us_pubs, non_us, alternative="two-sided")
print(f"  US  (n={len(us_pubs)}):     median={us_pubs.median():.1f}, mean={us_pubs.mean():.1f}")
print(f"  Non-US (n={len(non_us)}):  median={non_us.median():.1f}, mean={non_us.mean():.1f}")
print(f"  Mann-Whitney U = {U_us:.0f},  p = {p_us:.3e}")

sub("Figure 2D regions: publication output by geographic region")
region_order = ["China", "Europe", "Others", "United States"]
region_stats = (
    df.groupby("region")
    .agg(
        n=("startup", "count"),
        n_pub=("has_pub", "sum"),
        median_pubs=("n_papers_total", "median"),
        mean_pubs=("n_papers_total", "mean"),
        median_cits=("total_citations", "median"),
        n_hc=("has_hc", "sum"),
    )
    .reindex(region_order)
)
region_stats["pct_sample"] = region_stats["n"] / N * 100
region_stats["pct_pub"] = region_stats["n_pub"] / region_stats["n"] * 100
region_stats["pct_hc"] = region_stats["n_hc"] / region_stats["n"] * 100
print(
    f"  {'Region':<16}  {'n':>4}  {'%Sample':>8}  {'Pub':>5}  {'%Pub':>6}  "
    f"{'Med pubs':>8}  {'Mean pubs':>9}  {'Med cits':>8}  {'HC firms':>8}"
)
print("  " + "─" * 88)
for region, row in region_stats.iterrows():
    print(
        f"  {region:<16}  {int(row['n']):>4}  {row['pct_sample']:>7.1f}%  "
        f"{int(row['n_pub']):>5}  {row['pct_pub']:>5.1f}%  "
        f"{row['median_pubs']:>8.1f}  {row['mean_pubs']:>9.1f}  "
        f"{row['median_cits']:>8.1f}  {int(row['n_hc']):>8}"
    )

region_groups = [
    df.loc[df["region"] == r, "n_papers_total"].values
    for r in region_order
]
H_region, p_region = kruskal(*region_groups)
print(
    f"  Kruskal-Wallis across Figure 2D regions: "
    f"H = {H_region:.2f},  p = {p_region:.3e}"
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  Scientific output by valuation tier
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 2  –  Scientific output by valuation tier")

sub("Tier definitions and firm counts")
tier_order = val_labels
for tier in tier_order:
    sub_t = df[df["val_tier"] == tier]
    print(f"  {tier:<20}  n = {len(sub_t):3d}  "
          f"(val range: ${sub_t['valuation_usd_m'].min():.0f}M – "
          f"${sub_t['valuation_usd_m'].max():.0f}M USD)")

sub("Publications, citations, and highly-cited papers by valuation tier")
print(f"  {'Tier':<20}  {'n':>4}  {'%Pub':>6}  {'Med pubs':>8}  "
      f"{'%HC':>6}  {'Med cits':>8}  {'Med HC pubs':>11}")
print("  " + "─" * 75)
tier_rows = []
for tier in tier_order:
    sub_t = df[df["val_tier"] == tier]
    n_t = len(sub_t)
    pct_pub = 100 * sub_t["has_pub"].sum() / n_t
    pct_hc  = 100 * sub_t["has_hc"].sum()  / n_t
    med_pub = sub_t["n_papers_total"].median()
    med_cit = sub_t["total_citations"].median()
    med_hc  = sub_t["n_highly_cited_total"].median()
    print(f"  {tier:<20}  {n_t:>4}  {pct_pub:>5.1f}%  {med_pub:>8.1f}  "
          f"{pct_hc:>5.1f}%  {med_cit:>8.1f}  {med_hc:>11.1f}")
    tier_rows.append(sub_t)

sub("Citation distribution by valuation tier")
for tier, sub_t in zip(tier_order, tier_rows):
    pub_t = sub_t[sub_t["has_pub"]]
    if len(pub_t) == 0:
        continue
    describe_series(pub_t["total_citations"], tier)

sub("Statistical tests (Kruskal-Wallis across all tiers)")
kw_groups = [df.loc[df["val_tier"] == t, "n_papers_total"].values for t in tier_order]
H_v, p_v = kruskal(*kw_groups)
print(f"  Publications:  H = {H_v:.2f},  p = {p_v:.3e}")
kw_cit = [df.loc[df["val_tier"] == t, "total_citations"].values for t in tier_order]
H_vc, p_vc = kruskal(*kw_cit)
print(f"  Citations:     H = {H_vc:.2f},  p = {p_vc:.3e}")

sub("Spearman correlation: valuation (USD M) ~ n_papers_total (all firms)")
rho_v, p_rv = spearmanr(df["valuation_usd_m"], df["n_papers_total"])
print(f"  ρ = {rho_v:.3f},  p = {p_rv:.3e}  (N = {N})")
rho_v2, p_rv2 = spearmanr(
    df.loc[df["has_pub"], "valuation_usd_m"],
    df.loc[df["has_pub"], "n_papers_total"],
)
n_pub = int(df["has_pub"].sum())
print(f"  ρ = {rho_v2:.3f},  p = {p_rv2:.3e}  (publishing firms only, n = {n_pub})")

sub("Spearman correlation: valuation (USD M) ~ highly-cited papers")
rho_vhc, p_rvhc = spearmanr(df["valuation_usd_m"], df["n_highly_cited_total"])
print(f"  ρ = {rho_vhc:.3f},  p = {p_rvhc:.3e}  (N = {N})")
rho_vhc_pub, p_rvhc_pub = spearmanr(
    df.loc[df["has_pub"], "valuation_usd_m"],
    df.loc[df["has_pub"], "n_highly_cited_total"],
)
print(
    f"  ρ = {rho_vhc_pub:.3f},  p = {p_rvhc_pub:.3e}  "
    f"(publishing firms only, n = {n_pub})"
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3  –  Funding raised vs scientific publishing
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 3  –  Funding raised vs scientific publishing")

fund_valid = df[df["amount_raised_usd_m"].notna()].copy()
n_fund = len(fund_valid)

sub(
    f"Funding tertile thresholds  "
    f"(n = {n_fund}; {N - n_fund} firms with missing funding excluded)"
)
print(f"  Low    (bottom group):   ≤ ${t33:,.1f} M")
print(f"  Medium (middle group):   > ${t33:,.1f} M and ≤ ${t67:,.1f} M")
print(f"  High   (top group):      > ${t67:,.1f} M")
print("  Note: boundary ties can make group sizes differ slightly from exact thirds.")

sub("Publications, citations, and highly-cited papers by funding tier")
fund_order = ["Low (bottom third)", "Medium (middle third)", "High (top third)"]
print(f"  {'Tier':<26}  {'n':>4}  {'%Pub':>6}  {'Med pubs':>8}  "
      f"{'%HC':>6}  {'Med cits':>8}  {'Preprint%':>9}")
print("  " + "─" * 80)
fund_tier_rows = []
for tier in fund_order:
    sub_t = fund_valid[fund_valid["fund_tier"] == tier]
    n_t = len(sub_t)
    pct_pub = 100 * sub_t["has_pub"].sum()     / n_t
    pct_hc  = 100 * sub_t["has_hc"].sum()      / n_t
    med_pub = sub_t["n_papers_total"].median()
    med_cit = sub_t["total_citations"].median()
    pub_only = sub_t[sub_t["n_papers_total"] > 0]
    pct_pre = (
        100 * pub_only["n_papers_preprint"].sum() / pub_only["n_papers_total"].sum()
        if len(pub_only) > 0 else float("nan")
    )
    print(f"  {tier:<26}  {n_t:>4}  {pct_pub:>5.1f}%  {med_pub:>8.1f}  "
          f"{pct_hc:>5.1f}%  {med_cit:>8.1f}  {pct_pre:>8.1f}%")
    fund_tier_rows.append(sub_t)

sub("Statistical tests (Kruskal-Wallis across funding tertiles)")
kw_f = [fund_valid.loc[fund_valid["fund_tier"] == t, "n_papers_total"].values for t in fund_order]
H_f, p_f = kruskal(*kw_f)
print(f"  Publications:  H = {H_f:.2f},  p = {p_f:.3e}")
kw_fc = [fund_valid.loc[fund_valid["fund_tier"] == t, "total_citations"].values for t in fund_order]
H_fc, p_fc = kruskal(*kw_fc)
print(f"  Citations:     H = {H_fc:.2f},  p = {p_fc:.3e}")

sub("Spearman correlation: amount_raised_usd_m ~ n_papers_total")
rho_f, p_rf = spearmanr(fund_valid["amount_raised_usd_m"], fund_valid["n_papers_total"])
print(f"  All firms with funding data:   ρ = {rho_f:.3f},  p = {p_rf:.3e}  (n = {n_fund})")
pub_fund = fund_valid[fund_valid["has_pub"]]
rho_f2, p_rf2 = spearmanr(pub_fund["amount_raised_usd_m"], pub_fund["n_papers_total"])
print(f"  Publishing firms only:         ρ = {rho_f2:.3f},  p = {p_rf2:.3e}  (n = {len(pub_fund)})")

sub("Spearman correlation: amount_raised_usd_m ~ highly-cited papers")
rho_fhc, p_rfhc = spearmanr(
    fund_valid["amount_raised_usd_m"],
    fund_valid["n_highly_cited_total"],
)
print(f"  All firms with funding data:   ρ = {rho_fhc:.3f},  p = {p_rfhc:.3e}  (n = {n_fund})")
rho_fhc_pub, p_rfhc_pub = spearmanr(
    pub_fund["amount_raised_usd_m"],
    pub_fund["n_highly_cited_total"],
)
print(
    f"  Publishing firms only:         ρ = {rho_fhc_pub:.3f},  "
    f"p = {p_rfhc_pub:.3e}  (n = {len(pub_fund)})"
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4  –  Startup age and scientific productivity
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 4  –  Startup age and scientific productivity")

age_valid = df[df["year_founded"].notna()].copy()
n_age = len(age_valid)
n_excluded = N - n_age
print(f"\n  Note: {n_excluded} firm(s) excluded due to unparseable founding year.")

# Publications per year since founding
age_valid["pubs_per_year"] = np.where(
    age_valid["age"] > 0,
    age_valid["n_papers_total"] / age_valid["age"],
    np.nan,
)
age_valid["cits_per_pub"] = np.where(
    age_valid["n_papers_total"] > 0,
    age_valid["total_citations"] / age_valid["n_papers_total"],
    np.nan,
)

sub("Summary: all firms with valid founding year")
describe_series(age_valid["age"],             "Age (years)")
describe_series(age_valid["n_papers_total"],  "Publications (total)")
ppy = age_valid["pubs_per_year"].dropna()
describe_series(ppy,                          "Publications per year")
cpp = age_valid["cits_per_pub"].dropna()
describe_series(cpp,                          "Citations per publication")

sub("Publishing activity by founding cohort")
cohort_order = ["Pre-2010", "2010–2014", "2015–2019", "2020–2025"]
print(f"  {'Cohort':<16}  {'n':>4}  {'%Pub':>6}  {'Med pubs':>8}  "
      f"{'Med cits':>8}  {'%HC':>6}  {'Med age':>7}")
print("  " + "─" * 68)
for coh in cohort_order:
    sub_c = age_valid[age_valid["cohort"] == coh]
    if len(sub_c) == 0:
        continue
    n_c     = len(sub_c)
    pct_pub = 100 * sub_c["has_pub"].sum() / n_c
    med_pub = sub_c["n_papers_total"].median()
    med_cit = sub_c["total_citations"].median()
    pct_hc  = 100 * sub_c["has_hc"].sum() / n_c
    med_age = sub_c["age"].median()
    print(f"  {coh:<16}  {n_c:>4}  {pct_pub:>5.1f}%  {med_pub:>8.1f}  "
          f"{med_cit:>8.1f}  {pct_hc:>5.1f}%  {med_age:>7.1f}")

sub("Spearman correlations with founding year")
rho_yr, p_yr = spearmanr(age_valid["year_founded"], age_valid["n_papers_total"])
print(f"  Founding year ~ publications (all firms, n={n_age}):  ρ = {rho_yr:.3f},  p = {p_yr:.3e}")
pub_age = age_valid[age_valid["has_pub"]]
rho_yr2, p_yr2 = spearmanr(pub_age["year_founded"], pub_age["n_papers_total"])
print(f"  Founding year ~ publications (publishers, n={len(pub_age)}):  ρ = {rho_yr2:.3f},  p = {p_yr2:.3e}")
rho_age, p_age = spearmanr(pub_age["age"], pub_age["n_papers_total"])
print(f"  Age ~ publications (publishers, n={len(pub_age)}):             ρ = {rho_age:.3f},  p = {p_age:.3e}")
ppy_valid = pub_age[pub_age["pubs_per_year"].notna()]
rho_ppy, p_ppy = spearmanr(ppy_valid["age"], ppy_valid["pubs_per_year"])
print(f"  Age ~ publications per year (publishers, n={len(ppy_valid)}):     ρ = {rho_ppy:.3f},  p = {p_ppy:.3e}")

sub("Kruskal-Wallis: publications across founding cohorts")
coh_groups = [
    age_valid.loc[age_valid["cohort"] == c, "n_papers_total"].values
    for c in cohort_order
    if (age_valid["cohort"] == c).sum() > 0
]
H_coh, p_coh = kruskal(*coh_groups)
print(f"  H = {H_coh:.2f},  p = {p_coh:.3e}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5  –  Country × Valuation hybrid analysis
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 5  –  Country × Valuation hybrid analysis")

sub("Countries with ≥3 firms: median valuation and publication activity")
major_countries = (
    df.groupby("Country")["startup"].count()
    [lambda s: s >= 3].index.tolist()
)
print(f"  {'Country':<24}  {'n':>4}  {'Med val ($M)':>12}  {'Med pubs':>8}  "
      f"{'%Pub':>6}  {'Med cits':>8}  {'%HC':>6}")
print("  " + "─" * 80)
for c in sorted(major_countries, key=lambda x: df[df["Country"] == x]["valuation_usd_m"].median(), reverse=True):
    sub_c = df[df["Country"] == c]
    n_c   = len(sub_c)
    med_v = sub_c["valuation_usd_m"].median()
    med_p = sub_c["n_papers_total"].median()
    pct_p = 100 * sub_c["has_pub"].sum() / n_c
    med_c = sub_c["total_citations"].median()
    pct_h = 100 * sub_c["has_hc"].sum()  / n_c
    print(f"  {c:<24}  {n_c:>4}  {med_v:>12,.0f}  {med_p:>8.1f}  "
          f"{pct_p:>5.1f}%  {med_c:>8.1f}  {pct_h:>5.1f}%")

sub("US firms: publication activity within valuation tiers")
us_df = df[df["Country"] == "United States"]
print(f"  {'Tier':<20}  {'n':>4}  {'%Pub':>6}  {'Med pubs':>8}  {'%HC':>6}")
print("  " + "─" * 50)
for tier in tier_order:
    sub_t = us_df[us_df["val_tier"] == tier]
    if len(sub_t) == 0:
        continue
    n_t     = len(sub_t)
    pct_pub = 100 * sub_t["has_pub"].sum() / n_t
    med_pub = sub_t["n_papers_total"].median()
    pct_hc  = 100 * sub_t["has_hc"].sum()  / n_t
    print(f"  {tier:<20}  {n_t:>4}  {pct_pub:>5.1f}%  {med_pub:>8.1f}  {pct_hc:>5.1f}%")

sub("Spearman: valuation ~ publications, by country (≥3 firms)")
for c in sorted(major_countries):
    sub_c = df[df["Country"] == c]
    if len(sub_c) < 3:
        print(f"  {c:<24}  (fewer than 3 firms, skipped)")
        continue
    if sub_c["valuation_usd_m"].nunique() < 2:
        print(f"  {c:<24}  (constant valuation, skipped)")
        continue
    if sub_c["n_papers_total"].nunique() < 2:
        print(f"  {c:<24}  (constant publication count, skipped)")
        continue
    rho_c, p_c = spearmanr(sub_c["valuation_usd_m"], sub_c["n_papers_total"])
    print(f"  {c:<24}  ρ = {rho_c:+.3f},  p = {p_c:.3e}  (n = {len(sub_c)})")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6  –  Startup leadership/collaboration trends vs overall AI publications
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 6  –  Startup leadership/collaboration trends vs overall AI publications")

sub("Annual counts used in Figure 2C")
print(
    f"  {'Year':<6}  {'Lead startup':>13}  {'Collaborative-only':>20}  "
    f"{'Startup total':>14}  {'All AI publications':>20}  {'Startup share':>14}"
)
print("  " + "─" * 100)
for _, row in trend_table.iterrows():
    print(
        f"  {int(row['year']):<6}  "
        f"{int(row['lead_startup_papers']):>13}  "
        f"{int(row['collaborative_only_startup_papers']):>20}  "
        f"{int(row['startup_involved_papers']):>14}  "
        f"{int(row['all_ai']):>20,}  "
        f"{row['startup_share_of_all_ai_pct']:>13.3f}%"
    )

sub(f"Coefficient of evolution ({TREND_YEARS[0]}→{TREND_YEARS[-1]})")
trend_labels = {
    "lead_startup_papers": "Lead startup papers",
    "collaborative_only_startup_papers": "Collaborative-only startup papers",
    "startup_involved_papers": "All startup-involved papers",
    "all_ai": "All AI publications",
}
for col, label in trend_labels.items():
    start = int(trend_table[col].iloc[0])
    end = int(trend_table[col].iloc[-1])
    print(
        f"  {label:<40}  {start:>8,} → {end:>8,}  "
        f"{trend_growth[col]:>6.1f}×  CAGR={100*trend_cagr[col]:>5.1f}%"
    )

sub("Growth index data (2016 = 1)")
print(
    f"  {'Year':<6}  {'Lead idx':>9}  {'Collaborative idx':>18}  "
    f"{'Startup idx':>12}  {'All AI idx':>11}"
)
print("  " + "─" * 67)
for _, row in trend_table.iterrows():
    print(
        f"  {int(row['year']):<6}  "
        f"{row['lead_startup_papers_index_2016']:>9.2f}  "
        f"{row['collaborative_only_startup_papers_index_2016']:>18.2f}  "
        f"{row['startup_involved_papers_index_2016']:>12.2f}  "
        f"{row['all_ai_index_2016']:>11.2f}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7  –  Preprints vs peer-reviewed outputs across financial magnitude
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 7  –  Preprints vs peer-reviewed outputs across financial magnitude")

# Restrict to publishing firms
pub_df = df[df["has_pub"]].copy()
pub_df["preprint_ratio"] = pub_df["n_papers_preprint"] / pub_df["n_papers_total"]

sub("Overall (publishing firms only)")
print(f"  N publishing firms: {len(pub_df)}")
describe_series(pub_df["preprint_ratio"] * 100, "Preprint share (%)")
preprint_heavy = int((pub_df["preprint_ratio"] > 0.5).sum())
print(f"  Firms where >50% of output is preprints: {preprint_heavy} ({100*preprint_heavy/len(pub_df):.1f}%)")
pure_preprint  = int((pub_df["n_papers_published"] == 0).sum())
print(f"  Firms with preprints only (no peer-review): {pure_preprint} ({100*pure_preprint/len(pub_df):.1f}%)")

sub("Preprint share by valuation tier (publishing firms)")
print(f"  {'Tier':<20}  {'n pub':>5}  {'Med preprint%':>13}  {'Mean preprint%':>14}")
print("  " + "─" * 57)
ppr_groups_val = []
for tier in tier_order:
    sub_t = pub_df[pub_df["val_tier"] == tier]
    if len(sub_t) == 0:
        continue
    med_pp = sub_t["preprint_ratio"].median() * 100
    mn_pp  = sub_t["preprint_ratio"].mean()   * 100
    print(f"  {tier:<20}  {len(sub_t):>5}  {med_pp:>12.1f}%  {mn_pp:>13.1f}%")
    ppr_groups_val.append(sub_t["preprint_ratio"].values)

if len(ppr_groups_val) >= 2:
    H_pv, p_pv = kruskal(*ppr_groups_val)
    print(f"\n  Kruskal-Wallis (preprint ratio across val. tiers):  H = {H_pv:.2f},  p = {p_pv:.3e}")

sub("Preprint share by funding tier (publishing firms with funding data)")
pub_fund_df = pub_df[pub_df["amount_raised_usd_m"].notna()].copy()
print(f"  {'Tier':<26}  {'n pub':>5}  {'Med preprint%':>13}  {'Mean preprint%':>14}")
print("  " + "─" * 63)
ppr_groups_fund = []
for tier in fund_order:
    sub_t = pub_fund_df[pub_fund_df["fund_tier"] == tier]
    if len(sub_t) == 0:
        continue
    med_pp = sub_t["preprint_ratio"].median() * 100
    mn_pp  = sub_t["preprint_ratio"].mean()   * 100
    print(f"  {tier:<26}  {len(sub_t):>5}  {med_pp:>12.1f}%  {mn_pp:>13.1f}%")
    ppr_groups_fund.append(sub_t["preprint_ratio"].values)

if len(ppr_groups_fund) >= 2:
    H_pf, p_pf = kruskal(*ppr_groups_fund)
    print(f"\n  Kruskal-Wallis (preprint ratio across funding tiers): H = {H_pf:.2f},  p = {p_pf:.3e}")

sub("Mann-Whitney: preprint ratio – high-valuation (>$10B) vs. lower")
hi_val_pp  = pub_df.loc[pub_df["val_tier"] == ">$10B", "preprint_ratio"]
lo_val_pp  = pub_df.loc[pub_df["val_tier"] != ">$10B", "preprint_ratio"]
U_pv, p_pv2 = mannwhitneyu(hi_val_pp, lo_val_pp, alternative="two-sided")
print(f"  >$10B (n={len(hi_val_pp)}): median preprint% = {hi_val_pp.median()*100:.1f}%")
print(f"  ≤$10B (n={len(lo_val_pp)}): median preprint% = {lo_val_pp.median()*100:.1f}%")
print(f"  U = {U_pv:.0f},  p = {p_pv2:.3e}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8  –  Concentration analysis of scientific impact
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 8  –  Concentration analysis of scientific impact")

sub("Top-share statistics (all N=317 firms)")
W7a, W7b, W7c = 42, 14, 14
print(f"  {'Metric':<{W7a}}  {'Publications':>{W7b}}  {'Citations':>{W7c}}")
print("  " + "─" * (W7a + W7b + W7c + 4))
g_pubs = gini(df["n_papers_total"])
g_cits = gini(df["total_citations"])
metrics_7 = [
    ("Gini coefficient",
     f"{g_pubs:.3f}", f"{g_cits:.3f}"),
    (f"Top 1%  (n={max(1,int(N*0.01))}) share",
     f"{100 * top_share(df['n_papers_total'], 0.01):.1f} %",
     f"{100 * top_share(df['total_citations'], 0.01):.1f} %"),
    (f"Top 5%  (n={max(1,int(N*0.05))}) share",
     f"{100 * top_share(df['n_papers_total'], 0.05):.1f} %",
     f"{100 * top_share(df['total_citations'], 0.05):.1f} %"),
    (f"Top 10% (n={max(1,int(N*0.10))}) share",
     f"{100 * top_share(df['n_papers_total'], 0.10):.1f} %",
     f"{100 * top_share(df['total_citations'], 0.10):.1f} %"),
    (f"Top 20% (n={max(1,int(N*0.20))}) share",
     f"{100 * top_share(df['n_papers_total'], 0.20):.1f} %",
     f"{100 * top_share(df['total_citations'], 0.20):.1f} %"),
    (f"Top 25% (n={max(1,int(N*0.25))}) share",
     f"{100 * top_share(df['n_papers_total'], 0.25):.1f} %",
     f"{100 * top_share(df['total_citations'], 0.25):.1f} %"),
]
for m, v1, v2 in metrics_7:
    print(f"  {m:<{W7a}}  {v1:>{W7b}}  {v2:>{W7c}}")

sub("Lorenz curve data points (publications, by decile)")
pts = lorenz_points(df["n_papers_total"])
print(f"  {'Decile':<8}  {'Cum. pop%':>10}  {'Cum. pub%':>10}")
print("  " + "─" * 32)
for pop_pct, share_pct in pts:
    print(f"  {pop_pct*100:>6.0f}%    {pop_pct*100:>8.1f}%  {share_pct*100:>9.1f}%")

sub("Lorenz curve data points (citations, by decile)")
pts_c = lorenz_points(df["total_citations"])
print(f"  {'Decile':<8}  {'Cum. pop%':>10}  {'Cum. cit%':>10}")
print("  " + "─" * 32)
for pop_pct, share_pct in pts_c:
    print(f"  {pop_pct*100:>6.0f}%    {pop_pct*100:>8.1f}%  {share_pct*100:>9.1f}%")

sub("Gini coefficient by valuation tier")
print(f"  {'Tier':<20}  {'n':>4}  {'Gini pubs':>10}  {'Gini cits':>10}")
print("  " + "─" * 50)
for tier in tier_order:
    sub_t = df[df["val_tier"] == tier]
    g_p = gini(sub_t["n_papers_total"].values)
    g_c = gini(sub_t["total_citations"].values)
    print(f"  {tier:<20}  {len(sub_t):>4}  {g_p:>10.3f}  {g_c:>10.3f}")

sub("Spearman: valuation ~ total citations (all firms)")
rho_vc, p_vc = spearmanr(df["valuation_usd_m"], df["total_citations"])
print(f"  ρ = {rho_vc:.3f},  p = {p_vc:.3e}  (N = {N})")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9  –  Publication participation typology
# ═════════════════════════════════════════════════════════════════════════════
banner("SECTION 9  –  Publication participation typology")

# Define four groups based on output volume and highly-cited status
def assign_typology(row):
    if row["n_papers_total"] == 0:
        return "T0 – No output"
    high_output = row["n_papers_total"] > 5
    high_impact = row["n_highly_cited_total"] > 0
    if high_output and high_impact:
        return "T3 – High output / high impact"
    if high_output and not high_impact:
        return "T2 – High output / low impact"
    if not high_output and high_impact:
        return "T1 – Low output / high impact"
    return "T1 – Low output / low impact"

df["typology"] = df.apply(assign_typology, axis=1)

# Rename to keep ordering clean
type_order = [
    "T0 – No output",
    "T1 – Low output / low impact",
    "T1 – Low output / high impact",
    "T2 – High output / low impact",
    "T3 – High output / high impact",
]

sub("Group definitions")
print("  T0: 0 papers")
print("  T1 (low output, low impact):   1–5 papers, 0 highly-cited papers")
print("  T1 (low output, high impact):  1–5 papers, ≥1 highly-cited paper")
print("  T2 (high output, low impact):  >5 papers,  0 highly-cited papers")
print("  T3 (high output, high impact): >5 papers,  ≥1 highly-cited paper")

sub("Group sizes and publication statistics")
print(f"  {'Typology':<38}  {'n':>4}  {'%':>6}  "
      f"{'Med pubs':>8}  {'Med HC':>7}  {'Med cits':>8}")
print("  " + "─" * 82)
for typ in type_order:
    sub_t = df[df["typology"] == typ]
    n_t = len(sub_t)
    if n_t == 0:
        continue
    pct  = 100 * n_t / N
    m_p  = sub_t["n_papers_total"].median()
    m_hc = sub_t["n_highly_cited_total"].median()
    m_c  = sub_t["total_citations"].median()
    print(f"  {typ:<38}  {n_t:>4}  {pct:>5.1f}%  "
          f"{m_p:>8.1f}  {m_hc:>7.1f}  {m_c:>8.1f}")

sub("Financial profile by typology")
print(f"  {'Typology':<38}  {'n':>4}  {'Med val ($M)':>12}  {'Med raised ($M)':>15}")
print("  " + "─" * 75)
for typ in type_order:
    sub_t = df[df["typology"] == typ]
    n_t = len(sub_t)
    if n_t == 0:
        continue
    med_val  = sub_t["valuation_usd_m"].median()
    med_fund = sub_t["amount_raised_usd_m"].median()
    fund_str = f"${med_fund:,.0f}" if pd.notna(med_fund) else "N/A"
    print(f"  {typ:<38}  {n_t:>4}  ${med_val:>11,.0f}  {fund_str:>15}")

sub("Founding year profile by typology")
print(f"  {'Typology':<38}  {'n':>4}  {'Med year':>8}  {'Min year':>8}  {'Max year':>8}")
print("  " + "─" * 73)
for typ in type_order:
    sub_t = df[df["typology"] == typ].dropna(subset=["year_founded"])
    n_t = len(sub_t)
    if n_t == 0:
        continue
    med_yr = sub_t["year_founded"].median()
    min_yr = int(sub_t["year_founded"].min())
    max_yr = int(sub_t["year_founded"].max())
    print(f"  {typ:<38}  {n_t:>4}  {med_yr:>8.0f}  {min_yr:>8}  {max_yr:>8}")

sub("Country distribution by typology (top-4 countries)")
top4 = ["United States", "China", "United Kingdom", "Germany"]
print(f"  {'Typology':<38}", end="")
for c in top4:
    print(f"  {c[:12]:>12}", end="")
print()
print("  " + "─" * (38 + 4 + len(top4) * 14))
for typ in type_order:
    sub_t = df[df["typology"] == typ]
    n_t = len(sub_t)
    if n_t == 0:
        continue
    print(f"  {typ:<38}", end="")
    for c in top4:
        c_n = int((sub_t["Country"] == c).sum())
        c_pct = 100 * c_n / n_t
        print(f"  {c_n:>4} ({c_pct:>4.0f}%)", end="")
    print()

sub("Chi-square: typology × country (US vs. non-US)")
df["is_us"] = (df["Country"] == "United States").astype(int)
contingency = pd.crosstab(df["typology"], df["is_us"])
chi2_t, p_t, dof_t, _ = chi2_contingency(contingency)
print(f"  χ²({dof_t}) = {chi2_t:.2f},  p = {p_t:.3e}")

print(f"\n{'─' * 72}")
print(f"  Source files:")
print(f"    {os.path.basename(STARTUPS_CSV)}")
print(f"    {os.path.basename(PAPERS_CSV)}")
print(f"    {os.path.basename(ALL_AI_XLSX)}")
print(f"  Reference year for age calculations: {REFERENCE_YEAR}")
print(f"  N = {N} AI startups")
print(f"{'─' * 72}\n")
