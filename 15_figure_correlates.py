"""
Figure 2 – Economic scale versus scientific contribution
=========================================================
A – Valuation vs publication count (log-log scatterplot)
B – Funding raised vs publication count (log-log scatterplot)
C – Startup lead/collaboration trends vs all AI publications (2016–2025)
D – Median publications per firm by geographic region, stacked by type

Rules applied (figures_rules):
  - No figure title or panel sub-titles in figure
  - No grid lines, no minor tick marks
  - No grayscale: open circles for no-output, blue solid for publishers, amber for HC firms
  - Distinct hues: peer-reviewed / lead = blue (#2C6E9B), highly-cited = amber (#E8A830),
    preprint / collaboration = teal (#3D9E8C), all AI = purple (#7A5195)
  - Helvetica / sans-serif, 10 pt bold part labels
  - Minimal in-figure text; details in caption
  - Figure sized at ~7 in wide (print-ready)
"""

import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "figure2-matplotlib")
)

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy import stats

# ── Paths and constants ───────────────────────────────────────────────────────
# Inputs are read, and figure2.png is written, into a single, user-defined
# folder — set the AI_UNICORN_DATA_DIR environment variable to point at it
# (it defaults to the current working directory, i.e. run this script from
# inside that folder).
DATA_DIR = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
STARTUPS_CSV = os.path.join(DATA_DIR, "startups_data.csv")
PAPERS_CSV = os.path.join(DATA_DIR, "included_papers.csv")
ALL_AI_CSV = os.path.join(DATA_DIR, "all_ai_publication.xlsx")
OUT_PATH = os.path.join(DATA_DIR, "figure2.png")

REFERENCE_YEAR = 2025
HC_CITATION_THRESHOLD = 200

FIRM_COUNT_COLS = [
    "n_papers_total", "n_papers_published", "n_papers_preprint",
    "n_highly_cited_total", "n_highly_cited_published",
    "n_highly_cited_preprint", "total_citations",
]
REQUIRED_STARTUP_COLS = [
    "startup", "Country", "has_any_output", "has_highly_cited",
    "valuation_usd_m", "amount_raised_usd_m",
]
REQUIRED_PAPER_COLS = [
    "UID", "Year", "canonical_name", "Total Times Cited", "Document Type",
    "first_author", "middle_author", "last_author", "alpha_order",
    "is_highly_cited", "shared_startups", "edge_case_flag",
]


def require_columns(data, columns, source):
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")


def require_nonblank(series, col_name):
    blank = series.isna() | series.astype(str).str.strip().eq("")
    if blank.any():
        raise ValueError(f"{col_name} contains {int(blank.sum())} blank value(s).")


def normalize_yes_no(series, col_name):
    values = series.fillna("").astype(str).str.strip().str.lower()
    unexpected = sorted(set(values) - {"yes", "no", ""})
    if unexpected:
        raise ValueError(
            f"{col_name} contains unexpected value(s): {unexpected}. "
            "Expected yes/no or blank."
        )
    return values


def coerce_nonnegative_int(series, col_name):
    values = pd.to_numeric(series, errors="coerce")
    bad = values.isna() | (values < 0) | (values % 1 != 0)
    if bad.any():
        examples = series.loc[bad].head(5).tolist()
        raise ValueError(
            f"{col_name} must contain non-negative integers. "
            f"Invalid example value(s): {examples}"
        )
    return values.astype(int)


def coerce_nonnegative_float(series, col_name, allow_missing):
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


def validate_and_prepare(usd, papers):
    require_columns(usd, REQUIRED_STARTUP_COLS, os.path.basename(STARTUPS_CSV))
    require_columns(papers, REQUIRED_PAPER_COLS, os.path.basename(PAPERS_CSV))
    require_nonblank(usd["startup"], "startup")
    require_nonblank(usd["Country"], "Country")

    if usd["startup"].duplicated().any():
        dupes = usd.loc[usd["startup"].duplicated(), "startup"].head(5).tolist()
        raise ValueError(f"Duplicate startup names in firm file: {dupes}")

    usd["valuation_usd_m"] = coerce_nonnegative_float(
        usd["valuation_usd_m"], "valuation_usd_m", allow_missing=False
    )
    usd["amount_raised_usd_m"] = coerce_nonnegative_float(
        usd["amount_raised_usd_m"], "amount_raised_usd_m", allow_missing=True
    )
    if (usd["valuation_usd_m"] <= 0).any():
        raise ValueError("valuation_usd_m must be positive for the log-scale plot.")
    if (usd["amount_raised_usd_m"].dropna() <= 0).any():
        raise ValueError("amount_raised_usd_m must be positive for the log-scale plot.")
    usd["has_any_output_norm"] = normalize_yes_no(
        usd["has_any_output"], "has_any_output"
    )
    usd["has_highly_cited_norm"] = normalize_yes_no(
        usd["has_highly_cited"], "has_highly_cited"
    )

    papers["edge_case_flag"] = papers["edge_case_flag"].fillna("").astype(str)
    papers = papers[papers["edge_case_flag"] != "garbled_excluded"].copy()
    blank_uid = papers["UID"].isna() | papers["UID"].astype(str).str.strip().eq("")
    blank_startup_marker = (
        papers["canonical_name"].isna() |
        papers["canonical_name"].astype(str).str.strip().eq("")
    )
    papers = papers[~(blank_uid & blank_startup_marker)].copy()

    startup_names = set(usd["startup"])
    papers["has_leadership"] = (
        papers["first_author"].notna() | papers["last_author"].notna()
    )
    papers["shared_startups"] = papers["shared_startups"].fillna("").astype(str)

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

    papers_for_firms["is_peer_nohc"] = (
        (~papers_for_firms["is_preprint"]) & (~papers_for_firms["is_hc"])
    )
    papers_for_firms["is_pre_nohc"] = (
        papers_for_firms["is_preprint"] & (~papers_for_firms["is_hc"])
    )
    papers_for_firms["is_hc_published"] = (
        papers_for_firms["is_hc"] & (~papers_for_firms["is_preprint"])
    )
    papers_for_firms["is_hc_preprint"] = (
        papers_for_firms["is_hc"] & papers_for_firms["is_preprint"]
    )

    paper_counts = (
        papers_for_firms.groupby("canonical_name")
        .agg(
            n_papers_total=("UID", "size"),
            n_papers_published=("is_preprint", lambda s: int((~s).sum())),
            n_papers_preprint=("is_preprint", "sum"),
            n_highly_cited_total=("is_hc", "sum"),
            n_highly_cited_published=("is_hc_published", "sum"),
            n_highly_cited_preprint=("is_hc_preprint", "sum"),
            total_citations=("cit", "sum"),
        )
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"canonical_name": "startup"})
    )
    usd = usd.drop(columns=[c for c in FIRM_COUNT_COLS if c in usd.columns], errors="ignore")
    usd = usd.merge(paper_counts, on="startup", how="left")
    usd[FIRM_COUNT_COLS] = usd[FIRM_COUNT_COLS].fillna(0).astype(int)
    usd["has_any_output_norm"] = np.where(usd["n_papers_total"] > 0, "yes", "no")
    usd["has_highly_cited_norm"] = np.where(
        usd["n_highly_cited_total"] > 0, "yes", "no"
    )
    return usd, papers, papers_for_firms, papers_startup


# ── Load data ─────────────────────────────────────────────────────────────────
usd = pd.read_csv(STARTUPS_CSV)
papers = pd.read_csv(PAPERS_CSV, low_memory=False)
usd, papers, papers_for_firms, papers_startup = validate_and_prepare(usd, papers)

df_a = usd.copy()                                          # Panel A: all 317
df   = usd.dropna(subset=["amount_raised_usd_m"]).copy()  # Panel B: 315 with funding data

N = len(usd)   # 317 firms (denominator for all panels)
MAX_PUBS = int(usd["n_papers_total"].max())

# ── Color palette ─────────────────────────────────────────────────────────────
C_PUB = "#2C6E9B"   # peer-reviewed (blue)
C_HC  = "#E8A830"   # highly-cited  (amber)
C_PPR = "#3D9E8C"   # preprint      (teal)
C_ALL_AI = "#7A5195"

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         7,
    "axes.linewidth":    0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size":  0,
    "ytick.minor.size":  0,
})

fig = plt.figure(figsize=(7.0, 5.5))
gs  = GridSpec(2, 2, figure=fig,
               hspace=0.52, wspace=0.42,
               left=0.11, right=0.98, top=0.97, bottom=0.11)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])


def add_panel_label(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def clean_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A – Valuation vs publication count (log-log)
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_a

x_a = df_a["valuation_usd_m"].values
y_a = df_a["n_papers_total"].values + 1   # +1 for log scale

no_out    = df_a["has_any_output_norm"] == "no"
pub_no_hc = (df_a["has_any_output_norm"] == "yes") & (df_a["has_highly_cited_norm"] == "no")
hc        = df_a["has_highly_cited_norm"] == "yes"

ax.scatter(x_a[no_out], y_a[no_out],
           s=10, facecolors="none", edgecolors=C_PUB, linewidths=0.5,
           alpha=0.55, zorder=2)
ax.scatter(x_a[pub_no_hc], y_a[pub_no_hc],
           s=12, facecolors=C_PUB, edgecolors="none",
           alpha=0.65, zorder=3)
ax.scatter(x_a[hc], y_a[hc],
           s=18, facecolors=C_HC, edgecolors="none",
           alpha=0.90, zorder=4)

# OpenAI and Waymo: standard data-relative offsets (right side, no legend conflict)
for name, (ha, dx, dy) in [("OpenAI", ("right", 1.35, 0.60)),
                            ("Waymo",  ("right", 1.35, 1.0))]:
    row = df_a[df_a["startup"] == name].iloc[0]
    xi, yi = row["valuation_usd_m"], row["n_papers_total"] + 1
    ax.annotate(name, xy=(xi, yi), xytext=(xi * dx, yi * dy),
                fontsize=5.5, ha=ha, va="center", color="black",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5))

# MEGVII and Preferred Networks: text to the right of their data points
# Both land at ~$10B x so the arrows are short horizontal/diagonal leftward
for name, (dx, dy) in [("MEGVII",             (2.5, 1.05)),
                        ("Preferred Networks", (5.0, 1.05))]:
    row = df_a[df_a["startup"] == name].iloc[0]
    xi, yi = row["valuation_usd_m"], row["n_papers_total"] + 1
    ax.annotate(name, xy=(xi, yi),
                xytext=(xi * dx, yi * dy),
                fontsize=5.5, ha="left", va="center", color="black",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 2, 6, 11, 26, 101, MAX_PUBS + 1])
ax.set_yticklabels(["0", "1", "5", "10", "25", "100", str(MAX_PUBS)], fontsize=6)
ax.set_xticks([1000, 3000, 10000, 50000, 500000])
ax.set_xticklabels(["$1B", "$3B", "$10B", "$50B", "$500B"], fontsize=6)
ax.set_xlabel("Valuation (USD)", labelpad=3)
ax.set_ylabel("Publications per firm", labelpad=3)
ax.set_xlim(600, 900000)
ax.set_ylim(0.65, 650)

r_a, _ = stats.spearmanr(df_a["valuation_usd_m"], df_a["n_papers_total"])
ax.text(0.97, 0.05, f"Spearman ρ = {r_a:.2f}",
        transform=ax.transAxes, fontsize=6, ha="right", va="bottom")

legend_a = [
    mpatches.Patch(facecolor="none", edgecolor=C_PUB, linewidth=0.8,
                   label="No output"),
    mpatches.Patch(facecolor=C_PUB, edgecolor="none",
                   label="Publishes"),
    mpatches.Patch(facecolor=C_HC, edgecolor="none",
                   label="Has highly-cited paper"),
]
ax.legend(handles=legend_a, fontsize=6, frameon=False, loc="upper left",
          handlelength=1.2)
clean_ax(ax)
add_panel_label(ax, "A")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B – Funding raised vs publication count (log-log)
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_b

x_b = df["amount_raised_usd_m"].values
y_b = df["n_papers_total"].values + 1   # +1 for log scale

no_out_b    = df["has_any_output_norm"] == "no"
pub_no_hc_b = (df["has_any_output_norm"] == "yes") & (df["has_highly_cited_norm"] == "no")
hc_b        = df["has_highly_cited_norm"] == "yes"

ax.scatter(x_b[no_out_b], y_b[no_out_b],
           s=10, facecolors="none", edgecolors=C_PUB, linewidths=0.5,
           alpha=0.55, zorder=2)
ax.scatter(x_b[pub_no_hc_b], y_b[pub_no_hc_b],
           s=12, facecolors=C_PUB, edgecolors="none", alpha=0.65, zorder=3)
ax.scatter(x_b[hc_b], y_b[hc_b],
           s=18, facecolors=C_HC, edgecolors="none", alpha=0.90, zorder=4)

to_label_b = {
    "OpenAI":       ("right", 1.30, 0.52),
    "MEGVII":       ("left",  0.60, 1.55),
    "Waymo":        ("right", 1.35, 1.60),
    "Hugging Face": ("right", 2.50, 2.20),
}
for name, (ha, dx, dy) in to_label_b.items():
    row = df[df["startup"] == name].iloc[0]
    xi = row["amount_raised_usd_m"]
    yi = row["n_papers_total"] + 1
    ax.annotate(name, xy=(xi, yi), xytext=(xi * dx, yi * dy),
                fontsize=5.5, ha=ha, va="center", color="black",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 2, 6, 11, 26, 101, MAX_PUBS + 1])
ax.set_yticklabels(["0", "1", "5", "10", "25", "100", str(MAX_PUBS)], fontsize=6)
ax.set_xticks([10, 100, 1000, 10000, 100000])
ax.set_xticklabels(["$10M", "$100M", "$1B", "$10B", "$100B"], fontsize=6)
ax.set_xlabel("Funding raised (USD)", labelpad=3)
ax.set_ylabel("Publications per firm", labelpad=3)

r_b, _ = stats.spearmanr(df["amount_raised_usd_m"], df["n_papers_total"])
ax.text(0.97, 0.05, f"Spearman ρ = {r_b:.2f}",
        transform=ax.transAxes, fontsize=6, ha="right", va="bottom")

legend_b = [
    mpatches.Patch(facecolor="none", edgecolor=C_PUB, linewidth=0.8,
                   label="No output"),
    mpatches.Patch(facecolor=C_PUB, edgecolor="none",
                   label="Publishes"),
    mpatches.Patch(facecolor=C_HC, edgecolor="none",
                   label="Has highly-cited paper"),
]
ax.legend(handles=legend_b, fontsize=6, frameon=False, loc="upper left",
          handlelength=1.2)
clean_ax(ax)
add_panel_label(ax, "B")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C – Startup lead/collaboration bars vs all AI publications (2016–2025)
#   Startup categories are mutually exclusive: lead if first/last/alphabetical;
#   collaborative-only if middle-author involvement without startup lead.
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_c

YEARS = list(range(2016, 2026))

all_ai = pd.read_excel(ALL_AI_CSV)
require_columns(all_ai, ["Publication Years", "Count"], os.path.basename(ALL_AI_CSV))
all_ai = all_ai.rename(columns={"Publication Years": "year", "Count": "all_ai"})
all_ai["year"] = coerce_nonnegative_int(all_ai["year"], "Publication Years")
all_ai["all_ai"] = coerce_nonnegative_int(all_ai["all_ai"], "Count")
all_ai = all_ai[all_ai["year"].isin(YEARS)][["year", "all_ai"]]

trend_rows = []
papers_startup["is_alphabetical"] = (
    papers_startup["alpha_order"].fillna("").astype(str).str.strip().str.upper()
    == "YES"
)
papers_startup["is_lead_role"] = (
    papers_startup["first_author"].notna() |
    papers_startup["last_author"].notna() |
    papers_startup["is_alphabetical"]
)
papers_startup["is_middle_role"] = papers_startup["middle_author"].notna()

for yr in YEARS:
    sub = papers_startup[papers_startup["paper_year"] == yr]
    trend_rows.append({
        "year": yr,
        "lead": int(sub.loc[sub["is_lead_role"], "UID"].nunique()),
        "collaborative_only": int(
            sub.loc[
                sub["is_middle_role"] & ~sub["is_lead_role"],
                "UID"
            ].nunique()
        ),
    })

yt = pd.DataFrame(trend_rows).merge(all_ai, on="year", how="left")
if yt[["lead", "collaborative_only", "all_ai"]].isna().any().any():
    raise ValueError("Panel C trend data contain missing values.")
baseline = yt.loc[yt["year"] == YEARS[0]].iloc[0]
for col in ["lead", "collaborative_only", "all_ai"]:
    if baseline[col] <= 0:
        raise ValueError(f"Panel C baseline for {col} must be positive.")
    yt[f"{col}_idx"] = yt[col] / baseline[col]

growth = {
    "lead": yt["lead"].iloc[-1] / yt["lead"].iloc[0],
    "collaborative_only": (
        yt["collaborative_only"].iloc[-1] / yt["collaborative_only"].iloc[0]
    ),
    "all_ai": yt["all_ai"].iloc[-1] / yt["all_ai"].iloc[0],
}

x = np.arange(len(YEARS))
bar_w = 0.34
startup_x = x - bar_w / 2
all_ai_x = x + bar_w / 2
ax2 = ax.twinx()

ax.bar(
    startup_x, yt["lead"], bar_w, color=C_PUB, zorder=3,
    label="Lead startup papers"
)
ax.bar(
    startup_x, yt["collaborative_only"], bar_w, bottom=yt["lead"],
    color=C_PPR, zorder=3,
    label="Collaborative-only startup papers"
)
ax2.bar(
    all_ai_x, yt["all_ai"], bar_w, color=C_ALL_AI, alpha=0.72, zorder=2,
    label="All AI publications"
)

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEARS], fontsize=6, rotation=45, ha="right")
ax.set_ylabel("Startup-involved papers", labelpad=3)
ax2.set_ylabel("All AI publications", labelpad=3)
ax2.yaxis.set_major_formatter(
    FuncFormatter(lambda v, _: f"{v/1_000_000:.1f}M" if v >= 1_000_000 else f"{v/1000:.0f}k")
)
ax.set_ylim(0, (yt["lead"] + yt["collaborative_only"]).max() * 1.18)
ax2.set_ylim(0, yt["all_ai"].max() * 1.18)
ax.set_xlim(-0.55, len(YEARS) - 0.45)

handles_1, labels_1 = ax.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(
    handles_1 + handles_2, labels_1 + labels_2,
    fontsize=5.5, frameon=False, loc="upper left", handlelength=1.2
)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(which="both", top=False, right=False)
ax2.spines[["top", "left"]].set_visible(False)
ax2.tick_params(which="both", top=False, left=False)
add_panel_label(ax, "C")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D – Median publications per firm by geographic region, stacked
#   Regional composition keeps the peer_nohc / pre_nohc / hc convention.
#   Medians used (not means) given highly skewed distributions
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_d

_europe = {"United Kingdom", "Germany", "France", "Sweden", "Ireland",
           "The Netherlands", "Finland", "Portugal", "Switzerland",
           "Belgium", "Norway", "Estonia"}

def _region(c):
    if c == "United States": return "United States"
    if c == "China":         return "China"
    if c in _europe:         return "Europe"
    return "Others"   # Japan folded in here

usd["_region"]    = usd["Country"].map(_region)
usd["_peer_nohc"] = (usd["n_papers_published"] - usd["n_highly_cited_published"]).clip(lower=0)
usd["_pre_nohc"]  = (usd["n_papers_preprint"]  - usd["n_highly_cited_preprint"]).clip(lower=0)
usd["_total"]     = usd["_peer_nohc"] + usd["_pre_nohc"] + usd["n_highly_cited_total"]

# Mean type proportions computed from publishing firms only
_pub = usd[usd["_total"] > 0].copy()
_pub["_prop_peer"] = _pub["_peer_nohc"] / _pub["_total"]
_pub["_prop_pre"]  = _pub["_pre_nohc"]  / _pub["_total"]
_pub["_prop_hc"]   = _pub["n_highly_cited_total"] / _pub["_total"]
_pub_props = _pub.groupby("_region").agg(
    mean_prop_peer=("_prop_peer", "mean"),
    mean_prop_pre =("_prop_pre",  "mean"),
    mean_prop_hc  =("_prop_hc",   "mean"),
)

by_r = (
    usd.groupby("_region")
    .agg(
        n            =("startup",            "count"),
        n_pub        =("has_any_output_norm", lambda x: (x == "yes").sum()),
        median_total =("n_papers_total",      "median"),
    )
    .assign(pct_pub=lambda x: (x["n_pub"] / x["n"] * 100).round(0).astype(int))
    .join(_pub_props)
    .sort_values("median_total", ascending=True)
    .reset_index()
)

# Scale segments so they sum exactly to median_total
by_r["seg_peer"] = by_r["median_total"] * by_r["mean_prop_peer"]
by_r["seg_pre"]  = by_r["median_total"] * by_r["mean_prop_pre"]
by_r["seg_hc"]   = by_r["median_total"] * by_r["mean_prop_hc"]

y  = np.arange(len(by_r))
bw = 0.62

ax.barh(y, by_r["seg_peer"], bw,
        left=0,
        color=C_PUB, edgecolor="none", zorder=3, label="Peer-reviewed (non-HC)")
ax.barh(y, by_r["seg_pre"],  bw,
        left=by_r["seg_peer"],
        color=C_PPR, edgecolor="none", zorder=3, label="Preprint (non-HC)")
ax.barh(y, by_r["seg_hc"],   bw,
        left=by_r["seg_peer"] + by_r["seg_pre"],
        color=C_HC, edgecolor="none", zorder=3, label="Highly-cited (any type)")

# Two-line annotation: total n on first line, publishers + % on second line
for i, (_, row) in enumerate(by_r.iterrows()):
    label = f"n = {int(row['n'])}\n{int(row['n_pub'])} pub. ({int(row['pct_pub'])}%)"
    ax.text(row["median_total"] * 1.02 + 0.05, i, label,
            va="center", ha="left", fontsize=5, color="black", linespacing=1.4)

ax.set_yticks(y)
ax.set_yticklabels(by_r["_region"], fontsize=6.5)
ax.set_xlabel("Publications per firm (median total)", labelpad=3)
ax.set_xlim(0, by_r["median_total"].max() * 1.70)

handles_d = [
    mpatches.Patch(facecolor=C_PUB, label="Peer-reviewed (non-HC)"),
    mpatches.Patch(facecolor=C_PPR, label="Preprint (non-HC)"),
    mpatches.Patch(facecolor=C_HC,  label="Highly-cited (any type)"),
]
ax.legend(handles=handles_d, fontsize=5.5, frameon=False,
          loc="lower right", handlelength=1.2)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(which="both", top=False, right=False)
add_panel_label(ax, "D")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT_PATH}")

# ── Summary for caption drafting ──────────────────────────────────────────────
print(f"\nPanel A — Spearman ρ (valuation vs publications, N={len(df_a)}): {r_a:.3f}")
print(f"Panel B — Spearman ρ (amount raised vs publications, n={len(df)}): {r_b:.3f}")
print("\nPanel C — 2016→2025 growth coefficients:")
print(f"Lead startup papers: {growth['lead']:.1f}×")
print(f"Collaborative-only startup papers: {growth['collaborative_only']:.1f}×")
print(f"All AI publications: {growth['all_ai']:.1f}×")
print(yt[["year", "lead", "collaborative_only", "all_ai"]].to_string(index=False))
print(f"\nPanel D — median total publications per firm by region:")
print(by_r[["_region", "n", "n_pub", "median_total", "seg_peer", "seg_pre", "seg_hc"]]
      .sort_values("median_total", ascending=False).round(2).to_string(index=False))
