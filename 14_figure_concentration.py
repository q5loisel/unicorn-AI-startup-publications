"""
Figure 1 – Publication participation and concentration
=======================================================
A – Distribution of publications per firm (HC / non-HC split)
B – Lorenz concentration curve by firm (publications and citations)
C – Top 10 firms by cumulated citations (peer-reviewed / preprint split)
D – Lorenz concentration curve by author-publication observations

Scope: same as 12_concentration_analysis.py — lead papers (first/last startup author) +
       alpha-mid papers (alphabetical order) + EVEN papers (tied attribution).
Panel D uses unique author-publication observations: lead authors,
alphabetical startup authors, and tied-startup authors on EVEN/shared papers.

Rules applied (figures_rules):
  - No figure title or panel sub-titles (all text goes in caption)
  - No grid lines, no minor tick marks
  - No grayscale: no-output category shown with hatching on white
  - Distinct hues: published / non-HC = blue (#2C6E9B), HC / preprint = amber (#E8A830)
  - Helvetica / sans-serif, 10 pt bold part labels in upper-left corners
  - Minimal in-figure text; details in caption
  - Figure sized at ~7 in wide (print-ready)
"""

import os
import re
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "figure1-matplotlib")
)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths and constants ───────────────────────────────────────────────────────
# Inputs are read, and figure1.png is written, into a single, user-defined
# folder — set the AI_UNICORN_DATA_DIR environment variable to point at it
# (it defaults to the current working directory, i.e. run this script from
# inside that folder).
DATA_DIR     = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())
STARTUPS_CSV = os.path.join(DATA_DIR, "startups_data.csv")
PAPERS_CSV   = os.path.join(DATA_DIR, "included_papers.csv")
AUTHORS_CSV  = os.path.join(DATA_DIR, "authors_data.csv")
OUT_PATH     = os.path.join(DATA_DIR, "figure1.png")

HC_CITATION_THRESHOLD = 200

# ── Helpers ───────────────────────────────────────────────────────────────────
def require_columns(data, columns, source):
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")


def lorenz_gini(values):
    """Full Lorenz curve (one point per observation) and Gini coefficient."""
    vals  = np.sort(np.asarray(values, dtype=float))
    cum   = np.cumsum(vals)
    total = cum[-1]
    if total == 0:
        xfrac = np.linspace(0.0, 1.0, len(vals) + 1)
        return xfrac * 100, np.zeros(len(vals) + 1), 0.0
    lorenz = np.concatenate([[0.0], cum / total])
    xfrac  = np.linspace(0.0, 1.0, len(vals) + 1)
    gini   = float((len(vals) + 1 - 2.0 * cum.sum() / total) / len(vals))
    return xfrac * 100, lorenz * 100, gini


def _parse_author_names(text):
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


def _split_pipe(text):
    if not text or pd.isna(text):
        return set()
    return {p.strip() for p in str(text).split(" | ") if p.strip()}


def add_panel_label(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def clean_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)


# ── Load data ─────────────────────────────────────────────────────────────────
df      = pd.read_csv(STARTUPS_CSV)
papers  = pd.read_csv(PAPERS_CSV, low_memory=False)
authors = pd.read_csv(AUTHORS_CSV)

N = len(df)

require_columns(df,     ["startup"],                          os.path.basename(STARTUPS_CSV))
require_columns(papers, ["UID", "canonical_name",
                          "Total Times Cited", "Document Type",
                          "first_author", "last_author",
                          "alpha_order", "is_highly_cited",
                          "shared_startups", "edge_case_flag"], os.path.basename(PAPERS_CSV))
require_columns(authors, ["author_name", "startup", "affil_type",
                          "n_alpha"],                          os.path.basename(AUTHORS_CSV))

if df["startup"].duplicated().any():
    raise ValueError("Duplicate startup names in firm file.")

# ── Paper-level preprocessing ─────────────────────────────────────────────────
papers["cit"]            = pd.to_numeric(papers["Total Times Cited"], errors="coerce").fillna(0).astype(int)
papers["is_preprint"]    = papers["Document Type"].str.strip().str.lower() == "preprint"
papers["hc_200"]         = papers["cit"] >= HC_CITATION_THRESHOLD
papers["has_leadership"] = papers["first_author"].notna() | papers["last_author"].notna()
papers["is_alphabetical"]= papers["alpha_order"].str.strip().str.upper() == "YES"
papers["edge_case_flag"] = papers["edge_case_flag"].fillna("").astype(str)
papers["shared_startups"]= papers["shared_startups"].fillna("").astype(str)

# ── Paper scope (mirrors 12_concentration_analysis.py) ────────────────────────────────────────
startup_names = set(df["startup"])

papers = papers[
    (
        papers["canonical_name"].isin(startup_names) &
        (papers["has_leadership"] | (papers["edge_case_flag"] == "alpha_mid"))
    ) |
    (papers["edge_case_flag"] == "even_shared")
].copy()

# ── Expand EVEN papers for per-startup attribution ────────────────────────────
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
papers_for_firms = pd.concat([_normal, _even_exp], ignore_index=True)

# ── Recompute per-startup stats ───────────────────────────────────────────────
_pf = papers_for_firms[papers_for_firms["canonical_name"].isin(startup_names)]
per_startup = (
    _pf.groupby("canonical_name")
    .agg(
        n_papers_total      =("UID",         "size"),
        n_papers_published  =("is_preprint", lambda s: int((~s).sum())),
        n_papers_preprint   =("is_preprint", "sum"),
        n_highly_cited_total=("hc_200",      "sum"),
        total_citations     =("cit",         "sum"),
    )
    .fillna(0).astype(int)
    .reset_index()
    .rename(columns={"canonical_name": "startup"})
)

count_cols = [
    "n_papers_total", "n_papers_published", "n_papers_preprint",
    "n_highly_cited_total", "total_citations",
]
df = df.drop(columns=[c for c in count_cols if c in df.columns], errors="ignore")
df = df.merge(per_startup, on="startup", how="left")
df[count_cols] = df[count_cols].fillna(0).astype(int)

# ── Precompute author-publication Lorenz data for Panel D ─────────────────────
# Mirrors 12_concentration_analysis.py Section 2.C:
#   - non-alpha first/last startup authors
#   - all startup-affiliated authors on alphabetical papers
#   - all tied-startup authors on EVEN/shared papers
author_meta = authors.set_index("author_name")[["startup", "affil_type"]]
alpha_author_names = set(authors.loc[authors["n_alpha"] > 0, "author_name"])
obs_rows = []

for _, row in papers.iterrows():
    uid = str(row["UID"])
    names = _parse_author_names(row.get("Authors & Affiliations"))
    if not names:
        continue

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
        first_name = names[0]
        last_name = names[-1]
        has_first = pd.notna(row["first_author"]) and str(row["first_author"]).strip()
        has_last  = pd.notna(row["last_author"])  and str(row["last_author"]).strip()
        if has_first and first_name in author_meta.index:
            selected.add(first_name)
        if has_last and last_name in author_meta.index:
            selected.add(last_name)

    for name in selected:
        obs_rows.append({
            "UID": uid,
            "author_name": name,
            "hc_200": bool(row["hc_200"]),
        })

author_pub_obs = (
    pd.DataFrame(obs_rows).drop_duplicates(["UID", "author_name"])
    if obs_rows else
    pd.DataFrame(columns=["UID", "author_name", "hc_200"])
)
all_ppa = (
    author_pub_obs.groupby("author_name")["UID"].size().astype(int)
)
hc_ppa = (
    author_pub_obs[author_pub_obs["hc_200"]]
    .groupby("author_name")["UID"].size()
    .reindex(all_ppa.index, fill_value=0)
    .astype(int)
)

# ── Color palette ─────────────────────────────────────────────────────────────
C_PUB  = "#2C6E9B"   # peer-reviewed / non-HC / all-authors  (blue)
C_PRE  = "#E8A830"   # preprint / HC                         (amber)
C_NONE = "white"
HATCH  = "///"

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":           7,
    "axes.linewidth":      0.6,
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.minor.size":    0,
    "ytick.minor.size":    0,
    "legend.handlelength": 1.2,
    "legend.handleheight": 0.8,
})

fig = plt.figure(figsize=(7.0, 5.5))
gs  = GridSpec(2, 2, figure=fig,
               hspace=0.52, wspace=0.40,
               left=0.10, right=0.98, top=0.97, bottom=0.11)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A – Distribution of publications per firm (HC / non-HC)
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_a

bins_edges  = [0, 1, 3, 6, 11, 16, 21, 26, 31, float("inf")]
bins_labels = ["0", "1–2", "3–5", "6–10", "11–15", "16–20", "21–25", "26–30", "31+"]

df["_type"] = "zero"
df.loc[(df["n_papers_total"] > 0) & (df["n_highly_cited_total"] == 0), "_type"] = "no_hc"
df.loc[df["n_highly_cited_total"] > 0,                                  "_type"] = "hc"
df["_bin"] = pd.cut(df["n_papers_total"], bins=bins_edges, right=False, labels=bins_labels)

cnt_zero  = (df[df["_type"] == "zero"]
             .groupby("_bin", observed=True).size()
             .reindex(bins_labels, fill_value=0))
cnt_no_hc = (df[df["_type"] == "no_hc"]
             .groupby("_bin", observed=True).size()
             .reindex(bins_labels, fill_value=0))
cnt_hc    = (df[df["_type"] == "hc"]
             .groupby("_bin", observed=True).size()
             .reindex(bins_labels, fill_value=0))
totals = cnt_zero + cnt_no_hc + cnt_hc

x  = np.arange(len(bins_labels))
bw = 0.65
ax.bar(x, cnt_zero,  bw,
       facecolor=C_NONE, edgecolor="black", linewidth=0.5, hatch=HATCH, zorder=3)
ax.bar(x, cnt_no_hc, bw, bottom=cnt_zero,
       facecolor=C_PUB, edgecolor="none", zorder=3)
ax.bar(x, cnt_hc,    bw, bottom=cnt_zero + cnt_no_hc,
       facecolor=C_PRE, edgecolor="none", zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(bins_labels, fontsize=6.5)
ax.set_xlabel("Publications per firm", labelpad=3)
ax.set_ylabel("Number of firms")
ax.set_xlim(-0.55, len(bins_labels) - 0.45)
ax.set_ylim(0, totals.max() * 1.30)
clean_ax(ax)

for xi, tot in enumerate(totals):
    pct = tot / N * 100
    ax.text(xi, tot + totals.max() * 0.015,
            f"n={int(tot)}\n{pct:.1f}%",
            ha="center", va="bottom", fontsize=5.5, color="black")

ax.legend(handles=[
    mpatches.Patch(facecolor=C_NONE, edgecolor="black",
                   linewidth=0.5, hatch=HATCH, label="No output"),
    mpatches.Patch(facecolor=C_PUB, edgecolor="none",
                   label="No highly-cited paper"),
    mpatches.Patch(facecolor=C_PRE, edgecolor="none",
                   label="≥1 highly-cited paper"),
], fontsize=6, frameon=False, loc="upper right")
add_panel_label(ax, "A")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B – Lorenz concentration curve by firm
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_b

xp, yp, gini_pap = lorenz_gini(df["n_papers_total"].values)
xc, yc, gini_cit = lorenz_gini(df["total_citations"].values)

ax.plot([0, 100], [0, 100], "--", color="black", linewidth=0.8,
        label="Equality", zorder=2)
ax.plot(xp, yp, color=C_PUB, linewidth=1.5,
        label=f"Publications (Gini = {gini_pap:.2f})", zorder=4)
ax.plot(xc, yc, color=C_PRE, linewidth=1.5,
        label=f"Citations (Gini = {gini_cit:.2f})", zorder=4)

ax.set_xlabel("Cumulative % of firms (ranked by measure)", labelpad=3)
ax.set_ylabel("Cumulative % of output")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
clean_ax(ax)
ax.legend(fontsize=6, frameon=False, loc="upper left")
add_panel_label(ax, "B")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C – Top 10 firms by cumulated citations (peer-reviewed / preprint)
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_c

cit_split = (
    papers_for_firms[papers_for_firms["canonical_name"].isin(startup_names)]
    .groupby(["canonical_name", "is_preprint"])["cit"]
    .sum()
    .unstack(fill_value=0)
)
for col in (False, True):
    if col not in cit_split.columns:
        cit_split[col] = 0
cit_split = (cit_split
             .rename(columns={False: "cit_pub", True: "cit_pre"})
             .reset_index()
             .rename(columns={"canonical_name": "startup"}))

top10 = (
    df.nlargest(10, "total_citations")[["startup", "total_citations"]]
    .merge(cit_split, on="startup", how="left")
    .fillna(0)
    .sort_values("total_citations", ascending=True)
    .reset_index(drop=True)
)

y  = np.arange(len(top10))
bh = 0.65
ax.barh(y, top10["cit_pub"], bh,
        facecolor=C_PUB, edgecolor="none", zorder=3, label="Peer-reviewed")
ax.barh(y, top10["cit_pre"], bh, left=top10["cit_pub"],
        facecolor=C_PRE, edgecolor="none", zorder=3, label="Preprint")

ax.set_yticks(y)
ax.set_yticklabels(top10["startup"], fontsize=6.5)
ax.set_xlabel("Cumulated citations", labelpad=3)
ax.set_xlim(0, top10["total_citations"].max() * 1.04)
ax.xaxis.set_major_formatter(plt.FuncFormatter(
    lambda v, _: f"{int(v/1000)}k" if v >= 1000 else f"{int(v)}"
))
clean_ax(ax)
ax.legend(fontsize=6, frameon=False, loc="lower right")
add_panel_label(ax, "C")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D – Lorenz concentration by author
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_d

xa, ya, gini_all = lorenz_gini(all_ppa.values)
xh, yh, gini_hc  = lorenz_gini(hc_ppa.values)

ax.plot([0, 100], [0, 100], "--", color="black", linewidth=0.8,
        label="Equality", zorder=2)
ax.plot(xa, ya, color=C_PUB, linewidth=1.5,
        label=f"All publications (Gini = {gini_all:.2f})", zorder=4)
ax.plot(xh, yh, color=C_PRE, linewidth=1.5,
        label=f"Highly-cited publications (Gini = {gini_hc:.2f})",  zorder=4)

ax.set_xlabel("Cumulative % of authors (ranked by measure)", labelpad=3)
ax.set_ylabel("Cumulative % of output")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
clean_ax(ax)
ax.legend(fontsize=6, frameon=False, loc="upper left")
add_panel_label(ax, "D")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT_PATH}")

# ── Summary for caption drafting ──────────────────────────────────────────────
n_zero = int((df["n_papers_total"] == 0).sum())
print(f"\nN firms total: {N}  |  No output: {n_zero} ({n_zero/N*100:.1f}%)")
print(f"Panel B — Gini: publications = {gini_pap:.2f},  citations = {gini_cit:.2f}")
print(f"Panel D — Gini: all author-pubs = {gini_all:.2f},  HC author-pubs = {gini_hc:.2f}")
print(f"Panel D — Authors (all / HC): {len(all_ppa):,} / {len(hc_ppa):,}")
print(f"Panel D — Author-publication obs. (all / HC): {int(all_ppa.sum()):,} / {int(hc_ppa.sum()):,}")
