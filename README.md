# AI Unicorn Startups — Bibliometric Analysis: Reproduction Guide

Code for the paper's bibliometric analysis of scientific publishing among AI unicorn
startups. Three stages, run in order:

```
01_download_papers (raw API retrieval)  
→ 02_data_processing (merge → clean → filter → identify → aggregate) 
→ 03_analysis (firm-level & author-level statistics, figures)
```

---

## 1. Setup

### 1.1 Data folder

Every script reads and writes its data files in **one folder**. Point at it with:

```bash
export AI_UNICORN_DATA_DIR="/path/to/your/data/folder"
```

If unset, each script defaults to the current working directory (i.e. `cd` into your data
folder and run the scripts by their repository path from there). All files live flat inside
that folder — no subfolders.

### 1.2 API key

`01_download_papers/*.py` need a Clarivate Web of Science Expanded API key:

```bash
export WOS_API_KEY="your-key-here"
```

See `01_download_papers/.env.example`.

### 1.3 Files you need to supply

These reference/config files are not included in this repository. Place them directly inside
`AI_UNICORN_DATA_DIR` before running Stage 2 (available on request / via the OSF repository
referenced in the manuscript):

| File | Used by | Purpose |
|---|---|---|
| `startup_affiliation_dictionary.csv` | `05_clean_papers.py`, `07_identify_authorship.py`, `11_aggregate_authors.py` | Institution-variant → canonical-startup mapping |
| `aliases.md` | `07_identify_authorship.py` | Canonical-name / alias crosswalk (mergers, rebrands) |
| `startups_data.csv` (seed version) | `05_clean_papers.py` and all of Stage 3 | Firm metadata: name, country, founding year, valuation, funding |
| `all_ai_publication.xlsx` | `13_characteristics_analysis.py`, `15_figure_correlates.py` | External annual AI-publication-count baseline for trend comparisons |
| `codebook.csv` | — | Variable-name documentation for the output CSVs |

If you already have `wos_papers.csv`, `inspec_papers.csv`, and `preprints_papers.csv` (or
`merged_papers.csv` / `treated_data.csv`), you can drop Stage 1 (or the early part of Stage 2)
and start from whichever file you have.

---

## 2. Scripts

### Stage 1 — `01_download_papers/`

Queries the Web of Science Expanded API (`AD="{startup name}" AND PY=(1998-2025)`) for each
of ~317 AI unicorn startups plus 17 historical name variants.

| Script | Writes | Database |
|---|---|---|
| `01_download_wos.py` | `wos_papers.csv` | Web of Science Core Collection |
| `02_download_inspec.py` | `inspec_papers.csv` | INSPEC |
| `03_download_preprints.py` | `preprints_papers.csv` | Preprint Citation Index |

### Stage 2 — `02_data_processing/`

| Script | Reads | Writes | What it does |
|---|---|---|---|
| `04_merge_exports.py` | the 3 Stage-1 CSVs | `merged_papers.csv` | Concatenates the three exports |
| `05_clean_papers.py` | `merged_papers.csv`, `startups_data.csv`, `startup_affiliation_dictionary.csv` | `treated_data.csv` | Applies 5 sequential filters (keyword, doc type, date, dedup, affiliation match), each as an IN/EX column |
| `06_filter_papers.py` | `treated_data.csv` | `included_papers.csv` | Keeps rows where all 5 filters = `IN`; drops helper columns |
| `07_identify_authorship.py` | `included_papers.csv` | `included_papers.csv` (in place) | Adds alphabetical-order, first/middle/last-author-startup, lead-startup, and canonical-name columns |
| `08_aggregate_startups.py` | `included_papers.csv` | `startups_data.csv`, updates `included_papers.csv`'s `is_highly_cited` | Aggregates paper/citation counts per startup (published/preprint × lead/middle), highly-cited threshold ≥200 citations |
| `09_apply_edge_cases.py` | `included_papers.csv`, `startups_data.csv` | Both, in place | Applies 5 manual edge-case decisions (see Section 3) |
| `10_update_startups.py` | `included_papers.csv`, `startups_data.csv` | `startups_data.csv`, in place | Propagates the edge-case decisions into firm-level aggregates |
| `11_aggregate_authors.py` | `included_papers.csv`, `startup_affiliation_dictionary.csv` | `authors_data.csv` | One row per startup-affiliated author, with affiliation-type and position counts |

### Stage 3 — `03_analysis/`

Read-only: none of these modify `AI_UNICORN_DATA_DIR` inputs except to write their own output.

| Script | Produces | Content |
|---|---|---|
| `12_concentration_analysis.py` | Console report | Firm- and author-level distribution/concentration (Gini, Lorenz, top-share) of publications and citations; top contributors |
| `13_characteristics_analysis.py` | Console report | Country, valuation tier, funding, firm age, region, temporal trends vs. all-AI baseline, preprint share, participation typology — Kruskal–Wallis, Mann–Whitney, Spearman, χ² |
| `14_figure_concentration.py` | `figure1.png` | Publications-per-firm distribution; Lorenz curves (firm & author); top-10 firms by citations |
| `15_figure_correlates.py` | `figure2.png` | Valuation/funding vs. publications; startup-led vs. collaborative trend vs. all-AI publications; median publications by region |
| `16_authorship_counts.py` | Console report | Sanity-check counts of leadership position, alphabetical order, and EVEN cases |

---

## 3. Edge cases applied in Stage 2

`09_apply_edge_cases.py` resolves five situations `07_identify_authorship.py` can't handle
automatically:

| Case | Decision |
|---|---|
| Alpha-mid (alphabetical paper, startup author in a middle position only) | Included — alphabetical order confers no positional seniority |
| EVEN (two+ startups tied on affiliated-author count) | Shared attribution: each tied startup gets +1 paper/citations in firm-level counts; counted once corpus-wide |
| `canonical_name` empty (no startup identifiable) | Excluded |
| "Plus" / "Plus.ai" name mismatch | `startups_data.csv` row renamed to match papers' `canonical_name` |
| Garbled rows (corrupted CSV fields) | Excluded, unrecoverable |

Startup aliases, mergers, and rebrands are resolved via `aliases.md` in
`07_identify_authorship.py` before any of the above.

---

## 4. How to run

```bash
export AI_UNICORN_DATA_DIR="/path/to/your/data/folder"
export WOS_API_KEY="your-key-here"

# Stage 1 — raw retrieval
python 01_download_papers/01_download_wos.py
python 01_download_papers/02_download_inspec.py
python 01_download_papers/03_download_preprints.py

# Stage 2 — merge → clean → filter → identify → aggregate → edge cases
python 02_data_processing/04_merge_exports.py
python 02_data_processing/05_clean_papers.py
python 02_data_processing/06_filter_papers.py
python 02_data_processing/07_identify_authorship.py
python 02_data_processing/08_aggregate_startups.py
python 02_data_processing/09_apply_edge_cases.py
python 02_data_processing/10_update_startups.py
python 02_data_processing/11_aggregate_authors.py

# Stage 3 — analysis and figures
python 03_analysis/12_concentration_analysis.py
python 03_analysis/13_characteristics_analysis.py
python 03_analysis/14_figure_concentration.py
python 03_analysis/15_figure_correlates.py
python 03_analysis/16_authorship_counts.py
```

Place the files listed in **Section 1.3** inside `AI_UNICORN_DATA_DIR` before running Stage 2.

---

## 5. Software environment

| Package | Version |
|---|---|
| Python | 3.9.6 |
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| scipy | 1.13.1 |
| matplotlib | 3.9.4 |
| requests | 2.32.5 |
| openpyxl | 3.1.5 |
