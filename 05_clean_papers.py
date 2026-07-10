"""
Round 1 data cleaning for merged_papers.csv.

Five filters are applied. Each adds one column (IN / EX):
  filter_keywords  : title or abstract contains at least one AI keyword
  filter_doctype   : peer-reviewed or preprint; retracted publications excluded
  filter_date      : year ≤ 2025 AND year ≥ startup founding year
  filter_dedup     : duplicates removed (priority WOS > INSPEC > PPRN)
                     + duplicate_of and duplicate_reason helper columns
  filter_affil     : Authors & Affiliations contains a strict match against
                     the startup affiliation dictionary
                     + affil_matched_startup helper column

Output: treated_data.csv

All input/output files are read from and written to a single, user-defined
folder — set the AI_UNICORN_DATA_DIR environment variable to point at it (it
defaults to the current working directory, i.e. run this script from inside
that folder).
"""

import csv
import os
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

csv.field_size_limit(sys.maxsize)

DATA_DIR = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())

FILE_IN       = os.path.join(DATA_DIR, "merged_papers.csv")
FILE_OUT      = os.path.join(DATA_DIR, "treated_data.csv")
FILE_STARTUPS = os.path.join(DATA_DIR, "startups_data.csv")
FILE_AFFIL    = os.path.join(DATA_DIR, "startup_affiliation_dictionary.csv")

MAX_YEAR       = 2025
FUZZY_THRESH   = 0.90   # title AND abstract must both reach this to be a duplicate
TITLE_ONLY_MIN = 0.95   # threshold when one side has no abstract
WINDOW_SIZE    = 40     # sliding-window neighbours for fuzzy step

PRIORITY = {"WOS": 0, "INSPEC": 1, "PPRN": 2}

# Strip platform tags like [arXiv], [ICLR 2024] before title comparison
TAG_RE = re.compile(r"\[.*?\]|\(arxiv[^)]*\)", re.IGNORECASE)


# =============================================================================
# Filter 1 – Keywords
# =============================================================================

# Trailing * in the original spec becomes \w* (zero or more word chars).
# Grouped by theme for readability.
_KW_PATTERNS = [
    # --- Original list ---
    r"artificial\s+intelligence",
    r"machine\s+learning",
    r"deep\s+learning",
    r"neural\s+network\w*",
    r"transformer\w*",
    r"large\s+language\s+model\w*",
    r"foundation\s+model\w*",
    r"natural\s+language\s+processing",
    r"computer\s+vision",
    r"reinforcement\s+learning",
    r"expert\s+system\w*",

    # --- Architectures ---
    r"generative\s+adversarial\s+network\w*",   # GAN family
    r"\bgan\b",                                   # GAN abbreviation
    r"convolutional\s+neural",                    # CNN
    r"recurrent\s+neural",                        # RNN
    r"\blstm\b",                                  # Long Short-Term Memory
    r"\bgru\b",                                   # Gated Recurrent Unit
    r"graph\s+neural\s+network\w*",               # GNN
    r"graph\s+convolutional",
    r"vision\s+transformer\w*",                   # ViT
    r"diffusion\s+model\w*",
    r"autoencoder\w*",
    r"variational\s+autoencoder\w*",

    # --- Language models & NLP ---
    r"language\s+model\w*",                       # broader than "large language model"
    r"\bllm\b",                                   # LLM abbreviation
    r"\bnlp\b",                                   # NLP abbreviation
    r"\bbert\b",                                  # BERT / RoBERTa / etc.
    r"\bgpt\b",                                   # GPT family
    r"pre-?trained\s+model\w*",
    r"fine-?tun\w*",                              # fine-tune, fine-tuning, fine-tuned
    r"transfer\s+learning",
    r"self-supervised\s+learning",
    r"contrastive\s+learning",
    r"few-?shot\s+learn\w*",
    r"zero-?shot\s+learn\w*",
    r"knowledge\s+distillation",
    r"federated\s+learning",
    r"prompt\s+learning",
    r"in-?context\s+learning",

    # --- NLP tasks ---
    r"machine\s+translation",
    r"question\s+answer\w*",
    r"sentiment\s+analysis",
    r"named\s+entity\s+recognition",
    r"text\s+classification",
    r"text\s+generation",
    r"text\s+summariz\w*",
    r"information\s+extraction",
    r"relation\s+extraction",
    r"dialogue\s+system\w*",
    r"conversational\s+ai",
    r"\bchatbot\w*",
    r"speech\s+recognition",
    r"speech\s+synthesis",
    r"text-?to-?speech",

    # --- Vision tasks ---
    r"object\s+detection",
    r"image\s+classification",
    r"image\s+recognition",
    r"image\s+segmentation",
    r"semantic\s+segmentation",
    r"instance\s+segmentation",
    r"image\s+generation",
    r"image\s+synthesis",
    r"face\s+recognition",
    r"face\s+detection",
    r"pose\s+estimation",
    r"action\s+recognition",
    r"image\s+captioning",
    r"visual\s+question\s+answer\w*",
    r"optical\s+character\s+recognition",
    r"\bocr\b",
    r"point\s+cloud",

    # --- Multimodal & recent paradigms ---
    r"multimodal\w*",
    r"vision.language\s+model\w*",
    r"generative\s+ai",
    r"stable\s+diffusion",

    # --- Applied AI domains ---
    r"autonomous\s+driving",
    r"autonomous\s+vehicle\w*",
    r"recommendation\s+system\w*",
    r"recommender\s+system\w*",
    r"knowledge\s+graph\w*",
    r"anomaly\s+detection",
    r"intelligent\s+system\w*",
]
KEYWORD_RE = re.compile("|".join(_KW_PATTERNS), re.IGNORECASE)


def passes_keywords(row) -> bool:
    title    = str(row.get("Title",    "") or "")
    abstract = str(row.get("Abstract", "") or "")
    return bool(KEYWORD_RE.search(title + " " + abstract))


# =============================================================================
# Filter 2 – Document type
# =============================================================================

ACCEPTED_DOC_TYPES = {
    "article",
    "preprint",
    "proceedings paper",
    "conference proceedings",
    "review",
    "journal paper",
    "conference paper",
}

RETRACTION_MARKERS = {
    "retracted publication",
    "retracted paper",
    "retraction",
}


def passes_doctype(raw) -> bool:
    if pd.isna(raw) or not str(raw).strip():
        return False
    parts = [p.strip().lower() for p in str(raw).split(";")]
    if any(p in RETRACTION_MARKERS for p in parts):
        return False
    return any(p in ACCEPTED_DOC_TYPES for p in parts)


# =============================================================================
# Filter 3 – Dates
# =============================================================================

def load_founding_years(filepath: str) -> dict:
    df = pd.read_csv(filepath)
    founding = {}
    for _, row in df.iterrows():
        name = str(row["startup"]).strip()
        raw  = str(row["Year Founded"]).replace("\t", "").strip()
        try:
            founding[name] = int(raw)
        except ValueError:
            founding[name] = None   # e.g. "Ñ" → unknown
    return founding


def passes_date(row, founding: dict):
    """Returns (bool, reason_string)."""
    try:
        paper_year = int(str(row["Year"]).strip())
    except (ValueError, TypeError):
        return True, ""

    if paper_year > MAX_YEAR:
        return False, f"year {paper_year} > {MAX_YEAR}"

    ms_raw = str(row.get("Matched Startups", "")).strip()
    if not ms_raw or ms_raw.lower() == "nan":
        return True, ""

    founding_years = [
        founding[n.strip()]
        for n in ms_raw.split(" | ")
        if founding.get(n.strip()) is not None
    ]
    if founding_years and paper_year < min(founding_years):
        return False, f"paper {paper_year} < startup founded {min(founding_years)}"

    return True, ""


# =============================================================================
# Filter 4 – Deduplication helpers
# =============================================================================

def normalize(text, strip_tags: bool = False) -> str:
    if pd.isna(text) or str(text).strip().lower() in ("nan", "none", ""):
        return ""
    t = str(text).lower()
    if strip_tags:
        t = TAG_RE.sub(" ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+",     " ", t).strip()
    return t


def remove_leading_article(text: str) -> str:
    for art in ("the ", "a ", "an "):
        if text.startswith(art):
            return text[len(art):]
    return text


def sim(a: str, b: str, threshold: float = FUZZY_THRESH) -> float:
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False)
    if m.quick_ratio() < threshold:
        return 0.0
    return m.ratio()


# =============================================================================
# Filter 5 – Affiliation match
# =============================================================================

def build_affil_index(filepath: str) -> dict:
    """
    Returns variant_lower → startup_name dict built from the dictionary CSV.
    """
    aff_df = pd.read_csv(filepath)
    variant_to_startup: dict = {}
    for _, row in aff_df.iterrows():
        variant = str(row["affiliation_variant"]).strip().lower()
        startup  = str(row["startup"]).strip()
        variant_to_startup[variant] = startup
    return variant_to_startup


_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def _extract_institution_names(addresses_text, affiliations_text) -> set:
    """
    Extract the institution-name token from every address segment.

    In WoS address strings, each segment has the form:
        InstitutionName, Department, City, Country
    so the institution identifier is the text *before the first comma*.

    Two sources are parsed:
      • All Document Addresses  – pipe-separated address segments
      • Authors & Affiliations  – affiliation text inside [...] brackets,
                                  which may themselves be pipe-separated
    """
    names: set = set()

    def _add_from_segment(segment: str):
        seg = segment.strip()
        if seg:
            inst = seg.split(",")[0].strip().lower()
            if inst:
                names.add(inst)

    # Source 1 – All Document Addresses
    addr = str(addresses_text or "")
    if addr and addr.lower() not in ("nan", "none", ""):
        for seg in addr.split(" | "):
            _add_from_segment(seg)

    # Source 2 – Authors & Affiliations (content inside [...])
    affil = str(affiliations_text or "")
    if affil and affil.lower() not in ("nan", "none", ""):
        for bracket_content in _BRACKET_RE.findall(affil):
            for seg in bracket_content.split(" | "):
                _add_from_segment(seg)

    return names


def check_affil(row, variant_to_startup: dict):
    """
    Returns (bool, matched_startup_name).

    Strategy: extract institution names (text before first comma in each
    address segment) then look them up **exactly** in the variant dictionary.
    Exact matching prevents generic names like 'Harvey', 'Ada', or 'Turing'
    from firing on 'William Harvey Res Inst', 'Ada Lovelace Inst', etc.
    """
    inst_names = _extract_institution_names(
        row.get("All Document Addresses",  ""),
        row.get("Authors & Affiliations",  ""),
    )
    for inst in inst_names:
        startup = variant_to_startup.get(inst)
        if startup:
            return True, startup
    return False, ""


# =============================================================================
# Main
# =============================================================================

t0 = time.time()
print("Loading data …")
df = pd.read_csv(FILE_IN, low_memory=False)
print(f"  {len(df):,} records  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Filter 1 – Keywords
# ---------------------------------------------------------------------------
print("\nFilter 1: Keywords …")
df["filter_keywords"] = df.apply(
    lambda r: "IN" if passes_keywords(r) else "EX", axis=1
)
kw_in = (df["filter_keywords"] == "IN").sum()
kw_ex = (df["filter_keywords"] == "EX").sum()
print(f"  IN: {kw_in:,}  |  EX: {kw_ex:,}")


# ---------------------------------------------------------------------------
# Filter 2 – Document type
# ---------------------------------------------------------------------------
print("\nFilter 2: Document type …")
df["filter_doctype"] = df["Document Type"].apply(
    lambda dt: "IN" if passes_doctype(dt) else "EX"
)
dt_in = (df["filter_doctype"] == "IN").sum()
dt_ex = (df["filter_doctype"] == "EX").sum()
print(f"  IN: {dt_in:,}  |  EX: {dt_ex:,}")

print("\n  Included document types:")
for doc_type, cnt in df[df["filter_doctype"] == "IN"]["Document Type"].value_counts().items():
    print(f"    {doc_type:<60} {cnt:>8,}")
print("\n  Excluded document types:")
for doc_type, cnt in df[df["filter_doctype"] == "EX"]["Document Type"].value_counts().items():
    print(f"    {doc_type:<60} {cnt:>8,}")


# ---------------------------------------------------------------------------
# Filter 3 – Dates
# ---------------------------------------------------------------------------
print("\nFilter 3: Dates …")
founding = load_founding_years(FILE_STARTUPS)
_date_results = df.apply(lambda r: passes_date(r, founding), axis=1)
df["filter_date"]        = _date_results.apply(lambda r: "IN" if r[0] else "EX")
df["filter_date_reason"] = _date_results.apply(lambda r: r[1])

d_in = (df["filter_date"] == "IN").sum()
d_ex = (df["filter_date"] == "EX").sum()
after_cutoff   = df["filter_date_reason"].str.startswith("year ").sum()
before_founded = d_ex - after_cutoff
print(f"  IN: {d_in:,}  |  EX: {d_ex:,}")
print(f"    - after {MAX_YEAR}               : {int(after_cutoff):,}")
print(f"    - before startup founded    : {int(before_founded):,}")


# ---------------------------------------------------------------------------
# Filter 4 – Deduplication
# ---------------------------------------------------------------------------
print("\nFilter 4: Deduplication …")

df["_title_norm"]    = df["Title"].apply(lambda t: normalize(t, strip_tags=True))
df["_abstract_norm"] = df["Abstract"].apply(normalize)
df["_priority"]      = (
    df["Found In (Searched DBs)"].str.strip().map(PRIORITY).fillna(9).astype(int)
)

df["filter_dedup"]     = "IN"
df["duplicate_of"]     = ""
df["duplicate_reason"] = ""

# Sort so WOS records always appear before INSPEC, INSPEC before PPRN.
# Within the same DB the original order is preserved.
df = df.sort_values("_priority", kind="stable").reset_index(drop=True)

# -- Step 4a: DOI match --
doi_groups: dict = defaultdict(list)
for idx, row in df.iterrows():
    doi = str(row.get("DOI", "")).strip()
    if doi and doi.lower() not in ("nan", "", "none", "n/a"):
        doi_groups[doi.lower()].append(idx)

doi_count = 0
for doi, idxs in doi_groups.items():
    if len(idxs) < 2:
        continue
    keep = idxs[0]
    for idx in idxs[1:]:
        if df.at[idx, "filter_dedup"] == "IN":
            df.at[idx, "filter_dedup"]     = "EX"
            df.at[idx, "duplicate_of"]     = str(df.at[keep, "UID"])
            df.at[idx, "duplicate_reason"] = "DOI match"
            doi_count += 1
print(f"  DOI duplicates        : {doi_count:,}")

# -- Step 4b: Exact normalized title --
title_groups: dict = defaultdict(list)
for idx, row in df.iterrows():
    if df.at[idx, "filter_dedup"] == "EX":
        continue
    t = row["_title_norm"]
    if t and len(t) > 5:
        title_groups[t].append(idx)

title_count = 0
for t, idxs in title_groups.items():
    if len(idxs) < 2:
        continue
    keep = idxs[0]
    for idx in idxs[1:]:
        if df.at[idx, "filter_dedup"] == "IN":
            df.at[idx, "filter_dedup"]     = "EX"
            df.at[idx, "duplicate_of"]     = str(df.at[keep, "UID"])
            df.at[idx, "duplicate_reason"] = "Exact title match"
            title_count += 1
print(f"  Exact title dups      : {title_count:,}")

# -- Step 4c: Fuzzy similarity (sliding window on sorted titles) --
active = df[df["filter_dedup"] == "IN"].copy()
active["_sort_key"] = active["_title_norm"].apply(remove_leading_article)
active = active.sort_values("_sort_key")
sorted_idxs = active.index.tolist()

fuzzy_count  = 0
n            = len(sorted_idxs)
report_every = max(n // 20, 1)

for pos, idx_i in enumerate(sorted_idxs):
    if pos % report_every == 0:
        print(f"  {100*pos/n:4.0f}%  ({pos:,}/{n:,})  fuzzy so far: {fuzzy_count:,}")

    if df.at[idx_i, "filter_dedup"] == "EX":
        continue

    title_i    = df.at[idx_i, "_title_norm"]
    abstract_i = df.at[idx_i, "_abstract_norm"]

    if not title_i or len(title_i) < 10:
        continue

    for idx_j in sorted_idxs[pos + 1 : min(pos + WINDOW_SIZE + 1, n)]:
        if df.at[idx_j, "filter_dedup"] == "EX":
            continue

        title_j = df.at[idx_j, "_title_norm"]
        if not title_j or len(title_j) < 10:
            continue

        title_sim = sim(title_i, title_j)
        if title_sim < FUZZY_THRESH:
            continue

        abstract_j         = df.at[idx_j, "_abstract_norm"]
        both_have_abstract = bool(abstract_i) and bool(abstract_j)

        if both_have_abstract:
            abs_sim = sim(abstract_i, abstract_j)
            if abs_sim < FUZZY_THRESH:
                continue
            reason = f"Fuzzy match (title={title_sim:.2f}, abstract={abs_sim:.2f})"
        else:
            if title_sim < TITLE_ONLY_MIN:
                continue
            reason = f"Fuzzy title-only (title={title_sim:.2f}, no abstract)"

        # Keep higher-priority DB (lower _priority value)
        if df.at[idx_i, "_priority"] <= df.at[idx_j, "_priority"]:
            keep, remove = idx_i, idx_j
        else:
            keep, remove = idx_j, idx_i

        df.at[remove, "filter_dedup"]     = "EX"
        df.at[remove, "duplicate_of"]     = str(df.at[keep, "UID"])
        df.at[remove, "duplicate_reason"] = reason
        fuzzy_count += 1

print(f"  100%  — fuzzy dups    : {fuzzy_count:,}")

dedup_in  = (df["filter_dedup"] == "IN").sum()
dedup_ex  = (df["filter_dedup"] == "EX").sum()
print(f"  IN: {dedup_in:,}  |  EX: {dedup_ex:,}")

print("\nDuplicates by source database:")
dup_df = df[df["filter_dedup"] == "EX"]
for db, cnt in dup_df["Found In (Searched DBs)"].value_counts().items():
    print(f"  {db}: {cnt:,}")


# ---------------------------------------------------------------------------
# Filter 5 – Affiliation match
# ---------------------------------------------------------------------------
print("\nFilter 5: Affiliation match …")
variant_to_startup = build_affil_index(FILE_AFFIL)
_affil_results = df.apply(
    lambda r: check_affil(r, variant_to_startup), axis=1
)
df["filter_affil"]          = _affil_results.apply(lambda r: "IN" if r[0] else "EX")
df["affil_matched_startup"] = _affil_results.apply(lambda r: r[1])

af_in = (df["filter_affil"] == "IN").sum()
af_ex = (df["filter_affil"] == "EX").sum()
print(f"  IN: {af_in:,}  |  EX: {af_ex:,}")

# -- Manual corrections: confirmed false positives --
# Ada       → ADA = American Diabetes Association (not Ada AI chatbot)
# Sierra    → SIERRA = INRIA/ENS ML research team (not Sierra AI)
# Cognition → French/Italian academic research labs (not Cognition AI)
# Miro      → Chilean/Belgian photonics lab acronym (not Miro whiteboard)
FP_STARTUPS = {"Ada", "Sierra", "Cognition", "Miro"}
fp_mask = (df["filter_affil"] == "IN") & (df["affil_matched_startup"].isin(FP_STARTUPS))
df.loc[fp_mask, "filter_affil"] = "EX"
fp_count = fp_mask.sum()
print(f"  Manual false-positive corrections: {fp_count:,} records marked EX")
print(f"  ({', '.join(sorted(FP_STARTUPS))})")
af_in = (df["filter_affil"] == "IN").sum()
af_ex = (df["filter_affil"] == "EX").sum()
print(f"  Final — IN: {af_in:,}  |  EX: {af_ex:,}")

print("\n  Top matched startups (by paper count):")
top = (
    df[df["filter_affil"] == "IN"]["affil_matched_startup"]
    .value_counts()
    .head(20)
)
for startup, cnt in top.items():
    print(f"    {startup:<45} {cnt:>8,}")


# =============================================================================
# Summary
# =============================================================================
all_pass = (
    (df["filter_keywords"] == "IN") &
    (df["filter_doctype"]  == "IN") &
    (df["filter_date"]     == "IN") &
    (df["filter_dedup"]    == "IN") &
    (df["filter_affil"]    == "IN")
)

print(f"""
=== Summary ===
Total records                    : {len(df):,}
Filter 1 (keywords)    IN / EX   : {kw_in:,} / {kw_ex:,}
Filter 2 (doc type)    IN / EX   : {dt_in:,} / {dt_ex:,}
Filter 3 (date)        IN / EX   : {d_in:,} / {d_ex:,}
  - after {MAX_YEAR}                   : {int(after_cutoff):,}
  - before startup founding       : {int(before_founded):,}
Filter 4 (dedup)       IN / EX   : {dedup_in:,} / {dedup_ex:,}
  - DOI match                     : {doi_count:,}
  - Exact title                   : {title_count:,}
  - Fuzzy match                   : {fuzzy_count:,}
Filter 5 (affil)       IN / EX   : {af_in:,} / {af_ex:,}

Records passing ALL filters      : {int(all_pass.sum()):,}
""")


# =============================================================================
# Save
# =============================================================================
df = df.drop(
    columns=["_title_norm", "_abstract_norm", "_priority", "_sort_key"],
    errors="ignore",
)

print(f"Saving to {FILE_OUT} …")
df.to_csv(FILE_OUT, index=False)
print(f"Done in {time.time()-t0:.1f}s")
