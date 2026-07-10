import os  # Used to read the API key and data-directory from environment variables
import requests  # Used to send HTTP requests to the Web of Science API
import time  # Used to pause the script so we don't hit rate limits
import csv  # Used to export our final data to a spreadsheet
import re  # Used for title cleaning in deduplication

# --- Configuration ---
# REDACTED FOR GITHUB: the original script had a literal Clarivate Web of
# Science Expanded API key hardcoded here. It has been replaced with an
# environment-variable read so the key is never committed to version control.
# Set it before running, e.g.: export WOS_API_KEY="your-key-here"
API_KEY = os.environ["WOS_API_KEY"]

# All scripts in this package read/write their data files in a single,
# user-defined folder. Set AI_UNICORN_DATA_DIR to point at it; if unset,
# the current working directory is used (i.e. run the script from inside
# that folder).
DATA_DIR = os.environ.get("AI_UNICORN_DATA_DIR", os.getcwd())

# The list of AI startups/affiliations
AFFILIATIONS = [
    "01.AI", "1KMXC", "6sense", "Abnormal AI", "Abnormal Security", "Abridge", "Ada", "Adept AI", "Advance Intelligence Group", "Afiniti",
"Agibot", "Agile Robots AG", "AI21 Labs", "Aibee", "Airbyte", "Aiven", "AIWAYS", "Alation", "AlphaSense", "Ambience Healthcare",
"Amperity", "Anaconda", "Continuum Analytics", "Anduril Industries", "Anthropic", "Anyscale", "Anysphere", "Apollo.io",
"Applied Intuition", "ASAPP", "Attentive", "Augury", "Aura", "iSubscribed", "Intersections", "Automation Anywhere",
"Avathon", "Ayar Labs", "Baichuan Intelligence", "Beijing Baichuan Intelligent Technology Co.", "Baseten", "BetterUp",
"BigID", "Biren Technology", "Bluecore", "Canva", "Celestial AI", "Celonis", "Cera", "Cerebras Systems", "Character.ai",
"Checkr", "Clari", "Clarify Health Solutions", "Clay", "ClickHouse", "Clio", "Cognite", "Cognition", "Cognition AI",
"Cohere", "Cohesity", "Collibra", "Commure", "ConcertAI", "Concerto HealthAI", "Contentsquare", "Coralogix", "Cresta",
"Cruise", "Cruise Automation", "Crusoe Energy Systems", "Cyberhaven", "Cyera", "Darwinbox", "Databricks",
"DataDirect Networks", "Dataiku", "Dataminr", "DataStax", "dbt Labs", "Fishtown Analytics", "Decagon", "Decart",
"DeepL", "Deeproute", "Dental Monitoring", "DevRev", "Dexterity", "DFINITY", "Dialpad", "Distyl AI", "Dream Security",
"DriveWealth", "DuerOS", "Eightfold AI", "ElevenLabs", "EliseAI", "Enflame", "Eve", "EvenUp", "Exabeam", "LogRhythm",
"Fal", "Feedzai", "Figure", "Filevine", "Fireworks AI", "Firmus Technologies", "Fiture", "Flexiv", "Flipdish",
"Flo Health", "Formation Bio", "FourKites", "Fractal", "Framer", "G42", "Galaxy Bot", "Gamma", "Gaussian Robot",
"Gecko Robotics", "Generate Biomedicines", "Genspark AI", "Glance", "Glean", "Gong", "Govini", "Groq", "GrubMarket",
"GupShup", "H2O.ai", "Hailo", "Halcyon", "Haomo.AI", "Harvey", "Helsing", "HighRadius", "Highspot", "Hippocratic AI",
"Hive", "HoneyBook", "Hugging Face", "Huma", "iCarbonX", "Icertis", "Iluvatar CoreX", "Imbue", "Immunai",
"Inflection AI", "InMobi", "Innovaccer", "Insider", "Insilico Medicine", "Inspur Cloud", "Interos",
"Invisible Technologies", "Invoca", "Iterable", "Iyuno-SDI", "Jasper", "Jarvis.ai", "Conversion.ai",
"KoBold Metals", "Krutrim", "Kunlun", "Lambda", "LangChain", "Legora", "Lendbuzz", "Lightmatter",
"Lila Sciences", "Liquid AI", "LogicMonitor", "Lovable", "Lucid Software", "MaintainX", "Mashgin", "Matillion",
"Meero", "MegaRobo", "MEGVII", "Mercor", "Metropolis", "MiniMax", "Miro", "Mistral AI", "Mobvoi", "Modular",
"Moka", "Momenta", "Moonshot AI", "Moore Threads", "Morning Consult", "Motive", "Multiverse", "n8n",
"Neko Health", "Neo4j", "Neo Technology", "Netradyne", "NewsBreak", "Nimble Robotics", "Ninja", "NotCo",
"Nscale", "Nuro", "o9 Solutions", "OneTrust", "Tugboat Logic", "Certification Automation", "OpenAI",
"OpenEvidence", "Opentrons", "Optibus", "Orbbec", "OrCam Technologies", "Outreach", "OutSystems", "Owkin",
"Parametrix.ai", "Parloa", "Pathos", "PatSnap", "Pendo", "People.ai", "Periodic Labs", "Perplexity",
"Phenom", "Physical Intelligence", "Pilot", "Placer.ai", "Plus", "Poolside", "Preferred Networks",
"Quantexa", "Quantum Systems", "Rebellion Defense", "Rebellions", "Redesign Health", "Redpanda Data",
"Reflection.Ai", "Reka AI", "Relativity Space", "Replit", "Rokid", "Runway", "Safe Superintelligence",
"Sakana AI", "Salt Security", "SambaNova", "SandboxAQ", "Saronic", "Scale AI", "Scribe", "SeekOut", "Seekr",
"Shield AI", "Shift Technology", "Shiprocket", "Sierra", "Silicon Box", "Sisense", "Skild AI", "Skydio",
"SmartMore Corporation Limited", "SmartNews", "Snorkel AI", "Socure", "Soterea", "Sourcegraph",
"SparkCognition", "Speak", "SpreeAI", "Spring Health", "Squirrel Ai Learning", "Stability AI",
"Standard Cognition", "StepStar", "Suno", "Supabase", "Superhuman", "Grammarly", "Sword Health",
"Synthesia", "Tabby", "Tala Health", "Talkdesk", "Tamara", "Tekion", "Tenstorrent",
"Terminus Technologies", "The Bot Company", "Thinking Machines Lab", "ThoughtSpot",
"Together AI", "Tractable", "Trax", "Tresata", "Tricentis", "TUNGEE", "Turing", "Typeface",
"Uniphore", "Uptake Technologies", "Vanta", "Vantaca", "VAST Data", "Vectra AI", "Verbit",
"VITAC", "Vercel", "Zeit", "Veriff", "Vise", "Viz", "Vultr", "Waymo", "Wayve", "WEKA",
"World Labs", "xAI", "Xaira Therapeutics", "Xiaoice", "XPANCEO", "Yinwang Smart Technology",
"Yitu Technology", "You.com", "Yuanfudao", "Zelos", "Zeta", "Zhipu AI", "Zuoyebang", 
    # Add your list here...
]

# The official Clarivate endpoint for the WoS Expanded API
SEARCH_URL = "https://wos-api.clarivate.com/api/wos"

# WoS Collections to be searched — all known database IDs.
# Unsubscribed ones are automatically skipped by the 4xx error handler.
TARGET_DATABASES = {
    "WOS": "Web of Science Core Collection",
}

# When the same paper is found in multiple databases, we keep the metadata from
# the highest-priority source. Lower number = higher priority.
DB_PRIORITY = {"WOS": 0, "INSPEC": 1, "PPRN": 2}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def safe_get_list(item):
    """
    Clarivate's API JSON returns a list if a paper has multiple authors. If it only has ONE author, it returns a dictionary.
    This helper function forces everything into a list so our loops don't crash.
    """
    if isinstance(item, list): return item
    if isinstance(item, dict): return [item]
    return []

def extract_abstract(static_data):
    """Extract abstract text from WoS Expanded API JSON."""
    try:
        ab_block = static_data.get("fullrecord_metadata", {}).get("abstracts", {}).get("abstract", {})
        if isinstance(ab_block, dict):
            ab_text = ab_block.get("abstract_text", {})
            if isinstance(ab_text, dict):
                p = ab_text.get("p", "")
                return " ".join(str(x) for x in p).strip() if isinstance(p, list) else str(p).strip()
        parts = []
        for ab in safe_get_list(ab_block):
            ab_text = ab.get("abstract_text", {}) if isinstance(ab, dict) else {}
            p = ab_text.get("p", "") if isinstance(ab_text, dict) else ""
            parts.append(" ".join(str(x) for x in p) if isinstance(p, list) else str(p))
        return " ".join(parts).strip()
    except Exception:
        return ""


def extract_keywords(static_data):
    """Extract author keywords and KeyWords Plus from WoS Expanded API JSON."""
    try:
        terms = []
        # Author keywords
        for kw in safe_get_list(static_data.get("fullrecord_metadata", {}).get("keywords", {}).get("keyword", [])):
            val = kw.get("value", "") if isinstance(kw, dict) else str(kw)
            if val:
                terms.append(str(val).strip())
        # KeyWords Plus
        for kw in safe_get_list(static_data.get("item", {}).get("keywords_plus", {}).get("keyword", [])):
            val = kw.get("value", "") if isinstance(kw, dict) else str(kw)
            if val and str(val).strip() not in terms:
                terms.append(str(val).strip())
        return "; ".join(terms)
    except Exception:
        return ""


def extract_doctype(summary):
    """Extract document type from WoS Expanded API JSON."""
    try:
        raw = summary.get("doctypes", {}).get("doctype", "")
        if isinstance(raw, str):
            return raw
        parts = []
        for dt in safe_get_list(raw):
            val = dt if isinstance(dt, str) else dt.get("content", "")
            if val:
                parts.append(str(val))
        return "; ".join(parts)
    except Exception:
        return ""



def fetch_wos_records_for_db(affiliation, db_id):
    """
    This function handles the actual communication with the Web of Science server.
    It asks for papers from a specific organization in a specific database,
    and handles the "pagination" (flipping through pages of results 100 at a time).
    """
    headers = {"X-ApiKey": API_KEY,
               "Accept": "application/json"  # Tell the API we want JSON formatted data back
               }
    # AD (Address) is the safest tag to search across all these different databases.
    # PY restricts results to the 1998–2025 publication window.
    query = f'AD="{affiliation}" AND PY=(1998-2025)'
    first_record, count, all_records = 1, 100, []  # Start at record 1, pull 100 records per page (the API maximum), store all the pages of results in a master list

    while True:
        # Define the parameters for our API request
        params = {"databaseId": db_id, "usrQuery": query, "count": count, "firstRecord": first_record}
        response = requests.get(SEARCH_URL, headers=headers, params=params)  # Send the request to Clarivate

        # Error Handling: 429 means we are requesting data too fast
        if response.status_code == 429:
            print("    Rate limit reached. Sleeping for 10 seconds...")
            time.sleep(10)
            continue

        # Error Handling: If the database is unsubscribed or broken, skip it
        elif response.status_code != 200:
            print(f"    [!] Skipping {db_id} - {response.status_code} Error")
            break

        # Convert the raw text response into a Python dictionary (JSON)
        try:
            data = response.json()
        except Exception:
            print(f"    [!] Empty or invalid response body for {db_id}, skipping page.")
            break

        # Navigate the nested JSON to find the actual list of papers ("REC")
        try:
            records_list = safe_get_list(data.get("Data", {}).get("Records", {}).get("records", {}).get("REC", []))
        except AttributeError:
            records_list = []

        # If there are no papers on this page, stop looping
        if not records_list: break
        # Add this page's papers to our master list
        all_records.extend(records_list)

        # Check how many total papers exist on Clarivate's servers for this search
        total_results = data.get("QueryResult", {}).get("RecordsFound", 0)

        # Print a status update, but only on the first page
        if first_record == 1:
            print(f"  -> {db_id}: Found {total_results} records")

        # If we have collected all the available papers, stop looping
        if len(all_records) >= total_results: break

        # Move to the next page of 100 records
        first_record += count

        # Hard limit: The API physically cannot return more than 100,000 records
        if first_record >= 100000:
            print(f"    [!] Reached maximum pagination limit (100k) for {db_id}.")
            break

        time.sleep(0.5)  # Be polite to the server between pages

    return all_records


# ==========================================
# MAIN EXECUTION & DATA EXTRACTION
# ==========================================

def main():
    # We use a dictionary to store our final papers.
    # This automatically deduplicates records: if we try to save a paper
    # using a key that already exists, it just updates the existing entry.
    unique_records = {}

    # Loop through every organization in our list
    for affil in AFFILIATIONS:
        print(f"\n=======================================")
        print(f"Fetching AI Startup Records from: {affil}")
        print(f"=======================================")

        # For each organization, search all of our target databases
        for db_id in TARGET_DATABASES.keys():
            # Call our fetcher function to get the raw JSON records
            records = fetch_wos_records_for_db(affil, db_id)
            if records:
                print(f"     -> {len(records)} records found")

            # Now, process every single paper we found
            for record in records:
                static_data = record.get("static_data", {})
                dynamic_data = record.get("dynamic_data", {})
                # Guard against malformed responses: sometimes Clarivate returns a string
                # in place of the expected object. If that happens we replace it with an empty
                # dict so our subsequent .get() calls don't blow up.
                if not isinstance(dynamic_data, dict):
                    dynamic_data = {}
                summary = static_data.get("summary", {})
                fullrecord_metadata = static_data.get("fullrecord_metadata", {})

                # --- 1. EXTRACT IDENTIFIERS FOR SMART DEDUPLICATION ---

                # Get the WoS Unique Identifier (UID) (Ensure string)
                uid = ""
                uid_data = record.get("UID", "")
                if isinstance(uid_data, list) and len(uid_data) > 0:
                    uid = str(uid_data[0])
                elif isinstance(uid_data, str):
                    uid = uid_data

                # Get the Title (and ensure it's a string, not a list)
                title = ""
                for t in safe_get_list(summary.get("titles", {}).get("title", [])):
                    if isinstance(t, dict) and t.get("type") == "item":
                        raw_title = t.get("content", "")
                        if isinstance(raw_title, list):
                            title = " ".join([str(x) for x in raw_title])
                        else:
                            title = str(raw_title)
                        break

                # Get the Digital Object Identifier (DOI) (Safely convert lists to strings)
                doi = ""
                # the API sometimes returns unexpected strings in place of objects;
                # wrap the lookup in a try/except to avoid AttributeError crashes.
                try:
                    identifiers = safe_get_list(
                        dynamic_data.get("cluster_related", {}).get("identifiers", {}).get("identifier", []))
                except AttributeError:
                    identifiers = []
                for ident in identifiers:
                    if isinstance(ident, dict) and ident.get("type") == "doi":
                        raw_doi = ident.get("value", "")
                        if isinstance(raw_doi, list) and len(raw_doi) > 0:
                            doi = str(raw_doi[0])
                        else:
                            doi = str(raw_doi)
                        break

                # --- 2. WATERFALL DEDUPLICATION ---
                # We need a master key to identify this paper in our dictionary.
                # Since DOIs are the best, we try that first. If missing, we
                # clean the title and try that. If missing, we use the UID.

                # Clean the title: remove all punctuation, spaces, and make lowercase
                cleaned_title = re.sub(r'[^a-z0-9]', '', title.lower()) if title else ""

                dedup_key = ""
                if doi:
                    dedup_key = f"doi_{doi.lower().strip()}"
                elif cleaned_title:
                    dedup_key = f"title_{cleaned_title}"
                elif uid:
                    dedup_key = f"uid_{uid}"
                else:
                    continue  # If it has literally no identifying info, skip it entirely

                db_name_str = f"{db_id}"

                # IF WE ALREADY HAVE THIS PAPER:
                # Always merge the source DB and startup name into the existing entry.
                # Then check priorities: if the stored version already comes from a
                # higher-priority DB, skip re-extraction. Otherwise fall through so
                # the better-quality metadata from this DB overwrites the old one.
                is_priority_upgrade = False
                if dedup_key in unique_records:
                    if db_name_str not in unique_records[dedup_key]["source_db_list"]:
                        unique_records[dedup_key]["source_db_list"].append(db_name_str)
                    unique_records[dedup_key]["matched_startups"].add(affil)
                    stored_priority = DB_PRIORITY.get(unique_records[dedup_key]["primary_db"], 99)
                    new_priority    = DB_PRIORITY.get(db_id, 99)
                    if new_priority >= stored_priority:
                        continue  # Stored metadata is already from a better or equal DB
                    # New DB has higher priority — fall through and overwrite metadata
                    is_priority_upgrade = True

                # --- 3. EXTRACT METADATA FOR NEW PAPERS ---
                # Extract Publication Year
                pub_year = ""
                pub_info = summary.get("pub_info", {})
                if isinstance(pub_info, dict):
                    pub_year = pub_info.get("pubyear", "")
                    # Preprints often lack a formal 'pubyear', so we grab the first 4 chars of 'sortdate'
                    if not pub_year:
                        sortdate = pub_info.get("sortdate", "")
                        if sortdate:
                            pub_year = sortdate[:4]

                # Map Addresses: WoS separates "Authors" and "Addresses".
                # We build a dictionary mapping the internal Address Number (e.g., "1")
                # to the full text string (e.g., "Stanford Univ, CA").
                address_mapping = {}
                address_names = safe_get_list(fullrecord_metadata.get("addresses", {}).get("address_name", []))

                for addr_name in address_names:
                    if isinstance(addr_name, dict):
                        addr_specs = safe_get_list(addr_name.get("address_spec", []))
                        for spec in addr_specs:
                            if isinstance(spec, dict):
                                addr_no = str(spec.get("addr_no", ""))
                                full_addr = spec.get("full_address", "")
                                if addr_no and full_addr:
                                    address_mapping[addr_no] = full_addr

                # Extract Authors and link them to the Addresses we just mapped
                authors_with_affiliations = []
                names_list = safe_get_list(summary.get("names", {}).get("name", []))

                for name in names_list:
                    if isinstance(name, dict) and name.get("role") == "author":
                        display_name = name.get("full_name", name.get("display_name", ""))
                        if display_name:
                            author_addrs = []
                            # Look up which address numbers belong to this specific author
                            addr_nos = name.get("addr_no", "")

                            # Handle inconsistencies in how WoS formats this data
                            if isinstance(addr_nos, str):
                                addr_no_list = addr_nos.split()
                            elif isinstance(addr_nos, list):
                                addr_no_list = [str(a) for a in addr_nos]
                            elif isinstance(addr_nos, int):
                                addr_no_list = [str(addr_nos)]
                            else:
                                addr_no_list = []

                            # Convert those numbers into the actual university/startup names
                            for a_no in addr_no_list:
                                if a_no in address_mapping:
                                    author_addrs.append(address_mapping[a_no])

                            # Format nicely: "John Doe [Anthropic | Stanford]"
                            if author_addrs:
                                affil_str = " | ".join(author_addrs)
                                authors_with_affiliations.append(f"{display_name} [{affil_str}]")
                            else:
                                authors_with_affiliations.append(f"{display_name} [No Affiliation Listed]")

                authors_str = "; ".join(authors_with_affiliations)

                # --- 4. EXTRACT ABSTRACT, KEYWORDS, DOCUMENT TYPE ---
                abstract      = extract_abstract(static_data)
                keywords      = extract_keywords(static_data)
                document_type = extract_doctype(summary)

                # --- 5. EXTRACT CITATIONS & PLATFORM-WIDE INDEXING ---
                times_cited = 0
                # We keep track of every database this paper exists in across the whole WoS platform.
                # Start with the one we are currently searching.
                platform_dbs = set([db_id])  # Start with the database we found it in

                # Check the citation silos
                # citations info may also be malformed; protect with try/except
                try:
                    tc_list = safe_get_list(dynamic_data.get("citation_related", {}).get("tc_list", {}).get("silo_tc", []))
                except AttributeError:
                    tc_list = []

                for tc in tc_list:
                    if isinstance(tc, dict):
                        # The "coll_id" (Collection ID) tells us other databases where this paper lives.
                        # We add them to our set to expose its full indexing footprint.
                        coll_id = tc.get("coll_id", "")
                        if coll_id and coll_id != "WOK":
                            platform_dbs.add(coll_id)

                        # Grab the highest citation count available.
                        # The "WOK" silo represents "All Databases" combined, which is the most accurate.
                        local_count = tc.get("local_count")

                        if local_count is not None:
                            try:
                                local_count = int(local_count)
                            except ValueError:
                                continue

                            if coll_id == "WOK":
                                times_cited = local_count
                            elif local_count > times_cited:
                                times_cited = local_count

                # Finally, save all this beautifully extracted data to our master dictionary!
                if is_priority_upgrade:
                    # Overwrite metadata with the higher-priority DB's version,
                    # but preserve the already-merged source/startup/platform lists.
                    existing = unique_records[dedup_key]
                    unique_records[dedup_key] = {
                        "original_uid": uid,
                        "source_db_list": existing["source_db_list"],
                        "platform_dbs": existing["platform_dbs"].union(platform_dbs),
                        "matched_startups": existing["matched_startups"],
                        "primary_db": db_id,
                        "pub_year": pub_year,
                        "title": title,
                        "doi": doi,
                        "authors": authors_str,
                        "times_cited": times_cited,
                        "all_document_addresses": " | ".join(address_mapping.values()),
                        "abstract": abstract,
                        "keywords": keywords,
                        "document_type": document_type,
                    }
                else:
                    unique_records[dedup_key] = {
                        "original_uid": uid,
                        "source_db_list": [db_name_str],
                        "platform_dbs": platform_dbs,
                        "matched_startups": {affil},
                        "primary_db": db_id,
                        "pub_year": pub_year,
                        "title": title,
                        "doi": doi,
                        "authors": authors_str,
                        "times_cited": times_cited,
                        "all_document_addresses": " | ".join(address_mapping.values()),
                        "abstract": abstract,
                        "keywords": keywords,
                        "document_type": document_type,
                    }

            time.sleep(1)  # Delay between checking different databases

    # ---------------------------------------------------------
    # STEP 5: EXPORT TO CSV
    # ---------------------------------------------------------
    csv_filename = os.path.join(DATA_DIR, "wos_papers.csv")
    print(f"\n--- Saving to {csv_filename} ---")

    # Define the headers for our spreadsheet
    fieldnames = [
        "UID", "Found In (Searched DBs)", "Platform-Wide Indexing (All DBs)",
        "Matched Startups",
        "Year", "Title", "DOI", "Authors & Affiliations",
        "Total Times Cited", "All Document Addresses",
        "Abstract", "Keywords", "Document Type",
    ]

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Loop through our dictionary and write each paper as a row in the CSV
        for dedup_key, data in unique_records.items():
            # Combine the databases we explicitly searched with the hidden ones we found
            all_known_dbs = set(data["source_db_list"]).union(data["platform_dbs"])

            writer.writerow({
                "UID": data["original_uid"],
                "Found In (Searched DBs)": " | ".join(data["source_db_list"]),
                "Platform-Wide Indexing (All DBs)": " | ".join(sorted(list(all_known_dbs))),
                "Matched Startups": " | ".join(sorted(data["matched_startups"])),
                "Year": data["pub_year"],
                "Title": data["title"],
                "DOI": data["doi"],
                "Authors & Affiliations": data["authors"],
                "Total Times Cited": data["times_cited"],
                "All Document Addresses": data["all_document_addresses"],
                "Abstract": data.get("abstract", ""),
                "Keywords": data.get("keywords", ""),
                "Document Type": data.get("document_type", ""),
            })

    print(f"Total records saved: {len(unique_records)}")


if __name__ == "__main__":
    main()