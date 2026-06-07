# Ghost Hotel Detector — London

An end-to-end intelligence system that identifies London properties illegally operating as full-time short-term rentals on Airbnb, scores them by enforcement priority, predicts risk for brand-new listings with a machine-learning model, and drafts AI enforcement notices — all served through an **agentic officer console**.

Runs on an **NVIDIA GB10 (DGX Spark)**.

---

## The Problem

Under the **Deregulation Act 2015, Section 44**, London homeowners may only rent a property on platforms like Airbnb for a maximum of **90 nights per year** without planning permission. Beyond that, the property is a commercial operation — a "ghost hotel."

- Ghost hotels permanently remove housing stock from the long-term rental market.
- Councils (e.g. Westminster) can issue fines up to **£100,000** per breach.
- Enforcement is near-nonexistent at scale — councils have no tooling to find cases.
- London has 90,000+ active listings; a significant share exceed the limit.

---

## What We Built

### 1. Data Pipeline (`data/process_data.py`)
Downloads, decompresses, filters, and converts a dozen public datasets to Parquet. **49.9 million rows** across the clean layer.

### 2. Scoring Pipeline (`pipeline/score.py`)
Assigns a **confidence score (0–0.97)** to each listing from weighted signals:

| Feature | Weight | Signal |
|---|---|---|
| F1 — 90-night breach | 35% | Estimated nights booked vs 90-night cap (sigmoid-scaled) |
| F2 — Operator portfolio | 25% | Number of listings under same host |
| F3 — Review velocity | 20% | 2024 reviews vs borough median (spike = full-time flip) |
| F4 — Calendar density | 20% | Actual booking % from calendar data |
| F5 — HMO licence | 0% | **Stub** — weighted when data arrives |
| F6 — Borough complaints | 0% | **Stub** — weighted when data arrives |
| F7 — Commercial language | — | LLM read of description + host bio (0 personal → 1 commercial) |
| F8 — Review anomaly | — | Burst timing + reviewer reuse (`pipeline/review_anomaly.py`) |

Priority tiers: **Priority** (>0.90) · **Review** (0.75–0.90) · **Monitor** (0.50–0.75) · **Low** (<0.50)

### 3. Entity Resolution (`pipeline/score.py`)
Fuzzy-matches host-name variants with `rapidfuzz` (token-sort, threshold 85) to collapse operator aliases into canonical entities — "Alex", "Alexander Prop", "Alex Rentals" → one operator.

### 4. Companies House Cross-Reference (`pipeline/companies_xref.py`)
Matches commercial operators against the Companies House bulk register filtered to property SIC codes (68xxx, 55xxx). Surfaces the corporate entity behind a host name.

### 5. ML Ghost-Hotel Classifier  ⭐ *new*
A `HistGradientBoostingClassifier` (`pipeline/train_model.py` → `pipeline/predict.py`) that scores **brand-new listings absent from our snapshot** using **only live-scrapeable listing-page features** (no calendar/operator joins that a new listing wouldn't have). Shared feature spec in `pipeline/features.py`. Per-prediction explanations via occlusion analysis. See [ML Model](#ml-model) below.

### 6. Live Airbnb Scraper  ⭐ *new*
`pipeline/airbnb_scraper.py` — a defensive, best-effort single-listing scraper (og/meta + embedded JSON + JSON-LD) that **never raises** and degrades gracefully to manual entry when Airbnb blocks it. Used to pull listings created after our data cut-off and feed them to the ML model.

### 7. Enforcement Generator (`pipeline/generate_enforcement.py`)
Drafts three-part enforcement packages (formal notice citing Deregulation Act 2015 s.44 + Housing Act 2004, evidence summary, recommended action) via the vLLM model.

### 8. Agentic Officer Console (`dashboard/`)  ⭐ *rebuilt*
A Streamlit app at **http://10.18.216.24:8501** with two surfaces:

- **📊 Dashboard** — KPIs, Folium risk map, per-borough breakdown, top-operators table.
- **🤖 Agent Console** — a chat interface. Officers ask in plain English ("worst offenders in Westminster", "investigate airbnb.com/rooms/13913", "draft an enforcement notice for the worst Camden case"). The LLM **routes** the request to the right tool and shows its work as live step cards. Tools: NL→pandas search, property lookup + ML, operator intel, top cases, enforcement drafting. A deterministic regex router is the fallback if the LLM response can't be parsed.

Module layout: `core.py` (config, cached loaders, LLM helpers, scraper/ML handles), `ui.py` (light card theme + components), `agent.py` (router + tools), `app.py` (the two tabs).

---

## Model / Inference

The application is served by an **NVIDIA Nemotron Nano VL** vision-language model on a local **vLLM** server:

| | |
|---|---|
| Endpoint | `http://10.18.216.24:8090/v1/chat/completions` |
| Model id | `nemotron-nano-vl` (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`) |
| Max context | 8,192 tokens |
| Used for | F7 commercial-language scoring, AI search reasoning, agent routing, enforcement drafting |

---

## ML Model

Trained on the 25 Sep 2025 snapshot. Positives = `priority_tier` ∈ {Priority, Review}; negatives = `Low` (down-sampled 4×); the ambiguous Monitor/NaN middle is dropped.

| Metric | Value |
|---|---|
| ROC AUC | 1.000 |
| PR AUC | 0.9999 |
| Train / test | 13,324 / 3,331 |
| Confusion @0.5 | TN 2661 · FP 4 · FN 1 · TP 665 |
| Top features | `number_of_reviews_ltm`, `host_listings_count`, `minimum_nights` |

> ⚠️ **Honest caveat (stored in the model bundle):** the near-perfect score is **label leakage, not magic**. The labels are the heuristic's own F1–F8 tiers, which are computed from these same listing fields. So the model effectively **re-learns the heuristic** from live-scrapeable features — a useful, explainable triage aid for listings we have no history for, **not** independent ground truth. Genuine predictive power needs council-confirmed outcomes as labels.

Artifacts: `data/models/ghost_clf.joblib` (300 KB) · `data/models/metrics.json`.

---

## Data

**Total footprint: ~2.6 GB** · raw 1.8 GB · clean 705 MB · scored 90 MB · models 308 KB · enforcement 64 KB.

### Raw downloads (`data/raw/` — 1.8 GB)

| File | Size | Source |
|---|---|---|
| `os_open_uprn.zip` | 589 MB | OS Open UPRN — all UK property reference numbers |
| `companies_house_basic.zip` | 471 MB | Companies House — full UK company register |
| `reviews.csv.gz` | 247 MB | Inside Airbnb — London reviews |
| `pp-2024.csv` | 155 MB | Land Registry — 2024 price-paid (UK) |
| `pp-2025.csv` | 146 MB | Land Registry — 2025 price-paid (UK) |
| `calendar.csv.gz` | 78 MB | Inside Airbnb — London booking calendar |
| `listings.csv.gz` | 49 MB | Inside Airbnb — London listings (25 Sep 2025 scrape) |
| `london_boundaries.zip` | 28 MB | London Datastore — GIS boundary shapefiles |
| `pp-2026.csv` | 25 MB | Land Registry — 2026 price-paid (UK, partial year) |
| `chain_quarterly_data.ods` | <1 MB | CHAIN — rough-sleeping quarterly data |
| `mopac_pas_q3_2526.xlsx` | <1 MB | MOPAC — public attitude survey Q3 2025-26 |
| `london_borough_profiles.csv` | <1 MB | London Datastore — borough demographic profiles |

### Clean Parquet (`data/clean/` — 705 MB · 49.9 M rows)

| File | Rows | Size | Notes |
|---|---|---|---|
| `calendar.parquet` | 35,357,974 | 15 MB | Full booking calendar for all London listings |
| `os_open_uprn_london.parquet` | 6,044,125 | 119 MB | London bbox filter |
| `companies_house.parquet` | 5,698,274 | 393 MB | Full UK company register |
| `reviews.parquet` | 2,097,996 | 41 MB | Listing id · date · reviewer (key cols) |
| `land_registry_london.parquet` | 297,779 | 7 MB | All London price-paid records |
| `land_registry_2024_london.parquet` | 144,162 | 3.6 MB | London 2024 subset |
| `listings.parquet` | 96,871 | 63 MB | Pre-filtered candidate listings |
| `london_boundaries/` | — | 65 MB | Extracted borough shapefiles |

### Scored outputs (`data/scored/` — 90 MB)

| File | Rows | Content |
|---|---|---|
| `scored_listings.parquet` | 96,871 | Every listing with confidence score + full F1–F8 breakdown (103 cols) |
| `priority_cases.parquet` / `.csv` | 3,331 | Listings scoring ≥ 0.75 |
| `operator_summary.parquet` / `.csv` | 53,528 | Entity-resolved operators |
| `operator_summary_enriched.parquet` / `.csv` | 53,528 | Operators + Companies House match |

### Models & enforcement

| Path | Content |
|---|---|
| `data/models/ghost_clf.joblib` | Trained ML classifier bundle |
| `data/models/metrics.json` | Training metrics + top feature importances |
| `data/enforcement/case_*.json` | 11 generated enforcement packages |
| `data/enforcement/enforcement_batch.csv` | Batch enforcement run |

### Missing / stubbed (would strengthen scoring)

| Dataset | Source | Impact |
|---|---|---|
| HMO licence register | Borough portals (Westminster, Southwark, Tower Hamlets) | Activates F5 |
| Borough complaints (noise/ASB) | London Datastore | Activates F6 |
| CCOD/OCOD company ownership | Land Registry (free account) | Anchors entity resolution to company numbers |
| Council-confirmed outcomes | Enforcement records | Real ML labels (removes the leakage caveat) |

---

## Results

```
Total listings scored:      96,871
Priority cases (>0.90):        566   ← prosecution-ready
Review cases (0.75–0.90):    2,765   ← flagged for monitoring
Monitor (0.50–0.75):        13,365
Entity-resolved operators:  53,528
Companies House matches:       489   ← named entity confirmed
```

### Largest operators by portfolio (entity-resolved)

| Operator | Listings | >90 nights | Max score | Companies House |
|---|---|---|---|---|
| LuxurybookingsFZE | 500 | 0 | 0.43 | — |
| Blueground | 405 | 10 | 0.76 | — |
| Tobias Mead | 305 | 25 | 0.97 | — |
| Maxim | 289 | 60 | 0.97 | MAXIM HOMES LTD |
| Matthew | 233 | 14 | 0.76 | MATTHEW PROPERTY LTD |
| Alex | 221 | 64 | 0.97 | ALEX PROPERTIES LIMITED |

---

## GPU Benchmark

Booking-density aggregation across **35.4 million calendar rows**:

| | Time |
|---|---|
| CPU (pandas) | 7.20 s |
| GPU (cuDF on GB10) | **1.88 s** |
| **Speedup** | **3.8×** |

Hardware: NVIDIA GB10 (DGX Spark) · CUDA 13.0 · cuDF 26.4.0

---

## Setup & Run

```bash
# Python deps (system Python on the GB10 uses PEP-668 — hence the flag)
pip install --break-system-packages \
    pandas pyarrow streamlit streamlit-folium folium plotly requests \
    scikit-learn joblib beautifulsoup4 lxml rapidfuzz

# (Re)train the ML classifier — writes data/models/ghost_clf.joblib
python3 pipeline/train_model.py

# Launch the agentic console
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

The vLLM model server must be reachable at `http://10.18.216.24:8090`. Light theme is pinned in `.streamlit/config.toml`.

---

## Stack

| Component | Technology |
|---|---|
| Data processing | pandas · pyarrow |
| GPU acceleration | cuDF 26.4.0 · CUDA 13.0 |
| Entity resolution | rapidfuzz (token-sort, threshold 85) |
| ML model | scikit-learn `HistGradientBoostingClassifier` · joblib |
| Scraping | requests · BeautifulSoup4 · lxml |
| LLM serving | vLLM · Nemotron Nano VL (`http://10.18.216.24:8090`) |
| Dashboard | Streamlit · Folium · Plotly |
| Hardware | NVIDIA GB10 DGX Spark |

---

## Project Structure

```
AIR_BNB/
├── data/
│   ├── raw/          # Downloaded source files (1.8 GB)
│   ├── clean/        # Parquet datasets (705 MB · 49.9M rows)
│   ├── scored/       # Pipeline outputs (90 MB)
│   ├── models/       # Trained ML classifier + metrics
│   └── enforcement/  # AI-generated enforcement packages
├── pipeline/
│   ├── score.py                  # Core scoring pipeline (F1–F8)
│   ├── review_anomaly.py         # F8 review-anomaly signal
│   ├── description_classifier.py # F7 commercial-language signal
│   ├── companies_xref.py         # Companies House cross-reference
│   ├── features.py               # Shared ML feature spec
│   ├── train_model.py            # Trains the ghost-hotel classifier
│   ├── predict.py                # Inference + occlusion explanations
│   ├── airbnb_scraper.py         # Defensive live listing scraper
│   └── generate_enforcement.py   # Enforcement notice generator
├── dashboard/
│   ├── app.py        # Dashboard + Agent Console (Streamlit)
│   ├── core.py       # Shared config, loaders, LLM helpers
│   ├── agent.py      # LLM router + agent tools
│   └── ui.py         # Light card theme + components
├── benchmarks/
│   ├── benchmark.py              # GPU vs CPU timing
│   └── calendar_benchmark.json  # Results: 3.8× speedup
├── .streamlit/config.toml        # Pinned light theme
└── README.md
```
