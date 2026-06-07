# Ghost Hotel Detector — London

An end-to-end intelligence pipeline that identifies London properties illegally operating as full-time short-term rentals on Airbnb, scores them by enforcement priority, and generates AI-drafted enforcement notices for council officers.

---

## The Problem

Under the **Deregulation Act 2015, Section 44**, London homeowners may only rent their property on platforms like Airbnb for a maximum of **90 nights per year** without planning permission. Beyond that, the property is classified as a commercial operation — a "ghost hotel."

This matters because:
- Ghost hotels permanently remove housing stock from the long-term rental market
- Westminster Council can issue fines up to **£100,000** per breach
- Enforcement is almost non-existent at scale — councils have no tooling to find cases
- London has an estimated 50,000+ active short-term listings; a significant portion exceed the limit

---

## What We Built

A four-stage pipeline running on an **NVIDIA GB10 (DGX Spark)**:

### 1. Data Pipeline
Downloads, decompresses, filters, and converts 10 public datasets to Parquet.

### 2. Scoring Pipeline (`pipeline/score.py`)
Assigns a **confidence score (0–0.97)** to each listing using four signals:

| Feature | Weight | Signal |
|---|---|---|
| F1 — 90-night breach | 35% | Estimated nights booked vs 90-night cap (sigmoid-scaled) |
| F2 — Operator portfolio | 25% | Number of listings under same host |
| F3 — Review velocity | 20% | 2024 reviews vs borough median (spike = full-time flip) |
| F4 — Calendar density | 20% | Actual booking % from calendar data (post-Sept 2024) |
| F5 — HMO licence | 0% | **Stub** — will be weighted when data arrives |
| F6 — Borough complaints | 0% | **Stub** — will be weighted when data arrives |

Priority tiers: **Priority** (>0.90) · **Review** (0.75–0.90) · **Monitor** (0.50–0.75) · **Low** (<0.50)

### 3. Entity Resolution (`pipeline/score.py`)
Fuzzy-matches host name variants using `rapidfuzz` to collapse operator aliases into canonical entities — e.g. "Alex", "Alexander Prop", "Alex Rentals" → one operator.

### 4. Companies House Cross-Reference (`pipeline/companies_xref.py`)
Matches commercial operators (3+ listings) against Companies House bulk data filtered to property SIC codes (68xxx, 55xxx). Surfaces the corporate entity behind the Airbnb host name.

### 5. Enforcement Generator (`pipeline/generate_enforcement.py`)
Calls **Nemotron Nano (30B)** on a local NIM endpoint to draft three-part enforcement packages per case: formal notice citing Deregulation Act 2015 s.44 + Housing Act 2004, evidence summary, and recommended action.

### 6. Officer Dashboard (`dashboard/app.py`)
Streamlit app at **http://10.18.216.24:8501** with four tabs:
- **Map** — all 51,996 scored listings as coloured dots on a Folium map, borough bar chart
- **Priority Cases** — filterable table with per-feature score breakdown, CSV export
- **Operators** — operator intelligence table with Companies House matches, revenue scatter
- **Enforcement** — live AI enforcement notice generation per case, pre-generated batch viewer

---

## Data

### Total data footprint: **2.44 GB**

| Layer | Size |
|---|---|
| Raw downloads | 1.69 GB |
| Clean Parquet | 0.70 GB |
| Scored outputs | 49 MB |

### Raw datasets (1.69 GB)

| File | Size | Source |
|---|---|---|
| `listings.csv.gz` | 81.6 MB | Inside Airbnb — London listings (Sep 2025 scrape) |
| `calendar.csv.gz` | 50.4 MB | Inside Airbnb — London calendar |
| `reviews.csv.gz` | 258.0 MB | Inside Airbnb — London reviews |
| `pp-2024.csv` | 161.7 MB | Land Registry — 2024 price paid data (UK) |
| `companies_house_basic.zip` | 493.4 MB | Companies House — full UK company register (Jun 2026) |
| `os_open_uprn.zip` | 617.1 MB | OS Open UPRN — all UK property reference numbers (May 2026) |
| `london_boundaries.zip` | 28.7 MB | London Datastore — GIS boundary shapefiles |
| `chain_quarterly_data.ods` | 0.3 MB | CHAIN — rough sleeping / shelter quarterly data |
| `mopac_pas_q3_2526.xlsx` | 0.1 MB | MOPAC — public attitude survey Q3 2025-26 |

### Clean Parquet (0.70 GB — 49.4 million rows total)

| File | Rows | Size | Notes |
|---|---|---|---|
| `calendar.parquet` | **35,357,974** | 15.3 MB | Full booking calendar for all London listings |
| `companies_house.parquet` | **5,698,274** | 411.4 MB | Full UK company register |
| `os_open_uprn_london.parquet` | **6,044,125** | 124.2 MB | London bbox filter (51.28–51.70°N) |
| `reviews.parquet` | **2,097,996** | 42.9 MB | Listing ID, date, reviewer — key cols only |
| `listings.parquet` | **51,996** | 33.0 MB | Pre-filtered: multi-host OR >60 nights |
| `land_registry_2024_london.parquet` | **144,162** | 3.7 MB | London postcodes only (from 927k UK rows) |

### Scored outputs (49 MB)

| File | Rows | Content |
|---|---|---|
| `scored_listings.parquet` | 51,996 | All listings with confidence score + feature breakdown |
| `priority_cases.parquet` | 3,644 | Listings scoring above 0.75 |
| `priority_cases.csv` | 3,644 | Same — for officer download |
| `operator_summary_enriched.parquet` | 8,653 | Operators with Companies House match |
| `operator_summary_enriched.csv` | 8,653 | Same — for officer download |

### Missing / stubbed (will strengthen scoring when added)

| Dataset | Source | Impact |
|---|---|---|
| HMO licence register | London borough portals (Westminster, Southwark, Tower Hamlets) | F5 weight will increase from 0% |
| Borough complaints (noise/ASB) | London Datastore | F6 weight will increase from 0% |
| CCOD/OCOD (company ownership) | Land Registry — free account required | Replaces name fuzzy-match with company number anchor |

---

## Results

```
Total listings scored:     51,996
Priority cases (>0.90):       715   ← prosecution-ready
Review cases (0.75–0.90):   2,929   ← flagged for monitoring
Unique operators found:     8,653
Companies House matches:      489   ← named entity confirmed
```

### Top operators by portfolio (entity-resolved)

| Operator | Listings | >90 nights | Avg score | CH match |
|---|---|---|---|---|
| Maxim | 289 | — | 0.97 | MAXIM HOMES LTD |
| Alex | 221 | — | 0.97 | ALEX PROPERTIES LIMITED |
| Ali | 178 | — | 0.97 | ALI HOMES LTD |
| James (x2) | 125 / 130 | 32 / 61 | 0.52 / 0.60 | HOME JAMES PROPERTY LIMITED |
| Maciej Jerzy | 95 | 73 | 0.74 | — |

---

## GPU Benchmark

Aggregation task: booking density across **35.4 million calendar rows**

| | Time |
|---|---|
| CPU (pandas) | 7.20s |
| GPU (cuDF on GB10) | **1.88s** |
| **Speedup** | **3.8×** |

Hardware: NVIDIA GB10 (DGX Spark) · CUDA 13.0 · cuDF 26.4.0

---

## Features to Add

### Data gaps
- HMO licence register — plug into F5, direct borough FOI requests
- CCOD/OCOD overseas company ownership — free Land Registry account, anchors entity resolution
- Borough ASB/complaints data — correlated with ghost hotel clusters

### Scoring
- Seasonal pattern analysis — ghost hotels booked year-round, home-sharers have gaps
- Reverse geocode lat/lng → UPRN → Land Registry address (makes cases concrete: "14 Wardour St")
- `license` field validation — cross-check any supplied licence numbers against council registers
- Price-per-night vs local rental yield — operators price at hotel rates, not home-share rates

### Dashboard
- PDF enforcement pack export — ready to file, no copy-paste
- Case tracking — "under review", "notice sent", "resolved" workflow states
- Operator network graph — same address / same director / same company → connected entities
- Time-series review history — proves when a listing flipped from occasional to full-time

### Infrastructure
- Run full scoring pipeline on GPU (currently only the benchmark uses cuDF — swap pandas → cuDF throughout for a larger speedup number)
- Nightly re-score as new Airbnb data is scraped
- REST API for council case management system integration

---

## Stack

| Component | Technology |
|---|---|
| Data processing | pandas · pyarrow |
| GPU acceleration | cuDF 26.4.0 · CUDA 13.0 |
| Entity resolution | rapidfuzz (token sort ratio, threshold 85) |
| AI enforcement drafting | Nemotron Nano 30B via NIM (local, http://10.18.216.24:30000) |
| Dashboard | Streamlit · Folium · Plotly |
| Hardware | NVIDIA GB10 DGX Spark |

---

## Project Structure

```
AIR_BNB/
├── data/
│   ├── raw/          # Downloaded source files (1.69 GB)
│   ├── clean/        # Parquet-converted datasets (0.70 GB, 49.4M rows)
│   ├── scored/       # Pipeline outputs (49 MB)
│   └── enforcement/  # AI-generated enforcement packages
├── pipeline/
│   ├── score.py                  # Core scoring pipeline
│   ├── companies_xref.py         # Companies House cross-reference
│   └── generate_enforcement.py   # Nemotron enforcement notice generator
├── benchmarks/
│   ├── benchmark.py              # GPU vs CPU timing
│   └── calendar_benchmark.json  # Results: 3.8x speedup
├── dashboard/
│   └── app.py                    # Streamlit officer dashboard
└── README.md
```
