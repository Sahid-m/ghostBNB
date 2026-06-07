"""
Ghost Hotel Detector — Companies House Cross-Reference
/home/nvidia/AIR_BNB/pipeline/companies_xref.py

Run AFTER score.py
"""

import pandas as pd
import re
from rapidfuzz import fuzz, process
from pathlib import Path

DATA = Path("/home/nvidia/AIR_BNB/data/clean")
OUT  = Path("/home/nvidia/AIR_BNB/data/scored")

print("Loading scored listings and Companies House...")
scored    = pd.read_parquet(OUT / "scored_listings.parquet")
companies = pd.read_parquet(DATA / "companies_house.parquet")
companies.columns = companies.columns.str.strip()

# ── Filter to property-related SIC codes ──────────────────────────────────────

PROPERTY_SICS = {
    '68100', '68201', '68202', '68209', '68310', '68320',
    '55100', '55201', '55209', '55900',
}

def sic_prefix(val):
    if pd.isna(val):
        return ''
    return str(val)[:5]

property_cos = companies[
    companies['SICCode.SicText_1'].apply(sic_prefix).isin(PROPERTY_SICS) |
    companies['SICCode.SicText_2'].apply(sic_prefix).isin(PROPERTY_SICS)
].copy()

print(f"Property-related companies: {len(property_cos):,}")

# ── Name normalisation ────────────────────────────────────────────────────────

_STOP = re.compile(
    r'\b(ltd|limited|plc|llp|properties|property|management|letting|lettings'
    r'|group|uk|london|homes|home|residential|rental|rentals|solutions'
    r'|services|accommodation)\b'
)

def normalise(name):
    if pd.isna(name):
        return ''
    name = str(name).lower()
    name = _STOP.sub('', name)
    name = re.sub(r'[^a-z0-9 ]', '', name)
    return name.strip()

# ── Airbnb operators with 3+ listings ─────────────────────────────────────────

airbnb_operators = (
    scored[scored['operator_total_listings'] >= 3][
        ['canonical_operator_id', 'operator_name']
    ]
    .drop_duplicates('canonical_operator_id')
    .copy()
)
airbnb_operators['name_norm'] = airbnb_operators['operator_name'].apply(normalise)
print(f"Airbnb commercial operators to match: {len(airbnb_operators):,}")

# ── Fuzzy match ───────────────────────────────────────────────────────────────

property_cos['name_norm'] = property_cos['CompanyName'].apply(normalise)
company_names    = property_cos['name_norm'].tolist()
company_numbers  = property_cos['CompanyNumber'].tolist()
company_statuses = property_cos['CompanyStatus'].tolist()
company_sics     = property_cos['SICCode.SicText_1'].tolist()
company_addrs    = property_cos['RegAddress.AddressLine1'].tolist()

print("Matching operators to Companies House...")
matches = []
for _, row in airbnb_operators.iterrows():
    if not row['name_norm']:
        continue
    results = process.extract(
        row['name_norm'],
        company_names,
        scorer=fuzz.token_sort_ratio,
        limit=3
    )
    for match_name, score, idx in results:
        if score >= 80:
            matches.append({
                'canonical_operator_id': row['canonical_operator_id'],
                'airbnb_host_name':      row['operator_name'],
                'company_name':          property_cos.iloc[idx]['CompanyName'],
                'company_number':        company_numbers[idx],
                'company_address':       company_addrs[idx],
                'company_status':        company_statuses[idx],
                'sic_code':              company_sics[idx],
                'match_score':           score,
            })

matches_df = pd.DataFrame(matches) if matches else pd.DataFrame(
    columns=['canonical_operator_id', 'airbnb_host_name', 'company_name',
             'company_number', 'company_address', 'company_status', 'sic_code', 'match_score']
)
print(f"Matches found: {len(matches_df):,}")

# ── Merge into operator summary ───────────────────────────────────────────────

operator_summary = pd.read_parquet(OUT / "operator_summary.parquet")

# Keep best match per operator
best_matches = (
    matches_df.sort_values('match_score', ascending=False)
    .drop_duplicates('canonical_operator_id')
)

operator_summary = operator_summary.merge(
    best_matches[['canonical_operator_id', 'company_name', 'company_number',
                  'company_status', 'sic_code', 'match_score']],
    on='canonical_operator_id',
    how='left'
)

operator_summary.to_parquet(OUT / "operator_summary_enriched.parquet", index=False)
operator_summary.to_csv(OUT / "operator_summary_enriched.csv", index=False)

# ── Summary print ─────────────────────────────────────────────────────────────

matched = operator_summary[operator_summary['company_number'].notna()].sort_values(
    'max_confidence', ascending=False
)

print(f"\nOperators matched to Companies House: {len(matched):,} / {len(operator_summary):,}")
print("\nTop 20 operators with Companies House matches:")
print(
    matched[['operator_name', 'company_name', 'company_status',
             'total_listings', 'max_confidence', 'match_score']]
    .head(20)
    .to_string(index=False)
)

print(f"\nOutputs saved to {OUT}/")
print("  operator_summary_enriched.parquet")
print("  operator_summary_enriched.csv")
