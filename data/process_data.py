#!/usr/bin/env python3
"""Process raw downloaded datasets into clean parquet files."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gzip
import zipfile
import os
import sys

RAW = "/home/nvidia/AIR_BNB/data/raw"
CLEAN = "/home/nvidia/AIR_BNB/data/clean"
os.makedirs(CLEAN, exist_ok=True)

LONDON_PREFIXES = (
    'SW', 'SE', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9',
    'EC', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',
    'WC', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9',
    'NW', 'NE', 'EN', 'HA', 'UB', 'TW', 'KT', 'SM', 'CR', 'BR', 'DA', 'RM', 'IG',
)

def is_london_postcode(pc):
    if not isinstance(pc, str):
        return False
    pc = pc.strip().upper()
    return any(pc.startswith(p) for p in LONDON_PREFIXES)

def save_parquet(df, name):
    out = os.path.join(CLEAN, f"{name}.parquet")
    df.to_parquet(out, index=False)
    print(f"  Saved {name}.parquet: {len(df):,} rows, {df.shape[1]} cols, {os.path.getsize(out)/(1024**2):.1f}MB")
    print(f"  Columns: {list(df.columns)[:10]}")
    if len(df) > 0:
        print(f"  Sample row 0: {df.iloc[0].to_dict()}")
    return df


def process_listings():
    print("\n=== Inside Airbnb Listings ===")
    df = pd.read_csv(os.path.join(RAW, "listings.csv.gz"), compression='gzip', low_memory=False)
    print(f"  Raw rows: {len(df):,}, cols: {df.shape[1]}")
    # No pre-filter — keep all listings so any Airbnb URL can be looked up
    save_parquet(df, "listings")


def process_calendar():
    print("\n=== Inside Airbnb Calendar ===")
    df = pd.read_csv(os.path.join(RAW, "calendar.csv.gz"), compression='gzip', low_memory=False)
    print(f"  Raw rows: {len(df):,}, cols: {df.shape[1]}")
    save_parquet(df, "calendar")


def process_reviews():
    print("\n=== Inside Airbnb Reviews ===")
    # Only keep key columns to reduce size
    cols = ['listing_id', 'id', 'date', 'reviewer_id', 'reviewer_name']
    df = pd.read_csv(os.path.join(RAW, "reviews.csv.gz"), compression='gzip',
                     usecols=lambda c: c in cols, low_memory=False)
    print(f"  Raw rows: {len(df):,}, cols: {df.shape[1]}")
    save_parquet(df, "reviews")


def process_land_registry():
    print("\n=== Land Registry 2024 Price Paid ===")
    # PP data has no header
    col_names = [
        'transaction_id', 'price', 'date', 'postcode', 'property_type',
        'new_build', 'tenure', 'paon', 'saon', 'street', 'locality',
        'town', 'district', 'county', 'ppd_cat', 'record_status'
    ]
    df = pd.read_csv(os.path.join(RAW, "pp-2024.csv"), header=None,
                     names=col_names, low_memory=False, on_bad_lines='skip')
    print(f"  Raw rows: {len(df):,}")
    # Filter to London postcodes
    london_mask = df['postcode'].apply(is_london_postcode)
    df = df[london_mask]
    print(f"  London rows: {len(df):,}")
    save_parquet(df, "land_registry_2024_london")


def process_companies_house():
    print("\n=== Companies House Basic Data ===")
    zpath = os.path.join(RAW, "companies_house_basic.zip")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print(f"  Files in zip: {names[:5]}")
        # Find the main CSV
        csvs = [n for n in names if n.lower().endswith('.csv')]
        if csvs:
            with z.open(csvs[0]) as f:
                df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
            print(f"  Raw rows: {len(df):,}, cols: {df.shape[1]}")
            save_parquet(df, "companies_house")


def process_london_boundaries():
    print("\n=== London Boundary Files ===")
    zpath = os.path.join(RAW, "london_boundaries.zip")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print(f"  Files in zip: {[n for n in names if not n.endswith('/')][:15]}")
        # Extract all to clean dir
        extract_dir = os.path.join(CLEAN, "london_boundaries")
        os.makedirs(extract_dir, exist_ok=True)
        z.extractall(extract_dir)
        print(f"  Extracted to {extract_dir}")
        # List extracted files
        for root, dirs, files in os.walk(extract_dir):
            for f in files[:5]:
                fpath = os.path.join(root, f)
                print(f"    {fpath} ({os.path.getsize(fpath)/(1024**2):.1f}MB)")


def process_uprn():
    print("\n=== OS Open UPRN ===")
    zpath = os.path.join(RAW, "os_open_uprn.zip")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print(f"  Files in zip: {names[:5]}")
        csvs = [n for n in names if n.lower().endswith('.csv')]
        if csvs:
            print(f"  Reading {csvs[0]}...")
            with z.open(csvs[0]) as f:
                # Read in chunks to filter London
                chunks = []
                for chunk in pd.read_csv(f, chunksize=500000, low_memory=False):
                    # Filter to London UPRN by lat/lng bounding box
                    # London roughly: lat 51.28-51.70, lon -0.54 to 0.33
                    if 'LATITUDE' in chunk.columns and 'LONGITUDE' in chunk.columns:
                        mask = (
                            (chunk['LATITUDE'] >= 51.28) & (chunk['LATITUDE'] <= 51.70) &
                            (chunk['LONGITUDE'] >= -0.54) & (chunk['LONGITUDE'] <= 0.33)
                        )
                        chunks.append(chunk[mask])
                    elif 'latitude' in chunk.columns.str.lower().tolist():
                        lat_col = [c for c in chunk.columns if c.lower() == 'latitude'][0]
                        lon_col = [c for c in chunk.columns if c.lower() == 'longitude'][0]
                        mask = (
                            (chunk[lat_col] >= 51.28) & (chunk[lat_col] <= 51.70) &
                            (chunk[lon_col] >= -0.54) & (chunk[lon_col] <= 0.33)
                        )
                        chunks.append(chunk[mask])
                    else:
                        chunks.append(chunk)
                        print(f"  Columns: {list(chunk.columns)}")
                        break  # just get cols from first chunk
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                print(f"  London UPRN rows: {len(df):,}, cols: {df.shape[1]}")
                save_parquet(df, "os_open_uprn_london")


if __name__ == "__main__":
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ['all']

    if 'all' in tasks or 'listings' in tasks:
        process_listings()
    if 'all' in tasks or 'calendar' in tasks:
        process_calendar()
    if 'all' in tasks or 'reviews' in tasks:
        process_reviews()
    if 'all' in tasks or 'land_registry' in tasks:
        process_land_registry()
    if 'all' in tasks or 'companies' in tasks:
        process_companies_house()
    if 'all' in tasks or 'boundaries' in tasks:
        process_london_boundaries()
    if 'all' in tasks or 'uprn' in tasks:
        process_uprn()

    print("\n=== Done ===")
    print("\nClean files:")
    for f in os.listdir(CLEAN):
        fpath = os.path.join(CLEAN, f)
        if os.path.isfile(fpath):
            print(f"  {f}: {os.path.getsize(fpath)/(1024**2):.1f}MB")
