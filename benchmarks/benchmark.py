"""
GPU vs CPU benchmark — calendar booking density aggregation
/home/nvidia/AIR_BNB/benchmarks/benchmark.py
"""

import cudf
import pandas as pd
import time
import json
from pathlib import Path

PATH = "/home/nvidia/AIR_BNB/data/clean/calendar.parquet"
OUT  = Path("/home/nvidia/AIR_BNB/benchmarks")
OUT.mkdir(exist_ok=True)


def run_aggregation_cpu():
    df = pd.read_parquet(PATH)
    df["date"] = pd.to_datetime(df["date"])
    return (
        df[df["date"] >= "2024-09-01"]
        .groupby("listing_id")["available"]
        .apply(lambda x: (x == "f").sum() / len(x))
    )


def run_aggregation_gpu():
    df = cudf.read_parquet(PATH)
    df["date"] = cudf.to_datetime(df["date"])
    recent = df[df["date"] >= "2024-09-01"].copy()
    recent["is_booked"] = (recent["available"] == "f").astype("int8")
    return recent.groupby("listing_id")["is_booked"].mean()


print(f"Dataset: 35.4M calendar rows | {PATH}")
print()

print("CPU run (pandas)...")
t0 = time.perf_counter()
cpu_result = run_aggregation_cpu()
cpu_time = time.perf_counter() - t0
print(f"  CPU: {cpu_time:.2f}s  ({len(cpu_result):,} listings aggregated)")

print()
print("GPU run (cuDF)...")
# warm-up
_ = cudf.DataFrame({"x": [1, 2, 3]}).sum()
t0 = time.perf_counter()
gpu_result = run_aggregation_gpu()
gpu_time = time.perf_counter() - t0
print(f"  GPU: {gpu_time:.2f}s  ({len(gpu_result):,} listings aggregated)")

speedup = cpu_time / gpu_time
print()
print(f"  Speedup: {speedup:.1f}x")

result = {
    "cpu_seconds": round(cpu_time, 3),
    "gpu_seconds": round(gpu_time, 3),
    "speedup": round(speedup, 1),
    "rows": 35_357_974,
    "dataset": "London Airbnb calendar — booking density aggregation",
}
json.dump(result, open(OUT / "calendar_benchmark.json", "w"), indent=2)
print(f"\nSaved → {OUT}/calendar_benchmark.json")
print(json.dumps(result, indent=2))
