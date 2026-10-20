"""Build the cleaned road-year dataset from the original MMDA-derived source data.

Run:  python src/data_prep.py
"""
import re
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C  # noqa: E402

YEARS = list(range(2015, 2025))
R6_NAME = "R-6 MAGSAYSAY / AURORA BLVD."


def export_original_csv() -> pd.DataFrame:
    """Export the SOURCE_DATA sheet of the original workbook to CSV (unchanged values)."""
    src = pd.read_excel(C.ORIGINAL_WORKBOOK, sheet_name="SOURCE_DATA", header=3)
    src = src.dropna(how="all")
    src = src[src["Road"].notna()]
    src.to_csv(C.ORIGINAL_CSV, index=False)
    return src


def build_clean(src: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide source table to one row per road-year."""
    src = src.copy()
    src["Road"] = src["Road"].str.strip()
    # Keep study roads only
    src = src[src["Role in study"].str.contains("Model development|TEST", na=False)]

    # Merge Magsaysay + Aurora (R-6)
    r6 = src[src["Road"].isin(["MAGSAYSAY BLVD.", "AURORA BLVD."])]
    merged = {"Road": R6_NAME}
    for y in YEARS:
        merged[f"AADT {y}"] = r6[f"AADT {y}"].mean()
        merged[f"MMARAS {y}"] = r6[f"MMARAS {y}"].sum(min_count=1)
    merged["Road Length in km"] = r6["Road Length in km"].sum()
    merged["Total Lanes-Kilometers"] = r6["Total Lanes-Kilometers"].sum()
    merged["Total Lanes (Avg)"] = merged["Total Lanes-Kilometers"] / merged["Road Length in km"]
    src = pd.concat([src[~src["Road"].isin(["MAGSAYSAY BLVD.", "AURORA BLVD."])],
                     pd.DataFrame([merged])], ignore_index=True)

    rows = []
    for _, r in src.iterrows():
        for y in YEARS:
            rows.append({
                "road": r["Road"], "year": y,
                "aadt": r[f"AADT {y}"], "crashes": r[f"MMARAS {y}"],
                "road_length_km": r["Road Length in km"],
                "total_lanes_avg": r["Total Lanes (Avg)"],
                "lane_km": r["Total Lanes-Kilometers"],
            })
    df = pd.DataFrame(rows).sort_values(["road", "year"]).reset_index(drop=True)
    df["prev_crashes"] = df.groupby("road")["crashes"].shift(1)
    df["pandemic"] = df["year"].isin(C.PANDEMIC_YEARS).astype(int)
    df["is_test_road"] = (df["road"] == C.TEST_ROAD).astype(int)
    df["road"] = df["road"].map(lambda s: re.sub(r"\s+", " ", s))
    for c in ["aadt", "crashes", "prev_crashes"]:
        df[c] = df[c].astype("Int64") if c != "aadt" else df[c]
    return df[["road", "year", "aadt", "crashes", "prev_crashes", "road_length_km",
               "total_lanes_avg", "lane_km", "pandemic", "is_test_road"]]


def verify_against_workbook(df: pd.DataFrame) -> None:
    """Cross-check the rebuilt data against the workbook's DATASETS sheet."""
    ref = pd.read_excel(C.ORIGINAL_WORKBOOK, sheet_name="DATASETS", header=3, usecols="A:I")
    ref.columns = ["road", "year", "aadt", "crashes", "cls", "len", "lanes", "lane_km", "prev"]
    ref["road"] = ref["road"].str.strip()
    m = df.merge(ref, on=["road", "year"], suffixes=("", "_ref"))
    assert len(m) == len(df) == len(ref), "row mismatch with workbook DATASETS"
    assert (m["aadt"] - m["aadt_ref"]).abs().max() < 1e-6
    assert (m["crashes"].astype(float) - m["crashes_ref"]).abs().max() < 1e-6
    assert (m["lane_km"] - m["lane_km_ref"]).abs().max() < 1e-6
    both = m["prev_crashes"].notna() & m["prev"].notna()
    assert (m.loc[both, "prev_crashes"].astype(float) - m.loc[both, "prev"]).abs().max() < 1e-6
    assert m["prev_crashes"].isna().equals(m["prev"].isna())


def main() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    src = export_original_csv()
    df = build_clean(src)
    verify_against_workbook(df)
    df.to_csv(C.PROCESSED_CSV, index=False)
    print(f"Original CSV : {C.ORIGINAL_CSV.name} ({len(src)} rows)")
    print(f"Clean CSV    : {C.PROCESSED_CSV.name} {df.shape}; roads={df['road'].nunique()}; "
          f"missing prev_crashes={df['prev_crashes'].isna().sum()} (2015, no 2014 record)")


if __name__ == "__main__":
    main()
