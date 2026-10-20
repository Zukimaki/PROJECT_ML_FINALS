"""Input validation and prediction helpers shared by the Streamlit app and tests."""
from __future__ import annotations

import json

import joblib
import pandas as pd

import config as C

REQUIRED = ["aadt", "prev_crashes", "lane_km", "year"]
HARD_LIMITS = {  # hard rejection limits
    "aadt": (1, 2_000_000),
    "prev_crashes": (0, 100_000),
    "lane_km": (0.1, 20_000),
    "year": (2000, 2100),
}


def load_artifacts():
    model = joblib.load(C.MODEL_PATH)
    meta = json.loads(C.METADATA_PATH.read_text(encoding="utf-8"))
    return model, meta


def validate_frame(df: pd.DataFrame, ranges: dict | None = None):
    """Return (clean_df, problems). problems maps row index -> list of messages.

    Rows listed with an 'ERROR' message are not predicted; 'WARNING' rows are
    predicted but flagged because they lie outside the training range.
    """
    problems: dict[int, list[str]] = {}
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}. "
                         f"Expected: {', '.join(REQUIRED)}.")
    work = df.copy()
    for c in REQUIRED:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if "pandemic" not in work.columns:
        work["pandemic"] = work["year"].isin(C.PANDEMIC_YEARS).astype(float)
    work["pandemic"] = pd.to_numeric(work["pandemic"], errors="coerce")

    for i, row in work.iterrows():
        msgs = []
        for c in REQUIRED:
            v = row[c]
            if pd.isna(v):
                msgs.append(f"ERROR: '{c}' is missing or not a number")
                continue
            lo, hi = HARD_LIMITS[c]
            if not lo <= v <= hi:
                msgs.append(f"ERROR: '{c}' = {v:g} is outside the allowed range {lo:,g} to {hi:,g}")
            elif c != "year" and ranges and not ranges[c][0] <= v <= ranges[c][1]:
                msgs.append(f"WARNING: '{c}' = {v:g} is outside the training range "
                            f"{ranges[c][0]:g} to {ranges[c][1]:g}; the estimate is less reliable")
        if not pd.isna(row["year"]) and row["year"] != int(row["year"]):
            msgs.append("ERROR: 'year' must be a whole number")
        if row["pandemic"] not in (0, 1):
            msgs.append("ERROR: 'pandemic' must be 0 or 1")
        if msgs:
            problems[i] = msgs
    return work, problems


def predict_frame(model, work: pd.DataFrame, problems: dict) -> pd.DataFrame:
    ok = [i for i in work.index if not any(m.startswith("ERROR") for m in problems.get(i, []))]
    result = work.copy()
    result["predicted_crashes"] = float("nan")
    if ok:
        result.loc[ok, "predicted_crashes"] = model.predict(work.loc[ok, C.FEATURES])
    result["notes"] = [" | ".join(problems.get(i, [])) for i in work.index]
    return result


def recursive_forecast(
    model,
    start_year: int,
    horizon_years: int,
    initial_prev_crashes: float,
    initial_aadt: float,
    lane_km: float,
    aadt_growth_pct: float = 0.0,
    ranges: dict | None = None,
) -> pd.DataFrame:
    """Perform recursive multi-year crash forecasting.

    For year t = start_year:
      - prev_crashes comes from the known/entered previous-year crash count.
      - aadt is initial_aadt.

    For year t > start_year:
      - prev_crashes is taken from the model's prediction for year t - 1.
      - aadt is compounded by aadt_growth_pct.
    """
    rows: list[dict] = []
    current_prev = float(initial_prev_crashes)
    current_aadt = float(initial_aadt)
    growth_multiplier = 1.0 + (float(aadt_growth_pct) / 100.0)

    for step in range(int(horizon_years)):
        yr = int(start_year) + step
        if step > 0:
            current_aadt = max(1.0, current_aadt * growth_multiplier)

        step_df = pd.DataFrame([{
            "aadt": current_aadt,
            "prev_crashes": current_prev,
            "lane_km": float(lane_km),
            "year": yr,
        }])

        work, problems = validate_frame(step_df, ranges)
        errors = [m for m in problems.get(0, []) if m.startswith("ERROR")]
        warnings = [m for m in problems.get(0, []) if m.startswith("WARNING")]

        if errors:
            rows.append({
                "step": step + 1,
                "year": yr,
                "aadt": current_aadt,
                "prev_crashes": current_prev,
                "prev_crashes_type": "Entered / Baseline" if step == 0 else f"Estimated from {yr - 1}",
                "predicted_crashes": float("nan"),
                "change_vs_prev": float("nan"),
                "notes": "ERROR: " + " | ".join(errors),
            })
            break

        pred_res = predict_frame(model, work, problems)
        raw_pred = float(pred_res["predicted_crashes"].iloc[0])
        pred_val = max(0.0, raw_pred)

        step_notes = warnings.copy()
        if step == 0:
            step_notes.append("1-year ahead (direct baseline)")
        else:
            step_notes.append(f"Recursive (fed {current_prev:,.0f} crashes from {yr - 1})")

        rows.append({
            "step": step + 1,
            "year": yr,
            "aadt": current_aadt,
            "prev_crashes": current_prev,
            "prev_crashes_type": "Entered / Baseline" if step == 0 else f"Estimated from {yr - 1}",
            "predicted_crashes": pred_val,
            "change_vs_prev": pred_val - current_prev,
            "notes": " | ".join(step_notes),
        })

        # Feed prediction forward
        current_prev = pred_val

    return pd.DataFrame(rows)
