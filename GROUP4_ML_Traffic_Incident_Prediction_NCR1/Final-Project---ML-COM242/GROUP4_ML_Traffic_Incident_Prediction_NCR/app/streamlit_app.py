"""Streamlit app: predict yearly vehicular crashes on an NCR road.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
import inference as I  # noqa: E402

st.set_page_config(page_title="NCR Road Crash Predictor", page_icon=":traffic_light:", layout="wide")

st.markdown(
    """
    <style>
    /* Dark theme */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stSidebar"], [data-testid="stDecoration"] {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }
    header, footer {
        background-color: #0E1117 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
        font-weight: 600 !important;
    }
    p, span, label {
        color: #FAFAFA;
    }
    [data-testid="stMetricValue"] {
        color: #FAFAFA !important;
    }
    [data-testid="stMetricLabel"] {
        color: #B8B8B8 !important;
    }
    .stButton > button,
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background-color: #FAFAFA !important;
        border: 1px solid #FAFAFA !important;
        color: #0E1117 !important;
    }
    .stButton > button *,
    button[kind="primary"] *,
    button[data-testid="baseButton-primary"] *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #0E1117 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover,
    button[kind="primary"]:hover {
        background-color: #D6D6D6 !important;
        border-color: #D6D6D6 !important;
    }
    .stButton > button:hover *,
    .stButton > button:hover p {
        color: #0E1117 !important;
    }
    .stDownloadButton > button,
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border: 1.5px solid #FAFAFA !important;
        border-radius: 6px !important;
    }
    .stDownloadButton > button *,
    .stDownloadButton > button p,
    .stDownloadButton > button span,
    button[kind="secondary"] *,
    button[kind="secondary"] p {
        color: #FAFAFA !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #3A3B47 !important;
        border-color: #FAFAFA !important;
    }
    .stDownloadButton > button:hover *,
    .stDownloadButton > button:hover p {
        color: #FAFAFA !important;
    }
    button[data-baseweb="tab"] {
        color: #B8B8B8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FAFAFA !important;
        border-bottom-color: #FAFAFA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifacts():
    return I.load_artifacts()


@st.cache_data
def get_data():
    return pd.read_csv(C.PROCESSED_CSV)


try:
    model, meta = get_artifacts()
    data = get_data()
except FileNotFoundError:
    st.error("Model or data file not found. Run `python src/data_prep.py` and "
             "`python src/train.py` first (see README).")
    st.stop()

ranges = meta["training_ranges"]
test_mae = meta["final_test_metrics"]["MAE"]

st.title("NCR Road Crash Predictor")
st.caption(f"Model: **{meta['model_name']}** | trained on {meta['train_years'][0]}-"
           f"{meta['train_years'][1]} MMDA AADT and MMARAS data | decision-support tool only")

tab_main, tab_csv, tab_about = st.tabs([
    "Predict crashes",
    "Batch CSV upload",
    "About the model",
])

MAX_HORIZON = 10

with tab_main:
    st.subheader("Estimate yearly crashes for one road")
    st.write(
        "Enter the last year with known crash data (start year) and the year to predict (end year). "
        "If they are one year apart you get a single prediction. If they are further apart, the model "
        "automatically predicts year by year, feeding each predicted year in as the previous-year "
        "crashes for the next."
    )

    roads = ["(enter values manually)"] + sorted(data["road"].unique())
    choice = st.selectbox("Start from an existing road (optional)", roads)
    defaults = {"aadt": 150000.0, "crashes": 2000.0, "lane_km": 100.0, "known_year": 2024}
    if choice != roads[0]:
        latest_row = data[data["road"] == choice].sort_values("year").iloc[-1]
        defaults = {
            "aadt": float(latest_row["aadt"]),
            "crashes": float(latest_row["crashes"]) if pd.notna(latest_row["crashes"]) else 0.0,
            "lane_km": float(latest_row["lane_km"]),
            "known_year": int(latest_row["year"]),
        }

    c1, c2 = st.columns(2)
    start_year = c1.number_input("Start year (last year with known crashes)",
                                 value=defaults["known_year"], step=1, format="%d")
    end_year = c2.number_input("End year (year to predict)",
                               value=defaults["known_year"] + 1, step=1, format="%d")
    prev = c1.number_input(f"Crashes recorded in {start_year}", value=defaults["crashes"], step=10.0,
                           help="MMARAS crash count for the same road in the start year.")
    aadt = c2.number_input("AADT: average vehicles per day", value=defaults["aadt"], step=1000.0,
                           help="Annual Average Daily Traffic from the MMDA counting stations.")
    lane_km = c1.number_input("Lane-kilometers (road length x average lanes)",
                              value=defaults["lane_km"], step=10.0)
    steps = int(end_year) - int(start_year)
    aadt_growth = 0.0
    if steps > 1:
        aadt_growth = c2.slider("Annual traffic (AADT) change (%)", min_value=-5.0, max_value=10.0,
                                value=0.0, step=0.5,
                                help="Applied each year after the first predicted year (0% = constant traffic).")

    if st.button("Predict crashes", type="primary"):
        if steps < 1:
            st.error("The end year must be later than the start year.")
        elif steps > MAX_HORIZON:
            st.error(f"Forecasts are limited to {MAX_HORIZON} years ahead; errors compound with each step.")
        else:
            fc_df = I.recursive_forecast(
                model=model,
                start_year=int(start_year) + 1,
                horizon_years=steps,
                initial_prev_crashes=prev,
                initial_aadt=aadt,
                lane_km=lane_km,
                aadt_growth_pct=aadt_growth,
                ranges=ranges,
            )
            has_error = fc_df["notes"].str.contains("ERROR").any()
            if has_error:
                for _, r in fc_df[fc_df["notes"].str.contains("ERROR")].iterrows():
                    st.error(f"Year {int(r['year'])}: {r['notes'].replace('ERROR: ', '')}")
            else:
                aadt_min, aadt_max = ranges["aadt"]
                outside_aadt = fc_df.loc[
                    (fc_df["aadt"] < aadt_min) | (fc_df["aadt"] > aadt_max)
                ]
                if not outside_aadt.empty:
                    outside_years = ", ".join(outside_aadt["year"].astype(int).astype(str))
                    warning = (
                        f"Forecast AADT is outside the training range "
                        f"({aadt_min:,.0f}–{aadt_max:,.0f} vehicles/day) in {outside_years}. "
                        "These estimates are less reliable because the model has not seen "
                        "traffic volumes at these levels."
                    )
                    above_range = outside_aadt[outside_aadt["aadt"] > aadt_max]
                    if not above_range.empty:
                        above_years = ", ".join(above_range["year"].astype(int).astype(str))
                        warning += (
                            f" In {above_years}, traffic exceeds the training maximum; "
                            "the tree-based model may level off instead of reflecting "
                            "continued crash growth."
                        )
                    st.warning(warning)

                pred = float(fc_df.iloc[-1]["predicted_crashes"])
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Predicted crashes in {int(end_year)}", f"{pred:,.0f}")
                m2.metric("Typical error per step (test MAE)", f"± {test_mae:,.0f}")
                m3.metric(f"Change vs {int(start_year)}", f"{pred - prev:+,.0f}")
                if steps == 1:
                    st.caption("This model returns a single estimate and does not provide probabilities. "
                               "The typical error is the average miss on the 2023-2024 test years.")
                else:
                    st.info(
                        f"**Recursive forecast over {steps} years.** {start_year + 1} is predicted from "
                        f"your {prev:,.0f} crashes in {int(start_year)}; each later year uses the previous "
                        "year's prediction as its input. Errors compound with every step, and tree models "
                        "tend to level off beyond the training range, so treat later years as rough scenarios."
                    )
                    fig, ax = plt.subplots(figsize=(7, 2.8))
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')
                    years_list = fc_df["year"].astype(int).tolist()
                    crashes_list = fc_df["predicted_crashes"].tolist()
                    ax.plot(years_list, crashes_list, marker='o', color='#FAFAFA', linewidth=2, markersize=6)
                    for yr_pt, cr_pt in zip(years_list, crashes_list):
                        ax.annotate(f"{cr_pt:,.0f}", (yr_pt, cr_pt), textcoords="offset points",
                                    xytext=(0, 7), ha='center', fontsize=9, fontweight='bold', color='#FAFAFA')
                    ax.set_xlabel("Year", color="#FAFAFA", fontsize=10)
                    ax.set_ylabel("Predicted Crashes", color="#FAFAFA", fontsize=10)
                    ax.set_xticks(years_list)
                    ax.tick_params(colors="#FAFAFA")
                    ax.grid(True, linestyle='--', alpha=0.25, color="#888888")
                    for sp in ax.spines.values():
                        sp.set_color('#B8B8B8')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                disp = fc_df[[
                    "year", "aadt", "prev_crashes", "prev_crashes_type",
                    "predicted_crashes", "change_vs_prev", "notes"
                ]].copy()
                disp.columns = [
                    "Year", "AADT (vehicles/day)", "Prev-Year Crashes Used",
                    "Input Source", "Predicted Crashes", "Change vs Prev Year", "Notes"
                ]
                disp["AADT (vehicles/day)"] = disp["AADT (vehicles/day)"].map("{:,.0f}".format)
                disp["Prev-Year Crashes Used"] = disp["Prev-Year Crashes Used"].map("{:,.0f}".format)
                disp["Predicted Crashes"] = disp["Predicted Crashes"].map("{:,.0f}".format)
                disp["Change vs Prev Year"] = disp["Change vs Prev Year"].map("{:+,.0f}".format)
                st.dataframe(disp, hide_index=True)

                st.download_button(
                    "Download forecast CSV",
                    fc_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"forecast_{int(start_year) + 1}_{int(end_year)}.csv",
                    mime="text/csv",
                    key="btn_dl_rec",
                )


with tab_csv:
    st.subheader("Predict for many road-years")
    st.write("Upload a CSV with columns `aadt`, `prev_crashes`, `lane_km`, `year` "
             "(an optional `road` column is kept in the output).")
    template = pd.DataFrame({"road": ["EXAMPLE ROAD"], "aadt": [150000], "prev_crashes": [2000],
                             "lane_km": [100.0], "year": [2025]})
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="template.csv",
        mime="text/csv",
        key="btn_dl_template",
    )
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            if raw.empty:
                raise ValueError("The uploaded file has no rows.")
            work, problems = I.validate_frame(raw, ranges)
            res = I.predict_frame(model, work, problems)
            bad = sum(any(m.startswith("ERROR") for m in v) for v in problems.values())
            if bad < len(res):
                st.success(f"{len(res) - bad} of {len(res)} rows predicted; {bad} rejected.")
            else:
                st.error("No valid rows to predict.")
            display_res = res
            try:
                st.dataframe(display_res, hide_index=True)
            except Exception as e:
                raise RuntimeError(f"Could not render prediction table: {e}") from e
            try:
                st.download_button(
                    "Download results",
                    display_res.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                    key="btn_dl_batch",
                )
            except Exception as e:
                raise RuntimeError(f"Could not prepare prediction download: {e}") from e
        except ValueError as e:
            st.error(str(e))
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            st.error(f"The CSV could not be read: {e}")
        except Exception as e:
            st.error(f"The CSV could not be processed: {type(e).__name__}: {e}")

with tab_about:
    st.subheader("How the model was chosen")
    st.write(f"Three algorithms were compared: Linear Regression, Random Forest and Gradient Boosting. "
             f"**{meta['model_name']}** had the lowest validation MAE (primary metric) and was scored "
             f"once on the untouched 2023-2024 test years.")
    val = pd.DataFrame(meta["validation"])[["model", "CV_MAE_mean", "CV_MAE_std", "CV_RMSE_mean", "CV_R2_mean"]]
    st.dataframe(val.round(2), hide_index=True)
    ft = meta["final_test_metrics"]
    st.write(f"Final test: MAE {ft['MAE']:,.0f} | RMSE {ft['RMSE']:,.0f} | R² {ft['R2']:.3f} | "
             f"MAPE {ft['MAPE_%']:.1f}%")
    nb = meta["naive_baseline_test_metrics"]
    st.info(f"Honest reference: simply repeating last year's count scored MAE {nb['MAE']:,.0f} on the "
            f"same test years, so the model does not beat that simple baseline.")
    st.subheader("Inputs")
    st.table(pd.DataFrame({"Feature": [C.FEATURE_LABELS[f] for f in C.FEATURES],
                           "Training range": [f"{ranges[f][0]:,.0f} to {ranges[f][1]:,.0f}" for f in C.FEATURES]}))
    st.subheader("Limitations")
    st.markdown("- Yearly road totals only; not a real-time or per-incident predictor.\n"
                "- Trained on 14 roads; tree models cannot extrapolate beyond the training range "
                "(e.g. a road far larger than EDSA).\n"
                "- Excludes weather, driver behaviour and road geometry.")
