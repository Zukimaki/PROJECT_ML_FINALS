# Traffic Incident Prediction for NCR Roads

Predicts the yearly number of vehicular crashes on a Metro Manila road from
traffic volume (MMDA AADT) and past crashes (MMARAS). Traditional ML only
(Linear Regression, Random Forest, Gradient Boosting).

## Folder structure
- `data/original/` – source workbook and `source_data_wide.csv`
- `data/processed/ncr_road_crashes_clean.csv` – clean CSV used by all code and the app
- `src/` – `config.py`, `data_prep.py`, `eda.py`, `modeling.py`, `train.py`, `inference.py`
- `app/streamlit_app.py` – Streamlit app
- `models/` – `best_model.joblib`, `model_metadata.json`
- `documentation/` – figures and result tables
- `notebooks/`, `paper/`

## Setup
```
pip install -r requirements.txt
```
Tested with Python 3.14.

## Run order (from the project root)
```
python src/data_prep.py     # workbook -> clean CSV (with verification)
python src/eda.py           # EDA figures
python src/train.py         # tune, evaluate, save the model
streamlit run app/streamlit_app.py
```

## Method
- Usable rows: 2016-2024 (2015 has no previous-year crashes).
- Chronological split: train 2016-2022, test 2023-2024 (final test used once).
- Tuning: GridSearchCV with 5 expanding-window year folds, MAE.
- Supplementary check: train on 13 roads, test on unseen road EDSA.

## Results (test 2023-2024)
Selected: Gradient Boosting, MAE 405.5, RMSE 526.2, R2 0.962.
A naive "same as last year" baseline scores MAE 305.6, so the model does not
beat it. On EDSA (unseen road) Linear Regression is best (MAE 2182, R2 0.20);
tree models cannot extrapolate to EDSA's much larger values.

## App input
- **Predict crashes:** start year (last year with known crashes), end year, crashes in the start year, AADT, lane-km. If the end year is one year after the start year, it gives a single prediction. If it is further away (e.g. 2024 to 2026), the app automatically forecasts recursively: each predicted year is fed as `prev_crashes` for the next, with an optional annual traffic change assumption. When forecast AADT goes outside the training range, the app warns that the estimates are less reliable; above the training maximum, the tree-based model may level off even as traffic continues to grow. Per-year notes are included in the table and downloaded CSV.
- **Batch prediction:** CSV with columns `aadt, prev_crashes, lane_km, year`. Invalid values (negative, missing, text) give an error; values outside the training range give a warning.

## Limitations
- Only 14 roads and 126 usable rows. Road length/lanes are group-compiled; their source must be documented.
- Predictions are less reliable for roads far larger than those seen in training.
- The saved model was trained on 2016-2022 only.

## Data source
MMDA AADT and MMARAS records (MMDA FOI). Confirm permission/licence before submission.
