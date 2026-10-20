"""Train, compare and evaluate the three regressors; save the selected pipeline.

Run:  python src/train.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

import config as C  # noqa: E402
import modeling as M  # noqa: E402

RESULTS_DIR = C.DOC_DIR / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = M.usable_rows(M.load_clean())
    train, test = M.chronological_split(df)
    print(f"Usable rows {len(df)} | train {len(train)} (2016-{C.LAST_TRAIN_YEAR}) | "
          f"test {len(test)} ({C.LAST_TRAIN_YEAR + 1}-{df['year'].max()})")

    # Main experiment
    table, fitted = M.tune_models(train)
    table = table.sort_values("CV_MAE_mean").reset_index(drop=True)
    table.to_csv(RESULTS_DIR / "validation_comparison.csv", index=False)
    print("\nValidation (expanding-window CV on training years, primary metric = MAE):")
    print(table[["model", "CV_MAE_mean", "CV_MAE_std", "CV_RMSE_mean", "CV_R2_mean",
                 "CV_MAPE_%_mean"]].round(2).to_string(index=False))

    best_name = table.loc[0, "model"]
    best = fitted[best_name]

    # Final test, selected model only
    pred_test = best.predict(test[C.FEATURES])
    test_metrics = M.regression_metrics(test[C.TARGET], pred_test)
    base_metrics = M.regression_metrics(test[C.TARGET], M.naive_baseline(test))
    print(f"\nSelected model: {best_name}  {table.loc[0, 'best_params']}")
    print("Final test (2023-2024):", {k: round(v, 3) for k, v in test_metrics.items()})
    print("Naive baseline (prev year):", {k: round(v, 3) for k, v in base_metrics.items()})

    out = test[["road", "year", C.TARGET]].copy()
    out["predicted"] = pred_test
    out["residual"] = out[C.TARGET] - out["predicted"]
    out.to_csv(RESULTS_DIR / "final_test_predictions.csv", index=False)

    # Unseen-road check (EDSA)
    non_edsa = df[df["road"] != C.TEST_ROAD].reset_index(drop=True)
    edsa = df[df["road"] == C.TEST_ROAD].reset_index(drop=True)
    table_u, fitted_u = M.tune_models(non_edsa)
    rows = []
    for name, mdl in fitted_u.items():
        m = M.regression_metrics(edsa[C.TARGET], mdl.predict(edsa[C.FEATURES]))
        rows.append({"model": name, **m})
    rows.append({"model": "Naive baseline (prev year)",
                 **M.regression_metrics(edsa[C.TARGET], M.naive_baseline(edsa))})
    unseen = pd.DataFrame(rows)
    unseen.to_csv(RESULTS_DIR / "edsa_unseen_road_check.csv", index=False)
    print("\nEDSA unseen-road check (trained on 13 other roads, tested on EDSA 2016-2024):")
    print(unseen.round(2).to_string(index=False))
    edsa_best = fitted_u[best_name].predict(edsa[C.FEATURES])
    pd.DataFrame({"year": edsa["year"], "actual": edsa[C.TARGET],
                  "predicted": edsa_best}).to_csv(RESULTS_DIR / "edsa_predictions.csv", index=False)

    # Figures
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(out[C.TARGET], out["predicted"], s=25, alpha=0.8)
    lim = [0, max(out[C.TARGET].max(), out["predicted"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="Actual crashes", ylabel="Predicted crashes",
           title=f"Final test 2023-2024: {best_name}")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_test_actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(out["predicted"], out["residual"], s=25, alpha=0.8)
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="Predicted crashes", ylabel="Residual (actual - predicted)",
           title="Residuals on the final test set")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_test_residuals.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(edsa["year"], edsa[C.TARGET], "o-", label="Actual")
    ax.plot(edsa["year"], edsa_best, "s--", label=f"Predicted ({best_name})")
    ax.set(xlabel="Year", ylabel="EDSA crashes", title="EDSA unseen-road check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_edsa_check.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(table["model"], table["CV_MAE_mean"], yerr=table["CV_MAE_std"], capsize=4)
    ax.set(ylabel="Validation MAE (crashes)", title="Model comparison (mean ± SD across folds)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_model_comparison.png", dpi=200)
    plt.close(fig)

    # Save model + metadata
    joblib.dump(best, C.MODEL_PATH)
    ranges = {c: [float(train[c].min()), float(train[c].max())] for c in C.FEATURES}
    meta = {
        "model_name": best_name,
        "best_params": table.loc[0, "best_params"],
        "features": C.FEATURES,
        "target": C.TARGET,
        "train_years": [int(train["year"].min()), int(train["year"].max())],
        "test_years": [int(test["year"].min()), int(test["year"].max())],
        "n_train": int(len(train)), "n_test": int(len(test)),
        "primary_metric": "MAE",
        "validation": table.drop(columns="best_params").round(4).to_dict(orient="records"),
        "final_test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "naive_baseline_test_metrics": {k: round(float(v), 4) for k, v in base_metrics.items()},
        "edsa_unseen_road": unseen.round(4).to_dict(orient="records"),
        "training_ranges": ranges,
        "random_state": C.RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
    }
    C.METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved {C.MODEL_PATH.name} and {C.METADATA_PATH.name}")


if __name__ == "__main__":
    main()
