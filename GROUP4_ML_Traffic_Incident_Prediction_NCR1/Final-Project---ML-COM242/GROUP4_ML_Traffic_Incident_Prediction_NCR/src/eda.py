"""Exploratory data analysis: figures and a numeric summary used by the paper.

Run:  python src/eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import config as C  # noqa: E402
import modeling as M  # noqa: E402

sns.set_theme(style="whitegrid", context="paper")


def vif(frame: pd.DataFrame) -> pd.Series:
    """Variance inflation factor computed from R^2 of each column on the others."""
    out = {}
    Z = (frame - frame.mean()) / frame.std(ddof=0)
    for c in Z.columns:
        X = np.column_stack([np.ones(len(Z)), Z.drop(columns=c).to_numpy()])
        beta, *_ = np.linalg.lstsq(X, Z[c].to_numpy(), rcond=None)
        resid = Z[c].to_numpy() - X @ beta
        r2 = 1 - resid.var() / Z[c].var(ddof=0)
        out[c] = 1 / (1 - r2)
    return pd.Series(out)


def short(road: str) -> str:
    return road.replace("C.P. GARCIA /KATIPUNAN AVE. / TANDANG SORA", "C.P. Garcia/Katipunan") \
               .replace("R-6 MAGSAYSAY / AURORA BLVD.", "R-6 Magsaysay/Aurora").title()


def main() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = C.DOC_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = M.load_clean()
    use = M.usable_rows(df)
    lines = []
    P = lines.append

    P("== DATASET OVERVIEW ==")
    P(f"Shape: {df.shape}; roads: {df['road'].nunique()}; years: {df['year'].min()}-{df['year'].max()}")
    P("Dtypes:\n" + df.dtypes.to_string())
    P("Summary statistics:\n" + df.describe().T.round(2).to_string())
    P("\n== DATA QUALITY ==")
    P("Missing values per column:\n" + df.isna().sum().to_string())
    P(f"Duplicate road-year rows: {df.duplicated(['road', 'year']).sum()}")
    P(f"Rows with non-positive AADT or crashes: {((df['aadt'] <= 0) | (df['crashes'] <= 0)).sum()}")
    P(f"Rows usable for modelling (prev_crashes known): {len(use)}")

    # Crashes by road
    tot = df.groupby("road")["crashes"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([short(r) for r in tot.index], tot.values,
            color=["#c0392b" if r == C.TEST_ROAD else "#4c72b0" for r in tot.index])
    ax.set(xlabel="Total crashes 2015-2024", title="Crashes by road (EDSA highlighted)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_1_crashes_by_road.png", dpi=200)
    plt.close(fig)
    share = tot[C.TEST_ROAD] / tot.sum() * 100
    P(f"\nEDSA share of all crashes: {share:.1f}%")

    # Yearly trend
    yr = df.assign(group=np.where(df["road"] == C.TEST_ROAD, "EDSA", "Other 13 roads")) \
           .groupby(["year", "group"])["crashes"].sum().unstack()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    yr.plot(marker="o", ax=ax)
    ax.axvspan(2019.5, 2021.5, color="grey", alpha=0.15, label="Pandemic years")
    ax.set(ylabel="Crashes per year", title="Yearly crashes: EDSA vs other roads")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_2_yearly_trend.png", dpi=200)
    plt.close(fig)
    P("\nTotal crashes per year (all roads):\n" + df.groupby("year")["crashes"].sum().to_string())

    # AADT vs crashes
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.scatterplot(data=df, x="aadt", y="crashes", hue=df["road"].map(short),
                    legend=False, ax=ax, s=28)
    ax.set(xscale="log", yscale="log", xlabel="AADT (vehicles/day, log)",
           ylabel="Crashes (log)", title="AADT vs crashes, one point per road-year")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_3_aadt_vs_crashes.png", dpi=200)
    plt.close(fig)
    P(f"\nCorrelation AADT vs crashes (all rows): pearson {df['aadt'].corr(df['crashes']):.3f}, "
      f"spearman {df['aadt'].corr(df['crashes'], method='spearman'):.3f}")
    ne = df[df['road'] != C.TEST_ROAD]
    P(f"Excluding EDSA: pearson {ne['aadt'].corr(ne['crashes']):.3f}, "
      f"spearman {ne['aadt'].corr(ne['crashes'], method='spearman'):.3f}")

    # Correlation heatmap
    cols = ["crashes"] + C.FEATURES
    corr = use[cols].corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation of features and target")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_4_correlation.png", dpi=200)
    plt.close(fig)
    P("\nCorrelation matrix:\n" + corr.round(3).to_string())
    P("\nVariance inflation factors (numeric features):\n" + vif(use[C.FEATURES]).round(2).to_string())

    # Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    sns.histplot(use["crashes"], bins=30, ax=axes[0])
    axes[0].set(title="Crashes per road-year", xlabel="Crashes")
    sns.histplot(np.log10(use["crashes"]), bins=30, ax=axes[1])
    axes[1].set(title="log10(crashes)", xlabel="log10 crashes")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_5_target_distribution.png", dpi=200)
    plt.close(fig)
    P(f"\nTarget skewness: raw {use['crashes'].skew():.2f}, log10 {np.log10(use['crashes']).skew():.2f}")

    # Outliers by road
    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = df.groupby("road")["crashes"].median().sort_values().index
    sns.boxplot(data=df, y=df["road"].map(short), x="crashes", order=[short(r) for r in order],
                ax=ax, color="#9ecae1")
    ax.set(xscale="log", xlabel="Crashes (log scale)", ylabel="", title="Crash counts by road")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_6_crashes_boxplot.png", dpi=200)
    plt.close(fig)
    q1, q3 = use["crashes"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outl = use[use["crashes"] > q3 + 1.5 * iqr]
    P(f"\nIQR outliers in crashes: {len(outl)} rows; roads: {sorted(outl['road'].unique())}")
    P(f"EDSA rows among outliers: {(outl['road'] == C.TEST_ROAD).sum()}")

    # Previous vs current crashes
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.scatterplot(data=use, x="prev_crashes", y="crashes", hue="pandemic", ax=ax, s=28)
    lim = [use["prev_crashes"].min(), use["prev_crashes"].max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xscale="log", yscale="log", title="Crashes vs previous-year crashes")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_7_prev_vs_current.png", dpi=200)
    plt.close(fig)
    P(f"Correlation prev_crashes vs crashes: {use['prev_crashes'].corr(use['crashes']):.3f}")
    ratio = (use["crashes"] / use["prev_crashes"])
    P(f"Year-over-year ratio: median {ratio.median():.2f}, 5th-95th pct "
      f"{ratio.quantile(.05):.2f}-{ratio.quantile(.95):.2f}")
    pand = df[df["year"].isin([2019, 2020, 2021])].groupby("year")["crashes"].sum()
    P("Pandemic effect (total crashes):\n" + pand.to_string())

    (out_dir / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
