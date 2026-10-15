"""
Step 3 - Exploratory data analysis and visualisations.

Inputs:
    data/processed/reports.csv
    data/processed/reactions.csv
Outputs:
    figures/*.png
    results/summary_by_class.csv
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import AGE_LABELS, CONC_LABELS, FIG_DIR, PROCESSED_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid")
TOP_N = 15  # number of reactions shown in heatmaps


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cleaned tables and restore the ordering of categorical columns."""
    reports = pd.read_csv(PROCESSED_DIR / "reports.csv")
    reactions = pd.read_csv(PROCESSED_DIR / "reactions.csv")

    for df in (reports, reactions):
        df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_LABELS, ordered=True)

    reports["concomitant_group"] = pd.Categorical(
        reports["concomitant_group"].astype(str), categories=CONC_LABELS, ordered=True
    )
    reports["hospitalized_pct"] = reports["hospitalized"] * 100
    return reports, reactions


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figures/{filename}")


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------
def summary_by_class(reports: pd.DataFrame) -> pd.DataFrame:
    summary = (reports.groupby("drug_class")
               .agg(reports=("safetyreportid", "count"),
                    serious_pct=("serious", "mean"),
                    hospitalized_pct=("hospitalized", "mean"),
                    death_pct=("death", "mean"),
                    median_age=("age_years", "median"),
                    female_pct=("sex", lambda s: (s == "Female").mean()),
                    hcp_reporter_pct=("reporter_type",
                                      lambda s: (s == "Healthcare professional").mean()))
               .round(3))
    pct_cols = [c for c in summary.columns if c.endswith("_pct")]
    summary[pct_cols] = (summary[pct_cols] * 100).round(1)
    summary.to_csv(RESULTS_DIR / "summary_by_class.csv")
    return summary


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_missingness(reports: pd.DataFrame) -> None:
    """Data-quality check: how complete are the key fields?"""
    missing = pd.Series({
        "Age": reports["age_years"].isna().mean(),
        "Sex": (reports["sex"] == "Unknown").mean(),
        "Weight": reports["weight_kg"].isna().mean(),
        "Dose (structured, mg)": reports["target_dose_mg"].isna().mean(),
    }).mul(100).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    missing.plot.barh(ax=ax, color="grey")
    ax.set_xlabel("% of reports with missing value")
    ax.set_title("Data completeness of key fields")
    for i, value in enumerate(missing):
        ax.text(value + 1, i, f"{value:.0f}%", va="center")
    ax.set_xlim(0, 110)
    save_figure(fig, "01_missingness.png")


def plot_outcomes_by_class(reports: pd.DataFrame) -> None:
    """Share of reports with serious outcomes, by drug class."""
    rates = (reports.groupby("drug_class")[["serious", "hospitalized", "death"]]
             .mean().mul(100).reset_index()
             .melt(id_vars="drug_class", var_name="outcome", value_name="pct"))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=rates, x="drug_class", y="pct", hue="outcome", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("% of reports")
    ax.set_title("Reported outcomes by drug class")
    ax.legend(title="Outcome")
    save_figure(fig, "02_outcomes_by_class.png")


def plot_reaction_heatmap(reactions: pd.DataFrame, reports: pd.DataFrame,
                          group_col: str, order: list, filename: str, title: str) -> None:
    """Heatmap: % of reports in each group that mention each of the top reactions."""
    group_sizes = reports.groupby(group_col, observed=True).size().reindex(order)
    rx = reactions.dropna(subset=[group_col])
    #top = rx["reaction"].value_counts().head(TOP_N).index
    if group_col == "drug_class":
        # Top 5 reactions within EACH class, so smaller classes are represented fairly
        top = pd.Index(rx.groupby(group_col)["reaction"]
                       .apply(lambda s: s.value_counts().head(5).index.tolist())
                       .explode().drop_duplicates())
    else:
        top = rx["reaction"].value_counts().head(TOP_N).index

    pct = (rx[rx["reaction"].isin(top)]
           .groupby(["reaction", group_col], observed=True)["safetyreportid"].nunique()
           .unstack(group_col)
           .reindex(index=top, columns=order)
           .fillna(0)
           .div(group_sizes, axis=1) * 100)

    #fig, ax = plt.subplots(figsize=(1.6 * len(order) + 4, 0.45 * TOP_N + 2))
    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 4, 0.45 * len(top) + 2))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Reds",
                cbar_kws={"label": "% of reports in group"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    save_figure(fig, filename)


def plot_hospitalization_by_polypharmacy(reports: pd.DataFrame) -> None:
    """Hospitalisation rate by number of concomitant drugs (95% CI)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=reports, x="concomitant_group", y="hospitalized_pct",
                order=CONC_LABELS, errorbar=("ci", 95), color="steelblue", ax=ax)
    counts = reports["concomitant_group"].value_counts().reindex(CONC_LABELS).fillna(0)
    ax.set_xticks(range(len(CONC_LABELS)))
    ax.set_xticklabels([f"{label}\n(n={int(n):,})" for label, n in counts.items()])
    ax.set_xlabel("Number of concomitant drugs reported")
    ax.set_ylabel("% of reports involving hospitalisation")
    ax.set_title("Hospitalisation rate by polypharmacy level")
    save_figure(fig, "05_hospitalization_by_polypharmacy.png")


def plot_hospitalization_by_age_and_class(reports: pd.DataFrame) -> None:
    """Hospitalisation rate by age group, one line per drug class (95% CI)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(data=reports.dropna(subset=["age_group"]),
                  x="age_group", y="hospitalized_pct", hue="drug_class",
                  order=AGE_LABELS, errorbar=("ci", 95), dodge=0.4, ax=ax)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("% of reports involving hospitalisation")
    ax.set_title("Hospitalisation rate by age group and drug class")
    ax.legend(title="Drug class", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, "06_hospitalization_by_age_and_class.png")


def plot_dose_vs_hospitalization(reports: pd.DataFrame) -> None:
    """Reported dose by hospitalisation status, per drug (doses are not comparable across drugs)."""
    coverage = (reports.groupby("drug_class")["target_dose_mg"]
                .apply(lambda s: s.notna().mean() * 100).round(1))
    print("\n  Structured dose (mg) available for % of reports:")
    print(coverage.to_string())

    dose = reports.dropna(subset=["target_dose_mg"])
    if dose.empty:
        print("  No structured dose data available - skipping dose plot.")
        return

    grid = sns.catplot(data=dose, x="hospitalized", y="target_dose_mg", col="drug_class",
                       kind="box", sharey=False, showfliers=False,
                       height=4, aspect=0.8, color="lightsteelblue")
    grid.set_axis_labels("Hospitalised (0 = no, 1 = yes)", "Dose per administration (mg)")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Reported dose by hospitalisation status (outliers hidden)", y=1.04)
    grid.savefig(FIG_DIR / "07_dose_vs_hospitalization.png", dpi=150, bbox_inches="tight")
    plt.close(grid.figure)
    print("  Saved figures/07_dose_vs_hospitalization.png")


def plot_correlations(reports: pd.DataFrame) -> None:
    """Spearman correlations between patient/report features and outcomes."""
    cols = ["age_years", "weight_kg", "n_drugs", "n_concomitant", "n_reactions",
            "serious", "hospitalized", "death"]
    corr = reports[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Spearman correlation matrix")
    save_figure(fig, "08_correlations.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    reports, reactions = load_data()
    classes = sorted(reports["drug_class"].unique())

    print(f"Loaded {len(reports):,} reports and {len(reactions):,} report-reaction rows.\n")
    print("Summary by drug class:")
    print(summary_by_class(reports).to_string(), "\n")

    print("Generating figures...")
    plot_missingness(reports)
    plot_outcomes_by_class(reports)
    plot_reaction_heatmap(reactions, reports, "age_group", AGE_LABELS,
                          "03_top_reactions_by_age.png",
                          f"Top {TOP_N} reported reactions by age group")
    plot_reaction_heatmap(reactions, reports, "drug_class", classes,
                          "04_top_reactions_by_class.png",
                          f"Top {TOP_N} reported reactions by drug class")
    plot_hospitalization_by_polypharmacy(reports)
    plot_hospitalization_by_age_and_class(reports)
    plot_dose_vs_hospitalization(reports)
    plot_correlations(reports)
    print("\nDone.")


if __name__ == "__main__":
    main()