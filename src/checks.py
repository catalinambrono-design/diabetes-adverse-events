"""
Step 7 - Robustness checks.

Check 1: how are the reports distributed over time?
Check 2: ablation study - how much of the model's performance comes from
         clinical content versus reporting characteristics?

Outputs:
    figures/13_reports_by_year.png
    figures/14_ablation.png
    results/year_distribution.csv
    results/ablation.csv
"""
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FIG_DIR, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR
from model import TARGET, add_reaction_features, build_models, select_reaction_terms

sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# Check 1: distribution of reports over time
# --------------------------------------------------------------------------
def check_years(df: pd.DataFrame) -> None:
    df = df.copy()
    df["receivedate"] = pd.to_datetime(df["receivedate"], errors="coerce")
    df["year"] = df["year"].astype("Int64")

    by_year = pd.crosstab(df["year"], df["drug_class"], margins=True, margins_name="Total")
    by_year["% of reports"] = (by_year["Total"] / len(df) * 100).round(1)
    by_year.to_csv(RESULTS_DIR / "year_distribution.csv")

    print("\n=== CHECK 1: reports by year and drug class ===")
    print(by_year.to_string())

    span = df.groupby("drug_class")["receivedate"].agg(earliest="min", latest="max")
    span["days_covered"] = (span["latest"] - span["earliest"]).dt.days
    print("\nTime window actually covered by each drug class:")
    print(span.to_string())

    hosp = df.groupby("year")["hospitalized"].agg(
        reports="count", hospitalised_pct=lambda s: round(s.mean() * 100, 1))
    print("\nHospitalisation rate by year:")
    print(hosp.to_string())

    ax = (pd.crosstab(df["year"], df["drug_class"])
          .plot(kind="bar", stacked=True, figsize=(9, 5), rot=0))
    ax.set_xlabel("Year report was received by the FDA")
    ax.set_ylabel("Number of reports in final dataset")
    ax.set_title("Reports by year and drug class")
    ax.legend(title="Drug class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_reports_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/13_reports_by_year.png")


# --------------------------------------------------------------------------
# Check 2: ablation study
# --------------------------------------------------------------------------
ALL_NUMERIC = ["age_years", "weight_kg", "n_concomitant", "n_reactions", "year", "is_us"]
ALL_CATEGORICAL = ["sex", "reporter_type", "drug_class"]

SCENARIOS = {
    "1. Full model (as in model.py)": dict(
        numeric=ALL_NUMERIC, categorical=ALL_CATEGORICAL, reactions=True, indicator=True),
    "2. Full model, no missing-value indicators": dict(
        numeric=ALL_NUMERIC, categorical=ALL_CATEGORICAL, reactions=True, indicator=False),
    "3. Without reporter type, US origin and year": dict(
        numeric=["age_years", "weight_kg", "n_concomitant", "n_reactions"],
        categorical=["sex", "drug_class"], reactions=True, indicator=True),
    "4. Clinical content only (reactions, counts, drug class)": dict(
        numeric=["n_concomitant", "n_reactions"],
        categorical=["drug_class"], reactions=True, indicator=True),
    "5. Non-clinical only (reporter, origin, year, age, weight, sex)": dict(
        numeric=["age_years", "weight_kg", "year", "is_us"],
        categorical=["sex", "reporter_type"], reactions=False, indicator=True),
}


def make_pipeline(cfg: dict, rx_cols: list[str], model) -> Pipeline:
    transformers = []
    if cfg["numeric"]:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=cfg["indicator"])),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, cfg["numeric"]))
    if cfg["categorical"]:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cfg["categorical"]))
    if cfg["reactions"]:
        transformers.append(("rx", "passthrough", rx_cols))
    return Pipeline([("pre", ColumnTransformer(transformers, sparse_threshold=0)),
                     ("clf", model)])


def run_ablation(df: pd.DataFrame) -> None:
    # Same split and reaction selection as model.py, so results are comparable
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET],
                                         random_state=RANDOM_STATE)
    terms = select_reaction_terms(train_df)
    train_df = add_reaction_features(train_df, terms)
    test_df = add_reaction_features(test_df, terms)
    rx_cols = [f"rx_{t}" for t in terms]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    print("\n=== CHECK 2: ablation study (this takes a few minutes) ===")
    for scenario, cfg in SCENARIOS.items():
        cols = cfg["numeric"] + cfg["categorical"] + (rx_cols if cfg["reactions"] else [])
        for model_name, model in build_models().items():
            pipe = make_pipeline(cfg, rx_cols, model)
            cv_auc = cross_val_score(pipe, train_df[cols], train_df[TARGET],
                                     cv=cv, scoring="roc_auc")
            pipe.fit(train_df[cols], train_df[TARGET])
            proba = pipe.predict_proba(test_df[cols])[:, 1]
            rows.append({
                "scenario": scenario,
                "model": model_name,
                "n_features": len(cols),
                "cv_roc_auc": round(float(cv_auc.mean()), 3),
                "cv_std": round(float(cv_auc.std()), 3),
                "test_roc_auc": round(float(roc_auc_score(test_df[TARGET], proba)), 3),
                "test_pr_auc": round(float(average_precision_score(test_df[TARGET], proba)), 3),
            })
            print(f"  {scenario[:55]:<55} | {model_name:<19} | "
                  f"test ROC-AUC {rows[-1]['test_roc_auc']:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "ablation.csv", index=False)

    print("\nAblation results:")
    print(results.to_string(index=False))
    print(f"\nBaseline PR-AUC (random classifier) = {test_df[TARGET].mean():.3f}")

    results["label"] = results["scenario"].apply(lambda s: textwrap.fill(s, 28))
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=results, x="label", y="test_roc_auc", hue="model", ax=ax)
    ax.axhline(0.5, color="black", ls="--", lw=0.8, label="Random (0.50)")
    ax.set_ylim(0.4, 0.9)
    ax.set_xlabel("")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Ablation: how much performance comes from each group of features?")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/14_ablation.png")


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "reports.csv")
    print(f"Loaded {len(df):,} reports.")
    check_years(df)
    run_ablation(df)
    print("\nDone.")


if __name__ == "__main__":
    main()