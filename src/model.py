"""
Step 4 - Predict whether a reported adverse event involved hospitalisation.

Models:
    - Logistic Regression (interpretable baseline)
    - Random Forest (non-linear model)

Inputs:
    data/processed/reports.csv
Outputs:
    figures/09_roc_pr_curves.png
    figures/10_confusion_matrices.png
    figures/11_lr_coefficients.png
    figures/12_rf_permutation_importance.png
    results/metrics.json
    results/rf_permutation_importance.csv
"""
import json
import matplotlib
matplotlib.use('Agg')  # Împiedică încărcarea interfeței Tkinter/GUI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FIG_DIR, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR

sns.set_theme(style="whitegrid")

TARGET = "hospitalized"
TOP_K_REACTIONS = 30
NUMERIC = ["age_years", "weight_kg", "n_concomitant", "n_reactions", "year", "is_us"]
CATEGORICAL = ["sex", "reporter_type", "drug_class"]

# Reaction terms that literally describe the outcome -> would leak the target.
# Other seriousness flags (serious, death, etc.) are also never used as features.
LEAKAGE_TERMS = {"Death", "Hospitalisation", "Hospitalization"}


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def select_reaction_terms(train_df: pd.DataFrame, k: int = TOP_K_REACTIONS) -> list[str]:
    """Pick the k most frequent reactions using the TRAINING set only."""
    terms = train_df["reactions"].fillna("").str.split("|").explode()
    terms = terms[(terms != "") & (~terms.isin(LEAKAGE_TERMS))]
    return terms.value_counts().head(k).index.tolist()


def add_reaction_features(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Add one binary column per reaction term (multi-hot encoding)."""
    reaction_sets = df["reactions"].fillna("").str.split("|").apply(set)
    features = pd.DataFrame(
        {f"rx_{t}": reaction_sets.apply(lambda s, t=t: int(t in s)) for t in terms},
        index=df.index,
    )
    return pd.concat([df, features], axis=1)


def make_preprocessor(rx_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
         ("rx", "passthrough", rx_cols)],
        sparse_threshold=0,
    )


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
    }


def pretty(name: str) -> str:
    """Readable feature names for plots."""
    name = name.split("__", 1)[-1]
    if name.startswith("rx_"):
        return "Reaction: " + name[3:]
    return name.replace("missingindicator_", "Missing: ")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_roc_pr(y_test, probas: dict, prevalence: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, proba in probas.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC = 0.50)")
    axes[1].axhline(prevalence, color="k", ls="--", lw=0.8,
                    label=f"Random (AP = {prevalence:.2f})")
    axes[0].set_title("ROC curve (test set)")
    axes[1].set_title("Precision-Recall curve (test set)")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figures/09_roc_pr_curves.png")


def plot_confusion_matrices(y_test, preds: dict) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 5))
    for ax, (name, pred) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Not hospitalised", "Hospitalised"],
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(f"{name} (threshold = 0.5)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figures/10_confusion_matrices.png")


def plot_lr_coefficients(pipe: Pipeline, top_n: int = 15) -> None:
    names = [pretty(n) for n in pipe.named_steps["pre"].get_feature_names_out()]
    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=names)
    top = coefs.loc[coefs.abs().sort_values(ascending=False).index[:top_n]].sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot.barh(ax=ax, color=["indianred" if c > 0 else "steelblue" for c in top])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (log-odds; numeric features standardised)")
    ax.set_title("Logistic Regression - strongest associations\n"
                 "(red = higher odds of hospitalisation, blue = lower)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_lr_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figures/11_lr_coefficients.png")


def plot_permutation_importance(pipe: Pipeline, X_test, y_test, top_n: int = 15) -> pd.DataFrame:
    print("  Computing permutation importance (this may take a minute)...")
    result = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                    n_repeats=5, random_state=RANDOM_STATE)
    importance = (pd.DataFrame({"importance_mean": result.importances_mean,
                                "importance_std": result.importances_std},
                               index=X_test.columns)
                  .sort_values("importance_mean", ascending=False))
    importance.to_csv(RESULTS_DIR / "rf_permutation_importance.csv")

    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([pretty(n) for n in top.index], top["importance_mean"],
            xerr=top["importance_std"], color="seagreen")
    ax.set_xlabel("Mean decrease in ROC-AUC when feature is shuffled")
    ax.set_title("Random Forest - permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_rf_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved figures/12_rf_permutation_importance.png")
    return importance


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "reports.csv")
    print(f"Loaded {len(df):,} reports. Hospitalisation rate: {df[TARGET].mean():.1%}\n")

    # 1. Split FIRST, then build reaction features from the training set only
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET],
                                         random_state=RANDOM_STATE)
    terms = select_reaction_terms(train_df)
    train_df = add_reaction_features(train_df, terms)
    test_df = add_reaction_features(test_df, terms)
    rx_cols = [f"rx_{t}" for t in terms]
    features = NUMERIC + CATEGORICAL + rx_cols

    X_train, y_train = train_df[features], train_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]
    prevalence = float(y_test.mean())

    # 2. Train and evaluate each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, probas, preds = {}, {}, {}
    metrics = {"n_train": len(X_train), "n_test": len(X_test),
               "test_prevalence": round(prevalence, 4),
               "reaction_features": terms, "models": {}}

    for name, model in build_models().items():
        print(f"Training {name}...")
        pipe = Pipeline([("pre", make_preprocessor(rx_cols)), ("clf", model)])

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fitted[name], probas[name], preds[name] = pipe, proba, pred
        metrics["models"][name] = {
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
            "test_precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        }
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(classification_report(y_test, pred, zero_division=0,
                                    target_names=["Not hospitalised", "Hospitalised"]))

    # 3. Figures
    print("Generating figures...")
    plot_roc_pr(y_test, probas, prevalence)
    plot_confusion_matrices(y_test, preds)
    plot_lr_coefficients(fitted["Logistic Regression"])
    plot_permutation_importance(fitted["Random Forest"], X_test, y_test)

    # 4. Save metrics
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSummary (test set):")
    print(pd.DataFrame(metrics["models"]).T.to_string())
    print(f"\nBaseline PR-AUC (random classifier) = prevalence = {prevalence:.3f}")
    print("Saved figures 09-12 and results/metrics.json")


if __name__ == "__main__":
    main()