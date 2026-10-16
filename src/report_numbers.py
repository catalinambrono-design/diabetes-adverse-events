"""Step 8 - Print every number needed to update the README (no figures)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import AGE_LABELS, CONC_LABELS, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR
from model import (CATEGORICAL, NUMERIC, TARGET, add_reaction_features,
                   build_models, make_preprocessor, pretty, select_reaction_terms)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

reports = pd.read_csv(PROCESSED_DIR / "reports.csv")
reactions = pd.read_csv(PROCESSED_DIR / "reactions.csv")
for df in (reports, reactions):
    df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_LABELS, ordered=True)
reports["concomitant_group"] = pd.Categorical(
    reports["concomitant_group"].astype(str), categories=CONC_LABELS, ordered=True)
classes = sorted(reports["drug_class"].unique())


def title(text: str) -> None:
    print(f"\n=== {text} ===")


def rate(df: pd.DataFrame, by) -> pd.DataFrame:
    """Hospitalisation rate (%), n and approximate 95% CI half-width."""
    g = df.groupby(by, observed=True)["hospitalized"]
    out = pd.DataFrame({"n": g.size(), "hosp_%": (g.mean() * 100).round(1)})
    p = out["hosp_%"] / 100
    out["+-95%CI"] = (196 * np.sqrt(p * (1 - p) / out["n"])).round(1)
    return out


def reaction_pct(group_col: str, order: list, top_n: int = 15) -> pd.DataFrame:
    sizes = reports.groupby(group_col, observed=True).size().reindex(order)
    rx = reactions.dropna(subset=[group_col])
    top = rx["reaction"].value_counts().head(top_n).index
    pct = (rx[rx["reaction"].isin(top)]
           .groupby(["reaction", group_col], observed=True)["safetyreportid"].nunique()
           .unstack(group_col).reindex(index=top, columns=order).fillna(0)
           .div(sizes, axis=1) * 100)
    return pct.round(1)


title("1. MISSINGNESS (% missing)")
print({"age": round(reports["age_years"].isna().mean() * 100),
       "sex unknown": round((reports["sex"] == "Unknown").mean() * 100),
       "weight": round(reports["weight_kg"].isna().mean() * 100),
       "dose": round(reports["target_dose_mg"].isna().mean() * 100)})
print("\nHospitalisation rate by whether age / weight is recorded:")
print(rate(reports.assign(age_recorded=reports["age_years"].notna()), "age_recorded"))
print(rate(reports.assign(weight_recorded=reports["weight_kg"].notna()), "weight_recorded"))

title("2. OVERALL RATE AND CLASS SIZE SHARE")
print(f"Overall hospitalisation: {reports['hospitalized'].mean() * 100:.1f}%")
print((reports["drug_class"].value_counts(normalize=True) * 100).round(1))

title("3. HOSPITALISATION BY POLYPHARMACY (concomitant drugs)")
print(rate(reports, "concomitant_group"))

title("4. HOSPITALISATION BY AGE GROUP")
adults = reports.dropna(subset=["age_group"])
print(rate(adults, "age_group"))
by_ac = rate(adults, ["drug_class", "age_group"])
print("\nBy class x age group (rate %):")
print(by_ac["hosp_%"].unstack("age_group"))
print("\nBy class x age group (n):")
print(by_ac["n"].unstack("age_group"))

title("5. HOSPITALISATION BY REPORTER TYPE AND BY US ORIGIN")
print(rate(reports, "reporter_type"))
print(rate(reports, "is_us"))

title("6. TOP 15 REACTIONS BY DRUG CLASS (% of reports in class)")
print(reaction_pct("drug_class", classes).to_string())

title("7. TOP 5 REACTIONS WITHIN EACH CLASS (% of reports in class)")
n_class = reports["drug_class"].value_counts()
for c in classes:
    top5 = (reactions[reactions["drug_class"] == c].groupby("reaction")["safetyreportid"]
            .nunique().sort_values(ascending=False).head(5))
    print(f"\n{c}")
    print((top5 / n_class[c] * 100).round(1).to_string())

title("8. TOP 15 REACTIONS BY AGE GROUP (% of reports in group)")
print(reaction_pct("age_group", AGE_LABELS).to_string())

title("9. SPEARMAN CORRELATIONS")
cols = ["age_years", "weight_kg", "n_drugs", "n_concomitant", "n_reactions",
        "serious", "hospitalized", "death"]
print(reports[cols].corr(method="spearman").round(2))

title("10. DOSE (mg) BY CLASS AND HOSPITALISATION")
dose = reports.dropna(subset=["target_dose_mg"])
print(dose.groupby(["drug_class", "hospitalized"])["target_dose_mg"].agg(
    n="count", q25=lambda s: s.quantile(0.25), median="median",
    q75=lambda s: s.quantile(0.75)).round(2))

title("11. CONFUSION MATRICES AND LOGISTIC REGRESSION COEFFICIENTS (same split as model.py)")
train_df, test_df = train_test_split(reports, test_size=0.2, stratify=reports[TARGET],
                                     random_state=RANDOM_STATE)
terms = select_reaction_terms(train_df)
train_df = add_reaction_features(train_df, terms)
test_df = add_reaction_features(test_df, terms)
rx_cols = [f"rx_{t}" for t in terms]
features = NUMERIC + CATEGORICAL + rx_cols
for name, model in build_models().items():
    pipe = Pipeline([("pre", make_preprocessor(rx_cols)), ("clf", model)])
    pipe.fit(train_df[features], train_df[TARGET])
    pred = (pipe.predict_proba(test_df[features])[:, 1] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_df[TARGET], pred).ravel()
    print(f"\n{name}: TN={tn} FP={fp} FN={fn} TP={tp} "
          f"(test hospitalisations: {tp + fn}, test reports: {len(test_df)})")
    if name == "Logistic Regression":
        names = [pretty(n) for n in pipe.named_steps["pre"].get_feature_names_out()]
        coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values()
        print("Largest positive coefficients:")
        print(coefs.tail(10).round(2).to_string())
        print("Largest negative coefficients:")
        print(coefs.head(6).round(2).to_string())

title("12. RANDOM FOREST PERMUTATION IMPORTANCE (top 12)")
print(pd.read_csv(RESULTS_DIR / "rf_permutation_importance.csv", index_col=0).head(12).round(4))