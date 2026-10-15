"""
Step 2 - Flatten the nested openFDA JSON into analysis-ready tables.

Outputs:
    data/processed/reports.csv    one row per report
    data/processed/reactions.csv  one row per (report, reaction)
"""
import json

import numpy as np
import pandas as pd

from config import (AGE_BINS, AGE_LABELS, CONC_BINS, CONC_LABELS, DRUGS,
                    PROCESSED_DIR, RAW_DIR)

# openFDA code lists (see the openFDA field reference)
AGE_UNIT_TO_YEARS = {"800": 10, "801": 1, "802": 1 / 12,
                     "803": 1 / 52, "804": 1 / 365, "805": 1 / 8760}
SEX_MAP = {"1": "Male", "2": "Female"}
REPORTER_MAP = {"1": "Healthcare professional",   # physician
                "2": "Healthcare professional",   # pharmacist
                "3": "Healthcare professional",   # other health professional
                "4": "Other",                     # lawyer
                "5": "Consumer"}
SUSPECT, CONCOMITANT = "1", "2"
DOSE_UNIT_MG = "003"


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def age_in_years(patient: dict) -> float:
    age = to_float(patient.get("patientonsetage"))
    factor = AGE_UNIT_TO_YEARS.get(str(patient.get("patientonsetageunit")))
    if np.isnan(age) or factor is None:
        return np.nan
    years = age * factor
    return years if 0 <= years <= 110 else np.nan


def is_target(drug_entry: dict, target: str) -> bool:
    """Match on the harmonised openFDA generic name or the reported product name."""
    generic = " ".join(drug_entry.get("openfda", {}).get("generic_name", [])).lower()
    product = str(drug_entry.get("medicinalproduct", "")).lower()
    return target in generic or target in product


def parse_report(report: dict, target: str, drug_class: str) -> dict:
    patient = report.get("patient", {}) or {}
    drugs = patient.get("drug", []) or []
    reactions = patient.get("reaction", []) or []
    source = report.get("primarysource") or {}

    target_suspect = [d for d in drugs
                      if is_target(d, target) and d.get("drugcharacterization") == SUSPECT]

    # Structured dose of the suspect target drug, only when reported in mg
    dose_mg = np.nan
    for entry in target_suspect:
        if entry.get("drugstructuredosageunit") == DOSE_UNIT_MG:
            dose_mg = to_float(entry.get("drugstructuredosagenumb"))
            break

    reaction_terms = sorted({r["reactionmeddrapt"].strip().capitalize()
                             for r in reactions if r.get("reactionmeddrapt")})

    weight = to_float(patient.get("patientweight"))
    if not 2 <= weight <= 300:
        weight = np.nan

    return {
        "safetyreportid": report.get("safetyreportid"),
        "target_drug": target,
        "drug_class": drug_class,
        "receivedate": report.get("receivedate"),
        "target_is_suspect": len(target_suspect) > 0,
        "serious": int(report.get("serious") == "1"),
        "hospitalized": int(report.get("seriousnesshospitalization") == "1"),
        "death": int(report.get("seriousnessdeath") == "1"),
        "age_years": age_in_years(patient),
        "sex": SEX_MAP.get(str(patient.get("patientsex")), "Unknown"),
        "weight_kg": weight,
        "reporter_type": REPORTER_MAP.get(str(source.get("qualification")), "Unknown"),
        "country": report.get("occurcountry") or source.get("reportercountry") or "Unknown",
        "n_drugs": len(drugs),
        "n_concomitant": sum(d.get("drugcharacterization") == CONCOMITANT for d in drugs),
        "n_reactions": len(reaction_terms),
        "target_dose_mg": dose_mg,
        "reactions": "|".join(reaction_terms),
    }


def main() -> None:
    rows = []
    for target, drug_class in DRUGS.items():
        path = RAW_DIR / f"{target}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"{target}: {len(data['results']):,} downloaded "
              f"of {data['meta_total']:,} matching reports")
        rows.extend(parse_report(r, target, drug_class) for r in data["results"])

    df = pd.DataFrame(rows)
    log = {"downloaded": len(df)}

    df = df.drop_duplicates(subset=["safetyreportid", "target_drug"])
    log["after_dedup"] = len(df)

    # Keep reports where the study drug is a SUSPECT (not just a concomitant) drug
    df = df[df["target_is_suspect"]]
    log["target_is_suspect"] = len(df)

    # Drop reports where 2+ study drugs are suspect, so classes stay comparable
    df = df[~df["safetyreportid"].duplicated(keep=False)].copy()
    log["single_study_drug"] = len(df)

    df["receivedate"] = pd.to_datetime(df["receivedate"], format="%Y%m%d", errors="coerce")
    df["year"] = df["receivedate"].dt.year
    df["age_group"] = pd.cut(df["age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    df["concomitant_group"] = pd.cut(df["n_concomitant"], bins=CONC_BINS, labels=CONC_LABELS)
    df["is_us"] = (df["country"] == "US").astype(int)
    df = df.drop(columns="target_is_suspect")

    reactions = (df[["safetyreportid", "drug_class", "age_group", "hospitalized", "reactions"]]
                 .assign(reaction=lambda d: d["reactions"].str.split("|"))
                 .explode("reaction")
                 .drop(columns="reactions"))
    reactions = reactions[reactions["reaction"].fillna("") != ""]

    df.to_csv(PROCESSED_DIR / "reports.csv", index=False)
    reactions.to_csv(PROCESSED_DIR / "reactions.csv", index=False)

    print("\nCleaning log:")
    for step, n in log.items():
        print(f"  {step:<20} {n:>7,}")
    print(f"\nSaved {len(df):,} reports and {len(reactions):,} report-reaction rows.")


if __name__ == "__main__":
    main()