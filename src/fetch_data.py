"""
Step 1 - Download adverse event reports from the openFDA Drug Adverse Event API.

Docs: https://open.fda.gov/apis/drug/event/
An API key is optional but raises the daily request limit.
Set it as an environment variable named OPENFDA_API_KEY.
"""
import json
import os
import time

import requests

from config import DATE_RANGE, DRUGS, MAX_RECORDS_PER_DRUG, PAGE_SIZE, RAW_DIR

BASE_URL = "https://api.fda.gov/drug/event.json"
API_KEY = os.getenv("OPENFDA_API_KEY")


def build_query(drug: str) -> str:
    """Search reports mentioning the drug, received within the date window."""
    return f'patient.drug.openfda.generic_name:"{drug}" AND receivedate:{DATE_RANGE}'


def fetch_drug(drug: str) -> dict:
    """Page through the API and return the raw reports plus the total match count."""
    records, skip, total = [], 0, 0

    while skip < MAX_RECORDS_PER_DRUG:
        params = {
            "search": build_query(drug),
            "limit": PAGE_SIZE,
            "skip": skip,
            "sort": "receivedate:desc",   # deterministic order: most recent first
        }
        if API_KEY:
            params["api_key"] = API_KEY

        response = requests.get(BASE_URL, params=params, timeout=60)

        if response.status_code == 404:          # no (more) matching results
            break
        if response.status_code == 429:          # rate limited: wait and retry
            print("  Rate limit reached, waiting 60 s...")
            time.sleep(60)
            continue
        response.raise_for_status()

        payload = response.json()
        batch = payload.get("results", [])
        total = payload["meta"]["results"]["total"]
        records.extend(batch)
        print(f"  {drug}: {len(records):,} / {min(total, MAX_RECORDS_PER_DRUG):,}")

        skip += PAGE_SIZE
        if not batch or skip >= total:
            break
        time.sleep(0.5)                          # be polite to the API

    return {"meta_total": total, "results": records}


def main() -> None:
    for drug in DRUGS:
        print(f"Fetching reports for {drug}...")
        data = fetch_drug(drug)
        out_path = RAW_DIR / f"{drug}.json"
        out_path.write_text(json.dumps(data), encoding="utf-8")
        print(f"  Saved {len(data['results']):,} reports "
              f"(of {data['meta_total']:,} matching) -> {out_path.name}\n")


if __name__ == "__main__":
    main()