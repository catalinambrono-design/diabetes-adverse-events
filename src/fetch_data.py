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


QUARTERS = [(y, m) for y in range(2020, 2025) for m in (1, 4, 7, 10)]
PER_QUARTER = 250   # 20 quarters x 250 = 5,000 reports per drug


def quarter_window(year: int, start_month: int) -> str:
    end_month = start_month + 2
    end_day = {3: 31, 6: 30, 9: 30, 12: 31}[end_month]
    return f"[{year}{start_month:02d}01 TO {year}{end_month:02d}{end_day}]"


def fetch_quarter(drug: str, window: str) -> tuple[list, int]:
    """Most recent PER_QUARTER reports inside one quarter, plus the quarter's total."""
    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug}" AND receivedate:{window}',
        "limit": PER_QUARTER,
        "sort": "receivedate:desc",
    }
    if API_KEY:
        params["api_key"] = API_KEY
    while True:
        response = requests.get(BASE_URL, params=params, timeout=60)
        if response.status_code == 404:
            return [], 0
        if response.status_code == 429:
            print("  Rate limit reached, waiting 60 s...")
            time.sleep(60)
            continue
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", []), payload["meta"]["results"]["total"]


def fetch_drug(drug: str) -> dict:
    records, total = [], 0
    for year, month in QUARTERS:
        batch, n_total = fetch_quarter(drug, quarter_window(year, month))
        records.extend(batch)
        total += n_total
        time.sleep(0.5)
    print(f"  {drug}: {len(records):,} reports downloaded across {len(QUARTERS)} quarters")
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