"""Shared configuration for the FAERS analysis project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

for directory in (RAW_DIR, PROCESSED_DIR, FIG_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Study drugs: one representative per major type 2 diabetes drug class
DRUGS = {
    "metformin": "Biguanide",
    "sitagliptin": "DPP-4 inhibitor",
    "empagliflozin": "SGLT2 inhibitor",
    "semaglutide": "GLP-1 receptor agonist",
}

DATE_RANGE = "[20200101 TO 20241231]"   # FDA receive date window
MAX_RECORDS_PER_DRUG = 5000             # openFDA allows skip up to 25,000
PAGE_SIZE = 100                       # openFDA maximum per request

AGE_BINS = [0, 18, 45, 65, 75, 120]
AGE_LABELS = ["0-17", "18-44", "45-64", "65-74", "75+"]

CONC_BINS = [-1, 0, 1, 2, 4, 9, float("inf")]
CONC_LABELS = ["0", "1", "2", "3-4", "5-9", "10+"]

RANDOM_STATE = 42