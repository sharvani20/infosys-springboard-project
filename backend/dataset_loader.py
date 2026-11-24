import csv
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "backend" / "uploads" / "insurance_dataset.csv"

NUMERIC_COLUMNS = {
    "Coverage_Amount",
    "No_of_People_Insured",
    "Premium_Rate_Percentage",
    "Annual_Premium",
    "Claim_Settlement_Ratio (%)",
    "Average_Claim_Processing_Time (in days)",
    "Max_Entry_Age",
    "Min_Entry_Age",
}


def to_number(value):
    """'1400000' -> 1400000, '49870.0' -> 49870.0, junk/blank -> None."""

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _read_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Insurance dataset not found: {CSV_PATH}"
        )

    rows = []

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {}

            for key, value in raw.items():
                if key is None:
                    continue

                name = key.strip()
                value = (value or "").strip()

                row[name] = (
                    to_number(value)
                    if name in NUMERIC_COLUMNS
                    else value
                )

            rows.append(row)

    if not rows:
        raise ValueError("Insurance dataset is empty.")

    return tuple(rows)


def load_insurance_dataset():
    """Return the dataset as a list of dicts (a fresh copy each call).

    The CSV is parsed once and cached, so requests no longer re-read it.
    """

    return [dict(row) for row in _read_rows()]


def get_dataset_summary():
    rows = load_insurance_dataset()

    return {
        "total_records": len(rows),
        "total_columns": len(rows[0]),
        "columns": list(rows[0].keys()),
        "companies": sorted(
            {row["Company"] for row in rows if row.get("Company")}
        )
    }


if __name__ == "__main__":
    print(get_dataset_summary())
