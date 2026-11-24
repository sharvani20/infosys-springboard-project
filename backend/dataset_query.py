from backend.dataset_loader import load_insurance_dataset


def search_policies(
    company=None,
    min_coverage=None,
    max_premium=None,
    cashless=None
):
    rows = load_insurance_dataset()

    if company:
        needle = str(company).lower()
        rows = [
            row for row in rows
            if needle in str(row.get("Company", "")).lower()
        ]

    if min_coverage is not None:
        rows = [
            row for row in rows
            if row.get("Coverage_Amount") is not None
            and row["Coverage_Amount"] >= min_coverage
        ]

    if max_premium is not None:
        rows = [
            row for row in rows
            if row.get("Annual_Premium") is not None
            and row["Annual_Premium"] <= max_premium
        ]

    if cashless is not None:
        wanted = str(cashless).lower()
        rows = [
            row for row in rows
            if str(row.get("Cashless_Available", "")).lower() == wanted
        ]

    return rows


def compare_companies():
    rows = load_insurance_dataset()

    numeric_columns = [
        "Annual_Premium",
        "Coverage_Amount",
        "Claim_Settlement_Ratio (%)",
        "Average_Claim_Processing_Time (in days)"
    ]

    grouped = {}

    for row in rows:
        company = row.get("Company")

        if not company:
            continue

        grouped.setdefault(company, []).append(row)

    result = []

    for company in sorted(grouped):
        entry = {"Company": company}

        for column in numeric_columns:
            values = [
                row[column] for row in grouped[company]
                if row.get(column) is not None
            ]

            entry[column] = (
                round(sum(values) / len(values), 2)
                if values else None
            )

        result.append(entry)

    return result


if __name__ == "__main__":
    print("POLICY SEARCH:")
    print(search_policies(company="HDFC").__repr__()[:2000])

    print("\nCOMPANY COMPARISON:")
    print(compare_companies())
