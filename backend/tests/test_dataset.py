from backend.dataset_loader import load_insurance_dataset
from backend.dataset_query import compare_companies, search_policies


def test_dataset_loads_with_numeric_types():
    rows = load_insurance_dataset()

    assert len(rows) == 200
    assert isinstance(rows[0]["Coverage_Amount"], int)
    assert isinstance(rows[0]["Annual_Premium"], float)


def test_search_by_company_and_filters():
    hdfc = search_policies(company="hdfc")

    assert hdfc and all("hdfc" in r["Company"].lower() for r in hdfc)

    cheap = search_policies(max_premium=50000)

    assert all(r["Annual_Premium"] <= 50000 for r in cheap)


def test_compare_companies_returns_one_row_per_company():
    result = compare_companies()
    companies = [r["Company"] for r in result]

    assert companies == sorted(set(companies))
    assert "Annual_Premium" in result[0]


def test_loader_returns_independent_copies():
    first = load_insurance_dataset()
    first[0]["Company"] = "CHANGED"

    assert load_insurance_dataset()[0]["Company"] != "CHANGED"
