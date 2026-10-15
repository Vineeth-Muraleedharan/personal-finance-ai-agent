import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (  # noqa: E402
    budget_planner,
    calculate,
    calculator,
    compute_budget,
    compute_emi,
    compute_required_sip,
    compute_sip_future_value,
    emi_calculator,
    inr,
    required_monthly_savings,
    sip_calculator,
)


def test_inr_formatting():
    assert inr(1234567.891) == "₹12,34,567.89"
    assert inr(999) == "₹999.00"
    assert inr(100000) == "₹1,00,000.00"


def test_calculator_percent_of():
    assert calculate("20% of ₹85,000") == pytest.approx(17000)
    assert "17,000.00" in calculator.invoke({"expression": "20% of 85000"})


def test_calculator_rejects_code():
    out = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert out.startswith("Error")


def test_calculator_div_zero():
    assert "division by zero" in calculator.invoke({"expression": "5/0"})


def test_emi_matches_known_value():
    # Rs 10 lakh, 9% p.a., 20 years -> standard calculators give about 8997.26
    d = compute_emi(1_000_000, 9, 20)
    assert d["emi"] == pytest.approx(8997.26, abs=0.01)
    assert d["months"] == 240


def test_emi_8_percent():
    d = compute_emi(1_000_000, 8, 20)
    assert d["emi"] == pytest.approx(8364.40, abs=0.01)


def test_emi_zero_rate():
    assert compute_emi(120000, 0, 1)["emi"] == pytest.approx(10000)


def test_sip_5000_10y_12pct():
    d = compute_sip_future_value(5000, 12, 10)
    # verified against a month-by-month simulation (balance*1.01 + 5000, 120 times)
    assert d["future_value"] == pytest.approx(1_150_193.45, abs=0.01)
    assert d["invested"] == 600_000


def test_sip_start_of_month_is_higher_by_one_period():
    end = compute_sip_future_value(5000, 12, 10)["future_value"]
    start = compute_sip_future_value(5000, 12, 10, start_of_month=True)["future_value"]
    assert start == pytest.approx(end * 1.01)


def test_sip_7000_15y_10pct():
    d = compute_sip_future_value(7000, 10, 15)
    assert d["future_value"] == pytest.approx(2_901_292.42, abs=0.01)


def test_required_sip_round_trip():
    req = compute_required_sip(2_000_000, 12, 10)
    fv = compute_sip_future_value(req["monthly"], 12, 10)["future_value"]
    assert fv == pytest.approx(2_000_000)


def test_budget_sums_to_income():
    b = compute_budget(80000)
    assert b == {"needs": 40000, "wants": 24000, "savings": 16000}
    assert math.isclose(sum(b.values()), 80000)


def test_tool_wrappers_return_strings_and_handle_bad_input():
    assert "monthly EMI" in emi_calculator.invoke(
        {"loan_amount": 1_000_000, "annual_interest_rate": 9, "tenure_years": 20}
    )
    assert "future value" in sip_calculator.invoke(
        {"monthly_investment": 3000, "annual_return_rate": 8, "investment_years": 5}
    )
    assert "per month" in required_monthly_savings.invoke(
        {"target_amount": 2_000_000, "annual_return_rate": 12, "years": 10}
    )
    assert "Needs" in budget_planner.invoke({"monthly_income": 90000})
    assert emi_calculator.invoke(
        {"loan_amount": -5, "annual_interest_rate": 9, "tenure_years": 20}
    ).startswith("Error")
