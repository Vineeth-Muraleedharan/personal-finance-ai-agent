"""Finance tools for the Personal Finance AI Agent.

Each calculation lives in a plain, testable Python function. A thin
LangChain @tool wrapper exposes it to the agent, so the LLM never does the
arithmetic itself: it only decides which tool to call and with what inputs.

Conventions (same as the assignment brief):
  * Annual rates are given in percent (12 means 12% per year).
  * Monthly rate r = annual_rate / 12 / 100.
  * Number of months n = years * 12.
"""

from __future__ import annotations

import ast
import operator
import re

from langchain_core.tools import tool

# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def inr(amount: float) -> str:
    """Format a number as rupees with Indian digit grouping (12,34,567.89)."""
    sign = "-" if amount < 0 else ""
    amount = abs(round(amount, 2))
    whole, frac = f"{amount:.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}₹{whole}.{frac}"


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if value is None or value <= 0:
            raise ValueError(f"{name} must be greater than 0 (got {value}).")


# --------------------------------------------------------------------------
# 1. Calculator (safe expression evaluation, no eval())
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large.")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only numbers and + - * / ** % // ( ) are allowed.")


def _normalise_expression(expression: str) -> str:
    expr = expression.strip().lower()
    expr = expr.replace("₹", "").replace(",", "").replace("^", "**").replace("x", "*")
    # "20% of 85000" -> "(20/100)*85000"
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", expr)
    # a bare "20%" -> "(20/100)"
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    return expr


def calculate(expression: str) -> float:
    """Safely evaluate an arithmetic expression."""
    tree = ast.parse(_normalise_expression(expression), mode="eval")
    return _eval_node(tree)


@tool
def calculator(expression: str) -> str:
    """General maths calculator. Use for percentages, sums, differences,
    products, divisions and powers that do NOT need the EMI, SIP or budget
    tools. Examples: '20% of 85000', '(1.08**5) * 1000', '250000 / 12'.
    Input is a single arithmetic expression string."""
    try:
        result = calculate(expression)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:  # noqa: BLE001 - report any bad input to the agent
        return f"Error: could not evaluate '{expression}'. {exc}"
    shown = f"{result:,.2f}" if abs(result) >= 1000 else f"{result:g}"
    return f"{expression} = {shown}"


# --------------------------------------------------------------------------
# 2. EMI calculator
# --------------------------------------------------------------------------


def compute_emi(principal: float, annual_rate_pct: float, years: float) -> dict:
    """EMI = P * r * (1+r)^n / ((1+r)^n - 1)."""
    _check_positive(principal=principal, years=years)
    if annual_rate_pct < 0:
        raise ValueError("annual_rate_pct cannot be negative.")
    n = round(years * 12)
    r = annual_rate_pct / 12 / 100
    emi = principal / n if r == 0 else principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total = emi * n
    return {
        "emi": emi,
        "months": n,
        "total_payment": total,
        "total_interest": total - principal,
    }


@tool
def emi_calculator(loan_amount: float, annual_interest_rate: float, tenure_years: float) -> str:
    """Calculate the monthly EMI for a loan.
    loan_amount: principal in rupees (convert lakh/crore first: 1 lakh = 100000,
    1 crore = 10000000). annual_interest_rate: percent per year, e.g. 9 for 9%.
    tenure_years: loan duration in years."""
    try:
        d = compute_emi(loan_amount, annual_interest_rate, tenure_years)
    except ValueError as exc:
        return f"Error: {exc}"
    return (
        f"Loan {inr(loan_amount)} at {annual_interest_rate}% for {tenure_years:g} years "
        f"({d['months']} months): monthly EMI = {inr(d['emi'])}. "
        f"Total payment = {inr(d['total_payment'])}, total interest = {inr(d['total_interest'])}."
    )


# --------------------------------------------------------------------------
# 3. SIP future value
# --------------------------------------------------------------------------


def compute_sip_future_value(
    monthly_investment: float,
    annual_return_pct: float,
    years: float,
    start_of_month: bool = False,
) -> dict:
    """FV = P * ((1+r)^n - 1) / r   (multiplied by (1+r) if paid at month start)."""
    _check_positive(monthly_investment=monthly_investment, years=years)
    if annual_return_pct < 0:
        raise ValueError("annual_return_pct cannot be negative.")
    n = round(years * 12)
    r = annual_return_pct / 12 / 100
    fv = monthly_investment * n if r == 0 else monthly_investment * ((1 + r) ** n - 1) / r
    if start_of_month and r != 0:
        fv *= 1 + r
    invested = monthly_investment * n
    return {"future_value": fv, "invested": invested, "gain": fv - invested, "months": n}


@tool
def sip_calculator(
    monthly_investment: float,
    annual_return_rate: float,
    investment_years: float,
    start_of_month: bool = False,
) -> str:
    """Calculate the future value of a monthly SIP (systematic investment plan).
    monthly_investment: rupees invested every month. annual_return_rate: expected
    return in percent per year, e.g. 12 for 12%. investment_years: duration in
    years. Leave start_of_month False unless the user says payments are made at
    the beginning of each month."""
    try:
        d = compute_sip_future_value(
            monthly_investment, annual_return_rate, investment_years, start_of_month
        )
    except ValueError as exc:
        return f"Error: {exc}"
    return (
        f"SIP of {inr(monthly_investment)}/month for {investment_years:g} years at "
        f"{annual_return_rate}% p.a. ({d['months']} months): future value = {inr(d['future_value'])}. "
        f"Total invested = {inr(d['invested'])}, estimated gain = {inr(d['gain'])}."
    )


# --------------------------------------------------------------------------
# 4. Required monthly SIP for a target (reverse of the SIP formula)
# --------------------------------------------------------------------------


def compute_required_sip(target_amount: float, annual_return_pct: float, years: float) -> dict:
    """P = FV * r / ((1+r)^n - 1)."""
    _check_positive(target_amount=target_amount, years=years)
    if annual_return_pct < 0:
        raise ValueError("annual_return_pct cannot be negative.")
    n = round(years * 12)
    r = annual_return_pct / 12 / 100
    monthly = target_amount / n if r == 0 else target_amount * r / ((1 + r) ** n - 1)
    invested = monthly * n
    return {"monthly": monthly, "invested": invested, "gain": target_amount - invested, "months": n}


@tool
def required_monthly_savings(
    target_amount: float, annual_return_rate: float, years: float
) -> str:
    """Work out how much to invest EVERY MONTH to reach a target corpus.
    target_amount: goal in rupees (1 lakh = 100000). annual_return_rate: expected
    return in percent per year (use 0 if the user mentions no return). years:
    time to reach the goal."""
    try:
        d = compute_required_sip(target_amount, annual_return_rate, years)
    except ValueError as exc:
        return f"Error: {exc}"
    return (
        f"To reach {inr(target_amount)} in {years:g} years at {annual_return_rate}% p.a., "
        f"invest {inr(d['monthly'])} per month. Total invested = {inr(d['invested'])}, "
        f"estimated gain = {inr(d['gain'])}."
    )


# --------------------------------------------------------------------------
# 5. Budget planner (50-30-20 rule)
# --------------------------------------------------------------------------


def compute_budget(monthly_income: float) -> dict:
    _check_positive(monthly_income=monthly_income)
    return {
        "needs": monthly_income * 0.50,
        "wants": monthly_income * 0.30,
        "savings": monthly_income * 0.20,
    }


@tool
def budget_planner(monthly_income: float) -> str:
    """Create a monthly budget using the 50-30-20 rule: 50% needs (rent, food,
    bills, EMIs), 30% wants (dining, travel, entertainment), 20% savings and
    investments. monthly_income: take-home income per month in rupees."""
    try:
        b = compute_budget(monthly_income)
    except ValueError as exc:
        return f"Error: {exc}"
    return (
        f"50-30-20 budget for monthly income {inr(monthly_income)}: "
        f"Needs (50%) = {inr(b['needs'])}; Wants (30%) = {inr(b['wants'])}; "
        f"Savings and investments (20%) = {inr(b['savings'])}."
    )


ALL_TOOLS = [calculator, emi_calculator, sip_calculator, required_monthly_savings, budget_planner]
