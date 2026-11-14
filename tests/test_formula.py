"""The rule editor accepts only arithmetic and reports meaningful calculus."""

import numpy as np
import pandas as pd
import pytest

from derivport.formula import (
    FormulaError,
    default_crossovers,
    evaluate_formula,
    formula_curve,
    local_slope,
)


VALUES = {
    "predicted_return": 5.0,
    "uncertainty": 3.0,
    "momentum": 7.0,
    "volatility": 4.0,
}


def test_rule_uses_percentage_point_arithmetic_and_all_supported_variables():
    expression = "(predicted_return + momentum) / 2 - k * uncertainty + volatility / 4"
    assert evaluate_formula(expression, VALUES, 2) == pytest.approx(1.0)
    assert evaluate_formula("-predicted_return + (+momentum)", VALUES, 2) == 2.0


@pytest.mark.parametrize("expression", [
    "predicted_return / 0",
    "predicted_return / (uncertainty - uncertainty)",
    "__import__('os').system('whoami')",
    "predicted_return.__class__",
    "[predicted_return][0]",
    "abs(predicted_return)",
    "predicted_return ** 2",
    "predicted_return // 2",
    "predicted_return % 2",
    "predicted_return if k else momentum",
    "unknown_variable",
    "True",
    "1e309",
    "1e308 * 1e308",
    "",
    "(",
])
def test_unsafe_or_undefined_expressions_are_rejected(expression):
    with pytest.raises(FormulaError) as error:
        evaluate_formula(expression, VALUES, 2)
    assert str(error.value).strip()


def test_division_zero_in_one_stock_rejects_entire_rule():
    values = pd.DataFrame([VALUES, dict(VALUES, uncertainty=0.0)])
    with pytest.raises(FormulaError, match="zero"):
        evaluate_formula("predicted_return / uncertainty", values, 2)


def test_nonfinite_input_rejected_even_without_overflow_literal():
    with pytest.raises(FormulaError, match="finite"):
        evaluate_formula("predicted_return", dict(VALUES, predicted_return=np.inf), 2)


def test_default_calculus_matches_two_percentage_point_change():
    expression = "predicted_return - k * uncertainty"
    score = evaluate_formula(expression, VALUES, 2)
    changed = evaluate_formula(expression, dict(VALUES, uncertainty=4.0), 2)
    assert changed - score == pytest.approx(-2.0)
    assert local_slope(expression, VALUES, 2) == pytest.approx(-2.0)
    curve = formula_curve(expression, VALUES, 2)
    np.testing.assert_allclose(curve["score"], 5 - 2 * curve["uncertainty"])


def test_custom_formula_slope_holds_other_inputs_fixed():
    expression = "predicted_return - k * uncertainty * uncertainty / volatility"
    expected = -2 * 2 * VALUES["uncertainty"] / VALUES["volatility"]
    assert local_slope(expression, VALUES, 2) == pytest.approx(expected)


def test_default_crossover_really_reverses_order():
    stocks = pd.DataFrame({
        "predicted_return": [8.0, 4.0], "uncertainty": [3.0, 1.0]
    }, index=["A", "B"])
    crossings = default_crossovers(stocks)
    crossing = crossings.iloc[0]
    assert crossing["k"] == pytest.approx(2.0)
    before = evaluate_formula("predicted_return - k * uncertainty", stocks, 1.9)
    after = evaluate_formula("predicted_return - k * uncertainty", stocks, 2.1)
    assert before["A"] > before["B"]
    assert after["A"] < after["B"]
    tied = evaluate_formula("predicted_return - k * uncertainty", stocks, 2.0)
    assert tied["A"] == pytest.approx(tied["B"])
