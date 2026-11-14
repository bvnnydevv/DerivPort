"""Student-rule calculus is exact locally and explicit about integration."""
import math

import numpy as np
import pandas as pd
import pytest

from algorithms.derivport_signal import FORMULA
from derivport.calculus import (
    LATEX_VARIABLES, VARIABLE_ORDER, calculus_experiment, differential,
    formula_to_latex, sensitivity_sweep,
)
from derivport.formula import FormulaError, evaluate_formula


VALUES = dict(predicted_return=5.0, uncertainty=3.0, momentum=7.0, volatility=4.0)


def test_default_gradient_and_mixed_partial_are_exact():
    result = differential("predicted_return - k * uncertainty", VALUES, k=2)
    assert result.value == -1
    assert result.gradient == dict(predicted_return=1, uncertainty=-2, momentum=0, volatility=0, k=-3)
    assert result.hessian["uncertainty"]["k"] == -1
    assert result.hessian["k"]["uncertainty"] == -1
    assert result.hessian["uncertainty"]["uncertainty"] == 0
    assert set(result.gradient) == set(VARIABLE_ORDER)
    assert result.method == "Exact automatic differentiation"


def test_named_balanced_rule_reports_its_explicit_arithmetic_partials():
    result = differential(FORMULA, VALUES, k=1.5)
    assert result.value == pytest.approx(evaluate_formula(FORMULA, VALUES, 1.5))
    assert result.gradient == pytest.approx(dict(
        predicted_return=.6, momentum=.4, uncertainty=-.75, volatility=-.375, k=-2.5))
    assert result.hessian["uncertainty"]["k"] == -.5
    assert result.hessian["volatility"]["k"] == -.25


def test_quotient_gradient_and_full_hessian():
    result = differential("predicted_return / uncertainty", VALUES)
    assert result.value == pytest.approx(5 / 3)
    assert result.gradient["predicted_return"] == pytest.approx(1 / 3)
    assert result.gradient["uncertainty"] == pytest.approx(-5 / 9)
    assert result.hessian["uncertainty"]["uncertainty"] == pytest.approx(10 / 27)
    assert result.hessian["predicted_return"]["uncertainty"] == pytest.approx(-1 / 9)
    assert result.hessian["uncertainty"]["predicted_return"] == pytest.approx(-1 / 9)


def test_product_rule_curvature_and_signs():
    result = differential("-k * uncertainty * uncertainty / volatility", VALUES, 2)
    assert result.value == -4.5
    assert result.gradient["uncertainty"] == -3
    assert result.hessian["uncertainty"]["uncertainty"] == -1
    assert result.hessian["uncertainty"]["volatility"] == .75
    assert result.hessian["volatility"]["uncertainty"] == .75
    assert result.hessian["volatility"]["volatility"] == pytest.approx(-.5625)


def test_constant_rule_has_zero_gradient_and_hessian():
    result = differential("1", {})
    assert result.value == 1
    assert all(value == 0 for value in result.gradient.values())
    assert all(value == 0 for row in result.hessian.values() for value in row.values())


@pytest.mark.parametrize("delta", [1.0, -.75, 0.0])
def test_linear_changes_and_fundamental_theorem_agree_in_both_directions(delta):
    result = calculus_experiment("predicted_return - k * uncertainty", VALUES, "uncertainty", delta, k=2)
    assert result["first_derivative"] == -2
    assert result["second_derivative"] == 0
    for key in ("actual_change", "linear_change", "quadratic_change", "integrated_change"):
        assert result[key] == pytest.approx(-2 * delta)
    assert result["integration_error"] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(result["curve"]["score"], result["curve"]["tangent"])


def test_second_order_taylor_improves_on_tangent_for_a_quadratic():
    result = calculus_experiment("uncertainty * uncertainty", VALUES, "uncertainty", 2)
    assert result["baseline"] == 9
    assert result["shifted"] == 25
    assert result["first_derivative"] == 6
    assert result["second_derivative"] == 2
    assert result["linear_change"] == 12
    assert result["quadratic_change"] == result["actual_change"] == 16
    assert result["integrated_change"] == pytest.approx(16)
    np.testing.assert_allclose(result["curve"]["score"], result["curve"]["quadratic"])


def test_integral_is_numerically_approximated_for_a_reciprocal():
    result = calculus_experiment("1 / uncertainty", VALUES, "uncertainty", 1, points=1001)
    assert result["actual_change"] == pytest.approx(-1 / 12)
    assert result["integrated_change"] == pytest.approx(result["actual_change"], abs=1e-8)
    assert result["integration_error"] == pytest.approx(result["integrated_change"] - result["actual_change"])
    assert result["integration_error"] != 0


def test_sweep_of_k_varies_k_instead_of_freezing_the_override():
    result = calculus_experiment("predicted_return - k * uncertainty", VALUES, "k", 1, k=2)
    assert result["curve"]["x"].iloc[0] == 2
    assert result["curve"]["x"].iloc[-1] == 3
    assert result["actual_change"] == -3
    assert result["first_derivative"] == -3
    pd.testing.assert_frame_equal(
        result["curve"], sensitivity_sweep("predicted_return - k * uncertainty", VALUES, "k", 1, k=2))


@pytest.mark.parametrize("expression", [
    "1 / (uncertainty - 3.007)",
    "(uncertainty - 3.007) / (uncertainty - 3.007)",
    "1 / (1 / (uncertainty - 3.007))",
    "1 / ((uncertainty - 3.007) * (uncertainty - 3.007))",
])
def test_poles_between_grid_points_and_cancelled_denominators_are_rejected(expression):
    # Both endpoints are finite and 3.007 is not on a 201-point [3,4] grid.
    assert math.isfinite(float(evaluate_formula(expression, VALUES)))
    assert math.isfinite(float(evaluate_formula(expression, dict(VALUES, uncertainty=4))))
    with pytest.raises(FormulaError, match="denominator|division"):
        calculus_experiment(expression, VALUES, "uncertainty", 1)


def test_a_pole_outside_the_interval_does_not_reject_the_experiment():
    result = calculus_experiment("1 / (uncertainty - 8)", VALUES, "uncertainty", 1)
    assert result["actual_change"] == pytest.approx(-.05)


@pytest.mark.parametrize("expression", [
    "1 / 0", "uncertainty ** 2", "abs(uncertainty)", "unknown_name", "True",
    "__import__('os')", "uncertainty.__class__", "1e309", "[uncertainty]",
])
def test_latex_and_derivatives_accept_only_the_existing_safe_language(expression):
    for call in (lambda: differential(expression, VALUES, 2), lambda: formula_to_latex(expression)):
        with pytest.raises(FormulaError):
            call()


def test_substitution_preserves_negative_values_and_subtraction_grouping():
    expression = "predicted_return - (momentum - uncertainty)"
    symbolic = formula_to_latex(expression)
    assert symbolic == r"\widehat{r}_{21} - \left(M_{21} - \varepsilon\right)"
    substituted = formula_to_latex(expression, dict(VALUES, predicted_return=-2))
    assert substituted == r"\left(-2\right) - \left(7 - 3\right)"


def test_latex_uses_fractions_and_parentheses_with_correct_precedence():
    assert formula_to_latex("predicted_return / (1 + k * uncertainty)") == (
        r"\frac{\widehat{r}_{21}}{1 + k\cdot \varepsilon}")
    assert formula_to_latex("(momentum + predicted_return) * uncertainty") == (
        r"\left(M_{21} + \widehat{r}_{21}\right)\cdot \varepsilon")
    assert formula_to_latex("1e10") == r"1\times 10^{10}"
    assert LATEX_VARIABLES["volatility"] == r"\sigma_{21}"


@pytest.mark.parametrize("variable,delta", [("uncertainty", -4), ("volatility", -5), ("k", -3)])
def test_physical_nonnegative_inputs_cannot_be_perturbed_below_zero(variable, delta):
    with pytest.raises(FormulaError, match="negative"):
        calculus_experiment("predicted_return - k * uncertainty", VALUES, variable, delta, k=2)


def test_nonfinite_or_missing_scalar_inputs_and_invalid_curve_settings_are_readable():
    for call in (
        lambda: differential("uncertainty", dict(VALUES, uncertainty=np.inf)),
        lambda: differential("k * uncertainty", VALUES),
        lambda: differential("predicted_return / uncertainty", dict(VALUES, uncertainty=0)),
        lambda: calculus_experiment("uncertainty", VALUES, "unknown", 1),
        lambda: calculus_experiment("uncertainty", VALUES, "uncertainty", np.nan),
        lambda: calculus_experiment("uncertainty", VALUES, "uncertainty", 1, points=2),
    ):
        with pytest.raises(FormulaError) as error:
            call()
        assert str(error.value).strip()
