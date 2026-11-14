"""Safe LaTeX and exact arithmetic derivatives for the student's ranking rule.

This differentiates the explicit ranking formula, not the regression model. The
gradient and Hessian use automatic differentiation; accumulated score changes
are separately estimated by numerical integration of the exact derivative.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from .formula import FormulaError, VARIABLES, _parse, evaluate_formula


VARIABLE_ORDER = ("predicted_return", "uncertainty", "momentum", "volatility", "k")
LATEX_VARIABLES = {
    "predicted_return": r"\widehat{r}_{21}",
    "uncertainty": r"\varepsilon",
    "momentum": r"M_{21}",
    "volatility": r"\sigma_{21}",
    "k": "k",
}
NONNEGATIVE_VARIABLES = frozenset({"uncertainty", "volatility", "k"})


def _number(value, name="Value"):
    if isinstance(value, (bool, np.bool_)):
        raise FormulaError(f"{name} must be a finite scalar number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormulaError(f"{name} must be a finite scalar number.") from exc
    if not math.isfinite(value):
        raise FormulaError(f"{name} must be finite.")
    return value


def _environment(values, k=None):
    env = {name: _number(values[name], name) for name in VARIABLE_ORDER if name in values}
    if k is not None:
        env["k"] = _number(k, "k")
    return env


def _validate_node(node):
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
    elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
        _number(node.value)
    elif isinstance(node, ast.Name):
        if node.id not in VARIABLES:
            raise FormulaError(f"Unknown variable '{node.id}'. Use {', '.join(VARIABLE_ORDER)}.")
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_node(node.operand)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        _validate_node(node.left)
        _validate_node(node.right)
        if isinstance(node.op, ast.Div) and not any(isinstance(part, ast.Name) for part in ast.walk(node.right)):
            # Reject an explicitly undefined constant expression even when no
            # numerical substitution has been requested for the LaTeX preview.
            denominator = evaluate_formula(ast.unparse(node.right), {})
            if float(denominator) == 0:
                raise FormulaError("Division by zero is undefined. Change the denominator.")
    else:
        raise FormulaError("Only numbers, the five listed variables, +, −, *, /, and parentheses are supported.")


def _validated_tree(expression):
    tree = _parse(expression)
    _validate_node(tree)
    return tree


def _latex_number(number):
    text = format(_number(number), ".4g")
    if "e" in text:
        coefficient, exponent = text.split("e")
        return coefficient + r"\times 10^{" + str(int(exponent)) + "}"
    return text


def formula_to_latex(expression: str, values: Mapping[str, float] | None = None) -> str:
    """Render the restricted expression without accepting arbitrary LaTeX.

    Optional values substitute known variables, rounded to four significant
    digits; omitted variables remain symbolic. Fractions and necessary grouping
    preserve the actual expression tree. Names and commands cannot be injected.
    """
    tree = _validated_tree(expression)
    env = _environment(values or {})
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if used.issubset(env):
        evaluate_formula(expression, env)

    def parenthesize(text):
        return r"\left(" + text + r"\right)"

    def render(node):
        if isinstance(node, ast.Expression):
            return render(node.body)
        if isinstance(node, ast.Constant):
            return _latex_number(node.value), 4
        if isinstance(node, ast.Name):
            if node.id in env:
                value = env[node.id]
                text = _latex_number(value)
                return (parenthesize(text) if value < 0 else text), 4
            return LATEX_VARIABLES[node.id], 4
        if isinstance(node, ast.UnaryOp):
            text, precedence = render(node.operand)
            if precedence < 3:
                text = parenthesize(text)
            return ("+" if isinstance(node.op, ast.UAdd) else "-") + text, 3
        left, lp = render(node.left)
        right, rp = render(node.right)
        if isinstance(node.op, ast.Div):
            return r"\frac{" + left + "}{" + right + "}", 4
        if isinstance(node.op, ast.Mult):
            if lp < 2:
                left = parenthesize(left)
            if rp < 2:
                right = parenthesize(right)
            return left + r"\cdot " + right, 2
        if isinstance(node.op, ast.Sub) and rp <= 1:
            right = parenthesize(right)
        return left + (" - " if isinstance(node.op, ast.Sub) else " + ") + right, 1

    return render(tree)[0]


@dataclass
class Differential:
    value: float
    gradient: dict[str, float]
    hessian: dict[str, dict[str, float]]
    method: str = "Exact automatic differentiation"


@dataclass
class _Dual:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray


def differential(expression: str, values: Mapping[str, float], k: float | None = None) -> Differential:
    """Exact first and second partials of supported scalar arithmetic.

    Floating-point arithmetic still applies. This measures only the rule's
    explicit mathematical sensitivity, not feature influence inside the model.
    """
    tree = _validated_tree(expression)
    env = _environment(values, k)
    n = len(VARIABLE_ORDER)

    def constant(value):
        return _Dual(float(value), np.zeros(n), np.zeros((n, n)))

    def checked(value):
        if not (np.isfinite(value.value) and np.isfinite(value.gradient).all() and np.isfinite(value.hessian).all()):
            raise FormulaError("The score or its derivatives are non-finite. Reduce extreme values or change the denominator.")
        return value

    def multiply(a, b):
        return checked(_Dual(a.value * b.value,
            a.gradient * b.value + b.gradient * a.value,
            a.hessian * b.value + b.hessian * a.value
            + np.outer(a.gradient, b.gradient) + np.outer(b.gradient, a.gradient)))

    def inverse(a):
        if a.value == 0:
            raise FormulaError("Division by zero is undefined. Change the denominator.")
        reciprocal = 1.0 / a.value
        if not np.any(a.gradient) and not np.any(a.hessian):
            return checked(constant(reciprocal))
        return checked(_Dual(reciprocal, -a.gradient * reciprocal * reciprocal,
            2 * np.outer(a.gradient, a.gradient) * reciprocal * reciprocal * reciprocal
            - a.hessian * reciprocal * reciprocal))

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return constant(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise FormulaError(f"A value for '{node.id}' is missing.")
            result = constant(env[node.id])
            result.gradient[VARIABLE_ORDER.index(node.id)] = 1.0
            return result
        if isinstance(node, ast.UnaryOp):
            result = walk(node.operand)
            return result if isinstance(node.op, ast.UAdd) else _Dual(-result.value, -result.gradient, -result.hessian)
        a, b = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Mult):
            return multiply(a, b)
        if isinstance(node.op, ast.Div):
            return multiply(a, inverse(b))
        sign = -1 if isinstance(node.op, ast.Sub) else 1
        return checked(_Dual(a.value + sign * b.value, a.gradient + sign * b.gradient, a.hessian + sign * b.hessian))

    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            result = checked(walk(tree))
    except FormulaError:
        raise
    except (ValueError, TypeError, OverflowError, ZeroDivisionError, RecursionError) as exc:
        raise FormulaError("The score or derivatives could not be evaluated. Check the arithmetic and input scale.") from exc
    return Differential(float(result.value),
        {name: float(result.gradient[i]) for i, name in enumerate(VARIABLE_ORDER)},
        {name: {other: float(result.hessian[i, j]) for j, other in enumerate(VARIABLE_ORDER)}
         for i, name in enumerate(VARIABLE_ORDER)})


def _check_interval(tree, env, variable, first, last):
    """Find denominator zeros, including poles between sampled curve points.

    With all other variables fixed, supported expressions are rational
    polynomials in one variable. Each division's numerator-of-denominator is
    checked separately so an algebraic cancellation cannot hide an undefined
    intermediate expression (for example 1/(1/(x-1))).
    """
    denominators = []

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return Polynomial([float(node.value)]), Polynomial([1.0])
        if isinstance(node, ast.Name):
            if node.id == variable:
                return Polynomial([0.0, 1.0]), Polynomial([1.0])
            if node.id not in env:
                raise FormulaError(f"A value for '{node.id}' is missing.")
            return Polynomial([env[node.id]]), Polynomial([1.0])
        if isinstance(node, ast.UnaryOp):
            numerator, denominator = walk(node.operand)
            return (numerator if isinstance(node.op, ast.UAdd) else -numerator), denominator
        a, b = walk(node.left)
        c, d = walk(node.right)
        if isinstance(node.op, ast.Add):
            return a * d + c * b, b * d
        if isinstance(node.op, ast.Sub):
            return a * d - c * b, b * d
        if isinstance(node.op, ast.Mult):
            return a * c, b * d
        denominators.append(c)
        return a * d, b * c

    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            walk(tree)
            lo, hi = sorted((float(first), float(last)))
            tolerance = 1e-9 * max(1.0, abs(lo), abs(hi))
            for polynomial in denominators:
                coefficients = polynomial.trim().coef
                if not np.isfinite(coefficients).all():
                    raise FormulaError("This interval is too extreme to verify safely. Reduce the input change.")
                if np.all(coefficients == 0):
                    raise FormulaError("The chosen interval contains division by zero. Choose a smaller change or another formula.")
                scale = float(np.max(np.abs(coefficients)))
                roots = Polynomial(coefficients / scale).roots()
                for root in roots:
                    real = float(np.real(root))
                    imaginary = abs(float(np.imag(root)))
                    if imaginary <= 1e-7 * max(1.0, abs(real)) and lo - tolerance <= real <= hi + tolerance:
                        raise FormulaError("The chosen interval crosses an undefined denominator. Choose a smaller change or another formula; integration across a pole is not shown.")
    except FormulaError:
        raise
    except (ValueError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError) as exc:
        raise FormulaError("The interval could not be checked for denominator zeros. Reduce the input change.") from exc


def calculus_experiment(expression: str, values: Mapping[str, float], variable: str,
                        delta: float, k: float | None = None, points: int = 201) -> dict:
    """Vary one input; compare actual, Taylor, and integrated score changes.

    The integral uses the trapezoid rule on the exact first derivative. It is
    explicitly a numerical approximation, unlike the local automatic partials.
    A singularity anywhere on the interval rejects the experiment safely.
    """
    if variable not in VARIABLES:
        raise FormulaError("Choose one of the five formula variables for the calculus experiment.")
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or not 3 <= points <= 2001:
        raise FormulaError("Use between 3 and 2,001 points for the calculus curve.")
    env = _environment(values, k)
    if variable not in env:
        raise FormulaError(f"A value for '{variable}' is missing.")
    delta = _number(delta, "Input change")
    initial = env[variable]
    end = initial + delta
    if not math.isfinite(end):
        raise FormulaError("The changed input must be finite. Try a smaller change.")
    if variable in NONNEGATIVE_VARIABLES and min(initial, end) < 0:
        raise FormulaError(f"{variable} cannot be negative in this experiment. Choose a smaller change.")
    tree = _validated_tree(expression)
    _check_interval(tree, env, variable, initial, end)
    baseline = differential(expression, env)
    first = baseline.gradient[variable]
    second = baseline.hessian[variable][variable]
    rows = []
    for x in np.linspace(initial, end, int(points)):
        current = differential(expression, dict(env, **{variable: float(x)}))
        dx = float(x) - initial
        rows.append({"x": float(x), "score": current.value,
                     "derivative": current.gradient[variable],
                     "tangent": baseline.value + first * dx,
                     "quadratic": baseline.value + first * dx + .5 * second * dx * dx})
    curve = pd.DataFrame(rows)
    integrated = float(np.trapezoid(curve["derivative"], curve["x"]))
    actual = float(curve.iloc[-1]["score"] - baseline.value)
    if not np.isfinite(curve.to_numpy()).all() or not math.isfinite(integrated):
        raise FormulaError("This interval produced non-finite calculus values. Reduce the input change.")
    return {"baseline": baseline.value, "shifted": float(curve.iloc[-1]["score"]),
            "first_derivative": first, "second_derivative": second,
            "linear_change": first * delta,
            "quadratic_change": first * delta + .5 * second * delta * delta,
            "actual_change": actual, "integrated_change": integrated,
            "integration_error": integrated - actual, "curve": curve}


def sensitivity_sweep(expression: str, values: Mapping[str, float], variable: str,
                      delta: float, k: float | None = None, points: int = 201) -> pd.DataFrame:
    """Return the verified one-variable score, derivative and Taylor curves."""
    return calculus_experiment(expression, values, variable, delta, k, points)["curve"]
