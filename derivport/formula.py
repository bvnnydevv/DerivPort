"""A deliberately small expression language; no Python execution or attributes."""

from __future__ import annotations

import ast
import math
from typing import Mapping

import numpy as np
import pandas as pd

VARIABLES = frozenset({"predicted_return", "uncertainty", "momentum", "volatility", "k"})


class FormulaError(ValueError):
    """A student expression cannot be safely or meaningfully evaluated."""


def _parse(expression: str) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise FormulaError("Enter a formula using the available variables.")
    if len(expression) > 400:
        raise FormulaError("Keep the formula under 400 characters.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise FormulaError("Check the formula's arithmetic and parentheses.") from exc
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise FormulaError("This expression is too complex. Use fewer terms.")
    return tree


def evaluate_formula(expression: str, values: Mapping[str, object] | pd.DataFrame, k: float | None = None):
    """Evaluate scalar or array inputs, rejecting all syntax except basic arithmetic.

    Return-like variables are numerical percentage points: 3.5 means 3.5%.
    Multiplication/division are allowed for experimentation; a custom expression's
    dimensional meaning is the student's responsibility, as documented in the UI.
    """
    tree = _parse(expression)
    env = {name: values[name] for name in VARIABLES if name in values}
    if k is not None:
        env["k"] = k

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value):
                raise FormulaError("All numbers must be finite.")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in VARIABLES:
                raise FormulaError(f"Unknown variable '{node.id}'. Use {', '.join(sorted(VARIABLES))}.")
            if node.id not in env:
                raise FormulaError(f"A value for '{node.id}' is missing.")
            return env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and np.any(np.asarray(right) == 0):
                raise FormulaError("Division by zero is undefined. Change the denominator.")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left / right
        raise FormulaError("Only numbers, the five listed variables, +, −, *, /, and parentheses are supported.")

    try:
        result = walk(tree)
        if not np.all(np.isfinite(np.asarray(result, dtype=float))):
            raise FormulaError("The formula produced a non-finite score. Try smaller values or another denominator.")
        return result
    except FormulaError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError, ZeroDivisionError) as exc:
        raise FormulaError("The formula could not be evaluated. Check its numbers and arithmetic.") from exc


def local_slope(expression: str, values: Mapping[str, float], k: float, step: float = 0.0001) -> float:
    """Central numerical derivative in score units per uncertainty percentage point."""
    lower, upper = dict(values), dict(values)
    lower["uncertainty"] -= step
    upper["uncertainty"] += step
    return float((evaluate_formula(expression, upper, k) - evaluate_formula(expression, lower, k)) / (2 * step))


def formula_curve(expression: str, values: Mapping[str, float], k: float, points: int = 101) -> pd.DataFrame:
    """Hold all other inputs fixed. Invalid locations are NaN gaps, not fabricated scores."""
    current = float(values["uncertainty"])
    x = np.linspace(max(0.0, current - max(4.0, current * 0.75)), current + max(4.0, current * 0.75), points)
    y = []
    for uncertainty in x:
        env = dict(values, uncertainty=float(uncertainty))
        try:
            y.append(float(evaluate_formula(expression, env, k)))
        except FormulaError:
            y.append(np.nan)
    return pd.DataFrame({"uncertainty": x, "score": y})


def default_crossovers(stocks: pd.DataFrame, max_k: float | None = None) -> pd.DataFrame:
    """Solve p_a - k*u_a == p_b - k*u_b for every nonparallel pair."""
    rows = []
    names = sorted(stocks.index)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = stocks.loc[first], stocks.loc[second]
            delta_u = float(a["uncertainty"] - b["uncertainty"])
            if abs(delta_u) < 1e-12:
                continue
            k = float((a["predicted_return"] - b["predicted_return"]) / delta_u)
            if np.isfinite(k) and k >= 0 and (max_k is None or k <= max_k):
                rows.append({"first": first, "second": second, "k": k,
                             "score": float(a["predicted_return"] - k * a["uncertainty"])})
    return pd.DataFrame(rows, columns=["first", "second", "k", "score"]).sort_values("k", ignore_index=True)
