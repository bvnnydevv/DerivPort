"""Interactive calculus for the editable score, separate from model attribution."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .calculus import (
    LATEX_VARIABLES, calculus_experiment, differential, formula_to_latex,
)
from .formula import FormulaError


INPUT_LABELS = {
    "uncertainty": "Uncertainty",
    "k": "Risk weight k",
    "momentum": "Trailing momentum",
    "volatility": "Volatility",
    "predicted_return": "Model return estimate",
}
NONNEGATIVE = {"uncertainty", "volatility", "k"}


def _number(value: float) -> str:
    """Short, valid LaTeX scalars, including very small or large derivatives."""
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-4:
        mantissa, exponent = f"{value:.4e}".split("e")
        return rf"{float(mantissa):g}\times 10^{{{int(exponent)}}}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _plot_style(fig: go.Figure, plotstyle, height=280):
    if plotstyle is not None:
        return plotstyle(fig, height)
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=11, color="#66706F"),
        legend=dict(orientation="h", y=1.15, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E6E4DD")
    fig.update_yaxes(gridcolor="#E6E4DD", zerolinecolor="#E6E4DD")
    return fig


def render_calculus_lab(expression: str, values: Mapping[str, float], k: float, plotstyle=None):
    """Render exact score derivatives, Taylor approximations, and a signed integral.

    The controls perturb one formula input while keeping the stock observations
    and all other inputs fixed. No ranking or historical-reveal state is changed.
    """
    st.subheader("Calculus lab")
    st.caption("Change one input and explore slope, curvature, and accumulated score change. These derivatives describe your ranking rule, not the regression model's feature influence.")
    with st.expander("Derivatives, curvature & accumulated change", expanded=False):
        env = {name: float(value) for name, value in values.items()}
        env["k"] = float(k)
        selector, perturbation = st.columns(2, gap="large")
        with selector:
            variable = st.selectbox("Input to vary", list(INPUT_LABELS),
                format_func=lambda name: INPUT_LABELS[name], key="calc_variable")
        current = env[variable]
        span = 2.0 if variable == "k" else max(2.0, min(10.0, abs(current) * .75))
        span = math.ceil(span * 10) / 10
        lower = -span
        if variable in NONNEGATIVE:
            # Round toward zero so even the lowest slider position stays valid.
            lower = -math.floor(min(max(current, 0.0), span) * 10) / 10
        upper = span
        if "calc_delta" in st.session_state:
            old_delta = float(st.session_state.calc_delta)
            if not lower <= old_delta <= upper:
                st.session_state.calc_delta = float(min(max(old_delta, lower), upper))
        unit = "dimensionless" if variable == "k" else "percentage points"
        with perturbation:
            delta = st.slider(f"Change in {INPUT_LABELS[variable].lower()} ({unit})",
                min_value=float(lower), max_value=float(upper), value=1.0,
                step=.1, key="calc_delta")
        symbol = LATEX_VARIABLES[variable]
        st.caption(f"Let g(x) be the score when only {INPUT_LABELS[variable].lower()} changes. All other inputs stay fixed. The current input is {current:.4f}; the changed input is {current + delta:.4f} ({unit}).")

        try:
            exact = differential(expression, env, k)
            result = calculus_experiment(expression, env, variable, delta, k, points=201)
            st.session_state.calc_result = {
                **result, "variable": variable, "current_input": current,
                "delta": float(delta), "expression": expression,
                "gradient": dict(exact.gradient),
                "hessian": {name: dict(row) for name, row in exact.hessian.items()},
            }
        except (FormulaError, ValueError, OverflowError) as exc:
            st.session_state.pop("calc_result", None)
            st.info(f"This calculus interval cannot be evaluated: {exc} Your active ranking scores are unchanged.")
            return

        st.latex(rf"S = {formula_to_latex(expression)}")
        st.latex(rf"x = {symbol},\qquad x_0 = {_number(current)},\qquad h=\Delta x={_number(delta)}")
        st.latex(rf"g'(x_0)={_number(result['first_derivative'])},\qquad g''(x_0)={_number(result['second_derivative'])}")
        st.caption(f"The first derivative is score units per unit of {INPUT_LABELS[variable].lower()}; the second derivative is score units per squared input unit. Derivatives use exact automatic differentiation rules, with displayed values rounded.")

        score_col, derivative_col = st.columns(2, gap="large")
        curve = result["curve"]
        with score_col:
            st.markdown("**The score and its approximations**")
            fig = go.Figure()
            for column, name, color, dash in (
                ("score", "Actual score", "#2454FF", "solid"),
                ("tangent", "Tangent", "#101820", "dash"),
                ("quadratic", "Quadratic", "#A48645", "dot"),
            ):
                fig.add_trace(go.Scatter(x=curve["x"], y=curve[column], name=name,
                    mode="lines", line=dict(color=color, width=2, dash=dash)))
            fig.add_trace(go.Scatter(x=[current, current + delta],
                y=[result["baseline"], result["shifted"]], mode="markers",
                marker=dict(color="#2454FF", size=8),
                name="Current / changed input", showlegend=False))
            fig.update_xaxes(title=f"{INPUT_LABELS[variable]} ({unit})")
            fig.update_yaxes(title="Ranking score")
            st.plotly_chart(_plot_style(fig, plotstyle, 290), width="stretch",
                config={"displayModeBar": False}, key="calc_score_plot")
            st.latex(r"T_1(h)=g(x_0)+g'(x_0)h")
            st.latex(r"T_2(h)=g(x_0)+g'(x_0)h+\frac{1}{2}g''(x_0)h^2")
            st.caption("The tangent uses the local slope. The quadratic approximation also includes curvature; it need not be exact away from the starting input.")
        with derivative_col:
            st.markdown("**The derivative and accumulated change**")
            fig = go.Figure(go.Scatter(x=curve["x"], y=curve["derivative"],
                name="Score derivative", mode="lines", fill="tozeroy",
                line=dict(color="#2454FF", width=2),
                fillcolor="rgba(36,84,255,.12)"))
            fig.update_xaxes(title=f"{INPUT_LABELS[variable]} ({unit})")
            fig.update_yaxes(title="Score derivative")
            fig.update_layout(showlegend=False)
            st.plotly_chart(_plot_style(fig, plotstyle, 290), width="stretch",
                config={"displayModeBar": False}, key="calc_derivative_plot")
            st.latex(r"\int_{x_0}^{x_0+h} g'(x)\,dx=g(x_0+h)-g(x_0)=\Delta S")
            st.latex(rf"\Delta S={_number(result['actual_change'])},\qquad \int g'(x)\,dx\approx {_number(result['integrated_change'])}")
            st.caption(f"The fundamental theorem of calculus links signed derivative area to score change. The displayed integral uses the trapezoidal rule on 201 samples; absolute numerical error is {abs(result['integration_error']):.3g}. A negative h reverses the direction of integration.")

        comparison = pd.DataFrame([
            {"Method": "Actual formula", "Score change": result["actual_change"], "Absolute error": 0.0},
            {"Method": "Tangent · first order", "Score change": result["linear_change"],
             "Absolute error": abs(result["actual_change"] - result["linear_change"])},
            {"Method": "Quadratic · second order", "Score change": result["quadratic_change"],
             "Absolute error": abs(result["actual_change"] - result["quadratic_change"])},
            {"Method": "Numerical integral", "Score change": result["integrated_change"],
             "Absolute error": abs(result["integration_error"])},
        ])
        st.dataframe(comparison, hide_index=True, width="stretch",
            column_config={name: st.column_config.NumberColumn(format="%.5f")
                           for name in ("Score change", "Absolute error")})
        if np.allclose(curve["score"], curve["tangent"], rtol=1e-9, atol=1e-9):
            st.caption("The tangent matches the sampled score curve across this interval. For a linear rule, the linear and quadratic approximations coincide with the actual score.")

        with st.expander("All partial derivatives"):
            ordered = list(INPUT_LABELS)
            gradient = r"\\".join(_number(exact.gradient[name]) for name in ordered)
            st.latex(rf"\nabla S=\begin{{bmatrix}}{gradient}\end{{bmatrix}}")
            st.caption("Gradient order: uncertainty, k, momentum, volatility, model return estimate. Each entry measures the explicit rule's score sensitivity while all other inputs are held fixed.")
            partials = pd.DataFrame([
                {"Input": INPUT_LABELS[name], "Current input": env[name],
                 "First partial derivative": exact.gradient[name],
                 "Pure second partial derivative": exact.hessian[name][name]}
                for name in ordered
            ])
            st.dataframe(partials, hide_index=True, width="stretch",
                column_config={name: st.column_config.NumberColumn(format="%.5f")
                    for name in ("Current input", "First partial derivative", "Pure second partial derivative")})
            st.latex(r"H_{ij}=\frac{\partial^2 S}{\partial x_i\,\partial x_j}")
            hessian = pd.DataFrame(exact.hessian).reindex(index=ordered, columns=ordered)
            hessian = hessian.rename(index=INPUT_LABELS, columns=INPUT_LABELS)
            st.dataframe(hessian, width="stretch")
            st.caption("The Hessian includes mixed partials: it shows whether changing one input changes the rule's sensitivity to another input. This is arithmetic sensitivity, not a causal claim about stock prices.")
