"""Point-in-time research tools for the DerivPort educational demo."""

from .analysis import (
    Analysis, TICKERS, DEFAULT_FORMULA, apply_rule, analyze_cutoff, build_features,
    evaluate_history, historical_forecasts, load_prices, rank_correlation,
    reveal_outcomes, validate_prices,
)
from .backtest import backtest_rule
from .formula import FormulaError, evaluate_formula, formula_curve, local_slope, default_crossovers
from .simulation import SCENARIOS, Simulation, simulate_paths, simulation_summary

__all__ = [
    "Analysis", "TICKERS", "DEFAULT_FORMULA", "apply_rule", "analyze_cutoff",
    "build_features", "evaluate_history", "historical_forecasts", "load_prices",
    "rank_correlation", "reveal_outcomes", "FormulaError", "evaluate_formula",
    "formula_curve", "local_slope", "default_crossovers", "SCENARIOS",
    "Simulation", "simulate_paths", "simulation_summary", "validate_prices", "backtest_rule",
]
