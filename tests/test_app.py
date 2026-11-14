"""Replay integrity through Streamlit's normal app.py entry point."""

import socket
from io import BytesIO
from datetime import date
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def has_future_table(app):
    return any(
        "Actual 21d return" in table.value.columns
        or "Realized rank" in table.value.columns
        for table in app.dataframe
    )


def choose_history_mode(app, *, frozen=False):
    """Choose the mode by meaning while allowing minor wording changes."""
    control = app.selectbox(key="history_mode")
    option = next(
        option for option in control.options
        if ("frozen" in option.lower()) == frozen
    )
    control.set_value(option).run()
    assert not app.exception


def test_offline_replay_reveal_and_invalid_rule_preservation(monkeypatch):
    """A complete student interaction while outbound network is unavailable."""
    def deny_network(*args, **kwargs):
        raise AssertionError("The main demonstration must not access the network.")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    choose_history_mode(app, frozen=True)
    app.date_input(key="decision_date").set_value(date(2024, 5, 3)).run()
    assert app.date_input(key="decision_date").value == date(2024, 5, 3)
    assert app.selectbox(key="stock").value == "AAPL"
    assert not has_future_table(app)

    app.date_input(key="decision_date").set_value(date(2020, 4, 1)).run()
    assert not app.exception
    assert app.session_state["cutoff"] == "2020-04-01"
    assert not has_future_table(app)
    assert any("No evaluation outcomes since 2022" in element.value for element in app.caption)
    assert any(metric.value == "Not available" for metric in app.metric)
    app.date_input(key="decision_date").set_value(date(2024, 5, 3)).run()
    assert not app.exception
    valid_scores = app.session_state["current_scores"].copy(deep=True)
    valid_formula = app.session_state["valid_formula"]
    valid_k = app.session_state["valid_k"]

    app.text_input(key="formula").set_value("predicted_return / 0").run()
    assert not app.exception
    assert app.error
    assert app.session_state["valid_formula"] == valid_formula
    assert app.session_state["valid_k"] == valid_k
    pd.testing.assert_frame_equal(app.session_state["current_scores"], valid_scores)
    assert not has_future_table(app)

    app.text_input(key="formula").set_value(valid_formula).run()
    app.slider(key="k").set_value(0.20).run()
    assert not app.exception
    assert app.slider(key="k").value == 0.20
    assert app.session_state["current_scores"].loc["AAPL", "student_rank"] == 3.0
    app.slider(key="k").set_value(0.25).run()
    assert not app.exception
    assert app.slider(key="k").value == 0.25
    assert app.session_state["current_scores"].loc["AAPL", "student_rank"] == 2.0
    app.slider(key="k").set_value(float(valid_k)).run()
    app.button(key="reveal").click().run()
    assert not app.exception
    assert has_future_table(app)
    reveal_key = app.session_state["active_replay_key"]
    assert "|2024-05-03" in reveal_key
    assert reveal_key in app.session_state["reveals"]
    original_reveal_scores = app.session_state["reveals"][reveal_key]["scores"].copy(deep=True)

    app.slider(key="k").set_value(float(valid_k) + 1.0).run()
    assert not app.exception
    visible_text = " ".join(
        str(element.value)
        for kind in ("markdown", "caption", "warning", "info")
        for element in getattr(app, kind)
    ).lower()
    assert "edited after seeing the outcome" in visible_text
    pd.testing.assert_frame_equal(
        app.session_state["reveals"][reveal_key]["scores"], original_reveal_scores
    )

    # A Saturday maps to the same previous common session; it must keep that
    # session's already-revealed snapshot instead of pretending it is unseen.
    app.date_input(key="decision_date").set_value(date(2024, 5, 4)).run()
    assert not app.exception
    assert app.session_state["active_replay_key"] == reveal_key
    assert has_future_table(app)

    # The same date with a different comparison universe is another experiment.
    original_universe = list(app.multiselect(key="universe_selection").value)
    app.multiselect(key="universe_selection").set_value(["AAPL", "MSFT"]).run()
    assert not app.exception
    assert len(app.session_state["current_scores"]) == 2
    assert app.session_state["active_replay_key"] != reveal_key
    assert not has_future_table(app)
    app.multiselect(key="universe_selection").set_value(original_universe).run()
    assert not app.exception
    assert app.session_state["active_replay_key"] == reveal_key
    assert has_future_table(app)

    app.date_input(key="decision_date").set_value(date(2023, 8, 7)).run()
    assert not app.exception
    assert not has_future_table(app)
    assert app.session_state["active_replay_key"] not in app.session_state["reveals"]

    # A constant student score ties all five stocks. It has no defined Spearman
    # correlation, and the regular app must explain that instead of crashing.
    app.text_input(key="formula").set_value("1").run()
    assert not app.exception
    assert app.session_state["current_scores"]["student_rank"].eq(3.0).all()
    app.button(key="reveal").click().run()
    assert not app.exception
    assert any(metric.value == "Undefined" for metric in app.metric)


def test_offline_global_comparison_and_algorithm_backtest(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Cached global research and backtests must work offline.")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.multiselect(key="universe_selection").set_value(["7203.T", "SAP.DE"]).run()
    assert not app.exception
    choose_history_mode(app, frozen=False)
    app.date_input(key="decision_date").set_value(date(2025, 2, 14)).run()
    assert not app.exception
    assert set(app.session_state["current_scores"].index) == {"7203.T", "SAP.DE"}
    assert app.session_state["cutoff"] == "2025-02-14"
    assert not has_future_table(app)
    assert any("No suitable dated company source" in element.value for element in app.info)
    assert not any("**Reported fact.**" in element.value or "**Interpretation.**" in element.value for element in app.markdown)

    app.text_input(key="formula").set_value("momentum").run()
    app.date_input(key="test_start").set_value(date(2023, 1, 1)).run()
    app.date_input(key="test_end").set_value(date(2023, 12, 31)).run()
    app.button(key="run_backtest").click().run()
    assert not app.exception
    result = app.session_state["backtest_result"]
    assert result["metrics"]["n_dates"] > 0
    assert result["metrics"]["n_forecasts"] == result["metrics"]["n_dates"] * 2
    assert pd.to_datetime(result["per_date"]["label_end"]).max() <= pd.Timestamp("2023-12-31")
    pd.testing.assert_series_equal(
        result["linked_returns"]["student"], result["linked_returns"]["momentum"], check_names=False
    )
    assert not has_future_table(app), "A past backtest must not silently reveal the later selected replay."

    app.text_input(key="formula").set_value("1").run()
    assert not app.exception
    assert app.session_state["backtest_result"]["metadata"]["formula"] == "momentum"
    assert any("Settings changed" in element.value for element in app.info)
    app.text_input(key="formula").set_value("momentum").run()

    app.date_input(key="decision_date").set_value(date(2023, 5, 10)).run()
    assert not app.exception
    assert any("overlaps a backtest" in element.value.lower() for element in app.info)

    app.multiselect(key="universe_selection").set_value(["7203.T"]).run()
    assert not app.exception
    assert any("at least two" in element.value.lower() for element in app.info)


def test_latest_2026_research_keeps_unobserved_outcomes_pending(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("The latest cached research must remain usable offline.")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert app.selectbox(key="history_mode").value == "Full cached history"
    latest = date.fromisoformat(app.session_state["latest_historical_date"])
    assert latest.year == 2026
    assert app.date_input(key="decision_date").value == latest
    assert app.session_state["cutoff"] == latest.isoformat()
    assert not has_future_table(app)
    assert app.button(key="reveal").disabled
    assert "pending" in app.button(key="reveal").label.lower()
    assert "actual_return" not in app.session_state["current_scores"]

    # The latest fully observed replay remains available, even though the newer
    # close can only be researched until its next 21 sessions have happened.
    completed = date.fromisoformat(app.session_state["latest_replay_date"])
    assert completed.year == 2026
    assert completed < latest
    app.date_input(key="decision_date").set_value(completed).run()
    assert not app.exception
    assert not app.button(key="reveal").disabled
    app.button(key="reveal").click().run()
    assert not app.exception
    assert has_future_table(app)
    app.date_input(key="decision_date").set_value(latest).run()
    assert not app.exception
    assert app.button(key="reveal").disabled
    assert not has_future_table(app), "An earlier reveal must not fill the latest date's pending outcome."


def test_2037_future_scenarios_are_explicit_hypotheses_not_actual_prices(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Future scenarios use the observed local price cache and local simulation.")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert not has_future_table(app)
    target = date(2037, 12, 31)
    app.date_input(key="projection_target").set_value(target).run()
    app.button(key="project_future").click().run()
    assert not app.exception
    result = app.session_state["projection_result"]
    assert result.dates[-1] == pd.Timestamp(target)
    assert pd.to_datetime(result.quantiles["date"]).max() == pd.Timestamp(target)
    assert result.terminal_returns.shape[0] >= 200
    assert {"date", "ticker", "p10", "p50", "p90"}.issubset(result.quantiles.columns)
    assert not any("actual" in column.lower() for column in result.quantiles.columns)
    visible_text = " ".join(
        str(element.value)
        for kind in ("markdown", "caption", "warning", "info")
        for element in getattr(app, kind)
    ).lower()
    assert "simulated scenarios" in visible_text
    assert "hypothetical" in visible_text
    assert "not historical prices" in str(result.assumptions).lower()
    assert not has_future_table(app)
    assert app.button(key="reveal").disabled
    assert not app.session_state["reveals"], "Simulating 2037 must never create a historical outcome snapshot."


def test_balanced_algorithm_preset_and_downloads_match_the_standalone_file(monkeypatch):
    """Load the preset normally, then compare its actual exported artifact/input."""
    import streamlit as st
    from algorithms.derivport_signal import DEFAULT_K, FORMULA, NAME, rank_stocks

    def deny_network(*args, **kwargs):
        raise AssertionError("The bundled algorithm and current cached inputs must work offline.")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    downloads = {}
    real_download_button = st.download_button

    def capture_download(*args, **kwargs):
        # AppTest does not drive browser download events. Record the bytes/name
        # passed to the real Streamlit download control, preserving its behavior.
        key = kwargs.get("key")
        if key in {"download_algorithm", "export_algorithm_inputs"}:
            downloads[key] = {
                "data": kwargs.get("data", args[1] if len(args) > 1 else None),
                "file_name": kwargs.get("file_name", args[2] if len(args) > 2 else None),
            }
        return real_download_button(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture_download)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.button(key="load_balanced_algorithm").click().run()
    assert not app.exception
    assert app.text_input(key="formula").value == FORMULA
    assert app.slider(key="k").value == DEFAULT_K
    assert app.session_state["active_algorithm_name"] == NAME
    assert app.session_state["valid_formula"] == FORMULA
    assert app.session_state["valid_k"] == DEFAULT_K

    current = app.session_state["current_scores"].copy(deep=True)
    inputs = ["predicted_return", "uncertainty", "momentum", "volatility"]
    rows = current[inputs].rename_axis("ticker").reset_index().to_dict("records")
    expected = pd.DataFrame(rank_stocks(rows, k=DEFAULT_K)).set_index("ticker").sort_index()
    pd.testing.assert_series_equal(
        current["score"].sort_index(), expected["score"], check_names=False
    )
    pd.testing.assert_series_equal(
        current["student_rank"].sort_index(), expected["rank"], check_names=False
    )

    algorithm = downloads["download_algorithm"]
    assert algorithm["file_name"] == "derivport_signal.py"
    algorithm_bytes = algorithm["data"].encode() if isinstance(algorithm["data"], str) else algorithm["data"]
    assert algorithm_bytes == (ROOT / "algorithms/derivport_signal.py").read_bytes()
    exported = downloads["export_algorithm_inputs"]
    assert exported["file_name"] == f"derivport_features_{app.session_state['cutoff']}.csv"
    csv_bytes = exported["data"].encode() if isinstance(exported["data"], str) else exported["data"]
    exported_inputs = pd.read_csv(BytesIO(csv_bytes))
    assert set(exported_inputs.columns) == {"ticker", "cutoff", *inputs}
    assert exported_inputs["cutoff"].eq(app.session_state["cutoff"]).all()
    pd.testing.assert_frame_equal(
        exported_inputs.set_index("ticker")[inputs].sort_index(),
        current[inputs].sort_index(),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not has_future_table(app)

    # Parser errors must retain the loaded algorithm's valid scores as well as
    # the original default rule's scores (covered in the earlier replay test).
    app.text_input(key="formula").set_value("predicted_return / 0").run()
    assert not app.exception
    assert app.error
    assert app.session_state["active_algorithm_name"] == NAME
    pd.testing.assert_frame_equal(app.session_state["current_scores"], current)
