"""Normal Streamlit entry-point checks for research evidence and dated sources."""
from datetime import date
import hashlib
from io import BytesIO
import json
from pathlib import Path
import socket
import zipfile

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from derivport.research_models import MODEL_NAMES


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offline(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Cached research and publication checks must not make network requests.")
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")


def frozen_app():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not app.exception
    mode = app.selectbox(key="history_mode")
    mode.set_value(next(option for option in mode.options if "frozen" in option.lower())).run()
    assert not app.exception
    return app


def replay_outcomes_are_visible(app):
    return any("Actual 21d return" in element.value.columns or "Realized rank" in element.value.columns
        for element in app.dataframe)


def test_research_experiment_exports_real_evidence_without_revealing_the_replay(offline, monkeypatch):
    downloads = {}
    original_download = st.download_button
    def capture_download(*args, **kwargs):
        if kwargs.get("key") == "benchmark_download":
            downloads["payload"] = kwargs.get("data", args[1] if len(args) > 1 else None)
            downloads["file_name"] = kwargs.get("file_name")
        return original_download(*args, **kwargs)
    monkeypatch.setattr(st, "download_button", capture_download)
    app = frozen_app()
    app.date_input(key="decision_date").set_value(date(2024, 12, 31)).run()
    assert not app.exception
    scores_before = app.session_state["current_scores"].copy(deep=True)
    assert not app.session_state["reveals"]
    app.button(key="benchmark_run").click().run()
    assert not app.exception
    assert not app.error
    result = app.session_state["research_experiment"]
    assert set(result["model_summary"].index) == set(MODEL_NAMES)
    assert result["manifest"]["final_test"]["n_dates"] >= 2
    assert pd.Timestamp(result["manifest"]["final_test"]["last_outcome"]) <= pd.Timestamp("2024-12-31")
    assert result["forecasts"].label_end.max() <= pd.Timestamp("2024-12-31")
    weight_fit = result["manifest"]["signals"]["weight_fit"]
    assert weight_fit["status"] == "available"
    assert pd.Timestamp(weight_fit["final_fit_last_label_end"]) < pd.Timestamp("2023-01-01")
    choices = set(app.selectbox(key="benchmark_signal").options)
    assert {"Balanced original", "Standardized weighted", "Risk-adjusted return",
        "Uncertainty-penalized", "Constrained weights"}.issubset(choices)
    assert app.selectbox(key="benchmark_signal").value == "Balanced original"
    pd.testing.assert_frame_equal(app.session_state["current_scores"], scores_before)
    assert not app.session_state["reveals"]
    assert not replay_outcomes_are_visible(app)
    assert any("interval_lower" in table.value.columns for table in app.dataframe)

    # Inspect bytes actually supplied to the real download widget, rather than
    # reconstructing an expected archive separately from the user-facing path.
    with zipfile.ZipFile(BytesIO(downloads["payload"])) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["run_id"] == result["manifest"]["run_id"]
        assert manifest["config"]["end"] == "2024-12-31"
        assert downloads["file_name"] == f"derivport_research_{manifest['run_id']}.zip"
        forecasts = pd.read_csv(BytesIO(archive.read("forecasts.csv")))
        assert len(forecasts) == len(result["forecasts"])
        for filename, digest in manifest["output_sha256"].items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == digest

    app.date_input(key="decision_date").set_value(date(2024, 5, 3)).run()
    assert not app.exception
    assert any("Previous results are hidden" in item.value for item in app.info)
    assert not any("interval_lower" in table.value.columns for table in app.dataframe)
    assert not app.session_state["reveals"]
    assert not replay_outcomes_are_visible(app)
    assert "actual_return" not in app.session_state["current_scores"]

    # A narrower international group clamps saved dates and hides incompatible
    # old evidence. This interaction must not launch another benchmark.
    old_run = result["manifest"]["run_id"]
    app.multiselect(key="universe_selection").set_value(["7203.T", "SAP.DE"]).run()
    assert not app.exception
    assert set(app.session_state["current_scores"].index) == {"7203.T", "SAP.DE"}
    assert app.session_state["research_experiment"]["manifest"]["run_id"] == old_run
    assert any("Previous results are hidden" in item.value for item in app.info)
    assert not any("interval_lower" in table.value.columns for table in app.dataframe)


def test_same_day_company_release_is_embargoed_until_a_later_cutoff(offline):
    app = frozen_app()
    app.date_input(key="decision_date").set_value(date(2024, 5, 2)).run()
    assert not app.exception
    captions = "\n".join(element.value for element in app.caption)
    # The latest actually cached earlier Apple release is November2023; there
    # is no February2024 source in this repository's curated evidence set.
    assert "Published 2023-11-02" in captions
    assert "Published 2024-05-02" not in captions
    assert "same-day releases are embargoed" in captions
    assert "actual_return" not in app.session_state["current_scores"]
    assert not app.session_state["reveals"]
    assert not replay_outcomes_are_visible(app)
    app.date_input(key="decision_date").set_value(date(2024, 5, 3)).run()
    assert not app.exception
    assert any("Published 2024-05-02" in element.value for element in app.caption)
    assert not replay_outcomes_are_visible(app)
