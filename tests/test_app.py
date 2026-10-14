"""User flows for the real Streamlit app with a small test-only release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from phishshield.features import FEATURE_VERSION, URLFeatureExtractor


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def ui_release(tmp_path, monkeypatch):
    """Fit a deterministic balanced model; this bundle never enters the real app."""
    model = Pipeline([
        ("features", URLFeatureExtractor()),
        ("classifier", DummyClassifier(strategy="prior")),
    ]).fit(
        pd.DataFrame({"url": ["https://example.com/", "http://example.org/login", "https://example.net/about", "http://192.0.2.1/verify"]}),
        [0, 1, 0, 1],
    )
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    manifest = {
        "model_name": "PhishShield", "model_version": "test-only-1", "algorithm": "Test-only dummy",
        "run_id": "test-only-run", "created_at": "test fixture", "feature_version": FEATURE_VERSION,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "threshold": .5,
        "metrics": {"accuracy": .5, "precision": .5, "recall": 1.0, "f1": 2 / 3, "roc_auc": .5},
        "training_rows": 4, "validation_rows": 4, "calibration_rows": 4, "test_rows": 4,
        "dataset_checksum": "test-only-data", "tld_statistics": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame([
        {"model": "Test-only dummy", "average_precision": .5, "recall": 1., "precision": .5, "f1": 2 / 3, "roc_auc": .5},
    ]).to_csv(tmp_path / "comparison.csv", index=False)
    pd.DataFrame([
        {"feature": "has_https", "importance_mean": .01, "importance_std": .001},
    ]).to_csv(tmp_path / "feature_importance.csv", index=False)
    monkeypatch.setenv("PHISHSHIELD_RELEASE_DIR", str(tmp_path))
    return tmp_path


def _app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=30).run()


def _analyze(app: AppTest, url: str) -> AppTest:
    app.text_input(key="url_input").set_value(url)
    next(button for button in app.button if button.label == "Analyze URL").click()
    return app.run()


def test_valid_url_shows_real_inference_result(ui_release):
    app = _analyze(_app(), "http://example.com@192.0.2.10/verify-account")
    assert not app.exception
    assert "Potential phishing" in [heading.value for heading in app.subheader]
    assert "192.0.2.10" in [element.value for element in app.text]
    assert app.session_state["prediction"].risk_score == pytest.approx(50.)
    assert app.session_state["prediction"].risk_band == "Medium"
    assert "IP address in the URL" in " ".join(element.value for element in app.markdown)
    assert len(app.get("plotly_chart")) == 2  # risk gauge and real artifact importance


@pytest.mark.parametrize("url,message", [
    ("", "Enter a URL"),
    ("example.com", "Include http:// or https://"),
    ("https://example.com:bad", "invalid hostname"),
])
def test_invalid_input_has_actionable_error_without_result(ui_release, url, message):
    app = _analyze(_app(), url)
    assert not app.exception
    assert any(message in error.value for error in app.error)
    assert "Potential phishing" not in [heading.value for heading in app.subheader]
    assert app.session_state.get("prediction") is None


def test_example_changes_input_and_clears_previous_analysis(ui_release):
    app = _analyze(_app(), "https://example.com/about")
    app.button(key="example_Account keywords").click().run()
    assert not app.exception
    assert app.text_input(key="url_input").value == "http://secure-account.example.org/login/verify?account=update"
    assert app.session_state.get("prediction") is None
    assert "Potential phishing" not in [heading.value for heading in app.subheader]
    app.button(key="example_Simple HTTPS").click().run()
    assert not app.exception
    assert app.text_input(key="url_input").value == "https://www.example.com/"


def test_new_release_clears_stale_prediction(ui_release):
    app = _analyze(_app(), "https://example.com/about")
    manifest_path = ui_release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_version"] = "test-only-2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    app.run()
    assert not app.exception
    assert app.session_state.get("prediction") is None
    assert any("test-only-2" in caption.value for caption in app.caption)


def test_unavailable_release_is_clear_and_does_not_simulate_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv("PHISHSHIELD_RELEASE_DIR", str(tmp_path / "missing"))
    app = _app()
    assert not app.exception
    submit = next(button for button in app.button if button.label == "Analyze URL")
    assert submit.disabled
    assert any("trained model release is needed" in warning.value for warning in app.warning)
    assert any("phishshield.cli train" in code.value for code in app.code)
    assert not app.metric
    assert not app.get("plotly_chart")


def test_lookalike_domain_warning_is_primary_even_if_model_disagrees(ui_release):
    app = _analyze(_app(), "https://mail.g0ogle.com/")
    assert not app.exception
    result = app.session_state["prediction"]
    assert result.category == "Suspicious domain — review required"
    assert result.assessment_basis == "domain_warning"
    assert result.assessment_band == "High"
    assert result.risk_score == pytest.approx(50.)  # unmodified fitted fixture
    assert result.risk_band == "Medium"
    assert result.domain_signals[0]["code"] == "brand_digit_substitution"
    assert any("Verify the address independently" in error.value for error in app.error)
    assert "Domain identity warnings" in [heading.value for heading in app.subheader]
    assert any("unmodified model estimate" in expander.label for expander in app.expander)
