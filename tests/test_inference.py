"""Release integrity and prediction semantics, without datasets or networking."""

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishshield.features import FEATURE_VERSION, URLFeatureExtractor, URLValidationError
from phishshield.inference import (
    ModelBundle,
    ModelLoadError,
    analyze_url,
    file_sha256,
    load_release,
    verify_release,
)


class ReversedProbabilityModel:
    """Expose phishing first, proving that inference follows class labels."""

    classes_ = np.array([1, 0])

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, urls):
        assert list(urls.columns) == ["url"]
        return np.array([[self.probability, 1 - self.probability]] * len(urls))


@pytest.fixture
def release_dir(tmp_path):
    directory = tmp_path / "release"
    directory.mkdir()
    pipeline = Pipeline([
        ("features", URLFeatureExtractor()),
        ("classifier", DummyClassifier(strategy="prior")),
    ])
    pipeline.fit(pd.DataFrame({"url": ["https://example.com/", "http://school.example/login"]}), [0, 1])
    joblib.dump(pipeline, directory / "model.joblib")
    manifest = {
        "feature_version": FEATURE_VERSION,
        "model_sha256": file_sha256(directory / "model.joblib"),
        "dependency_versions": {"scikit-learn": sklearn_version},
        "model_version": "12",
        "run_id": "test-run-id",
        "algorithm": "DummyClassifier",
        "threshold": 0.5,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fixtures = [{"url": "https://example.com/", "phishing_probability": 0.5, "predicted_label": 1}]
    (directory / "known_urls.json").write_text(json.dumps(fixtures), encoding="utf-8")
    return directory


def rewrite_manifest(directory, **changes):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_release_load_and_prediction_match_fitted_pipeline(release_dir):
    bundle = load_release(release_dir)
    result = analyze_url("HTTPS://EXAMPLE.COM/", bundle)
    assert result.normalized_url == "https://example.com"
    assert result.hostname == result.registered_domain == "example.com"
    assert result.phishing_probability == 0.5
    assert result.category == "Potential phishing"
    assert result.risk_band == "Medium"
    assert result.risk_score == 50
    assert result.model_version == "12"
    assert result.algorithm == "DummyClassifier"
    assert result.to_dict()["phishing_probability"] == 0.5
    original = bundle.model.predict_proba(pd.DataFrame({"url": [result.normalized_url]}))[0, 1]
    assert result.phishing_probability == original


def test_missing_release_has_actionable_error(tmp_path):
    with pytest.raises(ModelLoadError, match="train first"):
        load_release(tmp_path / "does-not-exist")


def test_checksum_is_verified_before_deserializing(release_dir):
    with (release_dir / "model.joblib").open("ab") as stream:
        stream.write(b"unexpected modification")
    with patch("phishshield.inference.joblib.load", side_effect=AssertionError("Unsafe model was loaded")):
        with pytest.raises(ModelLoadError, match="checksum"):
            load_release(release_dir)


def test_feature_version_mismatch_is_rejected(release_dir):
    rewrite_manifest(release_dir, feature_version="999.0.0")
    with pytest.raises(ModelLoadError, match="feature version"):
        load_release(release_dir)


def test_dependency_mismatch_is_rejected(release_dir):
    rewrite_manifest(release_dir, dependency_versions={"scikit-learn": "0.0.0"})
    with pytest.raises(ModelLoadError, match="requires scikit-learn 0.0.0"):
        load_release(release_dir)


@pytest.mark.parametrize("threshold", [0, 1, -0.5, float("nan"), float("inf")])
def test_invalid_release_threshold_is_rejected(release_dir, threshold):
    rewrite_manifest(release_dir, threshold=threshold)
    with pytest.raises(ModelLoadError, match="threshold"):
        load_release(release_dir)


def test_malformed_manifest_has_model_load_error(release_dir):
    (release_dir / "manifest.json").write_text("{broken json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="could not be loaded"):
        load_release(release_dir)


def test_unsupported_model_classes_are_rejected(release_dir):
    model = DummyClassifier().fit([[0], [1]], [0, 2])
    joblib.dump(model, release_dir / "model.joblib")
    rewrite_manifest(release_dir, model_sha256=file_sha256(release_dir / "model.joblib"))
    with pytest.raises(ModelLoadError, match="legitimate=0 and phishing=1"):
        load_release(release_dir)


@pytest.mark.parametrize("probability,category,band", [
    (0, "Likely legitimate", "Low"),
    (0.299999, "Likely legitimate", "Low"),
    (0.3, "Likely legitimate", "Medium"),
    (0.49999, "Likely legitimate", "Medium"),
    (0.5, "Potential phishing", "Medium"),
    (0.69999, "Potential phishing", "Medium"),
    (0.7, "Potential phishing", "High"),
    (1, "Potential phishing", "High"),
])
def test_probability_class_order_and_unrounded_boundaries(tmp_path, probability, category, band):
    bundle = ModelBundle(ReversedProbabilityModel(probability), {"threshold": 0.5}, tmp_path)
    result = analyze_url("https://example.com/", bundle)
    assert result.phishing_probability == probability
    assert result.risk_score == probability * 100
    assert result.category == category
    assert result.risk_band == band


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_probabilities_are_rejected(tmp_path, probability):
    bundle = ModelBundle(ReversedProbabilityModel(probability), {}, tmp_path)
    with pytest.raises(ModelLoadError, match="invalid phishing probability"):
        analyze_url("https://example.com/", bundle)


def test_analysis_rejects_invalid_url_before_prediction(tmp_path):
    bundle = ModelBundle(ReversedProbabilityModel(0.5), {}, tmp_path)
    with patch.object(bundle.model, "predict_proba", side_effect=AssertionError("Invalid URL reached the model")):
        with pytest.raises(URLValidationError):
            analyze_url("javascript:alert(1)", bundle)


def test_analysis_never_resolves_or_fetches_submitted_host(release_dir):
    bundle = load_release(release_dir)
    with patch("socket.getaddrinfo", side_effect=AssertionError("Host resolution is forbidden")), patch("requests.get", side_effect=AssertionError("Fetching submitted URL is forbidden")):
        result = analyze_url("https://a.b.c.secure-school.example/verify", bundle)
    assert result.hostname == "a.b.c.secure-school.example"
    assert result.indicators


def test_root_slash_probability_parity_with_fitted_lexical_model(tmp_path):
    model = Pipeline([
        ("features", URLFeatureExtractor()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(pd.DataFrame({"url": [
        "https://school.example", "https://www.example.com", "https://library.example",
        "http://verify-account.example/login", "http://192.0.2.1/password", "http://a.b.c.school.example/verify",
    ]}), [0, 0, 0, 1, 1, 1])
    bundle = ModelBundle(model, {"threshold": 0.5}, tmp_path)
    without_slash = analyze_url("https://www.example.com", bundle)
    with_slash = analyze_url("https://www.example.com/", bundle)
    assert with_slash.normalized_url == without_slash.normalized_url
    assert with_slash.features == without_slash.features
    assert with_slash.phishing_probability == without_slash.phishing_probability
    assert with_slash.category == without_slash.category


def test_verify_release_reports_identity_and_checked_examples(release_dir):
    result = verify_release(release_dir)
    assert result == {
        "status": "passed", "examples_checked": 1,
        "model_version": "12", "run_id": "test-run-id",
        "model_sha256": file_sha256(release_dir / "model.joblib"),
    }


@pytest.mark.parametrize("fixture_format", ["examples", "urls"])
def test_verify_supports_documented_fixture_wrappers(release_dir, fixture_format):
    path = release_dir / "known_urls.json"
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({fixture_format: fixtures}), encoding="utf-8")
    assert verify_release(release_dir)["examples_checked"] == 1


def test_verify_fails_when_probability_changes(release_dir):
    path = release_dir / "known_urls.json"
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    fixtures[0]["phishing_probability"] = 0.6
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="probability differs"):
        verify_release(release_dir)


def test_verify_fails_when_classification_changes(release_dir):
    path = release_dir / "known_urls.json"
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    fixtures[0]["predicted_label"] = 0
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="classification differs"):
        verify_release(release_dir)


@pytest.mark.parametrize("missing", [False, True])
def test_verify_requires_nonempty_fixtures(release_dir, missing):
    path = release_dir / "known_urls.json"
    path.unlink() if missing else path.write_text("[]", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="missing|empty"):
        verify_release(release_dir)
