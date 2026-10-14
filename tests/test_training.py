import json

import joblib
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phishshield import training
from phishshield.data import file_sha256


def _small_candidates(seed):
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=100, random_state=seed))]),
        "Random Forest": Pipeline([("classifier", RandomForestClassifier(n_estimators=5, max_depth=3, random_state=seed, n_jobs=1))]),
        "Histogram Gradient Boosting": Pipeline([("classifier", HistGradientBoostingClassifier(max_iter=5, min_samples_leaf=2, random_state=seed, early_stopping=False))]),
    }


def test_metrics_take_probability_class_and_threshold_seriously():
    result = training._metrics(np.array([0, 0, 1, 1]), np.array([.1, .7, .5, .9]))
    assert result["recall"] == 1
    assert result["false_positive_rate"] == .5
    assert result["accuracy"] == .75


def test_tiny_training_creates_a_reloadable_registered_release(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "_build_candidates", _small_candidates)
    path = tmp_path / "urls.csv"
    rows = []
    for index in range(160):
        is_phishing = index % 2
        url = (f"http://verify-account-{index}.example/login?password=12345678" if is_phishing
               else f"https://site{index}.example/articles")
        rows.append({"URL": url, "label": 1 - is_phishing})
    pd.DataFrame(rows).to_csv(path, index=False)
    output = tmp_path / "release"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    manifest = training.train_project(path, output, uri, seed=42, importance_samples=12)
    assert manifest["model_sha256"] == file_sha256(output / "model.joblib")
    assert manifest["dataset_checksum"] == file_sha256(path)
    assert sum(manifest[key] for key in ("training_rows", "validation_rows", "calibration_rows", "test_rows")) == 160
    comparison = pd.read_csv(output / "comparison.csv")
    assert len(comparison) == 3
    assert manifest["algorithm"] == comparison.iloc[0]["model"]
    assert "validation" in manifest["selection_metric"]
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    champion = client.get_model_version_by_alias("PhishShield", "champion")
    assert str(champion.version) == str(manifest["model_version"])
    assert champion.tags["validation_status"] == "passed"
    snapshot = json.loads((output / "known_urls.json").read_text())
    pipeline = joblib.load(output / "model.joblib")
    actual = training._probabilities(pipeline, pd.DataFrame({"url": [item["url"] for item in snapshot]}))
    np.testing.assert_allclose(actual, [item["phishing_probability"] for item in snapshot], atol=1e-12, rtol=0)
    for artifact in ("confusion_matrix.png", "calibration.png", "roc_curve.png", "pr_curve.png", "feature_importance.png"):
        assert (output / artifact).stat().st_size > 1000
    assert list(pd.read_csv(output / "feature_importance.csv").columns) == ["feature", "importance_mean", "importance_std"]
    assignments = pd.read_csv(output / "split_assignments.csv")
    assert assignments.groupby("domain_group")["split"].nunique().eq(1).all()
    assert client.get_run(manifest["run_id"]).data.metrics["test_average_precision"] == manifest["metrics"]["average_precision"]
