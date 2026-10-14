import json
from uuid import UUID

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
        rows.append({"URL": url, "status": "phishing" if is_phishing else "legitimate"})
    pd.DataFrame(rows).to_csv(path, index=False)
    output = tmp_path / "release"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    manifest = training.train_project(path, output, uri, seed=42, importance_samples=12)
    assert manifest["model_sha256"] == file_sha256(output / "model.joblib")
    assert manifest["dataset_checksum"] == file_sha256(path)
    assert manifest["dataset_kind"] == "hannousse"
    assert manifest["dataset_provenance"]["csv_sha256"] == file_sha256(path)
    assert manifest["dataset_url_profiles"]["legitimate"]["nonroot_path"] == 1
    assert manifest["assessment_policy"] == "independent_domain_warnings_v1"
    review = json.loads((output / "behavior_review.json").read_text())
    assert review["not_used_for_selection_or_calibration"] is True
    assert "benchmark" in review["purpose"]
    assert review["cases"][0]["domain_warnings"][0]["code"] == "brand_digit_substitution"
    assert all(0 <= case["model_probability"] <= 1 for case in review["cases"])
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

    # A training cycle identifies comparable runs even when the same corpus is retrained.
    assert UUID(manifest["training_cycle_id"]).version == 4
    assert set(manifest["candidate_run_ids"]) == set(_small_candidates(42))
    assert manifest["candidate_run_ids"][manifest["algorithm"]] == manifest["selection_run_id"]
    phase_runs = {manifest["baseline_run_id"]: "baseline", manifest["run_id"]: "release"}
    phase_runs.update({run_id: "candidate" for run_id in manifest["candidate_run_ids"].values()})
    assert len(phase_runs) == 5
    row_counts = {"training": manifest["training_rows"], "validation": manifest["validation_rows"],
                  "calibration": manifest["calibration_rows"], "test": manifest["test_rows"]}
    context_digests = {}
    for run_id, phase in phase_runs.items():
        tracked = client.get_run(run_id)
        assert tracked.info.status == "FINISHED"
        assert tracked.data.tags["phase"] == phase
        assert tracked.data.tags["training_cycle_id"] == manifest["training_cycle_id"]
        assert tracked.data.params["training_cycle_id"] == manifest["training_cycle_id"]
        inputs = tracked.inputs.dataset_inputs
        expected_contexts = set(row_counts) if phase == "release" else {"training", "validation"}
        by_context = {dict((tag.key, tag.value) for tag in item.tags)["mlflow.data.context"]: item.dataset
                      for item in inputs}
        assert set(by_context) == expected_contexts
        assert len(inputs) == len(expected_contexts)
        for context, dataset in by_context.items():
            partition = "train" if context == "training" else context
            assert dataset.name == f"hannousse-{partition}"
            assert json.loads(dataset.source)["uri"] == path.resolve().as_uri()
            assert dataset.source_type == "local"
            assert json.loads(dataset.profile)["num_rows"] == row_counts[context]
            assert [column["name"] for column in json.loads(dataset.schema)["mlflow_colspec"]] == ["url", "label"]
            assert dataset.digest
            if context in context_digests:
                assert dataset.digest == context_digests[context]
            context_digests[context] = dataset.digest
    assert len(set(context_digests.values())) == 4
