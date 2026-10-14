import json
import shutil
from types import SimpleNamespace
import zipfile

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phishshield import academic_review, mlflow_review, training


def _tiny_candidates(seed):
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200, random_state=seed)),
        ]),
        "Random Forest": Pipeline([("classifier", RandomForestClassifier(
            n_estimators=5, max_depth=3, random_state=seed, n_jobs=1,
        ))]),
        "Histogram Gradient Boosting": Pipeline([("classifier", HistGradientBoostingClassifier(
            max_iter=5, min_samples_leaf=2, random_state=seed, early_stopping=False,
        ))]),
    }


@pytest.fixture(scope="module")
def real_release(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mlflow-review")
    csv = directory / "urls.csv"
    rows = [{"URL": (f"http://verify-account-{index}.example/login?password=12345678" if index % 2
                     else f"https://site{index}.example/articles"),
             "status": "phishing" if index % 2 else "legitimate"}
            for index in range(160)]
    pd.DataFrame(rows).to_csv(csv, index=False)
    uri = f"sqlite:///{directory / 'mlflow.db'}"
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
        patch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
        patch.setattr(training, "_build_candidates", _tiny_candidates)
        try:
            manifest = training.train_project(csv, directory / "release", uri, seed=42, importance_samples=12)
        finally:
            mlflow.set_tracking_uri(old_tracking)
            mlflow.set_registry_uri(old_registry)
    return directory / "release", uri, manifest


class _ClientWrapper:
    def __init__(self, client, **overrides):
        self._client = client
        self._overrides = overrides

    def __getattr__(self, name):
        return self._overrides.get(name, getattr(self._client, name))


def _replace_client(monkeypatch, uri, **overrides):
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    wrapper = _ClientWrapper(client, **overrides)
    monkeypatch.setattr(mlflow_review, "MlflowClient", lambda **kwargs: wrapper)
    return client


def test_real_review_matches_native_model_and_restores_global_uris(real_release):
    directory, uri, manifest = real_release
    before = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    review = mlflow_review.inspect_mlflow_release(directory, uri)
    assert (mlflow.get_tracking_uri(), mlflow.get_registry_uri()) == before
    assert review["status"] == "passed"
    assert review["model_version"] == manifest["model_version"]
    assert review["run_id"] == manifest["run_id"]
    assert review["examples_checked"] == 20
    assert review["probability_parity"]["max_absolute_difference"] < 1e-12
    assert review["probability_parity"]["positive_class"] == 1
    assert len(review["candidates"]) == 3
    assert len(review["artifacts"]) == len(mlflow_review.REQUIRED_ARTIFACTS)
    assert review["baseline"]["validation_metrics"]["average_precision"] <= 1
    assert review["test_metrics"] == manifest["metrics"]
    json.dumps(review, allow_nan=False)


def test_wrong_champion_is_rejected(real_release, monkeypatch):
    directory, uri, _ = real_release
    _replace_client(monkeypatch, uri, get_model_version_by_alias=lambda *args: SimpleNamespace(version="999"))
    with pytest.raises(mlflow_review.MlflowReviewError, match="champion alias"):
        mlflow_review.inspect_mlflow_release(directory, uri)


def test_wrong_registry_run_is_rejected(real_release, monkeypatch):
    directory, uri, _ = real_release
    _replace_client(monkeypatch, uri, get_model_version=lambda *args: SimpleNamespace(run_id="unrelated"))
    with pytest.raises(mlflow_review.MlflowReviewError, match="different release run"):
        mlflow_review.inspect_mlflow_release(directory, uri)


def test_changed_release_algorithm_is_rejected(real_release, monkeypatch):
    directory, uri, manifest = real_release
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    def changed_run(run_id):
        run = client.get_run(run_id)
        if run_id == manifest["run_id"]:
            run.data.params["algorithm"] = "Fabricated algorithm"
        return run
    _replace_client(monkeypatch, uri, get_run=changed_run)
    with pytest.raises(mlflow_review.MlflowReviewError, match="algorithm.*manifest"):
        mlflow_review.inspect_mlflow_release(directory, uri)


def test_missing_remote_chart_is_rejected(real_release, monkeypatch):
    directory, uri, _ = real_release
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    def filtered_artifacts(run_id, path):
        return [item for item in client.list_artifacts(run_id, path) if not item.path.endswith("calibration.png")]
    _replace_client(monkeypatch, uri, list_artifacts=filtered_artifacts)
    with pytest.raises(mlflow_review.MlflowReviewError, match="missing.*calibration.png"):
        mlflow_review.inspect_mlflow_release(directory, uri)


def test_native_probability_mismatch_is_rejected_and_uris_are_restored(real_release, monkeypatch):
    directory, uri, _ = real_release
    class ChangedModel:
        classes_ = np.array([0, 1])
        def predict_proba(self, examples):
            return np.full((len(examples), 2), .5)
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda *args, **kwargs: ChangedModel())
    before = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    with pytest.raises(mlflow_review.MlflowReviewError, match="probabilities differ"):
        mlflow_review.inspect_mlflow_release(directory, uri)
    assert (mlflow.get_tracking_uri(), mlflow.get_registry_uri()) == before


def test_probability_column_is_selected_by_class_label():
    class ReversedModel:
        classes_ = np.array([1, 0])
        def predict_proba(self, examples):
            return np.array([[.9, .1], [.2, .8]])
    actual = mlflow_review._phishing_probabilities(ReversedModel(), pd.DataFrame({"url": ["one", "two"]}))
    np.testing.assert_allclose(actual, [.9, .2])


def test_invalid_native_probabilities_are_rejected():
    class InvalidModel:
        classes_ = np.array([0, 1])
        def predict_proba(self, examples):
            return np.array([[0, np.nan]])
    with pytest.raises(mlflow_review.MlflowReviewError, match="invalid phishing probabilities"):
        mlflow_review._phishing_probabilities(InvalidModel(), pd.DataFrame({"url": ["one"]}))


def test_ambiguous_legacy_candidate_linkage_is_rejected(real_release):
    directory, uri, manifest = real_release
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    release_run = client.get_run(manifest["run_id"])
    runs = list(client.search_runs([release_run.info.experiment_id]))
    class DuplicateRuns:
        def search_runs(self, *args, **kwargs):
            duplicate = next(run for run in runs if run.data.tags.get("phase") == "candidate")
            class Page(list):
                token = None
            return Page(runs + [duplicate])
    legacy = {key: value for key, value in manifest.items() if key not in ("candidate_run_ids", "baseline_run_id")}
    with pytest.raises(mlflow_review.MlflowReviewError, match="unambiguously identify"):
        mlflow_review._resolve_sources(DuplicateRuns(), legacy, release_run,
            pd.read_csv(directory / "comparison.csv"), json.loads((directory / "baseline_metrics.json").read_text()))


def test_incomplete_explicit_candidate_ids_are_rejected(real_release):
    directory, uri, manifest = real_release
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    changed = {**manifest, "candidate_run_ids": {manifest["algorithm"]: manifest["selection_run_id"]}}
    with pytest.raises(mlflow_review.MlflowReviewError, match="every comparison model"):
        mlflow_review._resolve_sources(client, changed, client.get_run(manifest["run_id"]),
            pd.read_csv(directory / "comparison.csv"), json.loads((directory / "baseline_metrics.json").read_text()))


def _historical_runs(client, manifest):
    experiment_id = client.get_run(manifest["run_id"]).info.experiment_id
    runs = client.search_runs([experiment_id])
    return {run.info.run_id: run for run in runs
            if run.data.tags.get("phase") in {"baseline", "candidate", "release"}}


def _run_snapshot(run):
    return {"start": run.info.start_time, "end": run.info.end_time, "status": run.info.status,
            "params": dict(run.data.params), "metrics": dict(run.data.metrics)}


def test_academic_workflow_records_lineage_promotes_and_creates_real_evidence(real_release, tmp_path):
    directory, uri, manifest = real_release
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    before = {run_id: _run_snapshot(run) for run_id, run in _historical_runs(client, manifest).items()}
    result = academic_review.run_academic_review(
        directory, tracking_uri=uri, dataset_path=directory.parent / "urls.csv",
        output_dir=tmp_path / "evidence", promote=True,
    )
    assert result["status"] == "passed"
    assert result["models_retrained"] is False
    assert result["training_runs_reopened"] is False
    assert len(result["dataset_lineage"]) == 5
    assert {run_id: _run_snapshot(client.get_run(run_id)) for run_id in before} == before
    for evidence in result["dataset_lineage"]:
        run = client.get_run(evidence["run_id"])
        contexts = {tag.value for item in run.inputs.dataset_inputs for tag in item.tags
                    if tag.key == "mlflow.data.context"}
        expected_contexts = ({"training", "validation", "calibration", "test"}
                             if evidence["run_id"] == manifest["run_id"] else {"training", "validation"})
        assert contexts >= expected_contexts
        assert set(evidence["contexts"]) == expected_contexts
        assert all(dataset["rows"] > 0 for dataset in evidence["datasets"])
    for alias in ("staging", "production"):
        promoted = client.get_model_version_by_alias(manifest["model_name"], alias)
        assert str(promoted.version) == str(manifest["model_version"])
        assert promoted.run_id == manifest["run_id"]
    review = client.get_run(result["review_run_id"])
    assert review.info.status == "FINISHED"
    assert review.data.tags["phase"] == "academic_review"
    assert review.data.tags["review_status"] == "passed"
    assert review.data.tags["source_release_run_id"] == manifest["run_id"]
    assert review.data.metrics["parity_examples_checked"] == 20
    assert review.data.metrics["dataset_input_runs_verified"] == 5
    assert review.data.params["model_sha256"] == manifest["model_sha256"]
    artifact_names = {artifact.path for artifact in client.list_artifacts(review.info.run_id, "academic_review")}
    assert "academic_review/PhishShield-MLflow-Evidence.zip" in artifact_names
    with zipfile.ZipFile(result["evidence_zip"]) as archive:
        assert archive.testzip() is None
        assert {"index.html", "review.json", "promotion.json", "comparison.csv", "calibration.png",
                "confusion_matrix.png", "feature_importance.png"} <= set(archive.namelist())
        saved_report = json.loads(archive.read("review.json"))
        assert saved_report["test_metrics"] == manifest["metrics"]
        html = archive.read("index.html").decode("utf-8")
        assert f'{manifest["metrics"]["accuracy"] * 100:.2f}%' in html
        assert manifest["run_id"] in html
        assert 'data:image/png;base64,' in html
        assert "decision aid" in html


def test_repeated_academic_review_does_not_duplicate_dataset_inputs(real_release, tmp_path):
    directory, uri, manifest = real_release
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    first = academic_review.run_academic_review(
        directory, tracking_uri=uri, dataset_path=directory.parent / "urls.csv",
        output_dir=tmp_path / "first", promote=False,
    )
    before = {run_id: len(run.inputs.dataset_inputs) for run_id, run in _historical_runs(client, manifest).items()}
    aliases = dict(client.get_registered_model(manifest["model_name"]).aliases)
    second = academic_review.run_academic_review(
        directory, tracking_uri=uri, dataset_path=directory.parent / "urls.csv",
        output_dir=tmp_path / "second", promote=False,
    )
    assert first["review_run_id"] != second["review_run_id"]
    assert all(evidence["added_inputs"] == 0 for evidence in second["dataset_lineage"])
    assert {run_id: len(client.get_run(run_id).inputs.dataset_inputs) for run_id in before} == before
    assert dict(client.get_registered_model(manifest["model_name"]).aliases) == aliases
    assert second["promotion"]["events"] == []


def _forbid_review_mutations(monkeypatch, uri):
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("An invalid review attempted to mutate MLflow state.")
    overrides = {name: forbidden for name in (
        "log_inputs", "create_run", "set_tag", "set_registered_model_alias", "set_model_version_tag",
    )}
    wrapper = _ClientWrapper(client, **overrides)
    monkeypatch.setattr(academic_review, "MlflowClient", lambda **kwargs: wrapper)
    return client, calls


def test_corrupt_export_aborts_before_review_logging_or_alias_mutation(real_release, tmp_path, monkeypatch):
    directory, uri, manifest = real_release
    damaged = tmp_path / "damaged-release"
    shutil.copytree(directory, damaged)
    with (damaged / "model.joblib").open("ab") as stream:
        stream.write(b"checksum mismatch")
    client, calls = _forbid_review_mutations(monkeypatch, uri)
    aliases = dict(client.get_registered_model(manifest["model_name"]).aliases)
    experiment_id = client.get_run(manifest["run_id"]).info.experiment_id
    run_count = len(client.search_runs([experiment_id]))
    with pytest.raises(mlflow_review.ModelLoadError, match="checksum"):
        academic_review.run_academic_review(
            damaged, tracking_uri=uri, dataset_path=directory.parent / "urls.csv",
            output_dir=tmp_path / "invalid-evidence", promote=True,
        )
    assert calls == []
    assert len(client.search_runs([experiment_id])) == run_count
    assert dict(client.get_registered_model(manifest["model_name"]).aliases) == aliases
    assert not (tmp_path / "invalid-evidence").exists()


def test_wrong_csv_checksum_aborts_before_dataset_logging_or_promotion(real_release, tmp_path, monkeypatch):
    directory, uri, manifest = real_release
    different_csv = tmp_path / "different-urls.csv"
    different_csv.write_bytes((directory.parent / "urls.csv").read_bytes() + b"\n")
    client, calls = _forbid_review_mutations(monkeypatch, uri)
    aliases = dict(client.get_registered_model(manifest["model_name"]).aliases)
    experiment_id = client.get_run(manifest["run_id"]).info.experiment_id
    runs_before = len(client.search_runs([experiment_id]))
    inputs_before = {run_id: len(run.inputs.dataset_inputs) for run_id, run in _historical_runs(client, manifest).items()}
    with pytest.raises(mlflow_review.MlflowReviewError, match="CSV checksum"):
        academic_review.run_academic_review(
            directory, tracking_uri=uri, dataset_path=different_csv,
            output_dir=tmp_path / "invalid-evidence", promote=True,
        )
    assert calls == []
    assert {run_id: len(client.get_run(run_id).inputs.dataset_inputs) for run_id in inputs_before} == inputs_before
    assert len(client.search_runs([experiment_id])) == runs_before
    assert dict(client.get_registered_model(manifest["model_name"]).aliases) == aliases


@pytest.mark.parametrize("relationship", ["same", "parent", "child"])
def test_evidence_output_overlap_is_rejected_before_inspection_or_mlflow_access(tmp_path, monkeypatch, relationship):
    release = tmp_path / "releases" / "current"
    release.mkdir(parents=True)
    marker = release / "frozen-marker.txt"
    marker.write_text("preserve this release", encoding="utf-8")
    output = {"same": release, "parent": release.parent, "child": release / "evidence"}[relationship]
    calls = []
    def forbidden_inspection(*args, **kwargs):
        calls.append("inspection")
        raise AssertionError("An overlapping output reached release inspection.")
    def forbidden_client(*args, **kwargs):
        calls.append("client")
        raise AssertionError("An overlapping output reached MLflow.")
    monkeypatch.setattr(academic_review, "inspect_mlflow_release", forbidden_inspection)
    monkeypatch.setattr(academic_review, "MlflowClient", forbidden_client)
    with pytest.raises(mlflow_review.MlflowReviewError, match="must be separate.*frozen release"):
        academic_review.run_academic_review(
            release, tracking_uri="sqlite:///unused.db", output_dir=output, promote=True,
        )
    assert calls == []
    assert marker.read_text(encoding="utf-8") == "preserve this release"
    assert list(release.iterdir()) == [marker]
