"""Read-only checks that a local release matches its immutable MLflow version.

The reviewed probabilities are the classifier's estimates. Independent domain
warnings never enter the parity calculation or the model evaluation metrics.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from phishshield.inference import ModelLoadError, file_sha256, load_release, verify_release


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ARTIFACTS = (
    "manifest.json", "model.joblib", "metrics.json", "comparison.csv",
    "baseline_metrics.json", "dataset_summary.json", "split_assignments.csv",
    "known_urls.json", "behavior_review.json", "feature_importance.csv",
    "feature_importance.png", "confusion_matrix.png", "calibration.png",
    "roc_curve.png", "pr_curve.png",
)
METRIC_NAMES = (
    "accuracy", "precision", "recall", "f1", "average_precision", "roc_auc",
    "false_positive_rate", "brier_score", "log_loss",
)
IDENTITY_PARAMS = (
    "dataset_checksum", "seed", "split_seed", "feature_version", "threshold",
    "dataset_kind", "model_scope", "train_rows", "validation_rows",
    "calibration_rows", "test_rows",
)


class MlflowReviewError(ModelLoadError):
    """MLflow lineage or release evidence failed its review checks."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlflowReviewError(message)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal_number(actual: Any, expected: Any, message: str) -> None:
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError) as exc:
        raise MlflowReviewError(message) from exc
    _require(bool(np.isfinite(left) and np.isfinite(right)
                  and np.isclose(left, right, rtol=1e-9, atol=1e-12)), message)


def _finished(run, phase: str) -> None:
    _require(run.info.status == "FINISHED" and run.info.lifecycle_stage == "active",
             f"The {phase} run must be finished and active.")
    _require(run.data.tags.get("phase") == phase, f"The {phase} run has the wrong phase tag.")
    expected_split = "test" if phase == "release" else "validation"
    _require(run.data.tags.get("evaluation_split") == expected_split,
             f"The {phase} run has the wrong evaluation split.")


def _fingerprint_matches(run, release_run) -> bool:
    return (run.info.experiment_id == release_run.info.experiment_id
            and all(run.data.params.get(key) == release_run.data.params.get(key)
                    for key in IDENTITY_PARAMS))


def _metrics_match(run, expected: dict, prefix: str) -> bool:
    for key in METRIC_NAMES:
        try:
            actual = float(run.data.metrics[f"{prefix}_{key}"])
            target = float(expected[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not (np.isfinite(actual) and np.isfinite(target)
                and np.isclose(actual, target, rtol=1e-9, atol=1e-12)):
            return False
    return True


def _search_all_runs(client: MlflowClient, experiment_id: str):
    runs, token = [], None
    while True:
        page = client.search_runs([experiment_id], max_results=1000, page_token=token)
        runs.extend(page)
        token = page.token
        if not token:
            return runs


def _resolve_sources(client, manifest: dict, release_run, comparison: pd.DataFrame, baseline: dict):
    """Prefer immutable IDs; legacy discovery requires an unambiguous full match."""
    explicit = manifest.get("candidate_run_ids")
    algorithms = comparison["model"].tolist()
    if explicit is not None:
        _require(isinstance(explicit, dict) and set(explicit) == set(algorithms),
                 "Candidate run IDs must identify every comparison model exactly once.")
        candidates = {name: client.get_run(explicit[name]) for name in algorithms}
    else:
        runs = _search_all_runs(client, release_run.info.experiment_id)
        candidates = {}
        for row in comparison.to_dict(orient="records"):
            matches = [run for run in runs
                       if run.data.tags.get("phase") == "candidate"
                       and run.data.params.get("algorithm") == row["model"]
                       and _fingerprint_matches(run, release_run)
                       and _metrics_match(run, row, "validation")
                       and run.info.end_time is not None
                       and run.info.end_time <= release_run.info.start_time]
            _require(len(matches) == 1,
                     f"Cannot unambiguously identify the {row['model']} candidate run. Record candidate_run_ids in the manifest.")
            candidates[row["model"]] = matches[0]
    _require(len({run.info.run_id for run in candidates.values()}) == len(algorithms),
             "Each comparison model must have its own candidate run.")
    selection_id = manifest.get("selection_run_id")
    _require(candidates[manifest["algorithm"]].info.run_id == selection_id,
             "The selected candidate run does not match the manifest's selection_run_id.")

    if manifest.get("baseline_run_id"):
        baseline_run = client.get_run(manifest["baseline_run_id"])
    else:
        runs = _search_all_runs(client, release_run.info.experiment_id)
        matches = [run for run in runs
                   if run.data.tags.get("phase") == "baseline"
                   and _fingerprint_matches(run, release_run)
                   and _metrics_match(run, baseline, "validation")
                   and run.info.end_time is not None
                   and run.info.end_time <= min(item.info.start_time for item in candidates.values())]
        _require(len(matches) == 1,
                 "Cannot unambiguously identify the baseline run. Record baseline_run_id in the manifest.")
        baseline_run = matches[0]
    return candidates, baseline_run


def _phishing_probabilities(model, examples: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.classes_)
    indices = np.flatnonzero(classes == 1)
    _require(set(classes) == {0, 1} and len(indices) == 1,
             "The native MLflow model must use legitimate=0 and phishing=1.")
    matrix = np.asarray(model.predict_proba(examples))
    _require(matrix.shape == (len(examples), 2), "The native model returned the wrong probability shape.")
    values = matrix[:, int(indices[0])]
    _require(bool(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()),
             "The native model returned invalid phishing probabilities.")
    return values


@contextmanager
def _model_uris(tracking_uri: str):
    """MLflow's native loader uses global configuration; always put it back."""
    previous_tracking, previous_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        yield
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


def inspect_mlflow_release(release_dir: Path | str, tracking_uri: str | None = None) -> dict[str, Any]:
    """Verify existing evidence without logging runs or changing registry state.

    Raises ``MlflowReviewError`` (a ``ModelLoadError``) if any check fails. The
    successful result is JSON safe and suitable for a CLI or academic review log.
    """
    try:
        return _inspect_mlflow_release(release_dir, tracking_uri)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise MlflowReviewError(f"The MLflow release review failed: {exc}") from exc


def _inspect_mlflow_release(release_dir, tracking_uri):
    bundle = load_release(release_dir)
    manifest, directory = bundle.manifest, bundle.release_dir
    local_verification = verify_release(directory)
    uri = tracking_uri or f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    name, version, run_id = manifest["model_name"], str(manifest["model_version"]), manifest["run_id"]
    registered = client.get_model_version(name, version)
    _require(registered.run_id == run_id, "The registry version points to a different release run.")
    _require(registered.status == "READY", "The registered model is not ready.")
    for key in ("feature_version", "dataset_checksum", "model_sha256"):
        _require(registered.tags.get(key) == str(manifest[key]), f"The registry {key} does not match the release manifest.")
    _require(registered.tags.get("validation_status") == "passed", "The registered model has not passed export validation.")
    champion = client.get_model_version_by_alias(name, "champion")
    _require(str(champion.version) == version, "The champion alias does not point to this release version.")
    release_run = client.get_run(run_id)
    _finished(release_run, "release")
    for key in ("algorithm", "dataset_checksum", "feature_version", "dataset_kind", "model_scope", "split_seed"):
        _require(release_run.data.params.get(key) == str(manifest[key]), f"The release run's {key} does not match its manifest.")
    for manifest_key, parameter in (("training_rows", "train_rows"), ("validation_rows", "validation_rows"),
                                    ("calibration_rows", "calibration_rows"), ("test_rows", "test_rows")):
        _require(release_run.data.params.get(parameter) == str(manifest[manifest_key]), f"The release run's {parameter} count differs.")
    _equal_number(release_run.data.params.get("threshold"), manifest["threshold"], "The release classification threshold differs.")
    _require(release_run.data.params.get("calibration") == "sigmoid", "The release run has the wrong calibration method.")
    if "seed" in manifest:
        _require(release_run.data.params.get("seed") == str(manifest["seed"]), "The release run's seed differs from its manifest.")
    for key in ("selection_run_id", "selection_metric"):
        _require(release_run.data.tags.get(key) == manifest[key], f"The release run's {key} differs from its manifest.")
    _require(manifest["selection_metric"] == "validation_average_precision", "The selection metric must come from validation data.")
    _require(_metrics_match(release_run, manifest["metrics"], "test"), "The logged test metrics differ from the manifest.")

    remote = {item.path.removeprefix("release/"): item
              for item in client.list_artifacts(run_id, "release") if not item.is_dir}
    artifacts = []
    for filename in REQUIRED_ARTIFACTS:
        local = directory / filename
        _require(local.is_file(), f"The local release is missing {filename}.")
        _require(filename in remote, f"MLflow is missing the release/{filename} artifact.")
        _require(remote[filename].file_size == local.stat().st_size,
                 f"The MLflow release/{filename} artifact size differs from the local release.")
        artifacts.append({"path": f"release/{filename}", "bytes": local.stat().st_size,
                          "local_sha256": file_sha256(local)})

    with TemporaryDirectory(prefix="phishshield-mlflow-review-") as temporary:
        logged_manifest = _read_json(Path(client.download_artifacts(run_id, "release/manifest.json", temporary)))
        for key in ("model_name", "model_version", "run_id", "algorithm", "feature_version", "dataset_checksum", "model_sha256", "metrics"):
            _require(logged_manifest.get(key) == manifest.get(key), f"The logged manifest's {key} differs from the local release.")
        comparison = pd.read_csv(directory / "comparison.csv")
        _require("model" in comparison and len(comparison) == 3 and comparison["model"].nunique() == 3,
                 "The release comparison must identify three distinct candidate models.")
        baseline = _read_json(directory / "baseline_metrics.json")
        _require(baseline.get("strategy") == "training_class_prior" and baseline.get("evaluation_split") == "validation",
                 "The baseline must be a training-class-prior evaluated on validation rows.")
        candidates, baseline_run = _resolve_sources(client, manifest, release_run, comparison, baseline)
        candidate_evidence = []
        for row in comparison.to_dict(orient="records"):
            run = candidates[row["model"]]
            _finished(run, "candidate")
            _require(_fingerprint_matches(run, release_run), "A candidate run has a different dataset, feature version, seed, or split.")
            _require(run.info.end_time is not None and run.info.end_time <= release_run.info.start_time,
                     "Candidate evaluation must finish before the release run begins.")
            _require(run.data.params.get("algorithm") == row["model"] and run.data.params.get("calibration") == "none",
                     "A candidate run has the wrong algorithm or calibration setting.")
            _require(_metrics_match(run, row, "validation"), "Candidate metrics differ from comparison.csv.")
            _equal_number(run.data.metrics.get("fit_seconds"), row["fit_seconds"], "Candidate fitting duration differs from comparison.csv.")
            _equal_number(row["validation_rows"], manifest["validation_rows"], "Candidate validation row count differs.")
            if manifest.get("training_cycle_id"):
                _require(run.data.tags.get("training_cycle_id") == manifest["training_cycle_id"], "A candidate belongs to a different training cycle.")
            candidate_evidence.append({"algorithm": row["model"], "run_id": run.info.run_id,
                                       "validation_metrics": {key: float(row[key]) for key in METRIC_NAMES},
                                       "fit_seconds": float(row["fit_seconds"])})
        ranked = comparison.sort_values(["average_precision", "recall"], ascending=False, kind="stable")
        _require(ranked.iloc[0]["model"] == manifest["algorithm"], "The selected algorithm is not the validation winner.")
        _equal_number(ranked.iloc[0]["average_precision"], manifest["selection_score"], "The selection score differs from comparison.csv.")
        _finished(baseline_run, "baseline")
        _require(_fingerprint_matches(baseline_run, release_run), "The baseline run belongs to a different training dataset or split.")
        _require(baseline_run.info.end_time is not None
                 and baseline_run.info.end_time <= min(run.info.start_time for run in candidates.values()),
                 "The baseline must finish before candidate training begins.")
        _require(baseline_run.data.params.get("algorithm") == "Class-prior baseline", "The baseline algorithm differs.")
        _require(_metrics_match(baseline_run, baseline, "validation"), "The baseline metrics differ from its saved artifact.")
        _equal_number(baseline_run.data.params.get("phishing_prior"), baseline["phishing_prior"], "The training-class prior differs.")
        if manifest.get("training_cycle_id"):
            for run in (release_run, baseline_run):
                _require(run.data.tags.get("training_cycle_id") == manifest["training_cycle_id"], "A run belongs to a different training cycle.")
        dataset = _read_json(directory / "dataset_summary.json")
        _require(dataset.get("dataset_checksum") == manifest["dataset_checksum"] and dataset.get("domain_overlap") == 0,
                 "Dataset evidence has a different checksum or overlapping domain groups.")
        _require(dataset.get("split_seed") == manifest["split_seed"], "The dataset split seed differs from the manifest.")
        assignments = pd.read_csv(directory / "split_assignments.csv")
        _require({"domain_group", "split", "label", "url_sha256"}.issubset(assignments.columns),
                 "The split assignments are missing their domain, label, split, or URL-hash evidence.")
        _require(bool(assignments.groupby("domain_group")["split"].nunique().eq(1).all()),
                 "The saved split assignments reuse a domain across partitions.")
        expected_counts = {"train": manifest["training_rows"], "validation": manifest["validation_rows"],
                           "calibration": manifest["calibration_rows"], "test": manifest["test_rows"]}
        _require(assignments["split"].value_counts().to_dict() == expected_counts,
                 "The saved split assignments differ from the manifest's partition counts.")
        _require(_read_json(directory / "metrics.json") == manifest["metrics"], "metrics.json differs from the release manifest.")
        behavior = _read_json(directory / "behavior_review.json")
        _require(behavior.get("not_used_for_selection_or_calibration") is True,
                 "The behavioral review must be separate from model selection and calibration.")
        fixtures = _read_json(directory / "known_urls.json")
        if isinstance(fixtures, dict):
            fixtures = fixtures.get("examples", fixtures.get("urls", []))
        examples = pd.DataFrame({"url": [item["url"] for item in fixtures]})
        expected = np.array([item.get("phishing_probability", item.get("expected_probability")) for item in fixtures], dtype=float)
        native_destination = Path(temporary) / "native"
        native_destination.mkdir()
        with _model_uris(uri):
            native = mlflow.sklearn.load_model(f"models:/{name}/{version}", dst_path=str(native_destination))
        local_probabilities = _phishing_probabilities(bundle.model, examples)
        native_probabilities = _phishing_probabilities(native, examples)
        _require(bool(np.allclose(native_probabilities, local_probabilities, rtol=0, atol=1e-12)),
                 "Native MLflow probabilities differ from the exported release.")
        _require(bool(np.allclose(native_probabilities, expected, rtol=0, atol=1e-12)),
                 "Native MLflow probabilities differ from the frozen regression snapshots.")
        max_difference = float(np.max(np.abs(native_probabilities - local_probabilities)))

    return {
        "status": "passed", "model_name": name, "model_version": int(version), "run_id": run_id,
        "model_sha256": manifest["model_sha256"], "tracking_uri": uri,
        "champion_alias": {"name": "champion", "version": int(champion.version)},
        "selection_run_id": manifest["selection_run_id"], "selection_metric": manifest["selection_metric"],
        "candidates": candidate_evidence,
        "baseline": {"algorithm": "Class-prior baseline", "run_id": baseline_run.info.run_id,
                     "validation_metrics": {key: float(baseline[key]) for key in METRIC_NAMES}},
        "test_metrics": manifest["metrics"], "dataset_checksum": manifest["dataset_checksum"],
        "feature_version": manifest["feature_version"], "artifacts": artifacts,
        "examples_checked": local_verification["examples_checked"],
        "probability_parity": {"status": "passed", "positive_class": 1, "atol": 1e-12,
                               "max_absolute_difference": max_difference,
                               "scope": "raw classifier probabilities; independent domain warnings excluded"},
    }
