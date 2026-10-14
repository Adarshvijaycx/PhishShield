"""Reproducible candidate comparison, held-out calibration, and MLflow releases."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from phishshield.data import (
    SPLIT_FRACTIONS, file_sha256, prepare_dataset, split_dataset, split_summary,
    training_tld_statistics,
)
from phishshield.features import FEATURE_NAMES, FEATURE_VERSION, URLFeatureExtractor
from phishshield.domain_signals import DOMAIN_SIGNAL_VERSION, get_domain_signals


MODEL_NAME = "PhishShield"
THRESHOLD = 0.5
RISK_BANDS = {"low": 30, "high": 70}
INFERENCE_DISTRIBUTIONS = ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "tldextract")


def _build_candidates(seed: int) -> dict[str, Pipeline]:
    """A small, documented comparison; tests replace this factory with tiny models."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "Random Forest": Pipeline([("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=2,
        ))]),
        "Histogram Gradient Boosting": Pipeline([("classifier", HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=seed, early_stopping=False,
        ))]),
    }


def _probabilities(model, features: pd.DataFrame) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(indices) != 1:
        raise ValueError("The model must have exactly one phishing class labeled 1.")
    probabilities = np.asarray(model.predict_proba(features))[:, int(indices[0])]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("The model returned invalid phishing probabilities.")
    return probabilities


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "false_positive_rate": float(fp / (tn + fp)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _dependency_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in INFERENCE_DISTRIBUTIONS}
    versions["mlflow"] = importlib.metadata.version("mlflow")
    versions["python"] = __import__("platform").python_version()
    return versions


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _configure_tracking(tracking_uri: str | None) -> str:
    uri = tracking_uri or f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MODEL_NAME)
    if experiment is None:
        # The default project writes all large artifacts outside its SQL database.
        # Explicit test/custom SQLite databases use a neighboring artifact folder.
        artifact_root = PROJECT_ROOT / "mlartifacts"
        if tracking_uri and tracking_uri.startswith("sqlite:////"):
            database_path = Path(tracking_uri.removeprefix("sqlite:///"))
            artifact_root = database_path.parent / "mlartifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        client.create_experiment(MODEL_NAME, artifact_location=artifact_root.resolve().as_uri())
    mlflow.set_experiment(MODEL_NAME)
    return uri


def _log_model(model: Pipeline, example: pd.DataFrame, versions: dict[str, str]):
    return mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="cloudpickle",
        signature=infer_signature(example, model.predict_proba(example)),
        input_example=example,
        pyfunc_predict_fn="predict_proba",
        code_paths=[str(Path(__file__).resolve().parent)],
        pip_requirements=[f"{name}=={versions[name]}" for name in INFERENCE_DISTRIBUTIONS],
        metadata={"feature_version": FEATURE_VERSION, "positive_class": 1, "threshold": THRESHOLD, "model_scope": "full_url"},
    )


def _save_evaluation_charts(output: Path, labels: np.ndarray, probabilities: np.ndarray) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "figure.dpi": 140})
    predicted = (probabilities >= THRESHOLD).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.4, 4.3), layout="constrained")
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legitimate", "Phishing"],
             yticklabels=["Legitimate", "Phishing"], xlabel="Predicted class", ylabel="Actual class",
             title="Held-out test confusion matrix")
    for (row, column), count in np.ndenumerate(matrix):
        axis.text(column, row, f"{count:,}", ha="center", va="center",
                  color="white" if count > matrix.max() / 2 else "#15324f", fontsize=14)
    figure.savefig(output / "confusion_matrix.png")
    plt.close(figure)

    true_fraction, mean_probability = calibration_curve(labels, probabilities, n_bins=10, strategy="quantile")
    figure, axis = plt.subplots(figsize=(5.8, 4.3), layout="constrained")
    axis.plot([0, 1], [0, 1], "--", color="#8293a6", label="Perfect calibration")
    axis.plot(mean_probability, true_fraction, "o-", color="#3974e8", label="Calibrated winner")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted phishing probability",
             ylabel="Observed phishing fraction", title="Held-out test calibration")
    axis.legend(loc="upper left")
    figure.savefig(output / "calibration.png")
    plt.close(figure)

    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5.8, 4.3), layout="constrained")
    axis.plot(false_positive, true_positive, color="#3974e8",
              label=f"ROC-AUC = {roc_auc_score(labels, probabilities):.3f}")
    axis.plot([0, 1], [0, 1], "--", color="#8293a6")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
             title="Held-out test ROC curve")
    axis.legend(loc="lower right")
    figure.savefig(output / "roc_curve.png")
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5.8, 4.3), layout="constrained")
    axis.plot(recall, precision, color="#3974e8",
              label=f"Average precision = {average_precision_score(labels, probabilities):.3f}")
    axis.axhline(labels.mean(), linestyle="--", color="#8293a6", label="Class prevalence")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Phishing recall", ylabel="Phishing precision",
             title="Held-out test precision–recall curve")
    axis.legend(loc="lower left")
    figure.savefig(output / "pr_curve.png")
    plt.close(figure)


def _save_importance(
    output: Path, model, features: pd.DataFrame, labels: np.ndarray,
    *, samples: int, seed: int,
) -> pd.DataFrame:
    if samples < 2:
        raise ValueError("importance_samples must be at least 2.")
    # Sampling is separate from model selection and never touches test rows.
    if len(features) > samples:
        from sklearn.model_selection import train_test_split
        chosen, _ = train_test_split(
            np.arange(len(features)), train_size=samples, random_state=seed, stratify=labels
        )
        features, labels = features.iloc[chosen], labels[chosen]
    result = permutation_importance(
        model, features, labels, scoring="average_precision", n_repeats=3,
        random_state=seed, n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": list(FEATURE_NAMES),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    table.to_csv(output / "feature_importance.csv", index=False)
    top = table.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axis.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3974e8")
    axis.axvline(0, color="#8293a6", linewidth=0.8)
    axis.set(xlabel="Decrease in validation average precision after permutation",
             title="Global feature importance — top 10")
    figure.savefig(output / "feature_importance.png")
    plt.close(figure)
    return table



def _behavior_review(model: Pipeline) -> dict:
    """Report known failure patterns separately from labeled test metrics.

    These cases never select a candidate, fit calibration, or label an unknown
    domain as malicious. They record probabilities and spelling observations.
    """
    urls = [
        "https://mail.g0ogle.com/",
        "https://mail.googletex.com/",
        "https://www.youtube.com/watch?v=HsZxZ8hoYdU",
        "https://mail.google.com/",
        "https://www.google.com/search?q=python",
        "https://www.example.com/about?topic=python#intro",
        "https://www.example.com/",
        "http://secure-account.example.org/login/verify?account=update",
        "http://example.com@192.0.2.10/verify-account",
        "https://youtube.com.unrelated.test/watch?v=example",
    ]
    probabilities = _probabilities(model, pd.DataFrame({"url": urls}))
    cases = []
    for url, probability in zip(urls, probabilities):
        score = float(probability * 100)
        cases.append({
            "url": url,
            "model_probability": float(probability),
            "model_score": score,
            "model_band": "Low" if score < 30 else "Medium" if score < 70 else "High",
            "domain_warnings": [signal.to_dict() for signal in get_domain_signals(url)],
        })
    return {
        "purpose": "Behavioral review; not an independently labeled phishing benchmark",
        "not_used_for_selection_or_calibration": True,
        "domain_warning_version": DOMAIN_SIGNAL_VERSION,
        "cases": cases,
    }


def _source_hashes() -> dict[str, str]:
    files = list((PROJECT_ROOT / "src" / "phishshield").glob("*.py"))
    files += [PROJECT_ROOT / name for name in ("app.py", "pyproject.toml", "requirements.lock")]
    return {str(file.relative_to(PROJECT_ROOT)): file_sha256(file) for file in sorted(files) if file.is_file()}


def train_project(
    csv_path: Path, output_dir: Path, tracking_uri: str | None = None,
    max_rows: int | None = None, seed: int = 43, importance_samples: int = 2000,
) -> dict:
    """Create one validated release without selecting on calibration/test data."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(csv_path, max_rows=max_rows, seed=seed)
    frame = split_dataset(prepared.frame, seed=seed)
    summaries = split_summary(frame)
    datasets = {name: frame.loc[frame["split"] == name] for name in SPLIT_FRACTIONS}
    feature_extractor = URLFeatureExtractor().fit(datasets["train"][["url"]])
    matrix = feature_extractor.transform(frame[["url"]])
    matrix.index = frame.index
    features = {name: matrix.loc[partition.index] for name, partition in datasets.items()}
    labels = {name: partition["label"].to_numpy(dtype=int) for name, partition in datasets.items()}
    versions = _dependency_versions()
    _configure_tracking(tracking_uri)
    training_cycle_id = uuid4().hex
    dataset_kind = prepared.summary.get("dataset_kind", "phiusiil")
    input_datasets = {
        name: mlflow.data.from_pandas(
            partition[["url", "label"]].reset_index(drop=True),
            source=Path(csv_path).expanduser().resolve().as_uri(),
            name=f"{dataset_kind}-{name}", targets="label",
        )
        for name, partition in datasets.items()
    }

    def log_inputs(partitions):
        # Record only URL/target lineage; no supplied webpage features enter the model.
        for partition in partitions:
            context = "training" if partition == "train" else partition
            mlflow.log_input(input_datasets[partition], context=context)

    common_tags = {"training_cycle_id": training_cycle_id, "dataset_source": prepared.source}
    common_params = {
        "training_cycle_id": training_cycle_id,
        "seed": seed, "split_seed": frame.attrs["split_seed"], "feature_version": FEATURE_VERSION,
        "dataset_checksum": prepared.checksum, "threshold": THRESHOLD, "model_scope": "full_url",
        "dataset_kind": prepared.summary.get("dataset_kind", "phiusiil"),
        **{f"{name}_rows": len(partition) for name, partition in datasets.items()},
    }
    summary = {**prepared.summary, "dataset_checksum": prepared.checksum,
               "requested_split_fractions": SPLIT_FRACTIONS, "splits": summaries,
               "split_seed": frame.attrs["split_seed"], "domain_overlap": 0}
    _write_json(output / "dataset_summary.json", summary)
    assignments = frame[["row_id", "domain_group", "label", "split"]].copy()
    assignments["url_sha256"] = frame["url"].map(lambda value: __import__("hashlib").sha256(value.encode()).hexdigest())
    assignments.to_csv(output / "split_assignments.csv", index=False)
    example = pd.DataFrame({"url": ["https://example.com/", "http://verify-account.example/login"]})

    candidate_models, candidate_rows, candidate_runs = {}, [], {}
    # Bound native-library threads to keep a local laptop responsive and runs reproducible.
    with threadpool_limits(limits=2):
        prior = float(labels["train"].mean())
        baseline_metrics = _metrics(labels["validation"], np.full(len(labels["validation"]), prior))
        _write_json(output / "baseline_metrics.json", {"strategy": "training_class_prior", "phishing_prior": prior,
                                                        "evaluation_split": "validation", **baseline_metrics})
        with mlflow.start_run(run_name="baseline-class-prior") as baseline_run:
            baseline_run_id = baseline_run.info.run_id
            mlflow.set_tags({**common_tags, "phase": "baseline", "evaluation_split": "validation"})
            log_inputs(("train", "validation"))
            mlflow.log_params({**common_params, "algorithm": "Class-prior baseline", "phishing_prior": prior})
            mlflow.log_metrics({f"validation_{key}": value for key, value in baseline_metrics.items()})
            mlflow.log_artifact(str(output / "baseline_metrics.json"))
        for algorithm, numeric_pipeline in _build_candidates(seed).items():
            print(f"Training {algorithm} on {len(datasets['train']):,} rows...", flush=True)
            with mlflow.start_run(run_name=f"candidate-{algorithm}") as run:
                mlflow.set_tags({**common_tags, "phase": "candidate", "evaluation_split": "validation"})
                log_inputs(("train", "validation"))
                mlflow.log_params({**common_params, "algorithm": algorithm, "calibration": "none"})
                mlflow.log_dict({key: str(value) for key, value in numeric_pipeline.get_params(deep=True).items()}, "estimator_parameters.json")
                started = time.perf_counter()
                numeric_pipeline.fit(features["train"], labels["train"])
                fit_seconds = time.perf_counter() - started
                probabilities = _probabilities(numeric_pipeline, features["validation"])
                validation_metrics = _metrics(labels["validation"], probabilities)
                row = {"model": algorithm, **validation_metrics, "fit_seconds": fit_seconds,
                       "validation_rows": len(datasets["validation"])}
                mlflow.log_metrics({f"validation_{key}": value for key, value in validation_metrics.items()})
                mlflow.log_metric("fit_seconds", fit_seconds)
                mlflow.log_dict(summary, "dataset_summary.json")
                raw_pipeline = Pipeline([("features", feature_extractor), ("classifier", numeric_pipeline)])
                _log_model(raw_pipeline, example, versions)
                candidate_models[algorithm], candidate_runs[algorithm] = numeric_pipeline, run.info.run_id
                candidate_rows.append(row)
        comparison = pd.DataFrame(candidate_rows)
        comparison = comparison.sort_values(["average_precision", "recall"], ascending=False, kind="stable").reset_index(drop=True)
        comparison.to_csv(output / "comparison.csv", index=False)
        algorithm = str(comparison.iloc[0]["model"])
        print(f"Selected {algorithm} by validation average precision; calibrating on held-out calibration rows...", flush=True)
        winner = candidate_models[algorithm]
        calibrated = CalibratedClassifierCV(FrozenEstimator(winner), method="sigmoid")
        calibration_started = time.perf_counter()
        calibrated.fit(features["calibration"], labels["calibration"])
        calibration_seconds = time.perf_counter() - calibration_started
        full_pipeline = Pipeline([("features", feature_extractor), ("classifier", calibrated)])
        # The frozen winner is evaluated once; these same probabilities drive every test chart and snapshot.
        test_probabilities = _probabilities(calibrated, features["test"])
        test_metrics = _metrics(labels["test"], test_probabilities)
        _write_json(output / "metrics.json", test_metrics)
        _save_evaluation_charts(output, labels["test"], test_probabilities)
        _save_importance(output, calibrated, features["validation"], labels["validation"],
                         samples=importance_samples, seed=seed)

        with mlflow.start_run(run_name=f"calibrated-winner-{algorithm}") as run:
            mlflow.set_tags({**common_tags, "phase": "release", "evaluation_split": "test",
                             "selection_run_id": candidate_runs[algorithm], "selection_metric": "validation_average_precision"})
            log_inputs(("train", "validation", "calibration", "test"))
            mlflow.log_params({**common_params, "algorithm": algorithm, "calibration": "sigmoid",
                               "importance_samples": min(importance_samples, len(datasets["validation"])),
                               "importance_repeats": 3})
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
            mlflow.log_metric("calibration_seconds", calibration_seconds)
            info = _log_model(full_pipeline, example, versions)
            registered = mlflow.register_model(info.model_uri, MODEL_NAME)
            client = MlflowClient()
            version = str(registered.version)
            client.set_registered_model_alias(MODEL_NAME, "candidate", version)
            client.set_model_version_tag(MODEL_NAME, version, "validation_status", "pending")
            client.set_model_version_tag(MODEL_NAME, version, "feature_version", FEATURE_VERSION)
            client.set_model_version_tag(MODEL_NAME, version, "dataset_checksum", prepared.checksum)

            # Export the actual registered candidate, so runtime lineage describes the shipped object.
            registered_pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
            model_path = output / "model.joblib"
            joblib.dump(registered_pipeline, model_path, compress=3)
            created_at = datetime.now(timezone.utc).isoformat()
            manifest = {
                "model_name": MODEL_NAME, "model_version": int(version), "run_id": run.info.run_id,
                "algorithm": algorithm, "created_at": created_at, "feature_version": FEATURE_VERSION,
                "dataset_checksum": prepared.checksum, "dataset_source": prepared.source,
                "dataset_kind": prepared.summary.get("dataset_kind", "phiusiil"),
                "dataset_provenance": prepared.summary.get("dataset_provenance", prepared.summary.get("provenance", {})),
                "dataset_url_profiles": prepared.summary.get("url_format_profile_by_class", {}),
                "model_scope": "full_url", "model_features": list(FEATURE_NAMES),
                "assessment_policy": "independent_domain_warnings_v1",
                "probability_scope": "Historical full-URL lexical estimate; no live reputation or website verification",
                "model_sha256": file_sha256(model_path), "threshold": THRESHOLD, "risk_bands": RISK_BANDS,
                "metrics": test_metrics, "tld_statistics": training_tld_statistics(frame),
                "dependency_versions": versions, "code_revision": _git_revision(),
                "source_file_sha256": _source_hashes(),
                "domain_warning_version": DOMAIN_SIGNAL_VERSION,
                "selection_run_id": candidate_runs[algorithm], "selection_metric": "validation_average_precision",
                "training_cycle_id": training_cycle_id, "baseline_run_id": baseline_run_id,
                "candidate_run_ids": candidate_runs,
                "selection_score": float(comparison.iloc[0]["average_precision"]),
                "split_seed": frame.attrs["split_seed"],
                "training_rows": len(datasets["train"]), "validation_rows": len(datasets["validation"]),
                "calibration_rows": len(datasets["calibration"]), "test_rows": len(datasets["test"]),
            }
            _write_json(output / "manifest.json", manifest)
            _write_json(output / "behavior_review.json", _behavior_review(registered_pipeline))
            snapshots = []
            test_partition = datasets["test"].reset_index(drop=True)
            selected_indices = []
            for label in (0, 1):
                selected_indices.extend(test_partition.index[test_partition["label"] == label][:10].tolist())
            for index in selected_indices:
                probability = float(test_probabilities[index])
                predicted = int(probability >= THRESHOLD)
                snapshots.append({
                    "url": str(test_partition.loc[index, "url"]), "label": predicted,
                    "phishing_probability": probability,
                    "category": "Potential phishing" if predicted else "Likely legitimate",
                    "source": prepared.source,
                })
            _write_json(output / "known_urls.json", snapshots)
            reloaded = joblib.load(model_path)
            snapshot_input = pd.DataFrame({"url": [item["url"] for item in snapshots]})
            expected = np.array([item["phishing_probability"] for item in snapshots])
            actual = _probabilities(reloaded, snapshot_input)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12,
                                       err_msg="Registered model export/reload predictions changed.")
            if file_sha256(model_path) != manifest["model_sha256"]:
                raise RuntimeError("The exported model checksum changed during release verification.")
            mlflow.log_artifacts(str(output), artifact_path="release")
            client.set_model_version_tag(MODEL_NAME, version, "validation_status", "passed")
            client.set_model_version_tag(MODEL_NAME, version, "model_sha256", manifest["model_sha256"])
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"Registered {MODEL_NAME} version {version} as champion; release saved to {output}", flush=True)
    return manifest
