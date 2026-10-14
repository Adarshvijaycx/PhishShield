"""Record a verified MLflow academic review and optional alias promotion."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shutil
import zipfile

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, InputTag
import pandas as pd

from phishshield.data import file_sha256, prepare_dataset, split_dataset
from phishshield.mlflow_review import MlflowReviewError, inspect_mlflow_release


ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_inputs(client, inspection, release, csv_path):
    """Attach reconstructed lineage without reopening or retraining old runs."""
    manifest = json.loads((release / "manifest.json").read_text())
    prepared = prepare_dataset(csv_path, seed=int(manifest["split_seed"]))
    if prepared.checksum != manifest["dataset_checksum"]:
        raise MlflowReviewError("The review CSV checksum differs from the trained dataset.")
    selected_rows = sum(int(manifest[key]) for key in (
        "training_rows", "validation_rows", "calibration_rows", "test_rows"
    ))
    # Sampling used the original seed, which can differ from the successful
    # split retry seed. Obtain it from the immutable source run parameters.
    run = client.get_run(inspection["run_id"])
    seed = int(run.data.params["seed"])
    if len(prepared.frame) > selected_rows:
        prepared = prepare_dataset(csv_path, max_rows=selected_rows, seed=seed)
    frame = split_dataset(prepared.frame, seed=seed)
    actual = frame[["row_id", "domain_group", "label", "split"]].copy()
    actual["url_sha256"] = frame["url"].map(lambda text: sha256(text.encode()).hexdigest())
    columns = ["row_id", "domain_group", "label", "split", "url_sha256"]
    expected = pd.read_csv(release / "split_assignments.csv", keep_default_na=False)
    try:
        pd.testing.assert_frame_equal(
            actual[columns].sort_values("row_id").reset_index(drop=True),
            expected[columns].sort_values("row_id").reset_index(drop=True),
        )
    except (AssertionError, KeyError) as exc:
        raise MlflowReviewError("Reconstructed dataset partitions differ from the saved training assignments.") from exc

    datasets = {}
    for partition in ("train", "validation", "calibration", "test"):
        rows = frame.loc[frame["split"] == partition, ["url", "label"]].reset_index(drop=True)
        data = mlflow.data.from_pandas(
            rows, source=Path(csv_path).resolve().as_uri(), targets="label",
            name=f"{manifest['dataset_kind']}-{partition}",
        )
        context = "training" if partition == "train" else partition
        datasets[partition] = DatasetInput(
            dataset=Dataset(**json.loads(data.to_json())),
            tags=[InputTag("mlflow.data.context", context)],
        )
    run_ids = [inspection["baseline"]["run_id"]] + [c["run_id"] for c in inspection["candidates"]]
    evidence = []
    for run_id in run_ids + [inspection["run_id"]]:
        partitions = ("train", "validation", "calibration", "test") if run_id == inspection["run_id"] else ("train", "validation")
        original = client.get_run(run_id)
        existing = {
            (item.dataset.name, item.dataset.digest, tag.value)
            for item in original.inputs.dataset_inputs
            for tag in item.tags if tag.key == "mlflow.data.context"
        }
        additions = [datasets[p] for p in partitions if (
            datasets[p].dataset.name, datasets[p].dataset.digest, datasets[p].tags[0].value
        ) not in existing]
        if additions:
            client.log_inputs(run_id, datasets=additions)
            client.set_tag(run_id, "dataset_lineage_recorded_post_training", "true")
            client.set_tag(run_id, "dataset_lineage_recorded_at", _now())
        after = client.get_run(run_id)
        if (after.info.start_time, after.info.end_time, after.info.status) != (
            original.info.start_time, original.info.end_time, original.info.status
        ):
            raise MlflowReviewError("Recording dataset lineage changed a historical run's duration or status.")
        observed = {
            (item.dataset.name, item.dataset.digest, tag.value)
            for item in after.inputs.dataset_inputs
            for tag in item.tags if tag.key == "mlflow.data.context"
        }
        for partition in partitions:
            item = datasets[partition]
            if (item.dataset.name, item.dataset.digest, item.tags[0].value) not in observed:
                raise MlflowReviewError("A native dataset Input was not recorded correctly.")
        evidence.append({"run_id": run_id, "contexts": [datasets[p].tags[0].value for p in partitions],
                         "datasets": [{"name": datasets[p].dataset.name,
                                       "digest": datasets[p].dataset.digest,
                                       "rows": json.loads(datasets[p].dataset.profile)["num_rows"]} for p in partitions],
                         "added_inputs": len(additions)})
    return evidence


def _promote(client, inspection, review_run_id):
    """Use aliases for the academic lifecycle; preserve any prior production."""
    name, version = inspection["model_name"], str(inspection["model_version"])
    if str(client.get_model_version_by_alias(name, "champion").version) != version:
        raise MlflowReviewError("The champion changed during review. Recheck the release before promotion.")
    old = dict(client.get_registered_model(name).aliases)
    if old.get("production") and str(old["production"]) != version:
        client.set_registered_model_alias(name, "previous_production", str(old["production"]))
    events = []
    for alias in ("staging", "production"):
        client.set_registered_model_alias(name, alias, version)
        actual = client.get_model_version_by_alias(name, alias)
        if str(actual.version) != version or actual.run_id != inspection["run_id"]:
            raise MlflowReviewError(f"The {alias} alias did not resolve to the reviewed model.")
        events.append({"alias": alias, "version": int(version), "timestamp_utc": _now(),
                       "previous_version": old.get(alias), "scope": "local academic demonstration"})
    client.set_model_version_tag(name, version, "academic_review_status", "passed")
    client.set_model_version_tag(name, version, "academic_review_run_id", review_run_id)
    client.set_model_version_tag(name, version, "promotion_scope", "local_academic_demo")
    return {"workflow": "verified staging alias -> production alias", "events": events,
            "aliases": dict(client.get_registered_model(name).aliases),
            "note": "Aliases demonstrate lifecycle management. They do not deploy a service or certify security performance."}


def _html(report, release):
    cards = "".join(f'<div class="card"><span>{escape(label)}</span><strong>{report["test_metrics"][key]*100:.2f}%</strong></div>'
                    for key, label in (("accuracy", "Test accuracy"), ("precision", "Phishing precision"),
                                       ("recall", "Phishing recall"), ("f1", "Phishing F1")))
    rows = "".join(
        f'<tr><td>{escape(candidate["algorithm"])}</td>'
        + "".join(f'<td>{candidate["validation_metrics"][key]:.4f}</td>' for key in ("average_precision", "recall", "f1"))
        + f'<td><code>{escape(candidate["run_id"])}</code></td></tr>' for candidate in report["candidates"]
    )
    figures = "".join(
        f'<figure><figcaption>{escape(label)}</figcaption><img alt="{escape(label)}" src="data:image/png;base64,'
        + base64.b64encode((release / filename).read_bytes()).decode() + '"></figure>'
        for filename, label in (("confusion_matrix.png", "Held-out test confusion matrix"),
                                ("feature_importance.png", "Global validation permutation importance"),
                                ("calibration.png", "Held-out test calibration"),
                                ("pr_curve.png", "Held-out test precision–recall"))
    )
    aliases = escape(json.dumps(report.get("promotion", {}).get("aliases", {}), sort_keys=True))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhishShield · MLflow academic review</title><style>
body{{margin:0;background:#08111f;color:#e8eef8;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 24px}}h1{{font-size:38px;line-height:1.15}}h2{{font-size:24px;margin-top:36px}}.muted,span{{color:#a9b9d0}}.accent{{color:#28d5b6}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{background:#101d30;border:1px solid #243650;padding:18px;border-radius:12px}}strong{{display:block;font-size:29px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #243650}}code{{font-size:12px;overflow-wrap:anywhere}}.figures{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0;background:#101d30;padding:14px;border-radius:12px}}img{{width:100%;height:auto}}figcaption{{margin-bottom:12px}}@media(max-width:700px){{.cards,.figures{{grid-template-columns:1fr}}table{{display:block;overflow:auto}}}}@media print{{body{{background:white;color:#15283e}}main{{padding:16px}}.card,figure{{background:white}}.muted,span{{color:#42546a}}}}
</style><main><p class="accent">PHISHSHIELD · VERIFIED MLFLOW EVIDENCE</p><h1>Model version {report['model_version']}<br>Academic review</h1>
<p class="muted">Reviewed {escape(report['reviewed_at'])}. This file includes its charts and opens offline.</p><div class="cards">{cards}</div>
<h2>Three candidate models</h2><p>Validation average precision selects the winner; final test metrics above describe its separately calibrated release.</p><table><thead><tr><th>Model</th><th>Validation AP</th><th>Recall</th><th>F1</th><th>Source run</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Traceable registry and application</h2><p>Source release run: <code>{escape(report['run_id'])}</code><br>Review run: <code>{escape(report['review_run_id'])}</code><br>Model SHA-256: <code>{escape(report['model_sha256'])}</code><br>Dataset SHA-256: <code>{escape(report['dataset_checksum'])}</code></p>
<p>Native registry/export prediction parity: <b>{report['examples_checked']} checks passed</b>; maximum absolute difference {report['probability_parity']['max_absolute_difference']:.3g}. Aliases: <code>{aliases}</code>.</p>
<p>Staging and production are MLflow aliases for this local academic demonstration. Model checks establish provenance and prediction parity; they do not certify real-world security accuracy or deploy a website.</p>
<h2>Evaluation artifacts</h2><div class="figures">{figures}</div>
<h2>What the reviewer can show</h2><ol><li>Filter experiment runs by the review cohort and compare the three candidate models.</li><li>Open Dataset Inputs to show training, validation, calibration, and test lineage.</li><li>Show the selected release metrics and charts, then the version {report['model_version']} registry aliases.</li><li>Open the analyzer and match its model version/run ID to MLflow.</li><li>Demonstrate reserved URL examples and a malformed-input error.</li></ol>
<p class="muted">Historical corpus metrics exclude independent lookalike-domain caution rules. Submitted websites are not fetched. PhishShield is a decision aid, not a guarantee.</p></main></html>'''


def run_academic_review(release_dir, tracking_uri=None, output_dir=None, dataset_path=None, promote=False):
    """Verify first, then record lineage/evidence and optionally promote aliases."""
    try:
        return _run_academic_review(release_dir, tracking_uri, output_dir, dataset_path, promote)
    except MlflowReviewError:
        raise
    except Exception as exc:
        raise MlflowReviewError(f"The academic MLflow review failed: {exc}") from exc


def _run_academic_review(release_dir, tracking_uri, output_dir, dataset_path, promote):
    release = Path(release_dir).expanduser().resolve()
    output = Path(output_dir or ROOT / "reports" / "mlflow_review").expanduser().resolve()
    if output == release or output in release.parents or release in output.parents:
        raise MlflowReviewError("The evidence output directory must be separate from the frozen release directory.")
    inspection = inspect_mlflow_release(release, tracking_uri)
    manifest = json.loads((release / "manifest.json").read_text())
    source = dataset_path or ROOT / "data" / "raw" / (
        "Hannousse_Phishing_URL_Dataset.csv" if manifest["dataset_kind"] == "hannousse" else "PhiUSIIL_Phishing_URL_Dataset.csv"
    )
    client = MlflowClient(tracking_uri=inspection["tracking_uri"], registry_uri=inspection["tracking_uri"])
    output.mkdir(parents=True, exist_ok=True)
    lineage = _dataset_inputs(client, inspection, release, source)
    cohort = f"version-{inspection['model_version']}-{inspection['run_id'][:8]}"
    for run_id in [inspection["run_id"], inspection["baseline"]["run_id"]] + [c["run_id"] for c in inspection["candidates"]]:
        client.set_tag(run_id, "review_cohort", cohort)
    experiment_id = client.get_run(inspection["run_id"]).info.experiment_id
    review_run = client.create_run(experiment_id, tags={
        "mlflow.runName": f"academic-review-version-{inspection['model_version']}",
        "phase": "academic_review", "review_cohort": cohort,
        "source_release_run_id": inspection["run_id"], "evaluation_scope": "integrity_and_lineage",
    })
    review_id = review_run.info.run_id
    try:
        client.log_param(review_id, "model_name", inspection["model_name"])
        client.log_param(review_id, "model_version", str(inspection["model_version"]))
        client.log_param(review_id, "dataset_checksum", inspection["dataset_checksum"])
        client.log_param(review_id, "model_sha256", inspection["model_sha256"])
        client.log_metric(review_id, "parity_examples_checked", inspection["examples_checked"])
        client.log_metric(review_id, "parity_max_absolute_difference", inspection["probability_parity"]["max_absolute_difference"])
        client.log_metric(review_id, "verified_release_artifacts", len(inspection["artifacts"]))
        client.log_metric(review_id, "dataset_input_runs_verified", len(lineage))
        report = {**inspection, "review_run_id": review_id, "reviewed_at": _now(),
                  "review_cohort": cohort, "dataset_lineage": lineage,
                  "training_runs_reopened": False, "models_retrained": False}
        if promote:
            report["promotion"] = _promote(client, inspection, review_id)
        else:
            report["promotion"] = {"workflow": "not requested", "events": [],
                                   "aliases": dict(client.get_registered_model(inspection["model_name"]).aliases)}
        output.mkdir(parents=True, exist_ok=True)
        (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output / "promotion.json").write_text(json.dumps(report["promotion"], indent=2) + "\n", encoding="utf-8")
        (output / "index.html").write_text(_html(report, release), encoding="utf-8")
        shutil.copy2(release / "comparison.csv", output / "comparison.csv")
        for filename in ("confusion_matrix.png", "feature_importance.png", "calibration.png", "roc_curve.png", "pr_curve.png"):
            shutil.copy2(release / filename, output / filename)
        bundle = output / "PhishShield-MLflow-Evidence.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(output.iterdir()):
                if file.is_file() and file != bundle:
                    archive.write(file, file.name)
        client.log_artifacts(review_id, str(output), artifact_path="academic_review")
        client.set_tag(review_id, "review_status", "passed")
        client.set_terminated(review_id, status="FINISHED")
        return {**report, "output_dir": str(output), "evidence_zip": str(bundle)}
    except Exception:
        client.set_tag(review_id, "review_status", "failed")
        client.set_terminated(review_id, status="FAILED")
        raise
