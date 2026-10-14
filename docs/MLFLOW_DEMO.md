# PhishShield: MLflow academic review

MLflow is running at <http://127.0.0.1:5001>. The analyzer is at <http://127.0.0.1:8501>. The verified release is **PhishShield version 3**, Random Forest with separate sigmoid calibration.

The completed review run is [`ca8c64644e534a72af8cad406b0f91f7`](http://127.0.0.1:5001/#/experiments/1/runs/ca8c64644e534a72af8cad406b0f91f7). Its `academic_review` artifacts include a portable HTML review, JSON verification results, model comparison, charts, promotion history, and ZIP evidence pack. Local copies are in `reports/mlflow_review/`.

## What is completed

| Review item | Recorded evidence |
|---|---|
| Model comparison | Logistic Regression, Random Forest, Histogram Gradient Boosting; same data/splits |
| Candidate selection | Random Forest validation AP **0.973470**, selected before calibration/test |
| Final evaluation | **82.00%** accuracy, **86.49%** precision, **75.80%** recall, **80.79%** F1; 1,944 test URLs |
| Explainability | Global validation permutation-importance PNG/CSV; excluded from fitting/selection |
| Evaluation charts | Confusion matrix, calibration, ROC, precision–recall |
| Dataset lineage | Native Dataset Inputs for train/validation; release also records calibration/test |
| Model registry | Immutable `models:/PhishShield/3`, source run and model checksum |
| Promotion | Verified `staging` alias → `production` alias; version 3 also remains `champion` |
| Application parity | 20 saved examples; maximum probability difference **3.33 × 10⁻¹⁶** |
| Evidence integrity | 15 required release artifacts, run metadata, candidate metrics, checksums and source links |

These classifier metrics exclude independent domain-warning rules. `validation_status=passed` and `academic_review_status=passed` mean the recorded integrity/lineage checks succeeded; they do not certify real-world security performance.

## Repeatable commands

From the project folder, start either service if it is stopped:

```sh
.venv/bin/phishshield mlflow
```

```sh
.venv/bin/phishshield serve
```

Verify the existing release, record dataset lineage, and create a fresh academic review:

```sh
.venv/bin/phishshield mlflow-review --tracking-uri http://127.0.0.1:5001
```

To demonstrate staging-to-production promotion after those checks:

```sh
.venv/bin/phishshield mlflow-review --tracking-uri http://127.0.0.1:5001 --promote
```

The command loads the immutable registered version and compares its raw `predict_proba` values with the exported release and frozen snapshots. It checks the run IDs, candidate ranking, baseline, split assignments, dataset checksum, and required artifacts. Failed model/data checks stop the workflow before alias promotion. Repeated lineage recording skips existing matching Inputs; historical run durations/statuses are preserved.

The default output is `reports/mlflow_review`. Use `--output`, `--release`, and `--data` for other local review folders/bundles. Leaving out `--tracking-uri` uses the project SQLite store directly, so checks and logging also work with the UI server stopped.

For a read-only programmatic inspection, use `phishshield.mlflow_review.inspect_mlflow_release(release_dir, tracking_uri)`. It logs no runs or aliases. `phishshield verify` remains the local checksum/snapshot check.

## Six-minute demonstration

| Time | What to show and explain |
|---|---|
| 0–1 min | Open PhishShield experiment. Filter `tags.review_cohort = 'version-3-e3c4c481'`. Open a candidate's Dataset Inputs: raw URL/label only, source CSV, digest, training/validation contexts. |
| 1–2 min | Add `tags.phase = 'candidate'`; compare the three runs' validation AP, recall/F1, parameters, and fit time. Explain selection by validation AP rather than final test accuracy. |
| 2–3 min | Open source release run `e3c4c481ca7346aba76ab9a6e8da87f4`. Show separate calibration, test metrics, and 1,944 domain-disjoint test URLs. Explain historical data limitations. |
| 3–4 min | Open **Artifacts → release**: confusion matrix, calibration/ROC/PR plots, global permutation importance, data summary, and behavior review. Behavioral cases are separate from labeled accuracy. |
| 4–5 min | Open **Models → PhishShield → version 3**. Show `staging`, `production`, `champion`, source run, checksum, and review tags. Open the academic review's `promotion.json` and parity evidence. |
| 5–6 min | Open analyzer **Model Insights** and match model version/run ID. Try reserved example buttons and malformed input. Show the ZIP/offline HTML evidence and successful verification. |

Future training cycles have a UUID `training_cycle_id` in run tags/parameters and exact candidate/baseline IDs in their manifests. Filter those cycles by `tags.training_cycle_id` instead of the review cohort. Dataset checksum alone cannot distinguish repeated training cycles.

## Registry terminology

The project uses modern MLflow aliases/tags. **`staging` and `production` are aliases**, preserving the academic lifecycle vocabulary. They do not change MLflow's former stage field or deploy the application. MLflow deprecated fixed stages in version 2.9 and recommends aliases/tags. [Official registry workflow and stage migration](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

Before promotion, the review verifies model version, raw probability parity, and data lineage. If replacing an existing production version, its pointer is preserved as `previous_production`. The prior version 2 model/runs remain available for traceability; their UCI evaluation cohort differs from version 3's corpus.

## Dataset lineage provenance

Version 3 was trained before native Dataset Inputs were added. This completed review reconstructs the exact saved assignments from the checksum-matched CSV, verifies every row/group/label/split/URL hash, and attaches Inputs through the MLflow client. Existing runs are tagged `dataset_lineage_recorded_post_training=true` with a timestamp; their original training durations and metrics remain intact. Future training records Inputs during fitting. [Official dataset logging API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_input)

## Saved evidence and backup

- `reports/mlflow_review/index.html`: portable offline review with embedded charts.
- `reports/mlflow_review/PhishShield-MLflow-Evidence.zip`: review, comparison, promotion history, and charts.
- `reports/mlflow_review/review.json`: exact run IDs, verified artifacts, native/export parity, and dataset contexts.
- `docs/results/mlflow-review-version-3.json`: saved repository summary of the completed review.
- `artifacts/release/`: complete fitted pipeline, manifest, charts, partitions, and snapshots.

Back up **`mlflow.db`, `mlartifacts/`, and `artifacts/release/` together**. Stop MLflow/training before making an ordinary file copy of the database; use SQLite's backup API for a consistent database snapshot while it is running. The evidence ZIP is a review export and does not replace this full backup. Generated data, model files, reports, and MLflow state remain ignored by Git.

Training logs to the local SQLite store even with the UI stopped. Local CLI commands disable MLflow telemetry and Streamlit usage statistics. Submitted URLs are processed offline without opening websites or persisting input history.
