# Local MLflow review guide

Start the services in separate terminals from the project folder:

```sh
.venv/bin/phishshield mlflow
```

```sh
.venv/bin/phishshield serve
```

Open MLflow at <http://127.0.0.1:5001> and the analyzer at <http://127.0.0.1:8501>. Training can log to the same SQLite store with the server stopped; it does not require an external account. Local commands disable MLflow telemetry and Streamlit usage statistics.

## Demonstration sequence

1. Select the **PhishShield** experiment. Each full training cycle creates a baseline, three candidate runs, and a final calibrated-release run. Earlier runs are retained.
2. Filter with `tags.phase = 'candidate'`. Select the latest Logistic Regression, Random Forest, and Histogram Gradient Boosting runs; compare `validation_average_precision`, `validation_recall`, `validation_f1`, and fit time. Candidate evaluation uses validation data, not final test data.
3. Open the latest run with `tags.phase = 'release'`. Show `test_accuracy`, phishing precision/recall/F1, ROC-AUC, false-positive rate, Brier score, and log-loss. These are the actual frozen calibrated model's results. The current champion is version 2 with feature version 1.1.0; use its metrics rather than the older version 1 figures.
4. Browse **Artifacts → release**. Show `comparison.csv`, dataset/split summary, confusion matrix, calibration plot, PR/ROC curves, and `feature_importance.png`. The feature chart measures global validation permutation importance; it does not explain a single submitted URL.
5. Open **Models → PhishShield**. Show the selected version's `candidate` and `champion` aliases, `validation_status=passed`, source run, feature version, and model checksum. Aliases are the modern registry workflow; they do not mean this student model has been validated for security operations.
6. Open the analyzer. Select **Simple HTTPS**, then **Analyze URL**. Repeat with **Account keywords** or **Misleading address**. The examples use reserved domains/documentation IPs and are not fetched. Their predicted categories are model outputs, not assigned labels.
7. Open **Model Insights**. Match the release version and run ID to MLflow, then show candidate validation results, held-out test results, and the importance chart. Explain the dataset's narrow legitimate-URL profile.
8. Show a malformed input's validation error. Run `.venv/bin/phishshield verify` to demonstrate checksum and regression verification.

## Saved evidence

- `artifacts/release/model.joblib`: the fitted shared-features/calibrated-classifier pipeline.
- `manifest.json`: model version, run IDs, feature version, dependencies, checksums, threshold, risk bands, and train/validation/calibration/test counts.
- `dataset_summary.json` and `split_assignments.csv`: cleaning and domain separation evidence.
- `comparison.csv` and `baseline_metrics.json`: validation comparison and prior baseline.
- `metrics.json`, evaluation PNGs, and importance CSV/PNG: final test results and global explanation.
- `known_urls.json`: documented held-out URL prediction snapshots for stability checks; these are distinct from accuracy evaluation.

Back up `mlflow.db`, `mlartifacts`, and `artifacts/release` together. These local folders are ignored by Git. Retraining produces a new model version and updates the explicit release bundle; feature changes require retraining. Do not load an older release with incompatible feature code.
