# PhishShield

**Phishing URL detection with an interactive Streamlit demo and reproducible MLflow experiments.**

PhishShield analyzes URL text, estimates phishing risk, and shows suspicious patterns alongside model evaluation evidence. It is an academic decision aid, not a guarantee of website safety. Submitted websites are never fetched and submitted URL history is not persisted.

## Features

- Predicted category and calibrated 0–100 risk gauge with Low, Medium, and High bands.
- Indicators for IP hosts, `@`, many subdomains, long URLs, keywords, hostname hyphens, HTTP, punycode, and training-derived domain-suffix associations.
- Logistic Regression, Random Forest, and Histogram Gradient Boosting comparison.
- MLflow runs, evaluation artifacts, model registry, and `candidate`/`champion` aliases.
- Global permutation-importance chart, confusion matrix, and calibration/ROC/PR curves.
- Unit, inference, data-split, Streamlit, and MLflow integration tests; GitHub Actions configuration.

## Recorded benchmark

The current local release is **Random Forest, version 2**, using feature schema **1.1.0**. Results below come from 34,204 held-out URLs in the historical UCI collection.

| Accuracy | Phishing precision | Phishing recall | Phishing F1 |
|---:|---:|---:|---:|
| 98.35% | 99.64% | 96.31% | 97.95% |

**Dataset limitation:** all legitimate examples in this collection use HTTPS, a `www` hostname, and a root page. This narrow profile can inflate benchmark results and cause false positives for other legitimate URL formats. These figures do not establish current-web performance. See [MODEL_CARD.md](MODEL_CARD.md) and [the saved evaluation summary](docs/results/version-2.json).

![Held-out test confusion matrix](docs/images/confusion-matrix.png)

## Quick start

Use Python **3.12 or 3.13**. The local build was tested with Python 3.13; CI is configured for both versions. From the cloned project folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

On Windows, use `python` to create the environment and replace `.venv/bin/` with `.venv\Scripts\` in commands below.

Download the documented dataset, train the candidates, and verify the release:

```sh
.venv/bin/phishshield download
.venv/bin/phishshield train
.venv/bin/phishshield verify
```

**Fresh clones require training:** raw data, serialized models, and local MLflow state are generated locally and are not committed to this repository. The setup and tests work without them; real predictions require a trained release.

For a smaller development run, use `phishshield train --max-rows 20000`. The sample size is recorded in metadata; use the full dataset for academic evaluation.

## Run the demo

Start the analyzer and MLflow in separate terminals:

```sh
.venv/bin/phishshield serve
```

Analyzer: <http://127.0.0.1:8501>

```sh
.venv/bin/phishshield mlflow
```

MLflow: <http://127.0.0.1:5001>. Port 5001 avoids macOS AirPlay's usual port 5000. Stop either service with Ctrl+C in its terminal.

The app reads `artifacts/release`; tracking uses the project-local `mlflow.db` and `mlartifacts`. These persist between launches. Retraining adds new runs and a model version. Use the reserved example buttons to demonstrate patterns without visiting submitted links.

```sh
.venv/bin/phishshield analyze 'https://www.example.com/'
```

## Methodology

1. Read only raw URL and label columns from [UCI PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bwebsite%2Bdataset). Remap original 0=phishing, 1=legitimate to project 1=phishing, 0=legitimate.
2. Normalize URLs conservatively, remove duplicates/conflicting labels, and extract 30 numeric features using the same offline transformer as inference. Empty root paths and `/` receive identical features; other paths retain case and encoding.
3. Keep domain groups disjoint across approximate 60/15/10/15 training/validation/calibration/test partitions, seed 42.
4. Compare three models with balanced class weights and a prior baseline. Select by validation average precision, with phishing recall as tie-breaker.
5. Freeze the selected estimator and fit sigmoid calibration only on calibration data. Evaluate the final model on untouched test data.
6. Register the fitted pipeline, verify its exported checksum and prediction parity, and assign the `champion` alias.

Risk is `100 × P(phishing)`. The category threshold is 0.5; risk bands are below 30, 30 to below 70, and at least 70. Indicators describe observed patterns and do not override model predictions or establish individual causality.

![Global feature importance on validation data](docs/images/feature-importance.png)

## Tests and CI

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

The suite uses small local fixtures and a temporary MLflow registry. It does not download the full dataset, visit example websites, or need a running server. The GitHub Actions [Tests workflow](.github/workflows/tests.yml) runs on pushes to `main`, pull requests, and manual dispatch, and saves JUnit test reports. Hosted CI results become available after publication.

## Project layout

```text
app.py                     Streamlit analyzer and model insights
src/phishshield/           Features, data preparation, training, inference, CLI
tests/                     Unit and integration tests
.github/workflows/         GitHub Actions test workflow
.streamlit/config.toml     Local theme and app settings
docs/                      MLflow demo, publishing guide, evaluation evidence
MODEL_CARD.md              Provenance, metrics, and limitations
PLAN.md                    Four-member academic project plan and rubric
requirements.lock          Pinned full environment
requirements.txt           Direct dependency pins
pyproject.toml             Package metadata and command entry point
```

## Academic review and publishing

Follow [the MLflow review guide](docs/MLFLOW_DEMO.md) to demonstrate candidate comparison, held-out evaluation, artifacts, registry aliases, and the matching app release. Back up `mlflow.db`, `mlartifacts`, and `artifacts/release` for the local review.

Use [the GitHub publishing guide](docs/GITHUB_PUBLISH.md) for the prepared initial repository. Jira, Hugging Face deployment, Jenkins setup, and the final report remain later phases in [the academic plan](PLAN.md). See [CONTRIBUTING.md](CONTRIBUTING.md) for team development.

## License and dataset attribution

Project code is licensed under the [MIT License](LICENSE). This does not change the dataset's separate CC BY 4.0 license.

Prasad, A. and Chandra, S. (2024). *PhiUSIIL Phishing URL (Website).* UCI Machine Learning Repository. [Dataset and attribution details](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bwebsite%2Bdataset).

URL-only models can miss convincing phishing URLs and flag legitimate complex URLs. This project does not inspect HTML, TLS certificates, WHOIS, DNS, redirects, or live reputation. Only load trusted model artifacts; the app does not accept uploaded model files.
