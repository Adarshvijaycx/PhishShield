# Contributing to PhishShield

Use Python 3.12 or 3.13 and follow the setup steps in the README. Create a feature branch, make a focused change, and run:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

The tests use small local fixtures and a temporary MLflow registry. They do not need the full dataset, the saved champion, an external account, or a running MLflow server.

Keep URL parsing and features shared between training and inference. A change to feature semantics requires a feature-version bump, compatibility checks, and a new trained release. Preserve domain-disjoint data partitions and use validation data for selection, calibration data for calibration, and untouched test data for final reporting.

Include a short description of the behavior changed and the checks run in a pull request. Use reserved example domains or documentation IPs in examples. Redact credentials and private query parameters from bug reports. Do not commit datasets, serialized models, local databases, environments, or secrets.

Code contributions use the repository's MIT license. The UCI dataset keeps its separate CC BY 4.0 attribution. Report actual results and contributions; do not claim a CI run, deployment, or training result occurred unless it has been verified.
