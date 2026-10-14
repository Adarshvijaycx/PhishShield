# PhishShield model card

## Current release

- Purpose: an academic demonstration of phishing URL classification and MLflow lifecycle tracking.
- Algorithm: Random Forest, followed by sigmoid probability calibration.
- Registered model: `PhishShield`, version **2**, aliases `candidate` and `champion`.
- MLflow run: `974bbf9d17a34b4287f70d2b77a315dd`.
- Feature version: **1.1.0**, 30 offline URL features.
- Decision: legitimate=0, phishing=1; predicted phishing when calibrated probability is at least 0.5.
- Risk bands: below 30 Low; 30 to below 70 Medium; at least 70 High.

## Data and evaluation

[UCI PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bwebsite%2Bdataset), donated in 2024 and licensed CC BY 4.0. Attribution: Prasad, A. and Chandra, S. (2024), *PhiUSIIL Phishing URL (Website)*, UCI Machine Learning Repository.

Only raw URLs and labels are used. The provided webpage/statistical features are excluded. Original 0=phishing and 1=legitimate labels are remapped to 1=phishing and 0=legitimate.

- Raw rows: 235,795.
- Normalized duplicates removed: 897.
- Conflicting-label rows removed: 2, belonging to one equivalent normalized URL.
- Usable rows: 234,896, including 134,849 legitimate and 100,047 phishing URLs.
- Training: 142,384; validation: 33,866; calibration: 24,442; test: 34,204.
- No domain group overlaps between partitions; seed 42.

Candidate selection uses validation average precision, with phishing recall as the tie-breaker. The winner's base estimator remains frozen; calibration uses only calibration rows. The final test partition is evaluated after selection. Feature importance uses a separate sampled validation calculation.

| Held-out metric | Result |
|---|---:|
| Accuracy | 98.35% |
| Phishing precision | 99.64% |
| Phishing recall | 96.31% |
| Phishing F1 | 97.95% |
| Average precision | 0.9981 |
| ROC-AUC | 0.9978 |
| False-positive rate | 0.2373% |
| Brier score | 0.01063 |
| Log-loss | 0.03964 |

Results refer to version 2 on this historical dataset. Version 1 was retained in MLflow for traceability; its earlier results must not be presented as this release's results.

## Limitations and intended use

Inspection of the raw dataset found that **all legitimate examples use HTTPS, a `www` hostname, and a root page**. Phishing examples have more varied URL formats. This strong collection bias can make the benchmark much easier than detecting phishing on today's web and can cause false positives for genuine URLs with other formats. High confidence can also be misleading outside this dataset's distribution. Domain-disjoint splitting reduces duplicate-domain leakage but does not remove collection bias.

The root empty-path and `/` representations are normalized consistently. This follows the HTTP(S) URI equivalence described in [RFC 9110 §4.2.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-4.2.3). Other paths retain their case and encoding. `www` and non-`www` hosts are not silently treated as equivalent.

This release provides URL-pattern estimates only. It does not inspect HTML, TLS certificates, WHOIS, DNS, redirects, or live reputation, and it does not establish that a website is safe to visit. Indicators are observed characteristics, not causal explanations or rules that override the model. TLD/domain-suffix associations are computed from training rows only, require at least 100 examples and a phishing fraction of at least 70%, and are dataset-specific.

Use for academic demonstrations and methodology discussion. A later evaluation should add independently sourced benign URLs with diverse paths/hostnames and recent phishing URLs, then evaluate on a separate source or time period before wider use.

## Reproduction and integrity

Use the environment in `requirements.lock`, then `phishshield download`, `phishshield train`, and `phishshield verify`. The current release manifest records exact dependencies, dataset checksum, model checksum, seed, partitions, and source run. This release was trained before the first Git commit, so its manifest records no code revision. Later training runs record the current Git revision.

The app validates checksum, feature version, class mapping, and scikit-learn compatibility before loading the project-owned release. Only trusted serialized model artifacts should be loaded. Submitted URLs are processed locally without visiting them or persisting their history.
