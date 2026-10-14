# PhishShield model card

## Current release

- Purpose: academic URL classification, transparent warning signals, and MLflow lifecycle tracking.
- Algorithm: Random Forest, followed by sigmoid probability calibration.
- Registered model: `PhishShield`, version **3**, aliases `candidate` and `champion`.
- MLflow run: `e3c4c481ca7346aba76ab9a6e8da87f4`.
- Feature version: **1.1.0**, 30 offline numeric URL features shared by training and inference.
- Model class: legitimate=0, phishing=1; phishing at calibrated probability ≥0.5.
- Model bands: below 30 Low; 30 to below 70 Medium; at least 70 High.
- Independent assessment: a domain warning gives High caution and a review-required result without changing the model probability. Medium model estimates show an uncertain assessment.

## Dataset and provenance

Hannousse, A. and Yahiouche, S. (2021), [*Web page phishing detection*, version 3](https://data.mendeley.com/datasets/c2gw7fy2j4/3), DOI 10.17632/c2gw7fy2j4.3, **CC BY 4.0**, collected May 2020. The original dataset includes webpage and external-service features; this project reads only raw URL and string `status` labels.

The public [Hugging Face mirror](https://huggingface.co/datasets/pirocheto/phishing-url/tree/1375d450d4540205ac11781b807249c9509739cd) is pinned at revision `1375d450d4540205ac11781b807249c9509739cd`. Its two original partitions are recombined before fresh domain-disjoint splitting. The downloader records both source-file hashes and the derived CSV hash in a provenance sidecar; cleaning verifies that metadata matches the CSV. This is a third-party mirror, rather than independently verified original-source byte equivalence.

- Raw rows: 11,430 (5,715 of each original class).
- Invalid URLs removed: 8; normalized duplicates removed: 11; conflicting labels: 0.
- Usable rows: 11,411 (5,710 legitimate, 5,701 phishing).
- Effective domain groups: 7,259.
- Training: 6,158; validation: 2,011; calibration: 1,298; test: 1,944.
- Domain overlap across partitions: 0; final release seed: 43.
- Legitimate profile after cleaning: 61.28% nonroot paths, 44.54% HTTPS, 66.85% `www`, 3.35% with queries. These fractions describe the collection and are not safety rules.

An exploratory seed-42 audit identified the source bias and compared full-URL versus domain-only features. This release uses a fixed seed-43 split and the full-URL family; candidate selection uses only validation average precision, with phishing recall as tie-breaker. The selected estimator remains frozen while sigmoid calibration fits on separate calibration rows. Final figures below refer to this release's test partition. Seed-42 audit figures are not reported as final results.

## Measured evaluation

| Held-out metric | Version 3 |
|---|---:|
| Accuracy | 82.00% |
| Phishing precision | 86.49% |
| Phishing recall | 75.80% |
| Phishing F1 | 80.79% |
| Average precision | 0.9078 |
| ROC-AUC | 0.9066 |
| False-positive rate | 11.82% |
| Brier score | 0.12806 |
| Log-loss | 0.40705 |

These metrics evaluate the calibrated classifier, excluding independent domain-warning rules. Permutation importance uses sampled validation data and describes global model reliance, not an individual URL's explanation. See [saved results](docs/results/version-3.json).

## Domain warnings and reported failures

Offline comparisons cover selected Google, YouTube, Microsoft, PayPal, and Apple domains, documented by primary sources in `domain_signals.py`. Checks include selected digit/Unicode confusables, one-character edits, brand affixes, unrelated-domain subdomains, and private public-suffix tenants. Generic-word checks are conservative. Official-domain comparisons suppress only spelling warnings; they never lower the model probability or certify a URL.

| Offline behavioral example | Version 3 behavior |
|---|---|
| `mail.g0ogle.com` | High caution; Google-like digit substitution |
| `mail.googletex.com` | High caution; added text in Google-like registered domain |
| User-reported YouTube `/watch?v=…` | 3.91/100 model score, Low; no domain warning |

The two lookalike spellings have no independently established phishing labels here. Their raw model scores remain low and are shown separately. Behavioral cases are not used for candidate selection/calibration and are not an accuracy benchmark. `behavior_review.json` is saved as an MLflow artifact.

## Historical comparison and practical limits

Versions 1 and 2 remain in MLflow. Version 2 used [UCI PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bwebsite%2Bdataset), with original 0=phishing, 1=legitimate remapped internally. Inspection found all legitimate examples were HTTPS, `www`, and root pages. Its 98.35% accuracy concealed false positives on common benign paths and missed brand lookalikes. [Version 2 results](docs/results/version-2.json) are historical; its different source/test cohort makes direct accuracy comparison inappropriate.

Version 3 addresses that collection bias but still misses phishing and flags legitimate URLs. Historical sigmoid calibration does not establish calibration on today's web. The brand catalogue and confusable handling are incomplete, and spelling warnings can be false positives. HTTPS, no warnings, or a low model estimate never guarantees safety. Malicious uploads, compromised domains, redirects, and page content remain outside verification.

No HTML, TLS certificate, WHOIS, DNS, redirect, or live reputation lookup occurs. URLs are never fetched or persisted. Empty HTTP(S) root paths and `/` are normalized consistently under [RFC 9110 §4.2.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-4.2.3); other paths retain their case and encoding. Intended use is academic demonstration and methodology discussion.

## Reproduction and integrity

Install `requirements.lock`, then run `phishshield download`, `phishshield train`, and `phishshield verify`. Default dataset is Hannousse; default seed is 43. Use explicit source/seed/output arguments for historical comparisons.

Manifest metadata includes exact dependencies, dataset provenance/checksum, source-file code hashes, current Git revision, partitions, threshold, model checksum, and run ID. Source hashes identify the training code even when the working tree differs from its Git commit. The app validates model checksum, feature version, class mapping, and scikit-learn compatibility. Load only trusted serialized artifacts.
