"""Load trusted release artifacts and run the same pipeline used in training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version

from phishshield.features import (
    FEATURE_VERSION, extract_features, get_indicators, parse_url,
)


class ModelLoadError(RuntimeError):
    """The locally exported model is absent, corrupt, or incompatible."""


@dataclass
class ModelBundle:
    model: Any
    manifest: dict[str, Any]
    release_dir: Path


@dataclass(frozen=True)
class PredictionResult:
    normalized_url: str
    hostname: str
    registered_domain: str
    category: str
    phishing_probability: float
    risk_score: float
    risk_band: str
    indicators: list[dict[str, str]]
    model_version: str
    algorithm: str
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release(release_dir: Path | str) -> ModelBundle:
    """Load a project-owned bundle only after checking its manifest and checksum."""
    directory = Path(release_dir).expanduser().resolve()
    model_path = directory / "model.joblib"
    manifest_path = directory / "manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise ModelLoadError("No trained release was found. Download the dataset and run phishshield train first.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("feature_version") != FEATURE_VERSION:
            raise ModelLoadError("The model's feature version does not match this application. Retrain the release.")
        if file_sha256(model_path) != manifest.get("model_sha256"):
            raise ModelLoadError("The model checksum does not match its manifest. Re-export the release.")
        recorded_version = manifest.get("dependency_versions", {}).get("scikit-learn")
        if recorded_version and recorded_version != sklearn_version:
            raise ModelLoadError(f"This model requires scikit-learn {recorded_version}; installed version is {sklearn_version}. Install the locked dependencies.")
        model = joblib.load(model_path)
        if not callable(getattr(model, "predict_proba", None)) or set(model.classes_) != {0, 1}:
            raise ModelLoadError("The release must expose probabilities for legitimate=0 and phishing=1.")
        threshold = float(manifest.get("threshold", 0.5))
        if not 0 < threshold < 1:
            raise ModelLoadError("The release has an invalid classification threshold.")
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"The trained release could not be loaded: {exc}") from exc
    return ModelBundle(model=model, manifest=manifest, release_dir=directory)


def analyze_url(url: str, bundle: ModelBundle) -> PredictionResult:
    parsed = parse_url(url)
    data = pd.DataFrame({"url": [parsed.normalized_url]})
    index = list(bundle.model.classes_).index(1)
    probability = float(bundle.model.predict_proba(data)[0, index])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ModelLoadError("The model returned an invalid phishing probability.")
    score = probability * 100
    threshold = float(bundle.manifest.get("threshold", 0.5))
    return PredictionResult(
        normalized_url=parsed.normalized_url,
        hostname=parsed.hostname,
        registered_domain=parsed.registered_domain,
        category="Potential phishing" if probability >= threshold else "Likely legitimate",
        phishing_probability=probability,
        risk_score=score,
        risk_band="Low" if score < 30 else "Medium" if score < 70 else "High",
        indicators=get_indicators(parsed, bundle.manifest.get("tld_statistics", {})),
        model_version=str(bundle.manifest.get("model_version", "unregistered")),
        algorithm=str(bundle.manifest.get("algorithm", "Unknown")),
        features=extract_features(parsed),
    )


def verify_release(release_dir: Path | str) -> dict[str, Any]:
    bundle = load_release(release_dir)
    fixture_path = bundle.release_dir / "known_urls.json"
    if not fixture_path.is_file():
        raise ModelLoadError("The release is missing its known-URL regression fixtures.")
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    if isinstance(fixtures, dict):
        fixtures = fixtures.get("examples", fixtures.get("urls", []))
    if not fixtures:
        raise ModelLoadError("The release regression fixture is empty.")
    for fixture in fixtures:
        result = analyze_url(fixture["url"], bundle)
        expected_probability = fixture.get("phishing_probability", fixture.get("expected_probability"))
        if expected_probability is None or not np.isclose(result.phishing_probability, expected_probability, atol=1e-6, rtol=1e-6):
            raise ModelLoadError("A known-URL probability differs from the frozen release.")
        expected_label = fixture.get("label", fixture.get("predicted_label"))
        if expected_label is not None and int(result.phishing_probability >= float(bundle.manifest.get("threshold", 0.5))) != int(expected_label):
            raise ModelLoadError("A known-URL classification differs from the frozen release.")
    return {
        "status": "passed", "examples_checked": len(fixtures),
        "model_version": bundle.manifest.get("model_version"),
        "run_id": bundle.manifest.get("run_id"),
        "model_sha256": bundle.manifest.get("model_sha256"),
    }
