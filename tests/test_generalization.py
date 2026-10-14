"""Domain-boundary and advisory contracts, without inferred phishing labels.

Synthetic fixture labels fit a real model for probability-parity checks only.
None of the URL spellings below is asserted to be a verified phishing site.
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phishshield.domain_signals import get_domain_signals
from phishshield.features import URLFeatureExtractor, parse_url
from phishshield.inference import ModelBundle, analyze_url


@pytest.fixture
def fitted_bundle(tmp_path):
    model = Pipeline([
        ("features", URLFeatureExtractor()),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(pd.DataFrame({"url": [
        "https://school.example/", "https://library.test/books?author=Smith",
        "https://museum.example/exhibitions/", "https://bookshop.test/catalog#new",
        "http://verify-account.test/login", "http://192.0.2.10/password",
        "http://a.b.c.school.example/verify", "http://secure-update.test/account?token=123",
    ]}), [0, 0, 0, 0, 1, 1, 1, 1])
    return ModelBundle(model, {"threshold": 0.5}, tmp_path)


@pytest.mark.parametrize("url,registered_domain", [
    ("https://mail.google.com/", "google.com"),
    ("https://youtube.com.attacker.test/", "attacker.test"),
    ("https://youtube.com@attacker.test/", "attacker.test"),
    ("https://www.google.com.attacker.co.uk/", "attacker.co.uk"),
    ("https://youtube.github.io/", "youtube.github.io"),
    ("https://google.example.com/", "example.com"),
])
def test_brand_comparison_uses_effective_registered_domain(url, registered_domain):
    parsed = parse_url(url)
    assert parsed.registered_domain == registered_domain
    for signal in get_domain_signals(parsed):
        assert signal.observed_domain == registered_domain


@pytest.mark.parametrize("url,expected_code", [
    ("https://mail.g0ogle.com/", "brand_digit_substitution"),
    ("https://mail.googletex.com/", "brand_affix"),
    ("https://youtube.com.attacker.test/", "brand_in_subdomain"),
    ("https://www.google.com.attacker.co.uk/", "brand_in_subdomain"),
    ("https://youtube.github.io/", "brand_domain_mismatch"),
])
def test_brand_warnings_describe_spelling_or_boundaries(url, expected_code):
    signals = get_domain_signals(url)
    assert expected_code in {signal.code for signal in signals}
    for signal in signals:
        payload = signal.to_dict()
        assert payload["observed_domain"] == parse_url(url).registered_domain
        assert not {"probability", "risk_score", "label", "is_phishing"} & payload.keys()
        json.dumps(payload)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=HsZxZ8hoYdU",
    "https://mail.google.com/login?continue=https%3A%2F%2Fother.example",
    "http://www.youtube.com:8080/redirect?target=https%3A%2F%2Fother.example",
    "https://bookshop.test/?next=https://youtube.com#google",
    "https://youtube.com@attacker.test/",
])
def test_brand_text_and_comparison_domains_do_not_imply_safe_verdict(url):
    # Official-domain matches suppress only spelling warnings. Brand text in
    # query, fragment, or user-info must not be mistaken for the effective host.
    assert not get_domain_signals(url)


@pytest.mark.parametrize("url", [
    "https://mail.g0ogle.com/",
    "https://mail.googletex.com/",
    "https://youtube.com.attacker.test/watch?v=example",
])
def test_domain_advisory_preserves_fitted_model_probability(fitted_bundle, url):
    assert get_domain_signals(url)
    parsed = parse_url(url)
    phishing_column = list(fitted_bundle.model.classes_).index(1)
    expected = float(fitted_bundle.model.predict_proba(
        pd.DataFrame({"url": [parsed.normalized_url]})
    )[0, phishing_column])
    result = analyze_url(url, fitted_bundle)
    assert result.phishing_probability == expected
    assert result.category == "Suspicious domain — review required"
    assert result.assessment_band == "High"
    assert result.assessment_basis == "domain_warning"
    assert result.domain_signals
    assert result.risk_score == expected * 100
    assert np.isfinite(result.phishing_probability)


def test_domain_advisory_never_fetches_or_resolves_host(fitted_bundle):
    with patch("socket.getaddrinfo", side_effect=AssertionError("Network resolution is forbidden")), patch(
        "requests.get", side_effect=AssertionError("Submitted URLs must not be fetched")
    ):
        signals = get_domain_signals("https://mail.g0ogle.com/")
        result = analyze_url("https://mail.g0ogle.com/", fitted_bundle)
    assert signals
    assert result.hostname == "mail.g0ogle.com"


def test_userinfo_keeps_effective_host_and_existing_warning(fitted_bundle):
    result = analyze_url("https://youtube.com@attacker.test/", fitted_bundle)
    assert result.hostname == result.registered_domain == "attacker.test"
    assert "at_symbol" in {indicator["code"] for indicator in result.indicators}
