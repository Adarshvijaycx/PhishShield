"""URL parsing and indicator contracts, using reserved example domains only."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from phishshield.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    MAX_URL_LENGTH,
    URLFeatureExtractor,
    URLValidationError,
    extract_features,
    get_indicators,
    parse_url,
)


def test_normalization_preserves_case_sensitive_components():
    result = parse_url("  HTTPS://User:PaSs@WWW.Example.COM:443/LogIn?Code=AbC#Section  ")
    assert result.normalized_url == "https://User:PaSs@www.example.com:443/LogIn?Code=AbC#Section"
    assert result.hostname == "www.example.com"
    assert result.registered_domain == "example.com"
    assert result.subdomain == "www"
    assert result.tld == "com"
    assert result.path == "/LogIn"
    assert result.query == "Code=AbC"
    assert result.port == 443
    assert result.userinfo == "User:PaSs"


@pytest.mark.parametrize("url", [
    "", "   ", "example.com", "ftp://example.com", "https:///login", "https://",
    "https://example.com:abc", "https://example.com:65536", "https://example.com:0",
    "https://example.com:", "https://[2001:db8::1", "https://999.999.999.999/",
    "https://127.1/", "https://-bad.example/", "https://bad-.example/",
    "https://bad..example/", "https://example.com../", "https://[v1.foo]/",
    "https://bad_host.example/", "https://example.com/a b",
    "https://example.com/\nverify", "https://example.com\t", "https://example.com/\x00",
    "https://example.com\\@other.example/", "https://[fe80::1%25en0]/",
])
def test_rejects_invalid_or_ambiguous_urls(url):
    with pytest.raises(URLValidationError):
        parse_url(url)


def test_rejects_non_text_and_excessive_length():
    for value in (None, 123, float("nan")):
        with pytest.raises(URLValidationError):
            parse_url(value)
    with pytest.raises(URLValidationError, match="8,192"):
        parse_url("https://example.com/" + "a" * MAX_URL_LENGTH)


@pytest.mark.parametrize("url,hostname", [
    ("http://192.0.2.1/login", "192.0.2.1"),
    ("https://[2001:0db8:0000::1]:8443/", "2001:db8::1"),
])
def test_ip_addresses_are_canonical_and_group_by_ip(url, hostname):
    result = parse_url(url)
    assert result.hostname == hostname
    assert result.registered_domain == hostname
    assert result.is_ip
    assert result.subdomain == result.tld == ""
    assert extract_features(result)["has_ip"] == 1
    assert "ip_address" in {item["code"] for item in get_indicators(result)}


def test_idna_and_trailing_dns_dot_are_normalized():
    parsed = parse_url("https://BÜCHER.example./Katalog?q=Größe")
    assert parsed.hostname == "xn--bcher-kva.example"
    assert parsed.registered_domain == "xn--bcher-kva.example"
    assert parsed.normalized_url.endswith("/Katalog?q=Größe")
    assert extract_features(parsed)["has_punycode"] == 1
    assert "punycode" in {item["code"] for item in get_indicators(parsed)}


def test_effective_host_follows_last_userinfo_at_symbol():
    parsed = parse_url("https://trusted.example@user@other.example/verify")
    assert parsed.hostname == "other.example"
    assert parsed.userinfo == "trusted.example@user"
    assert extract_features(parsed)["at_count"] == 2
    assert extract_features(parsed)["has_userinfo"] == 1


def test_public_suffix_snapshot_is_used_without_network():
    with patch("socket.getaddrinfo", side_effect=AssertionError("Network must not be used")):
        parsed = parse_url("https://one.two.three.example.co.uk/")
        reserved = parse_url("https://one.two.three.school.test/")
    assert parsed.registered_domain == "example.co.uk"
    assert parsed.subdomain == "one.two.three"
    assert parsed.tld == "co.uk"
    assert reserved.registered_domain == "school.test"
    assert reserved.tld == "test"


def test_features_match_observable_components_and_are_finite():
    url = "http://a.b.c.secure-account.example:8080/login/verify?x=123&token=%40"
    features = extract_features(url)
    assert tuple(features) == FEATURE_NAMES
    assert FEATURE_VERSION == "1.1.0"
    assert len(features) == 30
    assert features["subdomain_count"] == 3
    assert features["path_depth"] == 2
    assert features["query_parameter_count"] == 2
    assert features["keyword_count"] == 4
    assert features["has_nonstandard_port"] == 1
    assert features["has_https"] == 0
    assert features["percent_count"] == 1
    assert 0 <= features["digit_ratio"] <= 1
    assert np.isfinite(list(features.values())).all()


def test_standard_ports_are_scheme_specific():
    assert extract_features("https://example.com:443/")["has_nonstandard_port"] == 0
    assert extract_features("http://example.com:80/")["has_nonstandard_port"] == 0
    assert extract_features("https://example.com:80/")["has_nonstandard_port"] == 1


@pytest.mark.parametrize("base,suffix", [
    ("https://www.example.com", ""),
    ("http://school.example", "?Code=AbC#Section"),
    ("https://User:PaSs@school.example:443", "?Next=%2FLogin"),
    ("https://[2001:db8::1]:8443", ""),
    ("https://bücher.example", "#Größe"),
])
def test_root_slash_and_empty_path_share_all_features(base, suffix):
    without_slash = parse_url(base + suffix)
    with_slash = parse_url(base + "/" + suffix)
    assert without_slash == with_slash
    assert with_slash.path == ""
    assert extract_features(without_slash) == extract_features(with_slash)
    assert get_indicators(without_slash) == get_indicators(with_slash)


@pytest.mark.parametrize("path", ["/LogIn", "/Login/", "//", "/%2F"])
def test_root_canonicalization_preserves_other_paths(path):
    parsed = parse_url("https://example.com" + path + "?Code=AbC#Section")
    assert parsed.path == path
    assert parsed.normalized_url == "https://example.com" + path + "?Code=AbC#Section"


def test_expected_indicators_and_no_automatic_tld_assumptions():
    url = "http://a.b.c.secure-account.example/verify?password=" + "x" * 40
    indicators = get_indicators(url)
    codes = {indicator["code"] for indicator in indicators}
    assert codes == {"many_subdomains", "long_url", "suspicious_keywords", "hyphenated_hostname", "no_https"}
    assert all(set(indicator) == {"code", "title", "description"} for indicator in indicators)
    assert get_indicators("https://example.com/") == []


@pytest.mark.parametrize("count,rate,flagged", [
    (100, 0.7, True), (101, 0.99, True), (99, 0.9, False),
    (100, 0.69, False), (100, 1.1, False),
])
def test_tld_indicator_requires_sufficient_training_evidence(count, rate, flagged):
    stats = {"example": {"count": count, "phishing_rate": rate}}
    indicators = get_indicators("https://school.example/", stats)
    assert ("dataset_tld_association" in {item["code"] for item in indicators}) is flagged


def test_indicator_boundaries_use_unrounded_observations():
    base = "https://example.com/"
    assert "long_url" not in {item["code"] for item in get_indicators(base + "x" * (75 - len(base)))}
    assert "long_url" in {item["code"] for item in get_indicators(base + "x" * (76 - len(base)))}
    assert "many_subdomains" not in {item["code"] for item in get_indicators("https://a.b.school.example/")}
    assert "many_subdomains" in {item["code"] for item in get_indicators("https://a.b.c.school.example/")}


def test_transformer_preserves_dataframe_indices_and_consistent_schema():
    urls = pd.DataFrame({"url": ["https://example.com/", "http://192.0.2.1/login"]}, index=[42, 7])
    transformer = URLFeatureExtractor()
    output = transformer.fit_transform(urls)
    assert list(output.index) == [42, 7]
    assert tuple(output.columns) == FEATURE_NAMES
    assert output.shape == (2, 30)
    assert all(dtype == np.dtype("float64") for dtype in output.dtypes)
    pd.testing.assert_frame_equal(output, transformer.transform(urls["url"]))
    assert np.array_equal(transformer.get_feature_names_out(), FEATURE_NAMES)
    assert clone(transformer).get_params() == {}


def test_transformer_accepts_lists_and_empty_input_and_rejects_wrong_schema():
    transformer = URLFeatureExtractor()
    assert transformer.transform(["https://example.com/"]).shape == (1, 30)
    assert transformer.transform([]).shape == (0, 30)
    with pytest.raises(ValueError, match="'url'"):
        transformer.transform(pd.DataFrame({"website": ["https://example.com/"]}))
    with pytest.raises(ValueError, match="list"):
        transformer.transform("https://example.com/")
