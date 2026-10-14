"""Domain comparisons are offline structural warnings, never ground truth."""

import json
from unittest.mock import patch

import pytest

from phishshield.domain_signals import BRAND_CATALOG, get_domain_signals
from phishshield.features import parse_url


@pytest.mark.parametrize("url,brand,code", [
    ("https://mail.g0ogle.test/", "Google", "brand_digit_substitution"),
    ("https://mail.googletex.test/", "Google", "brand_affix"),
    ("https://paypa1.test/", "PayPal", "brand_digit_substitution"),
    ("https://app1e.test/", "Apple", "brand_digit_substitution"),
    ("https://micros0ft.test/", "Microsoft", "brand_digit_substitution"),
    ("https://gooogle.test/", "Google", "brand_typo"),
    ("https://paypall.test/", "PayPal", "brand_typo"),
    ("https://micorsoft.test/", "Microsoft", "brand_typo"),
    ("https://youtbue.test/", "YouTube", "brand_typo"),
    ("https://secure-paypal.test/signin", "PayPal", "brand_affix"),
    ("https://apple-login.test/", "Apple", "brand_affix"),
    ("https://aple.test/account", "Apple", "brand_typo"),
])
def test_generalized_lookalike_patterns(url, brand, code):
    signals = get_domain_signals(url)
    assert [(signal.brand, signal.code) for signal in signals] == [(brand, code)]
    assert signals[0].observed_domain == parse_url(url).registered_domain
    assert "not proof" in signals[0].description
    assert "risk_score" not in signals[0].to_dict()
    assert "phishing_probability" not in signals[0].to_dict()
    assert json.loads(json.dumps(signals[0].to_dict()))["official_domains"]


@pytest.mark.parametrize("url,expected_domain", [
    ("https://google.com.attacker.test/login", "attacker.test"),
    ("https://paypal.attacker.co.uk/", "attacker.co.uk"),
    ("https://paypal.attacker.github.io/", "attacker.github.io"),
    ("https://secure-google.attacker.test/", "attacker.test"),
])
def test_brand_subdomains_do_not_establish_registered_domain_ownership(url, expected_domain):
    signals = get_domain_signals(url)
    assert len(signals) == 1
    assert signals[0].code == "brand_in_subdomain"
    assert signals[0].observed_domain == expected_domain


def test_private_suffix_tenant_label_is_compared_as_registered_domain():
    signals = get_domain_signals("https://g0ogle.github.io/")
    assert len(signals) == 1
    assert signals[0].observed_domain == "g0ogle.github.io"
    assert signals[0].code == "brand_digit_substitution"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=HsZxZ8hoYdU",
    "https://mail.google.com/", "https://accounts.google.co.in/login",
    "https://accounts.google.co.uk/login", "https://youtu.be/example",
    "https://support.microsoft.com/", "https://login.microsoftonline.com/common/oauth2/authorize",
    "https://www.paypal.com/signin", "https://support.apple.com/", "https://www.icloud.com/",
])
def test_catalogue_domains_and_subdomains_have_no_lookalike_warning(url):
    assert get_domain_signals(url) == []


@pytest.mark.parametrize("url", [
    "https://www.example.test/", "https://example.test/google/login",
    "https://example.test/?url=https://paypal.com/", "https://applepie.test/",
    "https://appleton.test/", "https://pineapple.test/", "https://ample.test/login",
    "https://maple.test/", "https://apply.test/", "https://goggle.test/",
    "https://papal.test/", "https://governance.test/", "https://bucher.test/",
    "https://bücher.test/", "https://192.0.2.10/google", "https://unknown-intranet/",
    "https://updates.school.test/", "https://apple.school.test/recipes",
])
def test_generic_words_and_brand_mentions_in_content_are_not_impersonation(url):
    assert get_domain_signals(url) == []


def test_idna_confusable_warning_matches_unicode_and_ascii_forms():
    # The first 'a' is Cyrillic; reserved suffix prevents a live-demo address.
    unicode_url = "https://pаypal.test/"
    parsed = parse_url(unicode_url)
    assert parsed.hostname.startswith("xn--")
    unicode_signals = get_domain_signals(unicode_url)
    assert unicode_signals == get_domain_signals(parsed)
    assert unicode_signals == get_domain_signals(parsed.normalized_url)
    assert len(unicode_signals) == 1
    assert unicode_signals[0].code == "brand_unicode_confusable"
    assert unicode_signals[0].brand == "PayPal"


def test_detector_does_not_resolve_or_fetch_hosts():
    with patch("socket.getaddrinfo", side_effect=AssertionError("Network must not be used")), patch("requests.get", side_effect=AssertionError("Network must not be used")):
        signals = get_domain_signals("https://mail.g0ogle.test/")
    assert len(signals) == 1


def test_reference_catalogue_has_primary_source_attribution_and_is_small():
    assert len(BRAND_CATALOG) == 5
    assert all(reference.sources and reference.official_domains for reference in BRAND_CATALOG)
    assert all(source.startswith("https://") for reference in BRAND_CATALOG for source in reference.sources)
