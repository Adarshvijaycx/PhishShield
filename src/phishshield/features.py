"""Offline URL parsing, lexical features, and human-readable indicators.

The same transformer is used when fitting models and when analyzing a URL.
No function here resolves a host, visits a website, or updates the suffix list.
The Public Suffix List snapshot comes from the pinned ``tldextract`` package.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import ipaddress
import math
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import tldextract


FEATURE_VERSION = "2.0.0"
MAX_URL_LENGTH = 8192
SUSPICIOUS_KEYWORDS = ("login", "verify", "secure", "account", "update", "password")
FEATURE_NAMES = (
    "url_length", "hostname_length", "registered_domain_length", "path_length",
    "query_length", "fragment_length", "tld_length", "subdomain_count",
    "path_depth", "query_parameter_count", "digit_count", "digit_ratio",
    "hostname_digit_ratio", "special_character_count", "dot_count", "hyphen_count",
    "hostname_hyphen_count", "at_count", "percent_count", "slash_count",
    "question_mark_count", "equals_count", "ampersand_count", "keyword_count",
    "has_ip", "has_https", "has_userinfo", "has_nonstandard_port", "has_punycode",
    "hostname_entropy",
)

# Empty URLs disable downloads; cache_dir=None avoids any cache writes. Reserved
# suffixes let harmless academic examples receive the same domain breakdown.
_SUFFIX_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    fallback_to_snapshot=True,
    include_psl_private_domains=True,
    extra_suffixes=("example", "test", "invalid", "localhost"),
)
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


class URLValidationError(ValueError):
    """The submitted value cannot be safely interpreted as an HTTP(S) URL."""


@dataclass(frozen=True)
class ParsedURL:
    normalized_url: str
    hostname: str
    registered_domain: str
    subdomain: str
    tld: str
    scheme: str
    port: int | None
    path: str
    query: str
    fragment: str = ""
    userinfo: str = ""
    is_ip: bool = False


def parse_url(url: str) -> ParsedURL:
    """Validate and normalize a URL without touching the network.

    Path/query/fragment text and user-info are preserved, except that a lone
    root slash is canonicalized to an empty path: HTTP(S) treats both as the
    same resource. Hostnames become
    lowercase ASCII IDNA; IP addresses use their canonical representation.
    A terminal DNS dot is removed to group equivalent hostnames consistently.
    """
    if not isinstance(url, str):
        raise URLValidationError("Enter a URL as text.")
    if len(url) > MAX_URL_LENGTH:
        raise URLValidationError(f"The URL must be at most {MAX_URL_LENGTH:,} characters.")
    if _CONTROL_CHARACTER.search(url):
        raise URLValidationError("The URL must not contain control characters or line breaks.")
    text = url.strip()
    if not text:
        raise URLValidationError("Enter a URL to analyze.")
    if any(character.isspace() for character in text):
        raise URLValidationError("The URL must not contain spaces; encode spaces as %20.")
    if "\\" in text:
        raise URLValidationError("Use forward slashes in the URL, not backslashes.")
    if not re.match(r"^https?://", text, flags=re.IGNORECASE):
        raise URLValidationError("Include http:// or https:// at the start of the URL.")
    try:
        parts = urlsplit(text)
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise URLValidationError("The URL has an invalid hostname, IPv6 address, or port.") from exc
    if not hostname:
        raise URLValidationError("The URL must include a hostname, such as example.com.")
    if parts.netloc.endswith(":"):
        raise URLValidationError("Specify a port number after the colon, or remove it.")
    if port == 0:
        raise URLValidationError("A URL port must be between 1 and 65535.")
    hostname = hostname.removesuffix(".").lower()
    if not hostname:
        raise URLValidationError("The URL must include a valid hostname.")
    # Zone identifiers are device-specific and not ordinary public URL hosts.
    if "%" in hostname:
        raise URLValidationError("The hostname must not contain percent escapes or zone identifiers.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None:
        hostname = address.compressed
        registered_domain, subdomain, tld = hostname, "", ""
    else:
        if ":" in hostname or re.fullmatch(r"[0-9.]+", hostname) or parts.netloc.rsplit("@", 1)[-1].startswith("["):
            raise URLValidationError("The URL contains an invalid IP address.")
        try:
            hostname = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise URLValidationError("The URL contains an invalid international hostname.") from exc
        if len(hostname) > 253 or any(not _HOST_LABEL.fullmatch(label) for label in hostname.split(".")):
            raise URLValidationError("The URL contains an invalid hostname or domain label.")
        domain_parts = _SUFFIX_EXTRACTOR(hostname)
        tld = domain_parts.suffix
        subdomain = domain_parts.subdomain
        registered_domain = domain_parts.top_domain_under_public_suffix
        if not registered_domain:
            # Unknown suffixes and intranet hosts are accepted, but grouped by
            # their complete hostname rather than guessing an effective domain.
            registered_domain = hostname
            subdomain = ""
    userinfo = parts.netloc.rsplit("@", 1)[0] if "@" in parts.netloc else ""
    host_for_url = f"[{hostname}]" if address and address.version == 6 else hostname
    authority = f"{userinfo}@" if "@" in parts.netloc else ""
    authority += host_for_url
    if port is not None:
        authority += f":{port}"
    scheme = parts.scheme.lower()
    path = "" if parts.path == "/" else parts.path
    normalized_url = urlunsplit((scheme, authority, path, parts.query, parts.fragment))
    return ParsedURL(
        normalized_url=normalized_url, hostname=hostname,
        registered_domain=registered_domain, subdomain=subdomain, tld=tld,
        scheme=scheme, port=port, path=path, query=parts.query,
        fragment=parts.fragment, userinfo=userinfo, is_ip=address is not None,
    )


def _parsed(url: str | ParsedURL) -> ParsedURL:
    return url if isinstance(url, ParsedURL) else parse_url(url)


def _entropy(value: str) -> float:
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in Counter(value).values()) if length else 0.0


def extract_features(url: str | ParsedURL) -> dict[str, int | float]:
    """Return the stable, numeric lexical schema for a validated URL."""
    parsed = _parsed(url)
    text = parsed.normalized_url
    lowered = text.lower()
    digits = sum(character.isdigit() for character in text)
    hostname_digits = sum(character.isdigit() for character in parsed.hostname)
    values: dict[str, int | float] = {
        "url_length": len(text),
        "hostname_length": len(parsed.hostname),
        "registered_domain_length": len(parsed.registered_domain),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "fragment_length": len(parsed.fragment),
        "tld_length": len(parsed.tld),
        "subdomain_count": len(parsed.subdomain.split(".")) if parsed.subdomain else 0,
        "path_depth": sum(bool(segment) for segment in parsed.path.split("/")),
        "query_parameter_count": len(parse_qsl(parsed.query, keep_blank_values=True)),
        "digit_count": digits,
        "digit_ratio": digits / len(text),
        "hostname_digit_ratio": hostname_digits / len(parsed.hostname),
        "special_character_count": sum(not character.isalnum() for character in text),
        "dot_count": text.count("."),
        "hyphen_count": text.count("-"),
        "hostname_hyphen_count": parsed.hostname.count("-"),
        "at_count": text.count("@"),
        "percent_count": text.count("%"),
        "slash_count": text.count("/"),
        "question_mark_count": text.count("?"),
        "equals_count": text.count("="),
        "ampersand_count": text.count("&"),
        "keyword_count": sum(lowered.count(keyword) for keyword in SUSPICIOUS_KEYWORDS),
        "has_ip": int(parsed.is_ip),
        "has_https": int(parsed.scheme == "https"),
        "has_userinfo": int("@" in urlsplit(text).netloc),
        "has_nonstandard_port": int(parsed.port is not None and parsed.port != (443 if parsed.scheme == "https" else 80)),
        "has_punycode": int(any(label.startswith("xn--") for label in parsed.hostname.split("."))),
        "hostname_entropy": _entropy(parsed.hostname),
    }
    return {name: values[name] for name in FEATURE_NAMES}


def get_indicators(
    url: str | ParsedURL,
    tld_statistics: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Describe observed patterns; these are not proof a URL is phishing."""
    parsed = _parsed(url)
    features = extract_features(parsed)
    indicators: list[dict[str, str]] = []

    def add(code: str, title: str, description: str) -> None:
        indicators.append({"code": code, "title": title, "description": description})

    if parsed.is_ip:
        add("ip_address", "IP address in the URL", "The host is an IP address rather than a domain name.")
    if features["at_count"]:
        add("at_symbol", "@ symbol", "The URL contains @. In user-info syntax, the actual host follows the last @ before the path.")
    if features["subdomain_count"] >= 3:
        add("many_subdomains", "Many subdomains", f"The hostname contains {features['subdomain_count']} subdomain labels before the registered domain.")
    if features["url_length"] > 75:
        add("long_url", "Long URL", f"The normalized URL has {features['url_length']} characters, above the 75-character indicator threshold.")
    keywords = [keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in parsed.normalized_url.lower()]
    if keywords:
        add("suspicious_keywords", "Sensitive-action keywords", "Observed keywords: " + ", ".join(keywords) + ". Legitimate services can use these words too.")
    if features["hostname_hyphen_count"]:
        add("hyphenated_hostname", "Hyphenated hostname", "The hostname contains a hyphen; this pattern also occurs in legitimate domains.")
    if parsed.scheme == "http":
        add("no_https", "No HTTPS", "This URL uses HTTP rather than HTTPS. HTTPS alone does not establish legitimacy.")
    if features["has_punycode"]:
        add("punycode", "Internationalized hostname", "The hostname contains a punycode label. Check for visually similar letters; international domains can be legitimate.")
    statistics = (tld_statistics or {}).get(parsed.tld, {})
    count = statistics.get("count", 0)
    rate = statistics.get("phishing_rate", 0.0)
    if parsed.tld and isinstance(count, (int, float)) and isinstance(rate, (int, float)) and count >= 100 and 0.7 <= rate <= 1:
        add("dataset_tld_association", "TLD associated with phishing in this dataset", f".{parsed.tld} had a phishing share of {rate:.0%} among {int(count):,} training examples. This is dataset-specific evidence.")
    return indicators


class URLFeatureExtractor(TransformerMixin, BaseEstimator):
    """scikit-learn transformer accepting raw URLs and preserving row indices."""

    def fit(self, X: Any, y: Any = None) -> "URLFeatureExtractor":
        self._urls(X)
        self.n_features_in_ = 1
        return self

    @staticmethod
    def _urls(X: Any) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if "url" not in X.columns:
                raise ValueError("URLFeatureExtractor expects a DataFrame with a 'url' column.")
            return X["url"]
        if isinstance(X, pd.Series):
            return X
        if isinstance(X, str):
            raise ValueError("Pass URLs as a list, Series, or DataFrame with a 'url' column.")
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 1:
            X = X[:, 0]
        try:
            return pd.Series(list(X), dtype=object)
        except TypeError as exc:
            raise ValueError("Pass URLs as a list, Series, or DataFrame with a 'url' column.") from exc

    def transform(self, X: Any) -> pd.DataFrame:
        urls = self._urls(X)
        records = [extract_features(url) for url in urls]
        return pd.DataFrame(records, index=urls.index, columns=FEATURE_NAMES, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)


# The UCI collection has no benign paths, HTTP, or non-www hostnames. These
# features deliberately exclude all those collection proxies from the model.
# Full URL features remain available above as independent observations.
MODEL_SCOPE = "registered_domain"
MODEL_FEATURE_NAMES = (
    "registered_domain_length", "domain_label_length", "tld_length",
    "domain_digit_count", "domain_digit_ratio", "domain_hyphen_count",
    "domain_dot_count", "domain_entropy", "domain_keyword_count",
    "domain_vowel_ratio", "domain_max_digit_run", "has_ip", "has_punycode",
)


def extract_domain_features(url: str | ParsedURL) -> dict[str, int | float]:
    """Describe only the effective registered domain; never the URL path.

    This is a lexical domain model, not a safety verdict for the complete URL.
    Public-suffix private tenants remain separate effective domains.
    """
    parsed = _parsed(url)
    domain = parsed.registered_domain
    label = domain.removesuffix("." + parsed.tld) if parsed.tld else domain
    digits = sum(character.isdigit() for character in domain)
    return {
        "registered_domain_length": len(domain),
        "domain_label_length": len(label),
        "tld_length": len(parsed.tld),
        "domain_digit_count": digits,
        "domain_digit_ratio": digits / len(domain),
        "domain_hyphen_count": domain.count("-"),
        "domain_dot_count": domain.count("."),
        "domain_entropy": _entropy(domain),
        "domain_keyword_count": sum(domain.count(word) for word in SUSPICIOUS_KEYWORDS),
        "domain_vowel_ratio": sum(character in "aeiou" for character in label) / len(label),
        "domain_max_digit_run": max((len(run) for run in re.findall(r"[0-9]+", domain)), default=0),
        "has_ip": int(parsed.is_ip),
        "has_punycode": int(any(part.startswith("xn--") for part in domain.split("."))),
    }


class RegisteredDomainFeatureExtractor(URLFeatureExtractor):
    """Use the same domain-only schema during training and inference."""

    def transform(self, X: Any) -> pd.DataFrame:
        urls = self._urls(X)
        return pd.DataFrame(
            [extract_domain_features(url) for url in urls],
            index=urls.index, columns=MODEL_FEATURE_NAMES, dtype=float,
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(MODEL_FEATURE_NAMES, dtype=object)
