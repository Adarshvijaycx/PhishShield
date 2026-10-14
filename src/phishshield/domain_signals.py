"""Conservative offline comparisons with a small, documented brand catalogue.

These warnings describe spelling and domain-boundary patterns. They establish
neither ownership nor maliciousness, and they never change a model probability.
The catalogue is intentionally incomplete; an absent warning is not a verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from phishshield.features import ParsedURL


DOMAIN_SIGNAL_VERSION = "1.0.0"

# Google's own supported-domain list includes regional search domains. Keeping
# the documented list offline avoids falsely flagging google.fr, for example,
# when checking a same-brand label on an unfamiliar suffix.
_GOOGLE_DOMAINS = tuple("""
google.com google.ad google.ae google.com.af google.com.ag google.al google.am
google.co.ao google.com.ar google.as google.at google.com.au google.az google.ba
google.com.bd google.be google.bf google.bg google.com.bh google.bi google.bj
google.com.bn google.com.bo google.com.br google.bs google.bt google.co.bw
google.by google.com.bz google.ca google.cd google.cf google.cg google.ch
google.ci google.co.ck google.cl google.cm google.cn google.com.co google.co.cr
google.com.cu google.cv google.com.cy google.cz google.de google.dj google.dk
google.dm google.com.do google.dz google.com.ec google.ee google.com.eg google.es
google.com.et google.fi google.com.fj google.fm google.fr google.ga google.ge
google.gg google.com.gh google.com.gi google.gl google.gm google.gr google.com.gt
google.gy google.com.hk google.hn google.hr google.ht google.hu google.co.id
google.ie google.co.il google.im google.co.in google.iq google.is google.it
google.je google.com.jm google.jo google.co.jp google.co.ke google.com.kh
google.ki google.kg google.co.kr google.com.kw google.kz google.la google.com.lb
google.li google.lk google.co.ls google.lt google.lu google.lv google.com.ly
google.co.ma google.md google.me google.mg google.mk google.ml google.com.mm
google.mn google.com.mt google.mu google.mv google.mw google.com.mx
google.com.my google.co.mz google.com.na google.com.ng google.com.ni google.ne
google.nl google.no google.com.np google.nr google.nu google.co.nz google.com.om
google.com.pa google.com.pe google.com.pg google.com.ph google.com.pk google.pl
google.pn google.com.pr google.ps google.pt google.com.py google.com.qa google.ro
google.ru google.rw google.com.sa google.com.sb google.sc google.se
google.com.sg google.sh google.si google.sk google.com.sl google.sn google.so
google.sm google.sr google.st google.com.sv google.td google.tg google.co.th
google.com.tj google.tl google.tm google.tn google.to google.com.tr google.tt
google.com.tw google.co.tz google.com.ua google.co.ug google.co.uk google.com.uy
google.co.uz google.com.vc google.co.ve google.co.vi google.com.vn google.vu
google.ws google.rs google.co.za google.co.zm google.co.zw google.cat
""".split())


@dataclass(frozen=True)
class BrandReference:
    name: str
    token: str
    official_domains: tuple[str, ...]
    sources: tuple[str, ...]
    generic_word: bool = False


# These are comparison references, not a safe-domain list. Sources identify
# canonical domains and selected alternate domains, checked 2026-10-14.
BRAND_CATALOG = (
    BrandReference(
        "Google", "google", _GOOGLE_DOMAINS + ("google.org",),
        ("https://support.google.com/mail/answer/8253", "https://www.google.com/supported_domains", "https://www.google.org/"),
    ),
    BrandReference(
        "YouTube", "youtube", ("youtube.com", "youtu.be"),
        ("https://support.google.com/youtube/answer/57741", "https://blog.youtube/news-and-events/make-way-for-youtube-links/"),
    ),
    BrandReference(
        "Microsoft", "microsoft", ("microsoft.com", "microsoftonline.com", "microsoft365.com"),
        ("https://support.microsoft.com/en-us/security/protect-yourself-from-phishing", "https://learn.microsoft.com/en-us/entra/architecture/authenticate-applications-and-users", "https://www.microsoft365.com/"),
    ),
    BrandReference(
        "PayPal", "paypal", ("paypal.com", "paypal.me"),
        ("https://www.paypal.com/us/cshelp/article/how-do-i-spot-a-fake-fraudulent-or-phishing-paypal-email-or-website-help164", "https://www.paypal.com/us/cshelp/article/paypalme-frequently-asked-questions-help432"),
    ),
    BrandReference(
        "Apple", "apple", ("apple.com", "icloud.com"),
        ("https://support.apple.com/en-us/102568",),
        generic_word=True,
    ),
)


@dataclass(frozen=True)
class DomainSignal:
    code: str
    title: str
    description: str
    brand: str
    observed_domain: str
    official_domains: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return JSON-compatible fields suitable for an indicator list."""
        result = asdict(self)
        result["official_domains"] = list(self.official_domains)
        return result


# A limited set of visually similar characters, not a complete Unicode
# confusables implementation. Do not equate all internationalized domains with
# impersonation. Only a resulting catalogue-brand match raises a warning.
_UNICODE_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ԛ": "q",
    "һ": "h", "ӏ": "l", "α": "a", "ο": "o", "ρ": "p", "ι": "i",
})
_DIGIT_CONFUSABLES = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "7": "t", "8": "b", "9": "g"})
_ACTION_WORDS = frozenset({"account", "accounts", "auth", "billing", "login", "logon", "password", "payment", "secure", "security", "signin", "sign-in", "support", "update", "verification", "verify", "wallet"})
_COMMON_NEIGHBORS = frozenset({"goggle", "goggles", "googly", "papal", "ample", "apply", "maple", "apples", "applet"})


def _unicode_label(label: str) -> str:
    try:
        return label.encode("ascii").decode("idna").lower()
    except (UnicodeError, UnicodeEncodeError):
        return label.lower()


def _skeleton(label: str) -> str:
    text = _unicode_label(label).translate(_UNICODE_CONFUSABLES)
    text = "".join(character for character in unicodedata.normalize("NFKD", text) if not unicodedata.combining(character))
    return text.translate(_DIGIT_CONFUSABLES).replace("rn", "m").replace("vv", "w")


def _single_edit(left: str, right: str) -> bool:
    """Recognize exactly one insertion, deletion, substitution, or swap."""
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differing = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(differing) == 1:
            return True
        return len(differing) == 2 and differing[1] == differing[0] + 1 and left[differing[0]] == right[differing[1]] and left[differing[1]] == right[differing[0]]
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    first_difference = next((index for index, (a, b) in enumerate(zip(longer, shorter)) if a != b), len(shorter))
    return longer[:first_difference] + longer[first_difference + 1:] == shorter


def _action_context(parsed: ParsedURL, label: str) -> bool:
    # Token matching avoids generic substrings such as "catalogin".
    words = set(re.split(r"[^a-z0-9]+", f"{label} {parsed.subdomain} {parsed.path} {parsed.query}".lower()))
    return bool(words & _ACTION_WORDS)


def _brand_affix(label: str, reference: BrandReference) -> bool:
    token = reference.token
    for segment in label.split("-"):
        if segment == token and segment != label:
            return True
    if label.startswith(token):
        rest = label[len(token):]
    elif label.endswith(token):
        rest = label[:-len(token)]
    else:
        return False
    if not rest:
        return False
    # Short unexplained additions catch google+tex; ordinary generic-word
    # compounds such as applepie receive no warning without action context.
    return rest.strip("-") in _ACTION_WORDS or (not reference.generic_word and len(rest) <= 3)


def _label_reason(label: str, reference: BrandReference, context: bool) -> tuple[str, str] | None:
    token = reference.token
    if label == token and (not reference.generic_word or context):
        return "brand_domain_mismatch", "uses the brand name with a different registered-domain suffix from the comparison catalogue"
    unicode_label = _unicode_label(label)
    skeleton = _skeleton(label)
    if skeleton == token and unicode_label != token:
        if any(character.isdigit() for character in unicode_label):
            return "brand_digit_substitution", "uses digits in place of visually similar brand letters"
        if unicode_label != label or any(ord(character) > 127 for character in unicode_label):
            return "brand_unicode_confusable", "uses internationalized letters resembling the brand spelling"
        return "brand_confusable_spelling", "uses a visually similar spelling of the brand name"
    if label not in _COMMON_NEIGHBORS and _single_edit(label, token) and (not reference.generic_word or context):
        return "brand_typo", "differs from the brand spelling by one character edit or adjacent swap"
    if _brand_affix(label, reference) and (not reference.generic_word or context):
        return "brand_affix", "adds text to the brand name in the registered-domain label"
    return None


def get_domain_signals(url: str | ParsedURL) -> list[DomainSignal]:
    """Describe possible brand lookalikes without DNS, HTTP, or score changes.

    Checks the effective registered domain (including private PSL suffixes),
    rather than trusting brand text elsewhere in a URL. Paths and queries can
    provide context for generic words, but are never mistaken for the host.
    """
    if isinstance(url, str):
        from phishshield.features import parse_url
        parsed = parse_url(url)
    else:
        parsed = url
    if parsed.is_ip or not parsed.tld:
        return []
    label = parsed.registered_domain.removesuffix("." + parsed.tld)
    signals = []
    for reference in BRAND_CATALOG:
        # This skips a lookalike warning only; no statement about this URL's
        # content, credentials, path, redirects, or overall safety follows.
        if parsed.registered_domain in reference.official_domains:
            continue
        context = _action_context(parsed, label)
        reason = _label_reason(label, reference, context)
        if reason is None:
            for subdomain_label in parsed.subdomain.split(".") if parsed.subdomain else ():
                subdomain_reason = _label_reason(subdomain_label, reference, context)
                if (subdomain_label == reference.token and (not reference.generic_word or context)) or subdomain_reason is not None:
                    reason = ("brand_in_subdomain", "places brand-like text in a subdomain while the registered domain is different")
                    break
        if reason is None:
            continue
        code, wording = reason
        official = ", ".join(reference.official_domains[:3])
        if len(reference.official_domains) > 3:
            official += f" and {len(reference.official_domains) - 3} other documented comparison domains"
        signals.append(DomainSignal(
            code=code, title=f"Possible {reference.name} domain lookalike",
            description=f"The hostname {wording}. Its registered domain is {parsed.registered_domain}; comparison domains for {reference.name} are {official}. This spelling pattern is a warning to verify independently, not proof of phishing or ownership.",
            brand=reference.name, observed_domain=parsed.registered_domain,
            official_domains=reference.official_domains,
        ))
    return signals
