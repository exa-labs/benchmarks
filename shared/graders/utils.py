import re
from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing protocol, www, and trailing slashes."""
    if not url:
        return ""
    parsed = urlparse(url.lower())
    domain = parsed.netloc or parsed.path
    domain = re.sub(r"^www\.", "", domain)
    path = parsed.path.rstrip("/") if parsed.netloc else ""
    return f"{domain}{path}".rstrip("/")


def url_matches(result_url: str, gold_url: str) -> bool:
    """Check if result URL matches or contains the gold URL."""
    result_normalized = normalize_url(result_url)
    gold_normalized = normalize_url(gold_url)

    if not result_normalized or not gold_normalized:
        return False

    return gold_normalized in result_normalized or result_normalized in gold_normalized
