from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse


def is_weekly_note(url: Optional[str], title: Optional[str], rss_category: Optional[str] = None) -> bool:
    """
    Determine if an article belongs to the "weekly notes" cohort based on URL patterns
    and optional metadata.

    Rules (high precision):
    - URL path contains '/option-flows/' OR '/option-flow-'
    - If RSS category is provided, accept if it equals/contains 'option flow' or 'weekly notes'
    - Title alone is NOT sufficient to avoid false positives

    Args:
        url: Article URL
        title: Article title (unused for decision except for future expansion)
        rss_category: Optional RSS category string

    Returns:
        True if article is considered a weekly note, else False
    """
    if not url:
        return False

    try:
        path = urlparse(url).path.lower()
    except Exception:
        # If URL parsing fails, be conservative
        return False

    # Primary URL-based patterns
    if "/option-flows/" in path:
        return True
    if "/option-flow-" in path:
        return True

    # Secondary: RSS category hints (optional)
    if rss_category:
        cat = rss_category.strip().lower()
        if "option flow" in cat:
            return True
        if "weekly note" in cat or "weekly notes" in cat:
            return True

    return False
