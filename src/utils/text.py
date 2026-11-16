"""Token helpers shared by keyword extraction and the fallback embedder."""

from __future__ import annotations

import re

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "best",
    "can",
    "destination",
    "destinations",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "recommend",
    "rwanda",
    "see",
    "should",
    "suggest",
    "the",
    "there",
    "to",
    "visit",
    "want",
    "what",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def variants(token: str) -> set[str]:
    """Return light plural variants. The longest real match is chosen later."""
    forms = {token}
    if len(token) > 4 and token.endswith("ies"):
        forms.add(token[:-3] + "y")
    if len(token) > 4 and token.endswith("es"):
        forms.add(token[:-2])
        forms.add(token[:-1])
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        forms.add(token[:-1])
    return {form for form in forms if len(form) >= 3}


def content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall((text or "").lower()):
        if raw in STOPWORDS or len(raw) < 3:
            continue
        # Keep the surface token. Callers compare variants against stored text.
        tokens.append(raw)
    return tokens


def contains_term(text: str, term: str) -> bool:
    haystack = (text or "").lower()
    needle = term.lower().strip()
    if not needle:
        return False
    if len(needle) <= 4:
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return needle in haystack
