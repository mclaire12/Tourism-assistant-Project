"""Choose short Neo4j search terms from a natural-language question.

The original Cypher query checks whether a stored name or relationship
CONTAINS the search term. Passing the whole sentence therefore matches
nothing. These terms are substrings that actually occur in the graph.
"""

from __future__ import annotations

from src.database.knowledge_graph import graph_fields
from src.utils.text import contains_term, content_tokens, variants

GENERIC = {
    "activity",
    "activities",
    "centre",
    "center",
    "district",
    "experience",
    "lovers",
    "museum",
    "national",
    "park",
    "province",
    "tourism",
}


def extract_graph_keywords(question: str, places: list[dict], limit: int = 3) -> list[str]:
    question_text = question or ""
    lowered = question_text.lower()

    name_hits = [
        place["name"].lower()
        for place in places
        if place.get("name") and place["name"].lower() in lowered
    ]
    if name_hits:
        return _unique(name_hits)[:limit]

    fields = graph_fields(places)
    found: list[str] = []

    phrases = sorted({field.strip().lower() for field in fields if " " in field}, key=len, reverse=True)
    for phrase in phrases:
        if phrase and phrase in lowered:
            found.append(phrase)

    for token in content_tokens(question_text):
        term = _term_in_fields(token, fields)
        if term:
            found.append(term)

    specific = [term for term in _unique(found) if term not in GENERIC]
    chosen = specific or _unique(found)
    if not chosen:
        stripped = question_text.strip()
        return [stripped] if stripped else []

    pruned: list[str] = []
    for term in sorted(chosen, key=len, reverse=True):
        if any(term in longer or longer in term for longer in pruned):
            continue
        pruned.append(term)
        if len(pruned) >= limit:
            break
    return pruned


def _term_in_fields(token: str, fields: list[str]) -> str | None:
    best = None
    best_score = 0
    for candidate in variants(token):
        score = sum(1 for field in fields if contains_term(field, candidate))
        if score > best_score or (score == best_score and best is not None and len(candidate) > len(best)):
            best = candidate
            best_score = score
    return best if best_score else None


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered
