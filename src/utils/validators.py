"""Rwanda-only geographical validation from the original LangGraph workflow."""

from __future__ import annotations

import re

import pycountry

SUPPORTED_COUNTRY = "rwanda"

# Two-letter aliases such as "us" match ordinary English ("tell us").
# Longer aliases still mirror the notebook.
COUNTRY_ALIASES = {
    "usa": "united states",
    "u.s.a.": "united states",
    "u.s.": "united states",
    "u.k.": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",
    "drc": "democratic republic of the congo",
    "dr congo": "democratic republic of the congo",
    "ivory coast": "cote d'ivoire",
    "south korea": "korea, republic of",
    "north korea": "korea, democratic people's republic of",
    "uae": "united arab emirates",
}


def _known_countries() -> list[str]:
    names: set[str] = set()
    for country in pycountry.countries:
        names.add(country.name.lower())
        official = getattr(country, "official_name", None)
        if official:
            names.add(official.lower())
        common = getattr(country, "common_name", None)
        if common:
            names.add(common.lower())
    names.update(COUNTRY_ALIASES)
    return sorted(names, key=len, reverse=True)


KNOWN_COUNTRIES = _known_countries()


def _mask_places(question: str, place_names: list[str]) -> str:
    masked = question
    for name in sorted(place_names, key=len, reverse=True):
        if not name:
            continue
        masked = re.sub(re.escape(name), " ", masked, flags=re.IGNORECASE)
    return masked


def validate_geographical_scope(question: str, place_names: list[str] | None = None) -> dict:
    """Decide whether a question is inside the Rwanda tourism scope.

    A named destination from the dataset is masked first. That keeps questions
    about the Congo Nile Trail in scope, while a question about Tanzania stays
    out of scope. Behaviour otherwise follows the notebook: an unnamed country
    is treated as Rwanda, and a named foreign country is out of scope.
    """
    normalized_question = _mask_places(question or "", place_names or []).lower()
    detected_country = None
    matched_text = None

    for country in KNOWN_COUNTRIES:
        pattern = rf"(?<!\w){re.escape(country)}(?!\w)"
        match = re.search(pattern, normalized_question)
        if match:
            detected_country = COUNTRY_ALIASES.get(country, country)
            matched_text = match.group(0)
            break

    if detected_country == SUPPORTED_COUNTRY:
        in_scope = True
    elif detected_country is not None:
        in_scope = False
    else:
        in_scope = True

    return {
        "in_scope": in_scope,
        "detected_country": detected_country,
        "matched_text": matched_text,
        "original_question": question,
    }


def prepare_retrieval_question(question: str, detected_country: str | None, matched_text: str | None = None) -> str:
    """Replace a foreign country mention with Rwanda before retrieval."""
    if detected_country is None or detected_country == SUPPORTED_COUNTRY:
        return question

    surface = matched_text or detected_country
    pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)", re.IGNORECASE)
    retrieval_question = pattern.sub("Rwanda", question)
    return " ".join(retrieval_question.split())
