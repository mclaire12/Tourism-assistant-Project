"""Load the original Rwanda places dataset and build retrieval documents."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import DATA_PATH


def clean_value(value):
    """Strip surrounding whitespace without changing the meaning of a field."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    return value


def load_dataset(path: Path | None = None) -> dict:
    dataset_path = path or DATA_PATH
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return clean_value(data)


def build_document(place: dict) -> str:
    """Document text used by the original ChromaDB index."""
    tips = place.get("tips") or []
    text = f"""
Destination: {place['name']}

Type: {place['type']}

Location:
Region: {place['region']}
District: {place['district']}

Description:
{place['description']}

Activities:
{', '.join(place['activities'])}

Visitor Interests:
{', '.join(place['interests'])}

Best Season:
{place['best_season']}

Budget Level:
{place['budget_level']}

Typical Duration:
{place['typical_duration']}

Costs Information:
{place['costs_note']}

Safety Information:
{place['safety_note']}

Access from Kigali:
{place['access_from_kigali']}

Travel Tips:
{'; '.join(tips) if tips else 'No additional tips recorded.'}
""".strip()
    return text


def build_metadata(place: dict) -> dict:
    return {
        "id": place["id"],
        "name": place["name"],
        "type": place["type"],
        "region": place["region"],
        "district": place["district"],
        "budget_level": place["budget_level"],
        "source_confidence": place["source_confidence"],
        "last_verified": place["last_verified"],
    }


def embedding_source_text(place: dict, document: str, embedder_name: str) -> str:
    """Text that is actually embedded.

    BAAI/bge-m3 embeds the original document. The hashed fallback repeats the
    name and appends moods so short questions such as "nature" or "cultural"
    still meet the records that use those words only as moods.
    """
    if embedder_name.startswith("BAAI/"):
        return document
    moods = ", ".join(place.get("moods") or [])
    return f"{place['name']}\n{place['name']}\n{place['name']}\n{document}\nMoods: {moods}"
