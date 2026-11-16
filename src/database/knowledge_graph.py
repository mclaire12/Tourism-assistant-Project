"""In-memory copy of the tourism graph used by the Neo4j Cypher query.

Relationship rules follow the notebook:

- districts are split on commas
- regions are split on slashes
- Eastern, Southern, and Western are stored as provinces
- activities, interests, and moods become their own nodes
"""

from __future__ import annotations

REGION_MAPPING = {
    "Eastern": "Eastern Province",
    "Southern": "Southern Province",
    "Western": "Western Province",
}

DESTINATION_PROPERTIES = [
    "id",
    "name",
    "type",
    "region",
    "district",
    "best_season",
    "budget_level",
    "typical_duration",
    "description",
    "costs_note",
    "safety_note",
    "access_from_kigali",
    "tips",
    "source_urls",
    "source_confidence",
    "last_verified",
]


def split_districts(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def split_regions(value: str) -> list[str]:
    regions = []
    for part in (value or "").split("/"):
        name = part.strip()
        if name:
            regions.append(REGION_MAPPING.get(name, name))
    return regions


def destination_rows(places: list[dict]) -> list[dict]:
    rows = []
    for place in places:
        row = {}
        for key in DESTINATION_PROPERTIES:
            value = place.get(key)
            if value is None:
                continue
            row[key] = value
        rows.append(row)
    return rows


def relationship_rows(places: list[dict]) -> dict[str, list[dict]]:
    located_in = []
    in_region = []
    offers = []
    matches_interest = []
    has_mood = []
    for place in places:
        place_id = place["id"]
        for name in split_districts(place.get("district", "")):
            located_in.append({"id": place_id, "name": name})
        for name in split_regions(place.get("region", "")):
            in_region.append({"id": place_id, "name": name})
        for name in place.get("activities") or []:
            offers.append({"id": place_id, "name": name})
        for name in place.get("interests") or []:
            matches_interest.append({"id": place_id, "name": name})
        for name in place.get("moods") or []:
            has_mood.append({"id": place_id, "name": name})
    return {
        "located_in": located_in,
        "in_region": in_region,
        "offers": offers,
        "matches_interest": matches_interest,
        "has_mood": has_mood,
    }


def graph_fields(places: list[dict]) -> list[str]:
    fields: list[str] = []
    for place in places:
        fields.append(place.get("name", ""))
        fields.extend(split_districts(place.get("district", "")))
        fields.extend(split_regions(place.get("region", "")))
        fields.extend(place.get("activities") or [])
        fields.extend(place.get("interests") or [])
        fields.extend(place.get("moods") or [])
    return [field for field in fields if field]


class KnowledgeGraph:
    """Same containment match as neo4j_graph_retrieval, without a database."""

    def __init__(self, places: list[dict]):
        self.records = []
        for place in places:
            self.records.append(
                {
                    "id": place["id"],
                    "destination": place["name"],
                    "type": place["type"],
                    "regions": split_regions(place.get("region", "")),
                    "districts": split_districts(place.get("district", "")),
                    "activities": list(place.get("activities") or []),
                    "interests": list(place.get("interests") or []),
                    "moods": list(place.get("moods") or []),
                }
            )

    def search(self, search_term: str, top_k: int = 5) -> list[dict]:
        term = (search_term or "").strip().lower()
        if not term:
            return []
        matches = []
        for record in self.records:
            haystacks = [
                record["destination"],
                *record["regions"],
                *record["districts"],
                *record["activities"],
                *record["interests"],
                *record["moods"],
            ]
            if any(term in (item or "").lower() for item in haystacks):
                matches.append(
                    {
                        "id": record["id"],
                        "destination": record["destination"],
                        "type": record["type"],
                        "regions": list(record["regions"]),
                        "districts": list(record["districts"]),
                        "activities": list(record["activities"]),
                        "interests": list(record["interests"]),
                        "moods": list(record["moods"]),
                    }
                )
        matches.sort(key=lambda item: item["destination"].lower())
        return matches[:top_k]
