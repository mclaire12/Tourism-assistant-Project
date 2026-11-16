"""Turn retrieved records into a readable answer. Facts come only from the dataset."""

from __future__ import annotations

OFFICIAL_NOTE = (
    "Prices, permits, and safety advice change. Check official sources "
    "(RDB and your own government) before you travel."
)


def public_destination(place: dict, graph_row: dict | None, sources: list[str]) -> dict:
    regions = _joined(graph_row.get("regions") if graph_row else None, place.get("region"))
    districts = _joined(graph_row.get("districts") if graph_row else None, place.get("district"))
    urls = [url for url in (place.get("source_urls") or []) if _is_http(url)]
    return {
        "id": place["id"],
        "name": place["name"],
        "type": place.get("type", ""),
        "region": regions,
        "district": districts,
        "description": place.get("description", ""),
        "activities": list(place.get("activities") or []),
        "interests": list(place.get("interests") or []),
        "moods": list(place.get("moods") or []),
        "best_season": place.get("best_season", ""),
        "budget_level": place.get("budget_level", ""),
        "typical_duration": place.get("typical_duration", ""),
        "costs_note": place.get("costs_note", ""),
        "safety_note": place.get("safety_note", ""),
        "access_from_kigali": place.get("access_from_kigali", ""),
        "tips": list(place.get("tips") or []),
        "source_urls": urls,
        "source_confidence": place.get("source_confidence", ""),
        "last_verified": place.get("last_verified", ""),
        "retrieval_sources": sources,
    }


def destination_markdown(destination: dict) -> str:
    lines = [
        f"### {destination['name']}",
        f"{destination['district']}, {destination['region']}",
        "",
        destination["description"],
        "",
        f"**Activities:** {_comma(destination['activities'])}",
        f"**Budget:** {destination['budget_level']} · **Typical duration:** {destination['typical_duration']}",
        f"**Best season:** {destination['best_season']}",
        f"**Access from Kigali:** {destination['access_from_kigali']}",
    ]
    if destination["tips"]:
        lines.append("**Travel tips:** " + "; ".join(destination["tips"]))
    if destination["costs_note"]:
        lines.append(f"**Costs:** {destination['costs_note']}")
    if destination["safety_note"]:
        lines.append(f"**Safety:** {destination['safety_note']}")
    if destination["source_urls"]:
        links = ", ".join(destination["source_urls"])
        lines.append(f"**Sources:** {links}")
    else:
        lines.append("**Sources:** No source URL was recorded for this destination.")
    if destination["last_verified"]:
        lines.append(
            f"_Record last verified {destination['last_verified']}"
            f" ({destination['source_confidence']} confidence)._"
        )
    return "\n".join(lines)


def lead_for_answer(question: str, in_scope: bool, detected_country: str | None, destinations: list[dict], model_text: str | None, model_notice: str | None) -> str:
    if not in_scope:
        country = (detected_country or "another country").title()
        refusal = (
            f"This assistant covers tourism in Rwanda only, so I can't answer questions about {country}."
        )
        if destinations:
            names = ", ".join(item["name"] for item in destinations[:3])
            return (
                f"{refusal} If you are looking for a similar experience in Rwanda, "
                f"the dataset points to: {names}."
            )
        return (
            f"{refusal} The Rwanda tourism dataset does not contain a sourced answer to that question."
        )

    if model_text:
        return model_text.strip()

    if not destinations:
        return (
            "I couldn't find a matching destination in the Rwanda tourism dataset. "
            "Try asking about gorillas, lakes, culture, parks, or a district."
        )

    names = ", ".join(item["name"] for item in destinations[:3])
    notice = model_notice or (
        "Qwen3 is not connected, so this reply quotes the retrieved destination records directly."
    )
    return f"{notice} Matching destinations: {names}."


def _joined(values, fallback: str) -> str:
    cleaned = [str(value) for value in (values or []) if value]
    if cleaned:
        return ", ".join(cleaned)
    return fallback or ""


def _comma(values: list) -> str:
    return ", ".join(values) if values else "Not recorded."


def _is_http(url: str) -> bool:
    return isinstance(url, str) and url.startswith(("http://", "https://")) and " " not in url
