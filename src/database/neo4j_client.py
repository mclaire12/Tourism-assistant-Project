"""Neo4j AuraDB client. Credentials come from settings, never from source code."""

from __future__ import annotations

import logging

from src.database.knowledge_graph import destination_rows, relationship_rows

logger = logging.getLogger("rwanda_tourism.neo4j")

GRAPH_RETRIEVAL_QUERY = """
MATCH (d:Destination)

OPTIONAL MATCH (d)-[:OFFERS]->(a:Activity)
OPTIONAL MATCH (d)-[:MATCHES_INTEREST]->(i:Interest)
OPTIONAL MATCH (d)-[:HAS_MOOD]->(m:Mood)
OPTIONAL MATCH (d)-[:IN_REGION]->(r:Region)
OPTIONAL MATCH (d)-[:LOCATED_IN]->(dist:District)

WITH d,
     collect(DISTINCT a.name) AS activities,
     collect(DISTINCT i.name) AS interests,
     collect(DISTINCT m.name) AS moods,
     collect(DISTINCT r.name) AS regions,
     collect(DISTINCT dist.name) AS districts

WHERE
    toLower(d.name) CONTAINS toLower($search_term)
    OR any(x IN activities WHERE toLower(x) CONTAINS toLower($search_term))
    OR any(x IN interests WHERE toLower(x) CONTAINS toLower($search_term))
    OR any(x IN moods WHERE toLower(x) CONTAINS toLower($search_term))
    OR any(x IN regions WHERE toLower(x) CONTAINS toLower($search_term))
    OR any(x IN districts WHERE toLower(x) CONTAINS toLower($search_term))

RETURN
    d.id AS id,
    d.name AS destination,
    d.type AS type,
    regions,
    districts,
    activities,
    interests,
    moods

ORDER BY destination
LIMIT $top_k
"""


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=8,
            max_transaction_retry_time=10,
        )
        try:
            self.driver.verify_connectivity()
        except Exception:
            self.driver.close()
            raise

    def close(self) -> None:
        self.driver.close()

    def destination_count(self) -> int:
        with self.driver.session() as session:
            record = session.run("MATCH (d:Destination) RETURN count(d) AS total").single()
        return int(record["total"]) if record else 0

    def seed(self, places: list[dict]) -> None:
        """Create the graph with MERGE so an empty AuraDB can be rebuilt."""
        destinations = destination_rows(places)
        links = relationship_rows(places)
        statements = [
            (
                """
                CREATE CONSTRAINT destination_id_unique IF NOT EXISTS
                FOR (d:Destination)
                REQUIRE d.id IS UNIQUE
                """,
                {},
            ),
            (
                """
                UNWIND $destinations AS destination
                MERGE (d:Destination {id: destination.id})
                SET d += destination
                """,
                {"destinations": destinations},
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (d:Destination {id: row.id})
                MERGE (n:District {name: row.name})
                MERGE (d)-[:LOCATED_IN]->(n)
                """,
                {"rows": links["located_in"]},
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (d:Destination {id: row.id})
                MERGE (n:Region {name: row.name})
                MERGE (d)-[:IN_REGION]->(n)
                """,
                {"rows": links["in_region"]},
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (d:Destination {id: row.id})
                MERGE (n:Activity {name: row.name})
                MERGE (d)-[:OFFERS]->(n)
                """,
                {"rows": links["offers"]},
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (d:Destination {id: row.id})
                MERGE (n:Interest {name: row.name})
                MERGE (d)-[:MATCHES_INTEREST]->(n)
                """,
                {"rows": links["matches_interest"]},
            ),
            (
                """
                UNWIND $rows AS row
                MATCH (d:Destination {id: row.id})
                MERGE (n:Mood {name: row.name})
                MERGE (d)-[:HAS_MOOD]->(n)
                """,
                {"rows": links["has_mood"]},
            ),
        ]
        with self.driver.session() as session:
            for query, params in statements:
                session.run(query, **params)
        logger.info("Seeded Neo4j from the Rwanda places dataset")

    def search(self, search_term: str, top_k: int = 5) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                GRAPH_RETRIEVAL_QUERY,
                search_term=search_term,
                top_k=top_k,
            )
            rows = []
            for record in result:
                data = record.data()
                rows.append(
                    {
                        "id": data.get("id"),
                        "destination": data.get("destination"),
                        "type": data.get("type"),
                        "regions": _clean_list(data.get("regions")),
                        "districts": _clean_list(data.get("districts")),
                        "activities": _clean_list(data.get("activities")),
                        "interests": _clean_list(data.get("interests")),
                        "moods": _clean_list(data.get("moods")),
                    }
                )
            return rows


def _clean_list(values) -> list:
    if not values:
        return []
    return [value for value in values if value]


def connect_neo4j(settings, places: list[dict]):
    """Return a connected client, or None when AuraDB is not configured."""
    if not settings.neo4j_configured:
        return None, (
            "Neo4j AuraDB is not configured. Graph questions use the dataset "
            "knowledge graph until NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are set."
        )
    try:
        client = Neo4jClient(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    except Exception as exc:
        logger.warning("Neo4j connection failed (%s)", exc.__class__.__name__)
        return None, (
            "Neo4j AuraDB could not be reached. Graph questions use the dataset "
            "knowledge graph until the connection is available."
        )
    try:
        if client.destination_count() == 0:
            client.seed(places)
    except Exception:
        logger.exception("Neo4j seed failed")
        client.close()
        return None, (
            "Connected to Neo4j, but the tourism graph could not be prepared. "
            "Graph questions use the dataset knowledge graph for now."
        )
    return client, ""
