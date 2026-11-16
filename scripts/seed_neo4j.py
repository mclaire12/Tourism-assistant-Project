"""Merge the Rwanda places dataset into Neo4j AuraDB.

Requires NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD. The queries use MERGE,
so running this again does not duplicate destinations.
"""

from src.runtime import build_runtime


def main() -> int:
    runtime = build_runtime()
    if runtime.neo4j is None:
        print("Neo4j is not configured or could not be reached. Nothing was changed.")
        return 1
    runtime.neo4j.seed(runtime.places)
    print(f"Graph merge completed. Destinations: {runtime.neo4j.destination_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
