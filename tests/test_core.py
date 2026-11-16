"""Core behaviour of the Rwanda Tourism Assistant, without a hosted model."""

from __future__ import annotations

import os
import unittest

from src.agents.tourism_agent import answer_tourism_question, create_rag_prompt, route_retrieval, tourism_chat
from src.retrieval.hybrid_retrieval import hybrid_graph_vector_retrieval
from src.retrieval.vector_retrieval import hybrid_retrieval
from src.runtime import build_runtime
from src.utils.validators import validate_geographical_scope


class TourismAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("NEO4J_URI", None)
        os.environ.pop("NEO4J_USERNAME", None)
        os.environ.pop("NEO4J_PASSWORD", None)
        os.environ.pop("MODEL_PROVIDER", None)
        os.environ["EMBEDDING_PROVIDER"] = "hashed"
        cls.runtime = build_runtime()

    def test_dataset_has_the_original_nineteen_places(self):
        self.assertEqual(len(self.runtime.places), 19)
        required = {
            "id",
            "name",
            "type",
            "region",
            "district",
            "moods",
            "interests",
            "best_season",
            "budget_level",
            "typical_duration",
            "description",
            "activities",
            "costs_note",
            "safety_note",
            "access_from_kigali",
            "tips",
            "source_urls",
            "source_confidence",
            "last_verified",
        }
        for place in self.runtime.places:
            self.assertTrue(required <= set(place))

    def test_search_index_is_chromadb(self):
        self.assertEqual(self.runtime.vector_index.backend, "chromadb")
        self.assertEqual(self.runtime.vector_index.collection.count(), 19)

    def test_geographical_validation(self):
        names = [place["name"] for place in self.runtime.places]
        rwanda = validate_geographical_scope("Where can I see gorillas in Rwanda?", names)
        tanzania = validate_geographical_scope("What is the capital of Tanzania?", names)
        trail = validate_geographical_scope("What activities can I do on the Congo Nile Trail?", names)
        english_us = validate_geographical_scope("Can you tell us about gorillas?", names)
        self.assertTrue(rwanda["in_scope"])
        self.assertFalse(tanzania["in_scope"])
        self.assertEqual(tanzania["detected_country"], "tanzania")
        self.assertTrue(trail["in_scope"])
        self.assertTrue(english_us["in_scope"])

    def test_routing_rules(self):
        self.assertEqual(route_retrieval({"retrieval_question": "Where can I see gorillas in Rwanda?"}), "vector")
        self.assertEqual(
            route_retrieval({"retrieval_question": "Which district is Volcanoes National Park located in?"}),
            "graph",
        )
        self.assertEqual(
            route_retrieval({"retrieval_question": "Recommend lakes for water activities."}),
            "both",
        )

    def test_vector_retrieval_finds_gorillas_and_respects_metadata(self):
        gorillas = hybrid_retrieval("Where can I see gorillas in Rwanda?", top_k=3)
        names = [item["name"] for item in gorillas["metadatas"][0]]
        self.assertIn("Volcanoes National Park", names)
        parks = hybrid_retrieval(
            "national parks and wildlife",
            top_k=3,
            metadata_filter={"type": "national_park"},
        )
        self.assertTrue(parks["metadatas"][0])
        self.assertTrue(all(item["type"] == "national_park" for item in parks["metadatas"][0]))

    def test_graph_retrieval_returns_musanze_for_volcanoes(self):
        answer = tourism_chat("Which district is Volcanoes National Park located in?", [])
        self.assertIn("Musanze", answer)
        self.assertIn("Volcanoes National Park", answer)

    def test_hybrid_retrieval_combines_sources(self):
        combined = hybrid_graph_vector_retrieval(
            "I want to see gorillas and enjoy nature",
            "gorilla",
            top_k=3,
        )
        ids = {item["id"] for item in combined}
        self.assertIn("volcanoes-np", ids)
        volcanoes = next(item for item in combined if item["id"] == "volcanoes-np")
        self.assertIn("ChromaDB", volcanoes["source"])

    def test_representative_questions(self):
        cultural = tourism_chat("Where can I experience cultural tourism?", [])
        lakes = tourism_chat("Which lakes can I visit in Rwanda?", [])
        nyungwe = tourism_chat("What activities can I do in Nyungwe?", [])
        nature = tourism_chat("What are the best destinations for nature lovers?", [])
        self.assertTrue(
            any(name in cultural for name in ["Iby'Iwacu", "Ethnographic Museum", "King's Palace", "Inema"])
        )
        self.assertIn("Lake", lakes)
        self.assertIn("Nyungwe", nyungwe)
        self.assertTrue(any(name in nyungwe for name in ["chimpanzee", "canopy", "hiking", "colobus"]))
        self.assertTrue(
            any(name in nature for name in ["Nyungwe", "Akagera", "Volcanoes", "Gishwati", "Lake Kivu"])
        )

    def test_tanzania_is_refused_without_invented_facts(self):
        answer = tourism_chat("What is the capital of Tanzania?", [])
        lowered = answer.lower()
        self.assertIn("rwanda only", lowered)
        self.assertNotIn("dodoma", lowered)
        self.assertNotIn("dar es salaam", lowered)
        self.assertNotIn("serengeti", lowered)

    def test_follow_up_uses_conversation_history(self):
        history = [
            {"role": "user", "content": "Where can I see gorillas in Rwanda?"},
            {"role": "assistant", "content": "Volcanoes National Park is the gorilla destination."},
        ]
        answer = tourism_chat("what does it cost?", history)
        self.assertIn("Volcanoes National Park", answer)
        prompt = create_rag_prompt("what does it cost?", "context", history)
        self.assertIn("Where can I see gorillas in Rwanda?", prompt)

    def test_missing_model_is_explained_and_hides_secrets(self):
        os.environ["NEO4J_URI"] = "neo4j+s://invalid.example"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_PASSWORD"] = "super-secret-password"
        os.environ["MODEL_PROVIDER"] = ""
        runtime = build_runtime()
        self.assertIsNone(runtime.model)
        blob = " ".join(runtime.warnings) + tourism_chat("Where can I see gorillas in Rwanda?", [])
        self.assertNotIn("super-secret-password", blob)
        self.assertIn("Qwen3 is not connected", blob)
        os.environ.pop("NEO4J_URI", None)
        os.environ.pop("NEO4J_USERNAME", None)
        os.environ.pop("NEO4J_PASSWORD", None)

    def test_source_links_come_from_the_dataset(self):
        result = answer_tourism_question("Where can I see gorillas in Rwanda?", [])
        self.assertEqual(result["engine"], "langgraph")
        self.assertIn("https://www.volcanoesnationalpark.org/gorilla-permits/", result["text"])
        self.assertIn("Volcanoes National Park", result["text"])


if __name__ == "__main__":
    unittest.main()
