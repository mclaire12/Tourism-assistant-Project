"""LangGraph tourism agent.

The routing rules, retrieval nodes, and Rwanda-only prompt follow the Colab
workflow. Two documented adaptations keep that workflow usable outside Colab:

- Graph search terms are short phrases that occur in the graph. The original
  Cypher query uses CONTAINS, so a full sentence never used to match.
- When Qwen3 is not configured, the generation node quotes retrieved records
  instead of calling another language model.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from src.llm.model_provider import ModelError
from src.retrieval.graph_retrieval import neo4j_graph_retrieval
from src.retrieval.vector_retrieval import hybrid_retrieval
from src.utils.formatting import destination_markdown, lead_for_answer, public_destination
from src.utils.keywords import extract_graph_keywords
from src.utils.text import content_tokens
from src.utils.validators import prepare_retrieval_question, validate_geographical_scope

logger = logging.getLogger("rwanda_tourism.agent")

_RUNTIME = None


class TourismAgentState(TypedDict):
    question: str
    retrieval_question: str
    graph_keyword: str
    in_scope: bool
    detected_country: str
    vector_results: Annotated[list[dict[str, Any]], operator.add]
    graph_results: Annotated[list[dict[str, Any]], operator.add]
    final_answer: str
    route: str
    history: list
    citations: list


def bind_runtime(runtime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def route_retrieval(state: TourismAgentState) -> str:
    question = state["retrieval_question"].lower()

    graph_keywords = [
        "district",
        "region",
        "located",
        "connected",
        "relationships",
        "province",
        "where is",
        "activities",
        "activity",
        "lake",
        "near",
    ]

    vector_keywords = [
        "recommend",
        "suggest",
        "similar",
        "describe",
        "experience",
        "explore",
        "visit",
        "enjoy",
        "water",
        "nature",
        "want",
        "gorilla",
        "cultural",
    ]

    has_graph_keyword = any(keyword in question for keyword in graph_keywords)
    has_vector_keyword = any(keyword in question for keyword in vector_keywords)

    if has_graph_keyword and has_vector_keyword:
        return "both"
    if has_graph_keyword:
        return "graph"
    if has_vector_keyword:
        return "vector"
    return "both"


def geographical_validation_node(state: TourismAgentState) -> dict:
    runtime = _runtime()
    result = validate_geographical_scope(
        state["question"],
        [place["name"] for place in runtime.places],
    )
    retrieval_question = prepare_retrieval_question(
        state["question"],
        result["detected_country"],
        result.get("matched_text"),
    )
    if result["in_scope"] and not _has_specific_keyword(retrieval_question, runtime.places):
        retrieval_question = _expand_follow_up(retrieval_question, state.get("history") or [])

    keywords = extract_graph_keywords(retrieval_question, runtime.places)
    graph_keyword = keywords[0] if keywords else retrieval_question
    route = "both" if not result["in_scope"] else route_retrieval({"retrieval_question": retrieval_question})
    logger.info(
        "scope=%s country=%s route=%s graph_keyword=%s",
        result["in_scope"],
        result["detected_country"],
        route,
        graph_keyword,
    )
    return {
        "in_scope": result["in_scope"],
        "detected_country": result["detected_country"] or "",
        "retrieval_question": retrieval_question,
        "graph_keyword": graph_keyword,
        "route": route,
    }


def chroma_retrieval_node(state: TourismAgentState) -> dict:
    results = hybrid_retrieval(query=state["retrieval_question"], top_k=3)
    vector_results = []
    documents = results["documents"][0] if results.get("documents") else []
    for index, document in enumerate(documents):
        vector_results.append(
            {
                "id": results["ids"][0][index],
                "document": document,
                "metadata": results["metadatas"][0][index],
            }
        )
    return {"vector_results": vector_results}


def neo4j_retrieval_node(state: TourismAgentState) -> dict:
    runtime = _runtime()
    keywords = extract_graph_keywords(state["retrieval_question"], runtime.places) or [state["graph_keyword"]]
    merged = []
    seen = set()
    for keyword in keywords:
        if keyword.strip().lower() == state["retrieval_question"].strip().lower() and len(keyword.split()) > 4:
            continue
        for row in neo4j_graph_retrieval(keyword, top_k=5):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            merged.append(row)
    return {"graph_results": merged[:5]}


def qwen_generation_node(state: TourismAgentState) -> dict:
    runtime = _runtime()
    citations = _collect_citations(state, runtime)
    if not state.get("in_scope") and not _has_specific_keyword(state.get("retrieval_question", ""), runtime.places):
        citations = []

    context = _context_from_state(state, citations)
    prompt = create_rag_prompt(state["question"], context, state.get("history") or [])
    model_text = None
    if state.get("in_scope") and runtime.model is not None:
        try:
            model_text = runtime.model.generate(prompt)
            model_notice = None
        except ModelError as exc:
            logger.error("Model provider failed: %s", exc)
            model_notice = str(exc)
        except Exception:
            logger.exception("Unexpected model failure")
            model_notice = "The language model could not complete the request. Showing retrieved records instead."
    elif state.get("in_scope"):
        model_notice = "Qwen3 is not connected yet, so this reply quotes the retrieved records."
    else:
        model_notice = None

    lead = lead_for_answer(
        question=state["question"],
        in_scope=bool(state.get("in_scope")),
        detected_country=state.get("detected_country") or None,
        destinations=citations,
        model_text=model_text,
        model_notice=model_notice,
    )
    return {"final_answer": lead, "citations": citations}


def create_rag_prompt(question, context, history=None) -> str:
    history_block = ""
    if history:
        lines = []
        for turn in history[-6:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = " ".join((turn.get("content") or "").split())
            if content:
                lines.append(f"{role}: {content[:500]}")
        if lines:
            history_block = "PREVIOUS CONVERSATION:\n" + "\n".join(lines) + "\n\n"

    prompt = f"""
You are a Rwanda Tourism Assistant.

GEOGRAPHICAL SCOPE:
You provide tourism information ONLY about Rwanda.

STRICT INSTRUCTIONS:

1. Only recommend tourist destinations located in Rwanda.
2. Never provide tourism recommendations for other countries.
3. If the user asks about Tanzania, Burundi, Kenya, Uganda,
   or any other country outside Rwanda:
   - Politely explain that this assistant currently covers Rwanda only.
   - Do not provide information about the requested foreign destination.
   - Suggest relevant destinations in Rwanda based on the user's
     original interests and activities.
4. Use ONLY the provided context when recommending destinations.
5. Never invent destinations, activities, prices, or other facts.
6. If the context does not contain relevant Rwanda information,
   explain that the available dataset does not provide a suitable answer.
7. Keep responses clear, helpful, and friendly.

{history_block}RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
    return prompt.strip()


def compile_agent():
    from langgraph.graph import END, START, StateGraph
    try:
        from langgraph.types import Send
    except ImportError:
        from langgraph.constants import Send

    def geographical_dispatcher(state: TourismAgentState):
        if not state["in_scope"]:
            return [Send("chroma_retrieval", state), Send("neo4j_retrieval", state)]
        decision = route_retrieval(state)
        if decision == "vector":
            return [Send("chroma_retrieval", state)]
        if decision == "graph":
            return [Send("neo4j_retrieval", state)]
        return [Send("chroma_retrieval", state), Send("neo4j_retrieval", state)]

    workflow = StateGraph(TourismAgentState)
    workflow.add_node("geographical_validation", geographical_validation_node)
    workflow.add_node("chroma_retrieval", chroma_retrieval_node)
    workflow.add_node("neo4j_retrieval", neo4j_retrieval_node)
    try:
        workflow.add_node("qwen_generation", qwen_generation_node, defer=True)
    except TypeError:
        workflow.add_node("qwen_generation", qwen_generation_node)
    workflow.add_edge(START, "geographical_validation")
    workflow.add_conditional_edges(
        "geographical_validation",
        geographical_dispatcher,
        ["chroma_retrieval", "neo4j_retrieval"],
    )
    workflow.add_edge("chroma_retrieval", "qwen_generation")
    workflow.add_edge("neo4j_retrieval", "qwen_generation")
    workflow.add_edge("qwen_generation", END)
    return workflow.compile()


def run_sequential(state: dict) -> dict:
    """Same nodes as the graph, used if LangGraph cannot join the branches."""
    updates = geographical_validation_node(state)
    current = {**state, **updates}
    if current["route"] in {"vector", "both"}:
        current["vector_results"] = chroma_retrieval_node(current)["vector_results"]
    if current["route"] in {"graph", "both"}:
        current["graph_results"] = neo4j_retrieval_node(current)["graph_results"]
    current.update(qwen_generation_node(current))
    current["engine"] = "sequential"
    return current


def answer_tourism_question(message, history=None) -> dict:
    question = (message or "").strip()
    prior = _clean_history(history)
    if len(question) < 3:
        lead = "Please enter a question about visiting Rwanda."
        return _package(lead, [], True, None, "", False)

    state = {
        "question": question,
        "retrieval_question": question,
        "graph_keyword": question,
        "in_scope": True,
        "detected_country": "",
        "vector_results": [],
        "graph_results": [],
        "final_answer": "",
        "route": "",
        "history": prior,
        "citations": [],
    }
    runtime = _runtime()
    try:
        if runtime.agent is None:
            raise RuntimeError("LangGraph agent is not compiled")
        result = runtime.agent.invoke(state)
        result["engine"] = "langgraph"
        if not result.get("final_answer"):
            raise RuntimeError("LangGraph returned an empty answer")
    except Exception:
        logger.exception("LangGraph run failed; using the same retrieval nodes sequentially")
        result = run_sequential(state)

    citations = result.get("citations") or []
    lead = result.get("final_answer") or "I couldn't prepare an answer from the Rwanda dataset."
    text = _full_text(lead, citations)
    return _package(
        lead,
        citations,
        bool(result.get("in_scope", True)),
        result.get("detected_country") or None,
        result.get("route") or "",
        bool(runtime.model) and bool(result.get("in_scope")),
        text=text,
        engine=result.get("engine", ""),
    )


def tourism_chat(message, history):
    """Original chat entry point. History is kept and passed into the agent."""
    return answer_tourism_question(message, history)["text"]


def _package(lead, destinations, in_scope, detected_country, route, model_used, text=None, engine="") -> dict:
    return {
        "lead": lead,
        "text": text or _full_text(lead, destinations),
        "destinations": destinations,
        "in_scope": in_scope,
        "detected_country": detected_country,
        "route": route,
        "model_used": model_used,
        "engine": engine,
    }


def _full_text(lead: str, destinations: list[dict]) -> str:
    parts = [lead.strip()]
    parts.extend(destination_markdown(destination) for destination in destinations)
    return "\n\n".join(part for part in parts if part)


def _collect_citations(state: TourismAgentState, runtime) -> list[dict]:
    combined: dict[str, dict] = {}
    order: list[str] = []
    for item in state.get("vector_results") or []:
        destination_id = item.get("id")
        if not destination_id:
            continue
        order.append(destination_id)
        combined[destination_id] = {"sources": ["ChromaDB"], "graph": None}
    for item in state.get("graph_results") or []:
        destination_id = item.get("id")
        if not destination_id:
            continue
        if destination_id not in combined:
            order.append(destination_id)
            combined[destination_id] = {"sources": [], "graph": item}
        graph_label = "Neo4j" if runtime.neo4j is not None else "Knowledge graph"
        if graph_label not in combined[destination_id]["sources"]:
            combined[destination_id]["sources"].append(graph_label)
        combined[destination_id]["graph"] = item

    citations = []
    for destination_id in order:
        place = runtime.by_id.get(destination_id)
        if not place:
            continue
        info = combined[destination_id]
        citations.append(public_destination(place, info["graph"], info["sources"]))
    return citations[:5]


def _context_from_state(state: TourismAgentState, citations: list[dict]) -> str:
    if not citations:
        return "No matching Rwanda destinations were retrieved."
    parts = []
    by_vector = {item.get("id"): item for item in state.get("vector_results") or []}
    by_graph = {item.get("id"): item for item in state.get("graph_results") or []}
    for destination in citations:
        chunks = [f"Destination: {destination['name']}"]
        vector_item = by_vector.get(destination["id"])
        if vector_item:
            chunks.append(f"Document: {vector_item['document']}")
        graph_item = by_graph.get(destination["id"])
        if graph_item:
            chunks.append(
                "Graph information:\n"
                f"Destination: {graph_item.get('destination')}\n"
                f"Type: {graph_item.get('type')}\n"
                f"Regions: {graph_item.get('regions')}\n"
                f"Districts: {graph_item.get('districts')}\n"
                f"Activities: {graph_item.get('activities')}\n"
                f"Interests: {graph_item.get('interests')}\n"
                f"Moods: {graph_item.get('moods')}"
            )
        parts.append("\n".join(chunks))
    return "\n\n".join(parts)[:6000]


def _has_specific_keyword(question: str, places: list[dict]) -> bool:
    keywords = extract_graph_keywords(question, places)
    if not keywords:
        return False
    if len(keywords) == 1 and keywords[0].strip().lower() == (question or "").strip().lower():
        return False
    return True


def _expand_follow_up(question: str, history: list[dict]) -> str:
    if len(content_tokens(question)) >= 4:
        return question
    for turn in reversed(history):
        if turn.get("role") == "user" and (turn.get("content") or "").strip():
            return f"{turn['content'].strip()} {question}".strip()
    return question


def _clean_history(history) -> list[dict]:
    cleaned = []
    if not history:
        return cleaned
    for turn in history:
        if isinstance(turn, dict) and turn.get("role") in {"user", "assistant"}:
            content = turn.get("content") or ""
            if isinstance(content, str) and content.strip():
                cleaned.append({"role": turn["role"], "content": content.strip()})
    return cleaned[-6:]


def _runtime():
    if _RUNTIME is None:
        raise RuntimeError("The tourism assistant is not ready.")
    return _RUNTIME


__all__ = [
    "answer_tourism_question",
    "compile_agent",
    "create_rag_prompt",
    "route_retrieval",
    "tourism_chat",
]
