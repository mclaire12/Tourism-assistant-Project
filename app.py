"""Rwanda Tourism Assistant."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import streamlit as st

from config.settings import HERO_PATH, LOGO_PATH, ROOT
from src.agents.tourism_agent import answer_tourism_question
from src.runtime import build_runtime
from src.utils.formatting import OFFICIAL_NOTE

LOGGER = logging.getLogger("rwanda_tourism.app")
WEB_HERO = ROOT / "assets" / "rwanda_hero_web.jpg"

SUGGESTIONS = [
    "Where can I see gorillas in Rwanda?",
    "Recommend lakes for water activities.",
    "Which district is Volcanoes National Park located in?",
    "Where can I experience cultural tourism?",
]

ROUTE_LABELS = {
    "vector": "Semantic search",
    "graph": "Knowledge graph",
    "both": "Semantic search and knowledge graph",
}


st.set_page_config(
    page_title="Rwanda Tourism Assistant",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🇷🇼",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_runtime():
    return build_runtime()


@st.cache_data
def _image_data_uri(path: str) -> str:
    payload = Path(path).read_bytes()
    kind = "jpeg" if path.endswith((".jpg", ".jpeg")) else "png"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/{kind};base64,{encoded}"


def _hero_path() -> Path:
    if WEB_HERO.exists():
        return WEB_HERO
    return HERO_PATH


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rw-green: #0b6e4f;
            --rw-green-deep: #064833;
            --rw-blue: #0a3d62;
            --rw-gold: #e3b505;
            --rw-ink: #143028;
            --rw-muted: #5d6f66;
            --rw-paper: #f3f7f4;
            --rw-line: #d5e4da;
        }
        .stApp { background: var(--rw-paper); color: var(--rw-ink); }
        .block-container { max-width: 1080px; padding-top: 1.2rem; padding-bottom: 3rem; }
        header[data-testid="stHeader"] { background: transparent; }
        .hero {
            position: relative;
            min-height: 280px;
            border-radius: 22px;
            overflow: hidden;
            margin-bottom: 1rem;
            background: var(--rw-green-deep);
            box-shadow: 0 16px 40px rgba(6, 72, 51, 0.16);
        }
        .hero img {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center 55%;
        }
        .hero-shade {
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, rgba(4, 48, 32, 0.82), rgba(10, 61, 98, 0.42) 58%, rgba(6, 40, 28, 0.55));
        }
        .hero-copy {
            position: relative;
            z-index: 2;
            padding: 2.2rem 1.5rem 1.6rem;
            color: white;
            max-width: 760px;
        }
        .eyebrow {
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--rw-gold);
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0 0 0.4rem;
        }
        .hero-copy h1 {
            color: white !important;
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.05;
            margin: 0 0 0.35rem;
            font-weight: 800;
        }
        .subtitle { color: white; font-size: 1.15rem; font-weight: 650; margin: 0 0 0.55rem; }
        .lede { color: rgba(255,255,255,0.94); font-size: 1rem; line-height: 1.55; margin: 0; max-width: 640px; }
        .section-label { color: var(--rw-green); font-weight: 750; margin: 0.2rem 0 0.35rem; }
        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 74px;
            border-radius: 14px;
            border: 1px solid var(--rw-line);
            background: white;
            color: var(--rw-green-deep);
            text-align: left;
            padding: 0.75rem 0.9rem;
            line-height: 1.35;
            white-space: normal;
            font-weight: 650;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: var(--rw-green);
            color: var(--rw-green);
            background: #f4fbf7;
        }
        div[data-testid="stChatMessage"] {
            background: white;
            border: 1px solid var(--rw-line);
            border-radius: 16px;
            padding: 0.2rem 0.4rem;
        }
        .footer-note { color: var(--rw-muted); font-size: 0.86rem; line-height: 1.5; margin-top: 0.8rem; }
        @media (max-width: 800px) {
            .hero { min-height: 230px; border-radius: 16px; }
            .hero-copy { padding: 1.4rem 1rem 1.2rem; }
            .block-container { padding-top: 4.2rem; padding-left: 0.8rem; padding-right: 0.8rem; }
            div[data-testid="stHorizontalBlock"] { flex-direction: column; }
            div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        }
        @media (prefers-reduced-motion: reduce) {
            .hero img { transform: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(image_uri: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <img src="{image_uri}" alt="Green hills, a lake, and volcanoes in Rwanda">
            <div class="hero-shade"></div>
            <div class="hero-copy">
                <p class="eyebrow">Land of a Thousand Hills</p>
                <h1>Rwanda Tourism Assistant</h1>
                <p class="subtitle">Discover Rwanda. Explore More.</p>
                <p class="lede">
                    Ask about destinations, wildlife, lakes, culture, and how to get there.
                    Answers stay inside Rwanda and come from the project’s destination records.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(runtime) -> None:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=72)
        st.markdown("### Rwanda Tourism Assistant")
        st.markdown(
            "A guide to 19 destinations. It checks whether a question is about Rwanda, "
            "then searches a vector index and a knowledge graph before writing the answer."
        )
        st.markdown("**Search index**")
        st.write(f"{runtime.vector_index.backend} · {runtime.embedder.name}")
        st.markdown("**Knowledge graph**")
        st.write(runtime.graph_backend)
        st.markdown("**Language model**")
        if runtime.model is None:
            st.write(f"Not connected ({runtime.settings.model_name} is the configured model name)")
        else:
            st.write(runtime.model.label)
        if runtime.warnings:
            st.markdown("**Notices**")
            for warning in runtime.warnings:
                st.caption(warning)
        st.caption(OFFICIAL_NOTE)


def render_destination(destination: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{destination['name']}**")
        location = " · ".join(part for part in [destination.get("district"), destination.get("region")] if part)
        if location:
            st.caption(location)
        if destination.get("description"):
            st.write(destination["description"])
        facts = []
        if destination.get("activities"):
            facts.append(f"**Activities:** {', '.join(destination['activities'])}")
        budget = destination.get("budget_level") or "not recorded"
        duration = destination.get("typical_duration") or "not recorded"
        facts.append(f"**Budget:** {budget} · **Typical duration:** {duration}")
        if destination.get("best_season"):
            facts.append(f"**Best season:** {destination['best_season']}")
        if destination.get("access_from_kigali"):
            facts.append(f"**Access from Kigali:** {destination['access_from_kigali']}")
        if destination.get("tips"):
            facts.append("**Travel tips:** " + "; ".join(destination["tips"]))
        st.markdown("\n\n".join(facts))
        if destination.get("costs_note"):
            st.markdown(f"**Costs:** {destination['costs_note']}")
            st.caption(OFFICIAL_NOTE)
        if destination.get("safety_note"):
            st.markdown(f"**Safety:** {destination['safety_note']}")
        if destination.get("source_urls"):
            st.markdown("**Sources**")
            for url in destination["source_urls"]:
                st.markdown(f"- [{url}]({url})")
        else:
            st.caption("No source URL was recorded for this destination.")
        verified = destination.get("last_verified")
        confidence = destination.get("source_confidence")
        if verified:
            st.caption(f"Record last verified {verified} ({confidence} confidence).")


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            route = ROUTE_LABELS.get(message.get("route", ""))
            if route:
                st.caption(f"Retrieved with {route.lower()}.")
            for destination in message.get("destinations") or []:
                render_destination(destination)


def main() -> None:
    inject_styles()
    try:
        runtime = get_runtime()
    except Exception:
        LOGGER.exception("Startup failed")
        st.error(
            "The assistant could not start. The tourism data or search index is unavailable. "
            "Please try again in a moment."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None

    render_sidebar(runtime)
    hero = _hero_path()
    if hero.exists():
        render_hero(_image_data_uri(str(hero)))
    else:
        st.title("Rwanda Tourism Assistant")
        st.subheader("Discover Rwanda. Explore More.")

    if runtime.model is None:
        st.info(
            "Qwen3 is not connected yet. Answers quote the Rwanda destination records. "
            "Setup details are in the sidebar."
        )

    st.markdown('<p class="section-label">Suggested questions</p>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for index, question in enumerate(SUGGESTIONS):
        column = left if index % 2 == 0 else right
        if column.button(question, key=f"suggestion-{index}", width="stretch"):
            st.session_state.pending = question
            st.rerun()

    header_left, header_right = st.columns([4, 1])
    header_left.markdown('<p class="section-label">Conversation</p>', unsafe_allow_html=True)
    if header_right.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    render_messages()
    if not st.session_state.messages and not st.session_state.pending:
        st.caption("Your Rwanda travel conversation starts here.")

    if st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching Rwanda destinations..."):
                try:
                    prior = [
                        {"role": item["role"], "content": item["content"]}
                        for item in st.session_state.messages
                        if item.get("role") in {"user", "assistant"}
                    ]
                    result = answer_tourism_question(question, prior)
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["lead"],
                            "destinations": result["destinations"],
                            "route": result["route"],
                        }
                    )
                except Exception:
                    LOGGER.exception("Question failed")
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Something went wrong while preparing that answer. Please try again.",
                            "destinations": [],
                            "route": "",
                        }
                    )
        st.rerun()

    prompt = st.chat_input("Ask about tourism in Rwanda")
    if prompt:
        st.session_state.pending = prompt
        st.rerun()

    st.markdown(f'<p class="footer-note">🇷🇼 Rwanda Tourism Assistant · {OFFICIAL_NOTE}</p>', unsafe_allow_html=True)


main()
