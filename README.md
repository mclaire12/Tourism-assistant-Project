# Rwanda Tourism Assistant

A Streamlit application for questions about visiting Rwanda. It keeps the Colab project's agentic retrieval flow: a Rwanda-only check, then ChromaDB, a knowledge graph, or both, then an answer grounded in the original 19-destination dataset.

The original Gradio notebook is preserved in `legacy/Rwanda_Tourism.ipynb`. Notebook outputs were removed before it was added here because a Colab run had printed the Neo4j host and username. The copy in Downloads is unchanged. If that notebook has been shared, rotate the AuraDB password.

## What changed

| Piece | Notebook | This app |
| --- | --- | --- |
| Interface | Gradio in Google Colab | Streamlit |
| Language model | Qwen3-4B-Instruct-2507, 4-bit, on a T4 GPU | Same model name, through a separate provider. No model is loaded until an endpoint or a local GPU is configured. |
| Embeddings | BAAI/bge-m3 (1024 dimensions) | Same model when `sentence-transformers` is installed and `EMBEDDING_PROVIDER=bge-m3`. Otherwise a 1024-dimension token-overlap index in ChromaDB, labeled as such. |
| Vector store | ChromaDB collection `rwanda_tourism` | Same collection name and cosine space. Rebuilt from `data/rwanda_places.json` when the saved index does not match. |
| Graph | Neo4j AuraDB | Same Cypher query. If AuraDB is not configured, the same relationships are read from the dataset. |
| Orchestration | LangGraph `tourism_agent` | Same nodes and routing rules. |
| Dataset | `rwanda_places.json`, 19 places | Unchanged records and fields. |

Graph search terms are shortened to words that actually appear in the graph. The original Cypher uses `CONTAINS`, so sending the whole sentence matched nothing.

Out-of-scope countries are refused in code. A question that only names a foreign country does not get foreign facts added by a model. A question that names a Rwanda place, including the Congo Nile Trail, stays in scope.

## Free-hosting limit

Streamlit Community Cloud cannot run Qwen3-4B with 4-bit GPU quantization. This project does not replace Qwen3 with another model.

Until `MODEL_PROVIDER` is set, the app still answers. It quotes the retrieved destination records and says that Qwen3 is not connected. That is retrieval, not a second language model.

There is no permanent free API for `Qwen/Qwen3-4B-Instruct-2507` that this repository can promise. A Hugging Face router, Groq, OpenRouter, or another OpenAI-compatible server can be attached later by changing environment variables only. Local weights load only when `MODEL_PROVIDER=local` and the machine has the extra packages in `requirements-local.txt`.

`BAAI/bge-m3` is also too large for the free Streamlit memory limit. The default index is the hashed 1024-dimension embedder. Set `EMBEDDING_PROVIDER=bge-m3` on a computer with enough memory after installing `requirements-local.txt`.

## Project layout

```
app.py                      Streamlit interface
config/settings.py          Environment and Streamlit secrets
data/rwanda_places.json     Original dataset
assets/                     Logo, original hero photo, compressed hero
legacy/Rwanda_Tourism.ipynb
src/agents/tourism_agent.py LangGraph workflow and tourism_chat
src/retrieval/              hybrid_retrieval, graph retrieval, combined retrieval
src/database/               ChromaDB, Neo4j, dataset graph
src/llm/model_provider.py   Hosted or local Qwen3
src/utils/                  Validation, documents, formatting
tests/test_core.py
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

On macOS or Linux, use `cp .env.example .env`.

Open http://localhost:8501.

```bash
python -m unittest tests.test_core -v
```

The tests cover dataset shape, ChromaDB, routing, gorilla and district questions, cultural and lake questions, Nyungwe activities, nature destinations, a Tanzania refusal, follow-up history, source links, and missing credentials. They do not call a paid or hosted model.

## Configuration

Use `.env` locally or Streamlit secrets in the cloud. Never commit `.env` or `.streamlit/secrets.toml`.

```toml
NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-aura-password"
MODEL_PROVIDER = "huggingface"
MODEL_API_URL = "https://router.huggingface.co/v1"
MODEL_API_KEY = "hf_your_token"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
EMBEDDING_PROVIDER = "hashed"
EMBEDDING_MODEL = "BAAI/bge-m3"
```

`MODEL_PROVIDER` may be empty, `huggingface`, `openai_compatible`, or `local`.

If the AuraDB graph has no `Destination` nodes, the app runs the original `MERGE` queries once and fills it from the dataset. You can also run:

```bash
python scripts/seed_neo4j.py
```

## Deploy on Streamlit Community Cloud

The public app was not deployed from this session because that step needs your GitHub and Streamlit logins.

1. Push this repository to GitHub. The remote already named in git is `origin`.
2. Sign in at https://share.streamlit.io/.
3. Choose **Create app**.
4. Select this repository, branch `main`, and main file `app.py`.
5. Open **Advanced settings** and paste the secrets above into Secrets. Use your real AuraDB password and, if you have one, a model token.
6. Deploy. The first start rebuilds the ChromaDB collection from the dataset. It does not download Qwen3.
7. Ask “Where can I see gorillas in Rwanda?” and “What is the capital of Tanzania?”. The first should cite Volcanoes National Park. The second should say the assistant covers Rwanda only.

The app address will look like `https://your-app-name.streamlit.app/`.

## Behaviour to expect

Suggested questions fill the chat. History stays in the browser session and is sent with the next question. Clear conversation removes that session only.

A Tanzania question is refused. It is not answered with facts that are not in the Rwanda dataset.

Source links are the `source_urls` stored on each destination. The app does not invent links, prices, or places.
