# Divergent Agent

A Streamlit application that answers questions about _Divergent_ and handles
web research. It uses LangGraph to turn each message into a small, single-pass
workflow: clarify the question, select the appropriate source, retrieve
context, and generate a grounded answer.

## How it works

```mermaid
flowchart TD
    start([User question]) --> rephrase[Rephrase query]
    rephrase --> router{Choose route}

    router -->|rag| rag[Qdrant RAG]
    router -->|search| search[Tavily web search]
    router -->|scrape| scrape[Static URL scraper]
    router -->|playwright| playwright[Dynamic URL scraper]

    rag --> answer[Generate answer]
    search --> answer
    scrape --> answer
    playwright --> answer

    answer --> end([End])
```

Only one route runs for a request. After the selected tool produces context,
the graph proceeds directly to the answer node and ends—there are no tool
loops or retries in the graph.

### Routing behavior

| Request type                                           | Route        | Source                    |
| ------------------------------------------------------ | ------------ | ------------------------- |
| Questions about the indexed _Divergent_ material       | `rag`        | Qdrant                    |
| General, current, or web-based questions without a URL | `search`     | Tavily                    |
| A normal/static HTML URL                               | `scrape`     | Requests + Beautiful Soup |
| A JavaScript-heavy, dynamically rendered URL           | `playwright` | Playwright + Chromium     |

The query rephraser preserves URLs so the scraper nodes can extract and fetch
the supplied URL reliably.

## Models

All language-model calls go through [OpenRouter](https://openrouter.ai/).

- Routing uses `dots-studio/dots-3-note-preview:free` with deterministic
  output.
- Answer generation defaults to
  `nvidia/nemotron-3.5-lightning:free`.

The interface shows a model picker for Nemotron 3.5 Lightning and Dots 3 Note
Preview. At present, the selected value is retained by the UI but is not yet
passed to the LangGraph invocation, so answers use the default Nemotron model.

Free-model availability and rate limits are controlled by OpenRouter and may
change. If a model is retired, update `ANSWER_MODEL_MAP` in `graph/nodes.py`
or the router model in `graph/router.py`.

## Prerequisites

- Python 3.10 or later
- An [OpenRouter API key](https://openrouter.ai/keys)
- A [Tavily API key](https://app.tavily.com/)
- A running Qdrant instance containing the _Divergent_ collection
- Docker, if you run Qdrant locally

Playwright is required only for the dynamic-page route, but installing it is
recommended so every route is available.

## Setup

1. Create and activate a virtual environment.

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   On macOS/Linux, use `source venv/bin/activate`.

2. Install Python dependencies and the Chromium browser used by Playwright.

   ```powershell
   pip install -r requirements.txt
   playwright install chromium
   ```

3. Create a `.env` file in the project root.

   ```env
   OPENROUTER_API_KEY=your_openrouter_key
   TAVILY_API_KEY=your_tavily_key
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=divergent_children
   # QDRANT_API_KEY=required_only_for_a_secured_qdrant_instance
   ```

4. Start Qdrant if it is hosted locally. For an existing Docker container,
   this may be as simple as:

   ```powershell
   docker start <your-qdrant-container>
   ```

5. Start the application.

   ```powershell
   streamlit run app.py
   ```

## Qdrant and embeddings

The RAG route connects to `QDRANT_URL` and searches the collection named by
`QDRANT_COLLECTION`; its default is `divergent_children`.

Queries are embedded locally with
`sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings`.
Your collection must have been created with the same embedding model and
vector dimensions. The retrieval step requests five similar child chunks,
removes duplicate parent chunks, and returns the associated parent text plus
chapter/page metadata to the answer model.

On first use, Sentence Transformers may download the embedding model. This is
separate from OpenRouter and Tavily credentials.

## Project structure

```text
agentic-rag-router/
├── app.py                     # Streamlit UI, chat history, and debug trace
├── graph/
│   ├── graph.py                # LangGraph nodes and conditional routing
│   ├── nodes.py                # Rephrase, route, tool, and answer nodes
│   ├── router.py               # LLM route-decision prompt
│   └── state.py                # Shared graph state schema
├── tools/
│   ├── qdrant_rag.py           # Local embedding + Qdrant retrieval
│   ├── tavily_search.py        # Tavily search integration
│   ├── web_scraper.py          # Static-page scraper
│   └── playwright_scraper.py   # Dynamic-page scraper
├── ui/
│   ├── components.py           # Streamlit HTML helpers
│   └── styles.css              # Application styling
├── requirements.txt
└── .env                        # Local credentials; excluded from Git
```

## Debugging

After each request, expand **Debug: pipeline trace** in the UI to inspect the
original query, rephrased query, selected route, and a preview of the retrieved
tool output. API-key and tool errors are returned in the chat response, which
makes missing configuration visible during local development.
