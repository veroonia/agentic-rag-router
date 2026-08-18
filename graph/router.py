import os

from langchain_openai import ChatOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter automatically selects an available free model.
ROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"

VALID_ROUTES = {
    "playwright",
    "scrape",
    "search",
    "rag",
}


_router_llm = None


def get_router_llm() -> ChatOpenAI:
    global _router_llm

    if _router_llm is None:
        _router_llm = ChatOpenAI(
            model="dots-studio/dots-3-note-preview:free",
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0,
        )

    return _router_llm


def decide_route(expanded_query: str) -> list[str]:

    llm = get_router_llm()

    prompt = f"""
You are the routing component of an agentic RAG system.

Choose ALL routes that apply to this query. A single query can require
more than one tool if it asks about more than one thing.

Valid routes: playwright, scrape, search, rag

Reply with a comma-separated list of the routes that apply, e.g.:
scrape,rag

Definitions:

playwright:
The user provided a specific URL and the page likely requires
JavaScript rendering, such as an SPA, dashboard, dynamically
loaded page, or infinite scrolling page.

scrape:
The user provided a specific URL and the page is likely
normal/static HTML.

search:
The user asks a general question, current question, news question,
or internet-related question without a specific URL.

rag:
The user asks about information stored in the internal Qdrant
knowledge base.

Important:
- Specific JavaScript-heavy URL -> playwright
- Specific static URL -> scrape
- Internal knowledge -> rag
- Everything else -> search
- playwright and scrape are mutually exclusive for the SAME url — never
  pick both for one link.
- If the query has multiple distinct parts (e.g. a URL AND a question
  about internal knowledge), include a route for each part.

Examples:

"What faction was Tris born into?" → rag
"What is the weather today?" → search
"Open https://example.com and tell me what's there" → scrape
"Analyze this JavaScript dashboard: https://example.com" → playwright
"Check https://example.com and tell me who Tris is" → scrape,rag
"Summarize this SPA https://example.com and search the latest news about it" → playwright,search

Query:
{expanded_query}

Reply with ONLY the comma-separated route list.
"""

    result = llm.invoke(prompt)

    reply = (
        result.content or ""
    ).strip().lower()

    routes = [
        r.strip()
        for r in reply.split(",")
        if r.strip() in VALID_ROUTES
    ]

    return routes or ["search"]