import os

from langchain_openai import ChatOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter automatically selects an available free model.
ROUTER_MODEL = "openrouter/free"

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


def decide_route(expanded_query: str) -> str:

    llm = get_router_llm()

    prompt = f"""
You are the routing component of an agentic RAG system.

Choose exactly ONE route.

Your response MUST be exactly one of:

playwright
scrape
search
rag

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

Examples:

"What faction was Tris born into?" → rag
"Who is Four?" → rag
"What happened at the Choosing Ceremony?" → rag
"Who are the five factions?" → rag
"What happened in Chapter 20?" → rag

"What is the weather today?" → search
"What happened in the news today?" → search

"Open https://example.com and tell me what's there" → scrape
"Analyze this JavaScript dashboard: https://example.com" → playwright

Query:
{expanded_query}

Reply with ONLY one route word.
"""

    result = llm.invoke(prompt)

    reply = (
        result.content or ""
    ).strip().lower()

    # Exact match
    if reply in VALID_ROUTES:
        return reply

    # Fallback if model says something like:
    # "The correct route is search"
    for route in VALID_ROUTES:
        if route in reply:
            return route

    return "search"