import os

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VALID_ROUTES = {"playwright", "scrape", "search", "rag"}

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
    """Ask the router model for exactly one route word and validate it.

    Free models on OpenRouter don't all support reliable function calling,
    so this asks for plain text instead of structured output and falls back
    to "search" if the reply doesn't clearly match one of the four routes.
    """
    llm = get_router_llm()
    prompt = (
        "Choose exactly one tool for this query. Reply with ONLY one word, "
        "nothing else: playwright, scrape, search, or rag.\n\n"
        "playwright = the query names a specific URL that is JS-heavy and "
        "needs a rendered browser (SPA, dashboard, infinite scroll).\n"
        "scrape = the query names a specific URL that is likely static HTML.\n"
        "search = a general or current-events question with no specific URL.\n"
        "rag = a question about internal knowledge already stored in Qdrant.\n\n"
        f"Query: {expanded_query}"
    )
    result = llm.invoke(prompt)
    reply = (result.content or "").strip().lower()

    for candidate in VALID_ROUTES:
        if candidate in reply:
            return candidate
    return "search"
