import os
import re

from langchain_openai import ChatOpenAI

from graph.router import decide_route
from graph.state import AgentState

from tools.playwright_scraper import scrape_dynamic
from tools.qdrant_rag import rag_retrieve
from tools.tavily_search import tavily_search
from tools.web_scraper import scrape_static


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Automatically choose an available free OpenRouter model.
MODEL = "openrouter/free"


URL_RE = re.compile(
    r"https?://[^\s]+"
)


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str:

    api_key = os.environ.get(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. "
            "Add it to your .env file."
        )

    return api_key


# ============================================================
# URL EXTRACTION
# ============================================================

def extract_url(
    text: str,
) -> str | None:

    match = URL_RE.search(
        text or ""
    )

    if match:
        return match.group(0).rstrip(
            ".,!?;:)"
        )

    return None


# ============================================================
# LLM #1 — REPHRASE
# ============================================================

_rephrase_llm = None


def get_rephrase_llm() -> ChatOpenAI:

    global _rephrase_llm

    if _rephrase_llm is None:

        _rephrase_llm = ChatOpenAI(
            model=MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=get_api_key(),
            temperature=0.2,
        )

    return _rephrase_llm


def rephrase_node(
    state: AgentState,
) -> AgentState:

    llm = get_rephrase_llm()

    original_query = state[
        "original_query"
    ]

    prompt = f"""
You are the query-understanding component
of an agentic AI system.

Rewrite the user's query into a clearer,
more complete query for downstream tools.

Rules:

1. Preserve the user's intent.
2. Make ambiguous wording clearer.
3. Add useful context only when it is safe.
4. Do not invent facts.
5. Preserve URLs exactly.
6. Keep the result concise.
7. Output ONLY the rewritten query.

User query:

{original_query}
"""

    result = llm.invoke(
        prompt
    )

    expanded_query = (
        result.content or ""
    ).strip()

    if not expanded_query:
        expanded_query = original_query

    return {
        "expanded_query": expanded_query
    }


# ============================================================
# ROUTER
# ============================================================

def router_node(
    state: AgentState,
) -> AgentState:

    route = decide_route(
        state["expanded_query"]
    )

    return {
        "route": route
    }


# ============================================================
# PLAYWRIGHT
# ============================================================

def playwright_node(
    state: AgentState,
) -> AgentState:

    url = (
        extract_url(
            state.get(
                "expanded_query",
                ""
            )
        )
        or
        extract_url(
            state.get(
                "original_query",
                ""
            )
        )
    )

    if not url:

        return {
            "tool_output":
                "No URL was found for Playwright."
        }

    return {
        "tool_output":
            scrape_dynamic(url)
    }


# ============================================================
# STATIC SCRAPER
# ============================================================

def scrape_node(
    state: AgentState,
) -> AgentState:

    url = (
        extract_url(
            state.get(
                "expanded_query",
                ""
            )
        )
        or
        extract_url(
            state.get(
                "original_query",
                ""
            )
        )
    )

    if not url:

        return {
            "tool_output":
                "No URL was found for scraping."
        }

    return {
        "tool_output":
            scrape_static(url)
    }


# ============================================================
# TAVILY SEARCH
# ============================================================

def search_node(
    state: AgentState,
) -> AgentState:

    return {
        "tool_output":
            tavily_search(
                state[
                    "expanded_query"
                ]
            )
    }


# ============================================================
# QDRANT RAG
# ============================================================

def rag_node(
    state: AgentState,
) -> AgentState:

    return {
        "tool_output":
            rag_retrieve(
                state[
                    "expanded_query"
                ]
            )
    }


# ============================================================
# LLM #2 — ANSWER
# ============================================================

_answer_llm = None


def get_answer_llm() -> ChatOpenAI:

    global _answer_llm

    if _answer_llm is None:

        _answer_llm = ChatOpenAI(
            model=MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=get_api_key(),
            temperature=0.3,
        )

    return _answer_llm


def answer_node(
    state: AgentState,
) -> AgentState:

    llm = get_answer_llm()

    original_query = state[
        "original_query"
    ]

    tool_output = state.get(
        "tool_output",
        "",
    )

    prompt = f"""
You are the final answering component
of an agentic AI system.

Answer the user's original question
using the retrieved information.

Rules:

1. Answer the question directly.
2. Use the retrieved information.
3. Do not invent unsupported facts.
4. If the retrieved information is insufficient,
   say so clearly.
5. Do not mention LangGraph, routing, Qdrant,
   Tavily, Playwright, or internal nodes unless
   the user asks about them.
6. Be concise and useful.

Original question:

{original_query}

Retrieved information:

{tool_output}
"""

    result = llm.invoke(
        prompt
    )

    final_answer = (
        result.content or ""
    ).strip()

    if not final_answer:

        final_answer = (
            "I couldn't generate an answer "
            "from the available information."
        )

    return {
        "final_answer": final_answer
    }