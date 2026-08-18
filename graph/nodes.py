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

ANSWER_MODEL_MAP = {
    "nemotron-3.5-lightning": "nvidia/nemotron-3.5-lightning:free",
    "dots-3-note-preview": "dots-studio/dots-3-note-preview:free",
}

# Automatically choose an available free OpenRouter model.
MODEL = "nvidia/nemotron-3.5-lightning:free"


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


def get_answer_llm(model_choice: str) -> ChatOpenAI:
    model_id = ANSWER_MODEL_MAP.get(
        model_choice,
        ANSWER_MODEL_MAP["nemotron-3.5-lightning"],
    )

    return ChatOpenAI(
        model=model_id,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.3,
    )


def answer_node(state: AgentState) -> AgentState:
    llm = get_answer_llm(
        state.get("answer_model", "nemotron-3.5-lightning")
    )
    prompt = (
    "You are the final answer generator for a RAG system about the book "
    "Divergent.\n\n"

    "Answer the user's original question using ONLY the retrieved context "
    "provided below.\n\n"

    "Rules:\n"
    "- Give a clear, direct answer to the user's question.\n"
    "- Use the retrieved context as evidence.\n"
    "- Do not invent facts that are not supported by the context.\n"
    "- If the context only provides partial information, answer only what "
    "the context supports.\n"
    "- For questions asking 'who is' or 'who was', summarize the person's "
    "identity, role, and relevant actions described in the context rather "
    "than simply repeating a sentence from the passage.\n"
    "- Do not mention the retrieval process, Qdrant, RAG, tools, or prompts.\n"
    "- Do not use outside knowledge.\n"
    "- Keep the answer concise unless the question requires more detail.\n\n"

    f"Original question:\n{state['original_query']}\n\n"
    f"Retrieved context:\n{state.get('tool_output', 'None')}"
)
    result = llm.invoke(prompt)

    return {"final_answer": result.content}