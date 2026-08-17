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

# Maps the UI's short model choice to the actual OpenRouter free model ID.
# Free model IDs rotate over time — verify at openrouter.ai/models?max_price=0
# and update here if either of these gets pulled.
ANSWER_MODEL_MAP = {
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
}

URL_RE = re.compile(r"https?://\S+")


def extract_url(text: str) -> str | None:
    match = URL_RE.search(text or "")
    return match.group(0) if match else None


# ---- Node 1: query rephraser (LLM #1) -------------------------------------

_rephrase_llm = None


def get_rephrase_llm() -> ChatOpenAI:
    global _rephrase_llm

    if _rephrase_llm is None:
        _rephrase_llm = ChatOpenAI(
            model="dots-studio/dots-3-note-preview:free",
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0.2,
        )

    return _rephrase_llm


def rephrase_node(state: AgentState) -> AgentState:
    llm = get_rephrase_llm()
    prompt = (
        "Rewrite the user's query into a fuller, unambiguous, self-contained "
        "instruction that a downstream tool-selection system can act on. "
        "Preserve any URLs exactly as written. Keep it to 2-3 sentences.\n\n"
        f"User query: {state['original_query']}"
    )
    result = llm.invoke(prompt)
    return {"expanded_query": result.content}


# ---- Node 2: router (decision only, no LLM call needed downstream) --------


def router_node(state: AgentState) -> AgentState:
    decision = decide_route(state["expanded_query"])
    return {"route": decision.route}


# ---- Tool nodes -------------------------------------------------------------


def playwright_node(state: AgentState) -> AgentState:
    url = extract_url(state.get("expanded_query")) or extract_url(state.get("original_query"))
    if not url:
        return {"tool_output": "No URL found for Playwright to render."}
    return {"tool_output": scrape_dynamic(url)}


def scrape_node(state: AgentState) -> AgentState:
    url = extract_url(state.get("expanded_query")) or extract_url(state.get("original_query"))
    if not url:
        return {"tool_output": "No URL found to scrape."}
    return {"tool_output": scrape_static(url)}


def search_node(state: AgentState) -> AgentState:
    return {"tool_output": tavily_search(state["expanded_query"])}


def rag_node(state: AgentState) -> AgentState:
    return {"tool_output": rag_retrieve(state["expanded_query"])}


# ---- Node 3: answering LLM (LLM #2, user-selectable model) -----------------


def get_answer_llm(model_choice: str) -> ChatOpenAI:
    model_id = ANSWER_MODEL_MAP.get(model_choice, ANSWER_MODEL_MAP["llama-3.3-70b"])
    return ChatOpenAI(
        model=model_id,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.3,
    )


def answer_node(state: AgentState) -> AgentState:
    llm = get_answer_llm(state.get("answer_model", "llama-3.3-70b"))
    prompt = (
        "Answer the user's original question using the retrieved context below. "
        "Be direct and specific. If the context doesn't contain the answer, say so "
        "plainly instead of guessing.\n\n"
        f"Original question: {state['original_query']}\n\n"
        f"Retrieved context:\n{state.get('tool_output', 'None')}"
    )
    result = llm.invoke(prompt)
    return {"final_answer": result.content}
