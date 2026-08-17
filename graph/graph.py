from langgraph.graph import END, StateGraph

from graph.nodes import (
    answer_node,
    playwright_node,
    rag_node,
    rephrase_node,
    router_node,
    scrape_node,
    search_node,
)
from graph.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("rephrase", rephrase_node)
    graph.add_node("router", router_node)
    graph.add_node("playwright", playwright_node)
    graph.add_node("scrape", scrape_node)
    graph.add_node("search", search_node)
    graph.add_node("rag", rag_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("rephrase")
    graph.add_edge("rephrase", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "playwright": "playwright",
            "scrape": "scrape",
            "search": "search",
            "rag": "rag",
        },
    )

    # Every tool converges straight into the answering LLM — no cycling back
    # into retrieval.
    for tool in ["playwright", "scrape", "search", "rag"]:
        graph.add_edge(tool, "answer")

    graph.add_edge("answer", END)
    return graph.compile()


app_graph = build_graph()
