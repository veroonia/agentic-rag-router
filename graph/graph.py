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

    graph = StateGraph(
        AgentState
    )

    # Nodes
    graph.add_node(
        "rephrase",
        rephrase_node,
    )

    graph.add_node(
        "router",
        router_node,
    )

    graph.add_node(
        "playwright",
        playwright_node,
    )

    graph.add_node(
        "scrape",
        scrape_node,
    )

    graph.add_node(
        "search",
        search_node,
    )

    graph.add_node(
        "rag",
        rag_node,
    )

    graph.add_node(
        "answer",
        answer_node,
    )

    # Entry
    graph.set_entry_point(
        "rephrase"
    )

    # Rephrase -> Router
    graph.add_edge(
        "rephrase",
        "router",
    )

    # Router -> Tool
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "playwright":
                "playwright",

            "scrape":
                "scrape",

            "search":
                "search",

            "rag":
                "rag",
        },
    )

    # Tool -> Answer
    graph.add_edge(
        "playwright",
        "answer",
    )

    graph.add_edge(
        "scrape",
        "answer",
    )

    graph.add_edge(
        "search",
        "answer",
    )

    graph.add_edge(
        "rag",
        "answer",
    )

    # Answer -> END
    graph.add_edge(
        "answer",
        END,
    )

    return graph.compile()


app_graph = build_graph()