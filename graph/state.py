from typing import Literal, TypedDict


RouteType = Literal["playwright", "scrape", "search", "rag"]


class AgentState(TypedDict, total=False):
    """Shared state passed between every node in the LangGraph agent."""

    # User input
    original_query: str

    # LLM #1 output
    expanded_query: str

    # Router output
    route: RouteType

    # Tool output
    tool_output: str

    # Final LLM output
    final_answer: str