from typing import Literal, TypedDict

RouteType = Literal["playwright", "scrape", "search", "rag"]
AnswerModel = Literal["llama-3.3-70b", "gpt-oss-120b"]


class AgentState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""

    original_query: str
    expanded_query: str
    route: RouteType
    tool_output: str
    final_answer: str
    answer_model: AnswerModel
