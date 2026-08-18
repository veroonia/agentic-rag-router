from typing import Literal, TypedDict


RouteType = Literal["playwright", "scrape", "search", "rag"]

AnswerModel = Literal[
    "nemotron-3.5-lightning",
    "dots-3-note-preview",
]


class AgentState(TypedDict, total=False):
    original_query: str
    expanded_query: str
    route: RouteType
    tool_output: str
    final_answer: str
    answer_model: AnswerModel