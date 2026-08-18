from typing import Annotated, Literal, TypedDict


RouteType = Literal["playwright", "scrape", "search", "rag"]

AnswerModel = Literal[
    "nemotron-3.5-lightning",
    "dots-3-note-preview",
]


def merge_outputs(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


class AgentState(TypedDict, total=False):
    original_query: str
    expanded_query: str
    routes: list[RouteType]
    tool_output: Annotated[list[str], merge_outputs]
    final_answer: str
    answer_model: AnswerModel