import os

from tavily import TavilyClient


_client = None


def get_client() -> TavilyClient:
    global _client

    if _client is None:

        api_key = os.environ.get(
            "TAVILY_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is missing. "
                "Add it to the .env file."
            )

        _client = TavilyClient(
            api_key=api_key
        )

    return _client


def tavily_search(
    query: str,
    max_results: int = 5,
) -> str:

    client = get_client()

    response = client.search(
        query=query,
        max_results=max_results,
        include_answer=True,
    )

    parts = []

    if response.get("answer"):
        parts.append(
            f"Tavily summary: {response['answer']}"
        )

    for result in response.get("results", []):

        title = result.get(
            "title",
            "Untitled",
        )

        content = result.get(
            "content",
            "",
        )

        url = result.get(
            "url",
            "",
        )

        parts.append(
            f"- {title}: "
            f"{content[:500]} "
            f"(source: {url})"
        )

    if not parts:
        return "No search results found."

    return "\n".join(parts)