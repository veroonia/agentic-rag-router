import os

from tavily import TavilyClient

_client = None


def get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


def tavily_search(query: str, max_results: int = 5) -> str:
    client = get_client()
    response = client.search(query=query, max_results=max_results, include_answer=True)

    parts = []
    if response.get("answer"):
        parts.append(f"Tavily summary: {response['answer']}")
    for r in response.get("results", []):
        parts.append(f"- {r['title']}: {r['content'][:400]} (source: {r['url']})")

    return "\n".join(parts) if parts else "No results found."
