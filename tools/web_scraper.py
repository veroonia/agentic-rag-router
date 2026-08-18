import requests
import trafilatura


def scrape_static(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"Failed to fetch {url}: {e}"

    extracted = trafilatura.extract(
        resp.text,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if extracted and len(extracted.strip()) > 200:
        return extracted[:6000]

    # Fallback to raw text extraction if trafilatura finds nothing
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())

    if len(text.strip()) < 200:
        return (
            f"Only minimal content could be extracted from {url}. "
            "The page may require JavaScript rendering — consider using the Playwright route instead."
        )

    return text[:6000]