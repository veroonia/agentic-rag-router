from playwright.sync_api import sync_playwright


def scrape_dynamic(url: str, wait_ms: int = 2000) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=20000)
            page.wait_for_timeout(wait_ms)
            content = page.inner_text("body")
        except Exception as e:
            content = f"Failed to load {url}: {e}"
        finally:
            browser.close()

    return " ".join(content.split())[:4000]
