from playwright.sync_api import sync_playwright
import time

def get_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        # Give it a bit more time to render
        time.sleep(3)
        text = page.inner_text("body")
        browser.close()
        return text

urls = [
    "https://gemini.google.com/share/4b4104976eea",
    "https://claude.ai/share/7b9d19c8-76bb-400f-a964-0a7faabef57a"
]

for i, url in enumerate(urls):
    try:
        print(f"Fetching {url}...")
        text = get_text(url)
        with open(f"chat_{i}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved chat_{i}.txt")
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
