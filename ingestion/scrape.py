"""
Scrape documentation pages from n8n and MCP.

Strategy:
- Pull URL list from each source's sitemap (or a curated list).
- Fetch each page, extract clean text content (strip nav/footer/scripts).
- Yield Document records — no DB writes here, those happen in ingest.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

USER_AGENT = os.getenv("USER_AGENT", "n8n-mcp-copilot/0.1")
DELAY = float(os.getenv("REQUEST_DELAY_SECONDS", "0.5"))
MAX_PAGES = int(os.getenv("MAX_PAGES_PER_SOURCE", "150"))


@dataclass
class ScrapedDoc:
    source_url: str
    title: str
    doc_type: str            # 'n8n' or 'mcp'
    content: str


def _http_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )


def _extract_text(html: str) -> tuple[str, str]:
    """Return (title, clean_text) — strip nav, footer, scripts, styles.
    Unwrap inline elements before extraction to preserve sentence flow.
    """
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup

    # ⬇ Nuevo: unwrap inline elements to preserve natural sentence flow
    for inline_tag in main.find_all(['code', 'a', 'strong', 'em', 'span', 'b', 'i', 'kbd', 'mark', 'samp', 'var']):
        inline_tag.unwrap()

    main.smooth()   # ← Esto fusiona los text nodes adyacentes
    text = main.get_text(separator="\n", strip=True)

    text = main.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return title, text

def _prioritize_urls(urls: list[str]) -> list[str]:
    """
    Reorder URLs so that high-priority paths come first.
    This ensures that when MAX_PAGES_PER_SOURCE truncates the list,
    we keep the most informative pages.

    Returns: same URLs, reordered by priority (high → low).
    """
    high_priority_patterns = [
        "/integrations/builtin/core-nodes/",     # webhook, http request, etc.
        "/integrations/builtin/trigger-nodes/",  # cron, schedule triggers
        "/integrations/builtin/credentials/",
        "/hosting/configuration/",
        "/integrations/creating-nodes/",
    ]
    medium_priority_patterns = [
        "/integrations/builtin/app-nodes/",
	"/data/expression-reference/",
	"/courses/level-one/",
	"/hosting/installation/",
	"/advanced-ai/examples/",
	"/hosting/securing/",
	"/code/cookbook/",
	"/courses/level-two/",
	"/integrations/community-nodes/",
    ]

    def _score(url: str) -> int:
        for pattern in high_priority_patterns:
            if pattern in url:
                return 0
        for pattern in medium_priority_patterns:
            if pattern in url:
                return 1
        return 2

    return sorted(urls, key=_score)

def _urls_from_sitemap(sitemap_url: str, prefix_filter: str) -> list[str]:
    """Parse an XML sitemap and return all <loc> URLs that start with the prefix."""
    with _http_client() as client:
        r = client.get(sitemap_url)
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "xml")
    urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    return [u for u in urls if u.startswith(prefix_filter)]


def scrape_n8n() -> Iterator[ScrapedDoc]:
    sitemap = os.getenv("N8N_SITEMAP_URL", "https://docs.n8n.io/sitemap.xml")
    base = os.getenv("N8N_DOCS_BASE_URL", "https://docs.n8n.io")
    urls = _urls_from_sitemap(sitemap, base)
    urls = _prioritize_urls(urls)              # ← nuevo
    urls = urls[:MAX_PAGES]

    with _http_client() as client:
        for url in tqdm(urls, desc="Scraping n8n"):
            try:
                r = client.get(url)
                r.raise_for_status()
                title, text = _extract_text(r.text)
                if len(text) < 200:
                    continue            # skip near-empty pages
                yield ScrapedDoc(source_url=url, title=title, doc_type="n8n", content=text)
            except Exception as e:                          # noqa: BLE001
                print(f"⚠ skipped {url}: {e}")
            time.sleep(DELAY)


def scrape_mcp() -> Iterator[ScrapedDoc]:
    """
    MCP docs don't always expose a sitemap — fall back to a curated seed list
    plus following internal links one level deep.
    """
    base = os.getenv("MCP_DOCS_BASE_URL", "https://modelcontextprotocol.io")
    seeds = [
        f"{base}/introduction",
        f"{base}/quickstart/server",
        f"{base}/quickstart/client",
        f"{base}/docs/concepts/architecture",
        f"{base}/docs/concepts/resources",
        f"{base}/docs/concepts/prompts",
        f"{base}/docs/concepts/tools",
        f"{base}/docs/concepts/sampling",
        f"{base}/docs/concepts/transports",
        f"{base}/specification",
    ]

    visited: set[str] = set()
    queue: list[str] = list(seeds)

    with _http_client() as client:
        with tqdm(total=MAX_PAGES, desc="Scraping MCP") as pbar:
            while queue and len(visited) < MAX_PAGES:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)
                try:
                    r = client.get(url)
                    r.raise_for_status()
                    title, text = _extract_text(r.text)
                    if len(text) >= 200:
                        yield ScrapedDoc(
                            source_url=url, title=title, doc_type="mcp", content=text
                        )
                        pbar.update(1)

                    # one-level link expansion
                    soup = BeautifulSoup(r.text, "lxml")
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        if href.startswith("/"):
                            href = base + href
                        if href.startswith(base) and href not in visited and href not in queue:
                            queue.append(href)

                except Exception as e:                      # noqa: BLE001
                    print(f"⚠ skipped {url}: {e}")
                time.sleep(DELAY)


if __name__ == "__main__":
    # quick smoke test
    for doc in scrape_mcp():
        print(f"{doc.doc_type} | {doc.title[:60]} | {len(doc.content)} chars")
