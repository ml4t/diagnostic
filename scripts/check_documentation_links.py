"""Check README paths and rendered documentation links and anchors."""

from __future__ import annotations

import argparse
import posixpath
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).parents[1]
ORIGIN = "https://www.ml4trading.io"
PREFIX = "/docs/diagnostic/"
BOOK_ORIGIN = "https://github.com/stefan-jansen/machine-learning-for-trading/"
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)\s]+)\)")


class ContentParser(HTMLParser):
    """Collect article links and all rendered anchor IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.in_article = False
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "article":
            self.in_article = True
        if identifier := values.get("id"):
            self.ids.add(identifier)
        if self.in_article:
            link = values.get("src" if tag == "img" else "href" if tag == "a" else "")
            if link:
                self.links.append(link)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self.in_article = False


def _page_url(page: Path, site: Path) -> str:
    relative = page.relative_to(site).as_posix()
    if relative == "index.html":
        return ORIGIN + PREFIX
    if relative.endswith("/index.html"):
        return ORIGIN + PREFIX + relative[: -len("index.html")]
    return ORIGIN + PREFIX + relative


def _target(site: Path, path: str) -> Path:
    relative = unquote(path.removeprefix(PREFIX))
    normalized = posixpath.normpath(relative).lstrip("/")
    if path.endswith("/") or normalized == ".":
        normalized = posixpath.join(normalized, "index.html")
    target = (site / normalized).resolve()
    if not target.is_relative_to(site):
        raise ValueError("link escapes the documentation site")
    return target


def _readme_links() -> tuple[int, set[str]]:
    checked = 0
    external: set[str] = set()
    for match in MARKDOWN_LINK.finditer((ROOT / "README.md").read_text(encoding="utf-8")):
        url = urlsplit(match.group(1))
        checked += 1
        if url.scheme in {"http", "https"}:
            if not url.geturl().startswith(BOOK_ORIGIN):
                external.add(url._replace(fragment="").geturl())
        elif not url.scheme and url.path:
            target = (ROOT / unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                raise ValueError(f"README link has no file: {match.group(1)}")
    return checked, external


def _check_external(url: str) -> None:
    request = Request(url, headers={"User-Agent": "ml4t-diagnostic-docs/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            if response.status >= 400:
                raise ValueError(f"{url}: HTTP {response.status}")
    except (HTTPError, URLError, TimeoutError) as error:
        raise ValueError(f"External link failed: {url}: {error}") from error


def check_links(site: Path, *, external: bool = False) -> tuple[int, int]:
    """Validate article targets and optional relevant external destinations."""
    site = site.resolve()
    pages: dict[Path, ContentParser] = {}
    for page in sorted(site.rglob("*.html")):
        parser = ContentParser()
        parser.feed(page.read_text(encoding="utf-8"))
        pages[page.resolve()] = parser
    if not pages:
        raise ValueError(f"No rendered HTML pages in {site}")
    checked, destinations = _readme_links()
    for page, content in pages.items():
        for link in content.links:
            checked += 1
            destination = urlsplit(urljoin(_page_url(page, site), link))
            if destination.scheme not in {"http", "https"}:
                raise ValueError(f"{page}: unsupported link: {link}")
            if destination.netloc not in {"www.ml4trading.io", "ml4trading.io"}:
                if not destination.geturl().startswith(BOOK_ORIGIN):
                    destinations.add(destination._replace(fragment="").geturl())
                continue
            if not destination.path.startswith(PREFIX):
                destinations.add(destination._replace(fragment="").geturl())
                continue
            target = _target(site, destination.path)
            if not target.is_file():
                raise ValueError(f"{page.relative_to(site)}: missing {link}")
            if destination.fragment and unquote(destination.fragment) not in pages[target].ids:
                raise ValueError(f"{page.relative_to(site)}: missing anchor {link}")
    if external:
        for url in sorted(destinations):
            _check_external(url)
    return checked, len(destinations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path("site"))
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    checked, destinations = check_links(args.site, external=args.external)
    print(f"Checked {checked} README/rendered links and {destinations} external destinations")


if __name__ == "__main__":
    main()
