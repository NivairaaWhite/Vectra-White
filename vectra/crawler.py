"""Polite, safe image crawler (standard library only).

The original was a single-page ``<img src>`` scraper with a 0.1 s sleep and no robots.txt, no
size cap and every error swallowed. This one:

  * honours robots.txt (and Crawl-delay) for pages *and* images, and rate-limits per host;
  * follows links breadth-first to a depth/page budget, optionally staying on the seed's host;
  * finds images in ``<img src/srcset/data-src>``, ``<picture><source>``, ``og:image``,
    ``<link rel=image_src>`` and ``<a href="full.jpg">`` (gallery thumbnails -> full size);
  * refuses private / loopback / link-local addresses by default and re-checks every redirect
    hop (a public URL that redirects to http://169.254.169.254/ is blocked);
  * caps download sizes, checks Content-Type, skips tiny icons, de-duplicates by content hash,
    and records provenance (source URL, page URL, alt text) for every image;
  * reports what happened instead of silently failing.

Only crawl sites you have the right to copy images from. Respect copyright and terms of use.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.parse import urljoin, urldefrag, urlparse

from .config import IMAGE_SUFFIXES

DEFAULT_UA = "VectraBot/2.0 (local reverse-image-search; honours robots.txt)"
_ASSET_SUFFIXES = IMAGE_SUFFIXES | {".css", ".js", ".json", ".xml", ".svg", ".ico", ".pdf", ".zip",
                                    ".mp4", ".mp3", ".woff", ".woff2", ".ttf", ".gz", ".exe", ".dmg"}


class FetchError(Exception):
    pass


class Blocked(FetchError):
    pass


class TooLarge(FetchError):
    pass


@dataclass
class Candidate:
    url: str
    alt: Optional[str] = None
    title: Optional[str] = None


class _Parser(HTMLParser):
    """Collects image candidates and outgoing links from one HTML page."""

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.images: List[Candidate] = []
        self.links: List[str] = []

    def _abs(self, u: Optional[str]) -> Optional[str]:
        if not u:
            return None
        u = u.strip()
        if not u or u.startswith(("data:", "javascript:", "mailto:", "tel:", "#")):
            return None
        full = urldefrag(urljoin(self.base, u))[0]
        return full if full.startswith(("http://", "https://")) else None

    @staticmethod
    def best_from_srcset(srcset: str) -> Optional[str]:
        best, best_w = None, -1.0
        for m in re.finditer(r"\s*(\S+?)(?:\s+(\d+(?:\.\d+)?)([wx]))?\s*(?:,\s|,$|$)", srcset.strip() + " "):
            url, num, unit = m.group(1), m.group(2), m.group(3)
            if not url or url.startswith("data:"):
                continue
            weight = float(num) * (1000.0 if unit == "x" else 1.0) if num else 0.0
            if weight > best_w:
                best, best_w = url, weight
        return best

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "base" and a.get("href"):
            self.base = urljoin(self.base, a["href"])
        elif tag == "img":
            srcset = a.get("srcset") or a.get("data-srcset")
            url = (self.best_from_srcset(srcset) if srcset else None) or \
                a.get("src") or a.get("data-src") or a.get("data-lazy-src") or a.get("data-original")
            url = self._abs(url)
            if url:
                self.images.append(Candidate(url, a.get("alt") or None, a.get("title") or None))
        elif tag == "source":
            srcset = a.get("srcset") or a.get("data-srcset")
            url = self._abs(self.best_from_srcset(srcset)) if srcset else None
            if url and (a.get("type", "").startswith("image/") or not a.get("type")):
                self.images.append(Candidate(url))
        elif tag == "meta":
            if (a.get("property") or a.get("name") or "").lower() in ("og:image", "twitter:image", "og:image:url"):
                url = self._abs(a.get("content"))
                if url:
                    self.images.append(Candidate(url))
        elif tag == "link" and "image_src" in (a.get("rel") or "").lower():
            url = self._abs(a.get("href"))
            if url:
                self.images.append(Candidate(url))
        elif tag == "a":
            url = self._abs(a.get("href"))
            if url:
                if _suffix(url) in IMAGE_SUFFIXES:
                    self.images.append(Candidate(url, None, a.get("title") or None))
                else:
                    self.links.append(url)


def _suffix(url: str) -> str:
    path = urlparse(url).path.lower()
    dot = path.rfind(".")
    return path[dot:] if dot != -1 and "/" not in path[dot:] else ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # we follow redirects ourselves, validating each hop
        return None


@dataclass
class CrawlReport:
    pages: int = 0
    images_seen: int = 0
    added: int = 0
    duplicates: int = 0
    too_small: int = 0
    invalid: int = 0
    blocked_robots: int = 0
    blocked_network: int = 0
    errors: List[str] = field(default_factory=list)
    new_ids: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.pages} pages, {self.images_seen} image links -> {self.added} new, "
                f"{self.duplicates} duplicates, {self.too_small} too small, {self.invalid} invalid, "
                f"{self.blocked_robots} blocked by robots.txt, {self.blocked_network} blocked (private network), "
                f"{len(self.errors)} errors")


class Crawler:
    def __init__(self, lib, *, user_agent: str = DEFAULT_UA, delay: float = 1.0, timeout: float = 15.0,
                 max_image_bytes: int = 20 * 1024 * 1024, max_page_bytes: int = 3 * 1024 * 1024,
                 min_side: int = 64, depth: int = 1, max_pages: int = 20, max_images: int = 200,
                 same_host: bool = True, respect_robots: bool = True, allow_private: bool = False,
                 log: Optional[Callable[[str], None]] = None):
        self.lib = lib
        self.ua = user_agent
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.max_image_bytes = max_image_bytes
        self.max_page_bytes = max_page_bytes
        self.min_side = min_side
        self.depth = max(0, depth)
        self.max_pages = max(1, max_pages)
        self.max_images = max(1, max_images)
        self.same_host = same_host
        self.respect_robots = respect_robots
        self.allow_private = allow_private
        self.log = log or (lambda _m: None)
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._last_hit: Dict[str, float] = {}
        self._robots: Dict[str, Optional[robotparser.RobotFileParser]] = {}
        self._robots_problem: Dict[str, str] = {}      # origin -> why robots.txt could not be read
        self._crawl_delay: Dict[str, float] = {}

    # --- network primitives ----------------------------------------------------------------
    def _check_url(self, url: str) -> None:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise Blocked(f"unsupported URL: {url}")
        if self.allow_private:
            return
        try:
            infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                       type=socket.SOCK_STREAM)
        except OSError as e:
            raise FetchError(f"cannot resolve {p.hostname}: {e}") from e
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if ip.version == 6 and ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            if not ip.is_global:
                raise Blocked(f"{p.hostname} resolves to a non-public address ({ip}); "
                              f"use --allow-private only for servers you control")

    def _throttle(self, host: str) -> None:
        wait = max(self.delay, self._crawl_delay.get(host, 0.0)) - (time.time() - self._last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.time()

    def _get(self, url: str, max_bytes: int, throttle: bool = True) -> Tuple[str, int, str, bytes]:
        """-> (final_url, status, content_type, body). Follows <=5 redirects, validating each hop."""
        for _hop in range(6):
            self._check_url(url)
            host = urlparse(url).netloc.lower()
            if throttle:
                self._throttle(host)
            req = urllib.request.Request(url, headers={
                "User-Agent": self.ua, "Accept": "text/html,image/*;q=0.9,*/*;q=0.1"})
            try:
                resp = self._opener.open(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                    url = urljoin(url, e.headers["Location"])
                    continue
                return url, e.code, e.headers.get("Content-Type", "") or "", b""
            except (urllib.error.URLError, socket.timeout, OSError) as e:
                raise FetchError(f"{url}: {getattr(e, 'reason', e)}") from e
            with resp:
                length = resp.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > max_bytes:
                    raise TooLarge(f"{url}: {int(length) / 1e6:.1f} MB exceeds the limit")
                body = bytearray()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    body += chunk
                    if len(body) > max_bytes:
                        raise TooLarge(f"{url}: larger than {max_bytes / 1e6:.0f} MB")
                return url, resp.status, resp.headers.get("Content-Type", "") or "", bytes(body)
        raise FetchError(f"{url}: too many redirects")

    # --- robots.txt --------------------------------------------------------------------------
    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._robots:
            rp: Optional[robotparser.RobotFileParser] = robotparser.RobotFileParser()
            try:
                _u, status, _ct, body = self._get(origin + "/robots.txt", 512 * 1024)
                if status == 200:
                    rp.parse(body.decode("utf-8", "replace").splitlines())
                    delay = rp.crawl_delay(self.ua)
                    if delay:
                        self._crawl_delay[p.netloc.lower()] = float(delay)
                elif 400 <= status < 500:
                    rp = None                     # no robots.txt: everything allowed
                else:
                    rp.disallow_all = True        # server trouble: be conservative
                    self._robots_problem[origin] = f"robots.txt returned HTTP {status}"
            except Blocked:
                raise
            except FetchError as e:
                rp.disallow_all = True
                self._robots_problem[origin] = str(e)
            self._robots[origin] = rp
        rp = self._robots[origin]
        return True if rp is None else rp.can_fetch(self.ua, url)

    # --- crawl -------------------------------------------------------------------------------
    def crawl(self, seeds: List[str]) -> CrawlReport:
        rep = CrawlReport()
        queue: Deque[Tuple[str, int]] = deque()
        hosts: Set[str] = set()
        for s in seeds:
            s = urldefrag(s.strip())[0]
            if not s.startswith(("http://", "https://")):
                s = "https://" + s
            queue.append((s, 0))
            hosts.add(urlparse(s).netloc.lower())
        seen_pages: Set[str] = set()
        seen_images: Set[str] = set()
        attempts = 0

        while queue and rep.pages < self.max_pages and rep.added < self.max_images:
            url, depth = queue.popleft()
            if url in seen_pages:
                continue
            seen_pages.add(url)
            try:
                if not self._allowed(url):
                    p = urlparse(url)
                    problem = self._robots_problem.get(f"{p.scheme}://{p.netloc}")
                    if problem:     # unreachable site / broken robots.txt: say why, don't call it a policy block
                        rep.errors.append(f"{url}: skipped, {problem}")
                    else:
                        rep.blocked_robots += 1
                        self.log(f"  robots.txt disallows {url}")
                    continue
                if _suffix(url) in IMAGE_SUFFIXES:           # a seed that is itself an image
                    self._take_image(Candidate(url), url, rep, seen_images)
                    continue
                final, status, ctype, body = self._get(url, self.max_page_bytes)
            except Blocked as e:
                rep.blocked_network += 1
                rep.errors.append(str(e))
                continue
            except FetchError as e:
                rep.errors.append(str(e))
                continue
            if status != 200:
                rep.errors.append(f"{url}: HTTP {status}")
                continue
            if "html" not in ctype.lower() and "xml" not in ctype.lower():
                continue
            rep.pages += 1
            parser = _Parser(final)
            try:
                parser.feed(body.decode("utf-8", "replace"))
                parser.close()
            except Exception as e:  # malformed HTML must not stop the crawl
                rep.errors.append(f"{url}: parse error {e}")
            self.log(f"[{rep.pages}] {final}: {len(parser.images)} image links, {len(parser.links)} links")

            for cand in parser.images:
                if rep.added >= self.max_images or attempts >= self.max_images * 4:
                    break
                if cand.url in seen_images:
                    continue
                attempts += 1
                self._take_image(cand, final, rep, seen_images)

            if depth < self.depth:
                for link in parser.links:
                    if link in seen_pages or _suffix(link) in _ASSET_SUFFIXES:
                        continue
                    if self.same_host and urlparse(link).netloc.lower() not in hosts:
                        continue
                    queue.append((link, depth + 1))
        return rep

    def _take_image(self, cand: Candidate, page_url: str, rep: CrawlReport, seen: Set[str]) -> None:
        seen.add(cand.url)
        rep.images_seen += 1
        try:
            if not self._allowed(cand.url):
                rep.blocked_robots += 1
                return
            final, status, ctype, data = self._get(cand.url, self.max_image_bytes)
        except Blocked as e:
            rep.blocked_network += 1
            rep.errors.append(str(e))
            return
        except FetchError as e:
            rep.errors.append(str(e))
            return
        if status != 200 or not data:
            rep.errors.append(f"{cand.url}: HTTP {status}")
            return
        ct = ctype.lower()
        if "svg" in ct or not (ct.startswith("image/") or "octet-stream" in ct):
            rep.invalid += 1
            return
        res = self.lib.add_image_bytes(
            data, source="crawl", source_url=final, page_url=page_url if page_url != final else None,
            alt=(cand.alt or None), title=(cand.title or None), min_side=self.min_side)
        if res.created:
            rep.added += 1
            rep.new_ids.append(res.id)
        elif res.reason == "duplicate":
            rep.duplicates += 1
        elif res.reason == "too small":
            rep.too_small += 1
        else:
            rep.invalid += 1
