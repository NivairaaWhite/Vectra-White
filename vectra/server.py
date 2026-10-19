"""Local web UI + JSON API (standard library only, read-only).

Security stance, because a local web server is still a web server:
  * binds to 127.0.0.1 by default; the Host header must match (blocks DNS-rebinding);
  * no CORS headers, and POST needs a custom ``X-Vectra`` header, so another website open in your
    browser cannot make it search or upload anything (a cross-origin request would need a
    preflight that we never approve);
  * images are served only by integer id from the database -- there is no path parameter, so no
    path traversal; thumbnails only in a fixed set of sizes;
  * strict Content-Security-Policy (no inline script/style, no third-party loads), nosniff;
  * request bodies are size-limited; the server never writes to the library.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .config import THUMB_SIZES
from .errors import VectraError
from .library import Library, Match

log = logging.getLogger("vectra.server")
WEB_DIR = Path(__file__).parent / "web"
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
MAX_K = 100
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _http_url(u: Optional[str]) -> Optional[str]:
    """Only http(s) links are ever exposed to the page (no javascript:/file: URLs)."""
    return u if u and re.match(r"^https?://", u, re.I) else None


def serialize(m: Match) -> Dict:
    name = m.title or Path(m.path).name
    return {
        "id": m.id, "score": round(float(m.score), 4), "kind": m.kind,
        "width": m.width, "height": m.height, "phash_distance": m.phash_distance,
        "name": name, "source": m.source, "source_url": _http_url(m.source_url),
        "page_url": _http_url(m.page_url), "alt": m.alt,
        "thumb": f"/thumb/{m.id}?s=320", "full": f"/img/{m.id}",
    }


class VectraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: Tuple[str, int], lib: Library, max_upload_mb: int = 25):
        self.lib = lib
        self.max_upload = int(max_upload_mb) * 1024 * 1024
        self.static = {p: ((WEB_DIR / f).read_bytes(), ct) for p, (f, ct) in _STATIC.items()}
        super().__init__(addr, Handler)
        host, port = self.server_address[:2]
        if host in LOOPBACK or host.startswith("127."):
            names = {"127.0.0.1", "localhost", "[::1]"}
            self.allowed_hosts = {f"{n}:{port}" for n in names} | ({n for n in names} if port == 80 else set())
        else:
            self.allowed_hosts = None   # explicit non-loopback bind: Host check cannot know the names


class Handler(BaseHTTPRequestHandler):
    server: VectraHTTPServer
    server_version = "Vectra"
    sys_version = ""

    # --- plumbing ------------------------------------------------------------------------
    def log_message(self, fmt, *args):  # quiet by default; enable with logging.DEBUG
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8",
                   {"Cache-Control": "no-store"})

    def _error(self, status: int, msg: str) -> None:
        self._json(status, {"error": msg})

    def _host_ok(self) -> bool:
        allowed = self.server.allowed_hosts
        return allowed is None or self.headers.get("Host", "").lower() in allowed

    @staticmethod
    def _int(params: Dict, key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(params.get(key, [default])[0])))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _flag(params: Dict, key: str) -> bool:
        return params.get(key, ["0"])[0] in ("1", "true", "yes", "on")

    # --- routes --------------------------------------------------------------------------
    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_GET(self):  # noqa: N802
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        url = urlparse(self.path)
        path, params = url.path, parse_qs(url.query)
        lib = self.server.lib
        try:
            if path in self.server.static:
                body, ctype = self.server.static[path]
                extra = {"Cache-Control": "no-cache"}
                if path == "/":
                    extra["Content-Security-Policy"] = CSP
                return self._send(200, body, ctype, extra)
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if path == "/api/info":
                return self._json(200, self._info())
            if path == "/api/samples":
                return self._json(200, {"samples": self._samples(self._int(params, "n", 8, 1, 24))})
            m = re.fullmatch(r"/api/similar/(\d{1,12})", path)
            if m:
                k = self._int(params, "k", 24, 1, MAX_K)
                res = lib.search(int(m.group(1)), k=k, flip=self._flag(params, "flip"), exclude_exact=True)
                return self._json(200, {"results": [serialize(r) for r in res]})
            m = re.fullmatch(r"/img/(\d{1,12})", path)
            if m:
                got = lib.image_bytes(int(m.group(1)))
                if got is None:
                    return self._error(404, "image not found")
                return self._send(200, got[0], got[1], {"Cache-Control": "private, max-age=3600"})
            m = re.fullmatch(r"/thumb/(\d{1,12})", path)
            if m:
                size = self._int(params, "s", 320, 1, 100000)
                if size not in THUMB_SIZES:
                    return self._error(400, f"thumbnail size must be one of {list(THUMB_SIZES)}")
                data = lib.thumbnail(int(m.group(1)), size)
                if data is None:
                    return self._error(404, "image not found")
                return self._send(200, data, "image/jpeg", {"Cache-Control": "private, max-age=3600"})
            return self._error(404, "not found")
        except VectraError as e:
            return self._error(400, str(e))
        except Exception:  # pragma: no cover - last-resort guard, details go to the log only
            log.exception("GET %s failed", self.path)
            return self._error(500, "internal error")

    def do_POST(self):  # noqa: N802
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        url = urlparse(self.path)
        if url.path != "/api/search":
            return self._error(404, "not found")
        if self.headers.get("X-Vectra") != "1":
            return self._error(403, "missing X-Vectra header")
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not (ctype.startswith("image/") or ctype == "application/octet-stream"):
            return self._error(415, "send the image file as the request body")
        length = self.headers.get("Content-Length")
        if not length or not length.isdigit():
            return self._error(411, "Content-Length required")
        n = int(length)
        if n == 0:
            return self._error(400, "empty upload")
        if n > self.server.max_upload:
            return self._error(413, f"image larger than {self.server.max_upload // (1024 * 1024)} MB")
        data = bytearray()
        while len(data) < n:
            chunk = self.rfile.read(min(65536, n - len(data)))
            if not chunk:
                return self._error(400, "upload ended early")
            data += chunk
        params = parse_qs(url.query)
        try:
            res = self.server.lib.search(
                bytes(data), k=self._int(params, "k", 24, 1, MAX_K), flip=self._flag(params, "flip"),
                exclude_exact=self._flag(params, "exclude_exact"))
            return self._json(200, {"results": [serialize(r) for r in res]})
        except VectraError as e:
            return self._error(400, str(e))
        except Exception:  # pragma: no cover
            log.exception("search failed")
            return self._error(500, "internal error")

    # --- helpers -------------------------------------------------------------------------
    def _info(self) -> Dict:
        lib = self.server.lib
        s = lib.stats()
        try:
            idx = lib.get_index()
            s["index_kind"], s["index_size"] = idx.kind, len(idx)
        except VectraError:
            s["index_kind"], s["index_size"] = None, 0
        keep = ("images", "backend", "active_model", "model_kind", "dim", "index_kind", "index_size",
                "embedded", "pending", "version")
        return {k: s.get(k) for k in keep}

    def _samples(self, n: int):
        ids = self.server.lib.store.all_ids()
        picked = random.sample(ids, min(n, len(ids)))
        return [{"id": i, "thumb": f"/thumb/{i}?s=160"} for i in picked]


def make_server(lib: Library, host: str = "127.0.0.1", port: int = 8765, max_upload_mb: int = 25) -> VectraHTTPServer:
    return VectraHTTPServer((host, port), lib, max_upload_mb)


def serve(lib: Library, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False,
          say=print) -> None:
    rep = lib.ensure_features()
    if rep.done:
        say(f"Embedded {rep.done} new images.")
    lib.get_index()      # fail early (and warm up) before accepting requests
    srv = make_server(lib, host, port)
    shown = "localhost" if host in LOOPBACK else host
    url = f"http://{shown}:{srv.server_address[1]}/"
    if srv.allowed_hosts is None:
        say(f"WARNING: listening on {host}, not just this machine. Anyone who can reach the port can "
            f"search and view your library. Prefer the default 127.0.0.1.")
    say(f"Vectra is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        say("\nStopped.")
    finally:
        srv.server_close()
