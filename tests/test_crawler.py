import http.server
import io
import threading
from pathlib import Path

import pytest
from PIL import Image

from vectra import Library
from vectra.crawler import Blocked, Crawler, _Parser
from vectra.demo import make_scene


def _save(path: Path, img: Image.Image, fmt: str, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, fmt, **kw)


@pytest.fixture()
def site(tmp_path, serve_in_thread):
    """A small gallery served from 127.0.0.1 exercising every discovery path and every refusal."""
    root = tmp_path / "www"
    img = root / "img"
    a, b, c, d, e, f, og, p2 = (make_scene(i) for i in range(20, 28))
    _save(img / "a.jpg", a, "JPEG")
    (img / "a_copy.jpg").write_bytes((img / "a.jpg").read_bytes())          # same bytes, other URL
    (img / "d.jpg").write_bytes((img / "a.jpg").read_bytes())
    _save(img / "b.png", b.resize((160, 120)), "PNG")
    _save(img / "c_320.jpg", c.resize((320, 200)), "JPEG")
    _save(img / "c_1280.jpg", c, "JPEG", quality=70)
    _save(img / "d.webp", d, "WEBP")
    _save(img / "e_thumb.jpg", e.resize((96, 80)), "JPEG")
    _save(img / "e_full.jpg", e, "JPEG", quality=60)
    _save(img / "og.jpg", og, "JPEG")
    _save(img / "p2.jpg", p2, "JPEG")
    _save(img / "small.png", Image.new("RGB", (32, 32), (9, 9, 9)), "PNG")
    _save(root / "private" / "secret.jpg", f, "JPEG")
    (img / "broken.jpg").write_bytes(b"this is not a jpeg")
    (img / "huge.jpg").write_bytes(b"\xff\xd8" + b"0" * 300_000)
    (root / "robots.txt").write_text("User-agent: *\nDisallow: /private/\n")
    (root / "index.html").write_text("""<!doctype html><html><head>
      <meta property="og:image" content="/img/og.jpg"></head><body>
      <img src="img/a.jpg" alt="Alpha picture" title="first">
      <img data-src="img/b.png" alt="lazy one">
      <img src="img/small.png">
      <img srcset="img/c_320.jpg 320w, img/c_1280.jpg 1280w" src="img/c_320.jpg">
      <picture><source srcset="img/d.webp" type="image/webp"><img src="img/d.jpg"></picture>
      <a href="img/e_full.jpg"><img src="img/e_thumb.jpg"></a>
      <img src="private/secret.jpg">
      <img src="img/a.jpg">
      <img src="img/a_copy.jpg">
      <img src="img/broken.jpg">
      <img src="img/huge.jpg">
      <img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=">
      <a href="page2.html">next</a> <a href="http://other.invalid/x">external</a>
      </body></html>""")
    (root / "page2.html").write_text('<html><body><img src="img/p2.jpg"></body></html>')

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def do_GET(self):
            if self.path == "/redir.jpg":
                self.send_response(302)
                self.send_header("Location", "/img/p2.jpg")
                self.end_headers()
                return
            super().do_GET()

        def log_message(self, *a):
            pass

    class Quiet(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):   # clients abort oversized downloads on purpose
            pass

    srv = Quiet(("127.0.0.1", 0), Handler)
    port = serve_in_thread(srv)
    return f"http://127.0.0.1:{port}"


def crawler_for(lib, **kw):
    opts = dict(delay=0, allow_private=True, max_image_bytes=100_000, depth=1)
    opts.update(kw)
    return Crawler(lib, **opts)


def test_crawl_discovers_everything_and_refuses_the_rest(tmp_path, site):
    lib = Library.create(tmp_path / "lib")
    rep = crawler_for(lib).crawl([site + "/index.html"])
    urls = {lib.store.get_image(i)["source_url"].replace(site, "") for i in lib.store.all_ids()}
    assert urls == {"/img/a.jpg", "/img/b.png", "/img/c_1280.jpg", "/img/d.webp", "/img/e_full.jpg",
                    "/img/e_thumb.jpg", "/img/og.jpg", "/img/p2.jpg"}
    assert rep.pages == 2 and rep.added == 8
    assert rep.duplicates == 2          # d.jpg and a_copy.jpg have the same bytes as a.jpg
    assert rep.too_small == 1           # small.png
    assert rep.invalid == 1             # broken.jpg
    assert rep.blocked_robots == 1      # /private/secret.jpg
    assert any("huge.jpg" in e for e in rep.errors)       # over the size cap
    assert {lib.store.get_image(i)["source"] for i in lib.store.all_ids()} == {"crawl"}


def test_provenance_is_recorded(tmp_path, site):
    lib = Library.create(tmp_path / "lib")
    crawler_for(lib).crawl([site + "/index.html"])
    rec = next(lib.store.get_image(i) for i in lib.store.all_ids()
               if lib.store.get_image(i)["source_url"].endswith("/img/a.jpg"))
    assert rec["alt"] == "Alpha picture" and rec["title"] == "first"
    assert rec["page_url"].endswith("/index.html")


def test_depth_zero_does_not_follow_links(tmp_path, site):
    lib = Library.create(tmp_path / "lib")
    rep = crawler_for(lib, depth=0).crawl([site + "/index.html"])
    assert rep.pages == 1
    assert not any(lib.store.get_image(i)["source_url"].endswith("p2.jpg") for i in lib.store.all_ids())


def test_max_images_budget(tmp_path, site):
    lib = Library.create(tmp_path / "lib")
    rep = crawler_for(lib, max_images=3).crawl([site + "/index.html"])
    assert rep.added == 3


def test_redirects_are_followed_and_direct_image_seeds_work(tmp_path, site):
    lib = Library.create(tmp_path / "lib")
    rep = crawler_for(lib).crawl([site + "/redir.jpg"])
    assert rep.added == 1


def test_private_addresses_blocked_by_default(tmp_path, site):
    lib = Library.create(tmp_path / "lib")
    rep = Crawler(lib, delay=0).crawl([site + "/index.html"])      # allow_private defaults to False
    assert rep.added == 0 and rep.blocked_network >= 1


@pytest.mark.parametrize("host", ["127.0.0.1", "169.254.169.254", "10.1.2.3", "192.168.0.9", "[::1]", "0.0.0.0", "100.64.0.1"])
def test_non_public_literals_are_refused(tmp_path, host):
    c = Crawler(Library.create(tmp_path / "lib"))
    with pytest.raises(Blocked):
        c._check_url(f"http://{host}/x.jpg")


def test_only_http_schemes(tmp_path):
    c = Crawler(Library.create(tmp_path / "lib"), allow_private=True)
    for bad in ("file:///etc/passwd", "ftp://example.com/a.jpg", "gopher://x/"):
        with pytest.raises(Blocked):
            c._check_url(bad)


def test_robots_disallow_all_blocks_pages(tmp_path, serve_in_thread):
    root = tmp_path / "www"; root.mkdir()
    (root / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    (root / "index.html").write_text('<img src="a.jpg">')
    _save(root / "a.jpg", make_scene(1), "JPEG")

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)
        def log_message(self, *a):
            pass
    port = serve_in_thread(http.server.ThreadingHTTPServer(("127.0.0.1", 0), H))
    lib = Library.create(tmp_path / "lib")
    rep = crawler_for(lib).crawl([f"http://127.0.0.1:{port}/index.html"])
    assert rep.added == 0 and rep.blocked_robots == 1


def test_srcset_parser_picks_largest_and_handles_commas_in_urls():
    best = _Parser.best_from_srcset
    assert best("a.jpg 1x, b.jpg 2x") == "b.jpg"
    assert best("s.jpg 320w, m.jpg 640w, l.jpg 1280w") == "l.jpg"
    assert best("https://cdn.x/i/w_100,h_100/a.jpg 100w, https://cdn.x/i/w_900,h_900/a.jpg 900w") == \
        "https://cdn.x/i/w_900,h_900/a.jpg"
    assert best("") is None


def test_parser_ignores_unsafe_urls():
    p = _Parser("http://h/x/")
    p.feed('<img src="javascript:alert(1)"><a href="mailto:a@b"><img src="data:image/png;base64,AA"><img src="ok.png">')
    assert [c.url for c in p.images] == ["http://h/x/ok.png"]


def test_unreachable_site_is_reported_as_unreachable_not_as_robots_block(tmp_path):
    lib = Library.create(tmp_path / "lib")
    rep = Crawler(lib, delay=0, timeout=2, allow_private=True).crawl(["http://127.0.0.1:9/page.html"])
    assert rep.added == 0 and rep.blocked_robots == 0
    assert rep.errors and "skipped" in rep.errors[0]
