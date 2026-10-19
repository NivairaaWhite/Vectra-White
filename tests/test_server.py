import http.client
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from vectra.augment import crop_center, jpeg
from vectra.imageio import sha256
from vectra.server import _http_url, make_server


@pytest.fixture()
def api(lib, serve_in_thread):
    srv = make_server(lib, "127.0.0.1", 0)
    port = serve_in_thread(srv)

    def call(method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, dict(r.getheaders()), data
    call.server = srv
    call.lib = lib
    return call


def j(resp):
    return json.loads(resp[2])


def test_static_pages_and_strict_csp(api):
    st, h, body = api("GET", "/")
    assert st == 200 and b"vectra" in body
    csp = h["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"
    assert api("GET", "/app.js")[1]["Content-Type"].startswith("text/javascript")
    assert api("GET", "/app.css")[0] == 200
    assert api("GET", "/favicon.ico")[0] == 204


def test_info_and_samples(api):
    info = j(api("GET", "/api/info"))
    assert info["images"] == 60 and info["backend"] == "handcrafted" and info["index_size"] == 60
    samples = j(api("GET", "/api/samples?n=5"))["samples"]
    assert len(samples) == 5 and samples[0]["thumb"].startswith("/thumb/")


def test_search_upload_finds_edited_copy(api, demo_dir):
    src = demo_dir / "demo_0012.jpg"
    q = jpeg(crop_center(Image.open(src).convert("RGB"), 0.7), 40)
    buf = io.BytesIO(); q.save(buf, "JPEG")
    st, _, body = api("POST", "/api/search?k=5", buf.getvalue(), {"Content-Type": "image/jpeg", "X-Vectra": "1"})
    res = json.loads(body)["results"]
    assert st == 200 and len(res) == 5
    top = res[0]
    assert api.lib.store.get_image(top["id"])["sha256"] == sha256(src.read_bytes())
    assert set(top) >= {"id", "score", "kind", "thumb", "full", "width", "height", "name"}


def test_search_similar_by_id_excludes_itself(api):
    res = j(api("GET", "/api/similar/3?k=6"))["results"]
    assert len(res) == 6 and all(r["id"] != 3 for r in res)


def test_post_requires_custom_header_and_image_type(api, demo_dir):
    data = (demo_dir / "demo_0001.jpg").read_bytes()
    assert api("POST", "/api/search", data, {"Content-Type": "image/jpeg"})[0] == 403        # no X-Vectra (CSRF)
    assert api("POST", "/api/search", data, {"Content-Type": "text/plain", "X-Vectra": "1"})[0] == 415
    st, _, body = api("POST", "/api/search", b"not an image", {"Content-Type": "image/png", "X-Vectra": "1"})
    assert st == 400 and "not an image" in json.loads(body)["error"] and "0x" not in json.loads(body)["error"]


def test_upload_size_limit_and_content_length(api, demo_dir):
    api.server.max_upload = 1000
    st, _, body = api("POST", "/api/search", b"x" * 5000, {"Content-Type": "image/png", "X-Vectra": "1"})
    assert st == 413
    import socket
    s = socket.create_connection(api.server.server_address, timeout=5)
    s.sendall(b"POST /api/search HTTP/1.0\r\nHost: 127.0.0.1:%d\r\nX-Vectra: 1\r\nContent-Type: image/png\r\n\r\n" % api.server.server_address[1])
    assert b" 411 " in s.recv(200)
    s.close()


def test_wrong_host_header_is_rejected_dns_rebinding(api):
    assert api("GET", "/api/info", headers={"Host": "evil.example.com"})[0] == 403
    assert api("GET", "/api/info", headers={"Host": "localhost:%d" % api.server.server_address[1]})[0] == 200


def test_no_cors_headers_so_other_sites_cannot_read_responses(api):
    st, h, _ = api("GET", "/api/info", headers={"Origin": "https://evil.example"})
    assert not any(k.lower().startswith("access-control-") for k in h)


def test_images_only_by_integer_id_no_path_traversal(api):
    for bad in ("/img/../../etc/passwd", "/img/%2e%2e/%2e%2e/etc/passwd", "/img/abc", "/img/1/../2",
                "/thumb/1?s=100", "/img/", "/api/similar/x"):
        assert api("GET", bad)[0] in (400, 404), bad
    st, h, body = api("GET", "/img/1")
    assert st == 200 and h["Content-Type"].startswith("image/") and len(body) > 100
    assert api("GET", "/img/999999")[0] == 404


def test_thumbnails_fixed_sizes_and_cached_on_disk(api):
    st, h, body = api("GET", "/thumb/2?s=160")
    assert st == 200 and body[:2] == b"\xff\xd8" and Image.open(io.BytesIO(body)).size[0] <= 160
    assert (api.lib.home / "thumbs" / "2_160.jpg").exists()
    assert api("GET", "/thumb/2?s=161")[0] == 400


def test_only_http_links_are_exposed_to_the_page(lib):
    assert _http_url("javascript:alert(1)") is None
    assert _http_url("file:///etc/passwd") is None
    assert _http_url("https://example.com/a.jpg") == "https://example.com/a.jpg"
    assert _http_url(None) is None


def test_ui_javascript_never_uses_innerhtml():
    """Crawled alt text / URLs are untrusted; the UI must only ever use textContent."""
    js = (Path(__file__).resolve().parents[1] / "vectra" / "web" / "app.js").read_text()
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "document.write" not in js
    assert "eval(" not in js


def test_server_is_read_only(api):
    n = api.lib.store.count()
    api("GET", "/api/similar/1?k=3")
    assert api("POST", "/api/other", b"x", {"X-Vectra": "1"})[0] == 404
    assert api("PUT", "/api/search", b"x")[0] in (501, 405, 404)
    assert api.lib.store.count() == n
