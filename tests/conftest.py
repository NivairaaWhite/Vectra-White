import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

from vectra import Library  # noqa: E402
from vectra.demo import make_demo_images  # noqa: E402


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    make_demo_images(d, n=60, seed=7)
    return d


@pytest.fixture()
def lib(tmp_path, demo_dir):
    """A fresh handcrafted-backend library with 60 images ingested and embedded."""
    library = Library.create(tmp_path / "lib")
    library.ingest_paths([demo_dir])
    library.ensure_features()
    yield library
    library.close()


@pytest.fixture()
def serve_in_thread():
    """Start an HTTP server (given a factory) on an ephemeral port; stop it afterwards."""
    started = []

    def start(server):
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        started.append((server, t))
        return server.server_address[1]

    yield start
    for server, t in started:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
