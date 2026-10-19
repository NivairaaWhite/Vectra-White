"""Command-line interface:  vectra <command> [options]   (also:  python -m vectra ...)"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from . import __version__
from .config import BACKENDS, INDEX_KINDS, Settings, resolve_home
from .demo import make_demo_images
from .embedder import list_models
from .errors import TrainingError, VectraError
from .library import Library


# --- small output helpers --------------------------------------------------------------------
def say(msg: str = "") -> None:
    print(msg, flush=True)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def progress_bar(label: str) -> Callable[[int, int], None]:
    """Overwriting bar on a terminal, a line every ~10 % otherwise (CI logs, pipes)."""
    tty = sys.stderr.isatty()
    state = {"t": 0.0, "decile": -1}

    def update(done: int, total: int) -> None:
        if total <= 0:
            return
        if tty:
            now = time.time()
            if done < total and now - state["t"] < 0.1:
                return
            state["t"] = now
            width = 28
            fill = int(width * done / total)
            sys.stderr.write(f"\r{label} [{'#' * fill}{'.' * (width - fill)}] {done}/{total}")
            if done >= total:
                sys.stderr.write("\n")
            sys.stderr.flush()
        else:
            decile = int(10 * done / total)
            if decile != state["decile"]:
                state["decile"] = decile
                say(f"{label}: {done}/{total}")

    return update


def open_lib(args) -> Library:
    return Library.open(resolve_home(args.home))


def pick_auto_backend(home: Path) -> str:
    """resnet18 when torch and local weights are both present, otherwise the zero-download backend."""
    try:
        from .backends.resnet import find_weights
        find_weights("resnet18", home, None)
        return "resnet18"
    except Exception:
        return "handcrafted"


def create_or_open(args, quiet: bool = False) -> Library:
    home = resolve_home(args.home)
    if Library.exists(home):
        return Library.open(home)
    backend = getattr(args, "backend", None) or "auto"
    if backend == "auto":
        backend = pick_auto_backend(home)
    lib = Library.create(
        home, backend=backend, embed_dim=getattr(args, "dim", None), index_kind=getattr(args, "index", None),
        weights_path=getattr(args, "weights", None),
        copy_images=not getattr(args, "no_copy", False),
    )
    if not quiet:
        say(f"Created library at {home}")
        say(f"  feature extractor: {backend}" + (
            "  (no downloads needed; pass --backend resnet18 after `vectra fetch-weights` for CNN features)"
            if backend == "handcrafted" else ""))
    return lib


# --- commands ---------------------------------------------------------------------------------
def cmd_init(args) -> int:
    home = resolve_home(args.home)
    if Library.exists(home):
        s = Settings.load(home)
        say(f"A library already exists at {home} (backend {s.backend}, active model {s.active_model or 'raw'}).")
        say("Nothing changed.")
        return 0
    create_or_open(args)
    say("Next:  vectra ingest <folder>   then   vectra train   then   vectra serve")
    return 0


def _embed_new(lib: Library) -> int:
    rep = lib.ensure_features(progress=progress_bar("Embedding"))
    if rep.failed:
        say(f"  {len(rep.failed)} images could not be read (moved or corrupt):")
        for i, why in rep.failed[:5]:
            say(f"    id {i}: {why}")
    return rep.done


def cmd_ingest(args) -> int:
    lib = create_or_open(args)
    rep = lib.ingest_paths(args.paths, copy=False if args.no_copy else None, progress=progress_bar("Reading"))
    say(f"Scanned {rep.scanned} files: {rep.added} added, {rep.duplicates} already in the library, "
        f"{len(rep.invalid)} unreadable.")
    for path, why in rep.invalid[:10]:
        say(f"  skipped {path}: {why}")
    if not args.no_embed and rep.added:
        _embed_new(lib)
    elif rep.added:
        say("Images stored but not embedded yet; run `vectra index` before searching.")
    return 0


def cmd_crawl(args) -> int:
    from .crawler import Crawler
    lib = create_or_open(args)
    crawler = Crawler(
        lib, depth=args.depth, max_pages=args.max_pages, max_images=args.max_images, delay=args.delay,
        min_side=args.min_size, same_host=not args.any_host, allow_private=args.allow_private,
        log=say if args.verbose else None)
    say("Crawling (robots.txt is honoured; only crawl sites you may copy images from)...")
    rep = crawler.crawl(args.urls)
    say(rep.summary())
    for e in rep.errors[:8]:
        say(f"  {e}")
    if len(rep.errors) > 8:
        say(f"  ... and {len(rep.errors) - 8} more (use -v to see pages as they are fetched)")
    if rep.added and not args.no_embed:
        _embed_new(lib)
    return 0


def cmd_index(args) -> int:
    lib = open_lib(args)
    t0 = time.time()
    if args.rebuild:
        dropped = lib.store.drop_features()
        lib._invalidate_index()
        say(f"Dropped {dropped} cached feature rows; recomputing.")
    _embed_new(lib)
    idx = lib.get_index()
    say(f"Index ready: {len(idx)} images, {idx.kind}, {idx.dim}-d, model {lib.embedder.model_id} "
        f"[{time.time() - t0:.1f}s]" + (f"  ({idx.note})" if idx.note else ""))
    return 0


def cmd_train(args) -> int:
    lib = open_lib(args)
    try:
        rep = lib.train(steps=args.steps, batch=args.batch, views=args.views, max_images=args.max_images,
                        seed=args.seed, dim=args.dim, force=args.force, log=say)
    except TrainingError as e:
        say(f"Cannot train: {e}")
        return 1
    say()
    say(rep.summary())
    if rep.chosen != "raw":
        say(f"Active model is now {rep.model_id}. Search uses it immediately; nothing needs re-indexing.")
    return 0


def cmd_search(args) -> int:
    lib = open_lib(args)
    if lib.store.ids_missing_features(lib.backend.backend_id):
        _embed_new(lib)
    res = lib.search(args.image, k=args.k, flip=args.flip, min_score=args.min_score,
                     exclude_exact=args.exclude_exact)
    if args.json:
        say(json.dumps([m.to_dict() for m in res], indent=2))
        return 0
    if not res:
        say("No matches.")
        return 0
    def short(s: str, n: int = 28) -> str:
        return s if len(s) <= n else s[: n - 1] + "…"

    say(f"{'#':>2}  {'score':>6}  {'match':<15} {'size':>11}  {'name':<28}  stored at")
    for i, m in enumerate(res, 1):
        name = short(m.title or Path(m.path).name)
        say(f"{i:>2}  {m.score:>6.3f}  {m.kind:<15} {m.width:>5}x{m.height:<5}  {name:<28}  {m.path}")
        if m.source_url and m.source_url.startswith("http"):
            say(f"{'':>2}  {'':>6}  {'':<15} {'':>11}  {'':<28}  from {m.source_url}")
    return 0


def cmd_eval(args) -> int:
    lib = open_lib(args)
    rep = lib.evaluate(n_queries=args.queries, seed=args.seed, k=10, progress=None)
    if args.json:
        say(json.dumps(rep.to_dict(), indent=2))
        return 0
    say(f"Library: {rep.gallery} images, features {rep.backend}, model {rep.model}; {rep.n_queries} queries per edit")
    say()
    say(rep.table())
    say()
    for n in rep.notes:
        say(f"note: {n}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    lib = open_lib(args)
    serve(lib, host=args.host, port=args.port, open_browser=args.open, say=say)
    return 0


def cmd_info(args) -> int:
    lib = open_lib(args)
    s = lib.stats()
    say(f"Vectra {__version__}   library: {s['home']}")
    say(f"  images           {s['images']}  ({human_bytes(s['bytes'])}; sources: "
        f"{', '.join(f'{k} {v}' for k, v in s['sources'].items()) or 'none'})")
    say(f"  feature backend  {s['backend']}" + (f"   [{s['backend_error']}]" if "backend_error" in s else ""))
    if "embedded" in s:
        say(f"  embedded         {s['embedded']}  ({s['pending']} pending)")
        say(f"  active model     {s['active_model']}  ({s['model_kind']}, {s['dim']}-d)")
    say(f"  index            {s['index_kind_setting']} (faiss {'available' if s['faiss'] else 'not installed'})")
    return 0


def cmd_models(args) -> int:
    lib = open_lib(args)
    models = list_models(lib.models_dir)
    active = lib.settings.active_model or "raw"
    say(f"{'':<2}{'model':<22}{'kind':<9}{'backend':<14}{'held-out MRR':>13}  trained")
    say(f"{'*' if active == 'raw' else ' ':<2}{'raw':<22}{'raw':<9}{lib.settings.backend:<14}{'':>13}  (no head: backend features only)")
    for m in models:
        mrr = (m.get("info", {}).get("heldout") or {}).get("mrr")
        when = datetime.fromtimestamp(m["created"]).strftime("%Y-%m-%d %H:%M")
        say(f"{'*' if m['model_id'] == active else ' ':<2}{m['model_id']:<22}{m['kind']:<9}{m['backend']:<14}"
            f"{(f'{mrr:.3f}' if mrr is not None else '-'):>13}  {when}")
    return 0


def cmd_use(args) -> int:
    lib = open_lib(args)
    emb = lib.use_model(args.model)
    say(f"Active model: {emb.model_id} ({emb.kind}, {emb.dim}-d)")
    return 0


def cmd_remove(args) -> int:
    lib = open_lib(args)
    n = lib.remove(args.ids)
    say(f"Removed {n} image(s) from the library (files you ingested with --no-copy are untouched).")
    return 0


def cmd_demo(args) -> int:
    paths = make_demo_images(args.dir, n=args.n, seed=args.seed)
    say(f"Wrote {len(paths)} procedural demo images to {args.dir}")
    say(f"Try:  vectra ingest {args.dir}   then   vectra train   then   vectra serve")
    return 0


def cmd_fetch_weights(args) -> int:
    from .backends.resnet import fetch_weights
    home = resolve_home(args.home)
    home.mkdir(parents=True, exist_ok=True)
    say(f"Downloading {args.name} ImageNet weights (one time, needs internet)...")
    path = fetch_weights(args.name, home)
    say(f"Saved {path}")
    say(f"Use them with:  vectra init --backend {args.name}   (for a new library)")
    return 0


def cmd_build(args) -> int:
    """One command: create -> ingest -> embed -> train."""
    lib = create_or_open(args)
    sources: List[str] = list(args.paths or [])
    tmp = None
    try:
        if args.demo:
            tmp = tempfile.mkdtemp(prefix="vectra_demo_")
            make_demo_images(tmp, n=args.demo, seed=0)
            say(f"Generated {args.demo} procedural demo images.")
            rep = lib.ingest_paths([tmp], copy=True, source="demo", progress=progress_bar("Reading"))
            say(f"Demo images: {rep.added} added.")
        if sources:
            rep = lib.ingest_paths(sources, copy=False if args.no_copy else None, progress=progress_bar("Reading"))
            say(f"Scanned {rep.scanned} files: {rep.added} added, {rep.duplicates} duplicates, {len(rep.invalid)} unreadable.")
        if lib.store.count() == 0:
            say("The library is empty. Give `vectra build` some folders, or use --demo 60.")
            return 1
        _embed_new(lib)
        if not args.no_train:
            try:
                say()
                say(lib.train(log=say).summary())
            except TrainingError as e:
                say(f"Skipping training: {e}")
        idx = lib.get_index()
        say()
        say(f"Ready: {len(idx)} images, {idx.kind} index, model {lib.embedder.model_id}.")
        say("Search:  vectra search path/to/image.jpg      Browser UI:  vectra serve")
        return 0
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# --- argument parsing --------------------------------------------------------------------------
def _common_init_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("auto",) + BACKENDS, default=None,
                   help="feature extractor for a NEW library (default: auto)")
    p.add_argument("--dim", type=int, default=None, help="embedding size for a new library (default 256)")
    p.add_argument("--index", choices=INDEX_KINDS, default=None, help="index type for a new library (default auto)")
    p.add_argument("--weights", default=None, help="path to CNN weights (.pth) for a new library")
    p.add_argument("--no-copy", action="store_true", help="reference images in place instead of copying them in")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vectra", description="Local, trainable reverse image search.",
        epilog="Quick start:  vectra build --demo 60  &&  vectra serve")
    ap.add_argument("--home", default=None, help="library folder (default: $VECTRA_HOME or ./vectra_data)")
    ap.add_argument("--version", action="version", version=f"vectra {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, fn, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        return sp

    p = add("init", cmd_init, "create an empty library")
    _common_init_args(p)

    p = add("build", cmd_build, "create, ingest, embed and train in one go")
    p.add_argument("paths", nargs="*", help="image files / folders")
    p.add_argument("--demo", type=int, default=0, metavar="N", help="also add N generated demo images")
    p.add_argument("--no-train", action="store_true")
    _common_init_args(p)

    p = add("ingest", cmd_ingest, "add images from files or folders (safe to re-run)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--no-embed", action="store_true", help="store only; embed later with `vectra index`")
    _common_init_args(p)

    p = add("crawl", cmd_crawl, "download images from web pages (honours robots.txt)")
    p.add_argument("urls", nargs="+")
    p.add_argument("--depth", type=int, default=1, help="how many link levels to follow (default 1)")
    p.add_argument("--max-pages", type=int, default=20)
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests to one host (default 1)")
    p.add_argument("--min-size", type=int, default=64, help="skip images with a side shorter than this (default 64)")
    p.add_argument("--any-host", action="store_true", help="follow links to other hosts (default: stay on the seed's host)")
    p.add_argument("--allow-private", action="store_true", help="allow localhost/LAN addresses (servers you control)")
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    _common_init_args(p)

    p = add("index", cmd_index, "embed any images that aren't yet, and build the search index")
    p.add_argument("--rebuild", action="store_true", help="discard cached features and recompute everything")

    p = add("train", cmd_train, "learn a head on your images (PCA + contrastive), validated on held-out images")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--views", type=int, default=6, help="edited copies generated per image (default 6)")
    p.add_argument("--max-images", type=int, default=4000)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="keep the contrastive head even if it doesn't beat the baseline")

    p = add("search", cmd_search, "find library images similar to a query image")
    p.add_argument("image")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--flip", action="store_true", help="also match mirrored copies")
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--exclude-exact", action="store_true", help="hide the stored copy of the query itself")
    p.add_argument("--json", action="store_true")

    p = add("eval", cmd_eval, "measure how well the library finds edited copies of its own images")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")

    p = add("serve", cmd_serve, "start the local web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open the browser")

    add("info", cmd_info, "show library statistics")
    add("models", cmd_models, "list trained models")
    p = add("use", cmd_use, "switch the active model ('raw' = no trained head)")
    p.add_argument("model")
    p = add("remove", cmd_remove, "remove images from the library by id")
    p.add_argument("ids", nargs="+", type=int)
    p = add("demo", cmd_demo, "write procedural demo images to a folder")
    p.add_argument("dir", nargs="?", default="demo_images")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p = add("fetch-weights", cmd_fetch_weights, "one-time download of CNN weights (the only command that needs internet)")
    p.add_argument("name", nargs="?", default="resnet18", choices=("resnet18", "mobilenet_v3_small"))
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    try:
        return int(args.fn(args) or 0)
    except VectraError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # e.g. `vectra search ... | head`
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
