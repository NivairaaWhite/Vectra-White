# Vectra

Local, trainable reverse image search. Point it at a folder (or a website you're allowed to copy
from), then search with any image to find the same picture again, even after it has been resized,
re-compressed, recoloured, watermarked or cropped. No cloud APIs, no pickles, one SQLite file.

```
pip install .                    # numpy + Pillow only
vectra build --demo 60           # try it on generated images
vectra serve --open              # browser UI at http://127.0.0.1:8765
```

For your own pictures:

```
vectra build ~/Pictures          # ingest, embed, train, index
vectra search query.jpg -k 5
vectra serve
```

## What you get

- **Search** by uploading or dragging an image, or from the command line. Each result is labelled
  `exact` (same bytes), `near-duplicate` (perceptual hash within 6 bits), `likely-copy`
  (calibrated, see below) or `similar`. "Find similar" on any result, mirrored-copy matching
  (`--flip`), compare dialog, light/dark/mobile layout.
- **Training** on your own library. A small linear head is trained so that an edited copy of an
  image lands next to its original. It is checked on images it never saw, and kept only if it beats
  the untrained baselines.
- **Calibrated "likely copy"**. At training time Vectra measures how high the best *wrong* match
  scores on held-out edited copies and sets the threshold from that, so the label means something on
  your data rather than being a fixed 0.9.
- **Crawler** for pages and image URLs: `srcset`, lazy-load, `og:image`, `<picture>`, depth and
  page budgets, robots.txt, per-host delay, size caps, and a guard against requests to local or
  private addresses. Provenance (page URL, alt text) is stored with each image.
- **Two feature extractors**: `handcrafted` (no downloads, no torch, ~2 ms/image on CPU) and
  `resnet18` / `mobilenet_v3_small` (needs torch and ImageNet weights).
- **Indexes**: exact (numpy) by default; HNSW or IVF through faiss for very large libraries.

## Commands

| command | what it does |
|---|---|
| `vectra build [paths] [--demo N] [--no-train]` | create, ingest, embed, train in one go |
| `vectra init [--backend --dim --index --weights --no-copy]` | create an empty library |
| `vectra ingest PATH...` | add images (idempotent; same bytes are never added twice) |
| `vectra crawl URL... [--depth --max-pages --max-images --delay --any-host]` | fetch images from the web |
| `vectra index [--rebuild]` | compute features for anything new |
| `vectra train [--steps --views --force]` | train the head and activate it if it wins |
| `vectra search IMAGE [-k --flip --min-score --exclude-exact --json]` | query |
| `vectra eval [--queries N]` | recall under 13 kinds of edits |
| `vectra serve [--host --port --open]` | browser UI |
| `vectra info`, `models`, `use MODEL\|raw`, `remove ID...` | housekeeping |
| `vectra fetch-weights [resnet18\|mobilenet_v3_small]` | download CNN weights (needs network) |

The library lives in `--home`, else `$VECTRA_HOME`, else `./vectra_data`. Python API:

```python
from vectra import Library
lib = Library.open_or_create("my_lib")
lib.ingest_paths(["photos/"]); lib.ensure_features(); lib.train()
for m in lib.search("query.jpg", k=5):
    print(m.title, round(m.score, 3), m.kind)
```

## How it works

1. **Backbone (frozen).** Each image becomes a feature vector once, and that vector is cached in
   SQLite. The handcrafted backbone is colour layout, RGB histograms, an oriented-gradient pyramid
   and a DCT block (1343 numbers). The CNN backbone is a ResNet with GeM pooling, always in eval
   mode so BatchNorm can't drift.
2. **Head (trained).** A linear map to `--dim` (default 256) numbers, L2-normalised. It starts from
   a partially-whitened PCA of your library, then is refined with a contrastive (InfoNCE) loss
   where the positives are randomly edited copies of the same image. Training runs on cached
   features, so it takes about a minute on a laptop CPU for thousands of images.
3. **Selection.** 20% of images are held out. For each, an edited copy is used as the query and
   the clean original must be found in the gallery. The head that is saved and activated is
   exactly the one that was scored. If training doesn't beat the best baseline, the baseline is
   kept.
4. **Index.** Cosine similarity on the normalised vectors. Rebuilt automatically when images or the
   active model change.

Models are stored as `models/<id>/{model.json, head.npz}`. The id is a hash of the numbers in the
head, and a model refuses to load against a different backbone.

## Measured results

Everything below was measured on this build, on a 2-core CPU container, handcrafted backend.

| | |
|---|---|
| Ingest (decode, hash, thumbnails) | ~770 images/s |
| Features | ~530 images/s |
| Train on 3,000 images (default settings) | 70 s |
| Search, 3,000 images, end to end | 3.6 ms |
| Exact index, 200k vectors | 1.3 ms/query |

**Held-out accuracy** (3,000 generated images, 600 held out, queries are randomly edited copies):

| head | recall@1 | recall@5 |
|---|---|---|
| raw features | 0.728 | 0.793 |
| PCA (best of three whitening settings) | 0.728 | 0.806 |
| **contrastive (kept)** | **0.948** | **0.982** |

**Real photos.** 23 real photographs bundled with scikit-image, scikit-learn and matplotlib, using
a head trained only on generated images. Mean recall@1 over the 12 edit types went from 0.917 to
0.989. The gallery is tiny, so treat this as a sanity check, not a benchmark; `vectra eval` warns
about this itself. The raw features struggled most with flips (0.61), borders (0.74) and mixed
edits (0.74); the trained head took all three to 0.91 or better.

## Limitations, honestly

- **The ResNet backend has not been validated.** There was no network access to the ImageNet
  weights while building this, so the CNN path is tested for correctness (frozen, deterministic,
  trained end to end with random weights) but not for retrieval quality. Run `vectra fetch-weights`,
  create a library with `--backend resnet18`, and compare `vectra eval` against the handcrafted
  one on your own images before relying on it.
- **Accuracy numbers come from generated images** and from 23 real ones. Your photos will differ.
  Run `vectra eval` on your own library; it is the only number that matters for you.
- **`vectra eval` on the library you trained on is optimistic.** It queries images the head saw
  during training. The held-out figure printed by `vectra train` is the honest one.
- **Mirrored copies** are handled in training and by `--flip`, so the eval's `flip` row overlaps
  with the training distribution.
- **"Likely copy" is calibrated on edited copies**, about 1% false alarms on held-out data. Two
  different photos from the same burst, or the same scene shot twice, can still score high.
- **Not a face, object or semantic search.** It finds the same picture, or close variants of it.
  With a CNN backbone it also finds visually similar pictures, but that is a bonus.
- **HNSW is approximate.** On 200k random vectors it found the true nearest neighbour 89% of the
  time (random vectors are a hard case). The exact index is the default up to 200k images for that
  reason.
- Tested on Python 3.13 (full suite, with torch and faiss) and Python 3.11 (everything that doesn't need torch, in a clean
  virtualenv with only numpy and Pillow). Declared support is 3.9+, but 3.9 and 3.10 were not run.
- **Crawl only sites you have the right to copy from.** Vectra obeys robots.txt and rate-limits,
  but that is politeness, not permission.

## Security notes

The web UI is meant for your own machine. It binds to `127.0.0.1`, is read-only, checks the `Host`
header (against DNS rebinding), requires a custom header on uploads (against cross-site posts),
sends a strict Content-Security-Policy, serves images only by integer id and thumbnails only in
three fixed sizes. The UI script never uses `innerHTML`, since crawled alt text is untrusted. The
crawler refuses non-http(s) URLs and non-public addresses, revalidating at every redirect. Nothing
is unpickled: arrays are loaded with `allow_pickle=False`, and weights with `weights_only=True`.

If you bind to anything other than localhost with `--host`, anyone on that network can search your
library.

## Development

```
pip install -e ".[dev]"      # add ,torch for the training and CNN tests
pytest                       # 87 tests; one to two minutes with torch installed
```

Without torch, the torch tests are skipped and the rest run.

Tests include a regression test for each defect found in v1: random projection heads that differed
between build and search, indexes rebuilt from only the newest batch, trained models discarded
after validation, BatchNorm drift in the backbone, and unvalidated IVF settings.

MIT licensed.
