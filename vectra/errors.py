"""Exception types. Everything user-facing derives from VectraError so the CLI can
print a clean one-line message instead of a traceback."""


class VectraError(Exception):
    """Base class for expected, user-facing failures."""


class ImageError(VectraError):
    """The bytes are not a usable image (corrupt, unsupported, absurdly large)."""


class LibraryNotFound(VectraError):
    """No Vectra library at the requested location."""


class BackendUnavailable(VectraError):
    """A backend / optional dependency is not installed."""


class WeightsNotFound(VectraError):
    """Pretrained weights are not available locally (we never download implicitly)."""


class ModelMismatch(VectraError):
    """The active model was trained for a different feature extractor."""


class TrainingError(VectraError):
    """Training cannot run (too little data, missing torch, ...)."""
