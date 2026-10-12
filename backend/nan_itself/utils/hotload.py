"""
Shared hot-reload primitives.

One module, two halves:

    detection   directory fingerprints and per-source failure
                bookkeeping with escalating retry (SourceTracker)
    loading     folder-as-package dynamic import with
                per-submodule vocabulary seeding

Folder contract (modules and local tools):

    <root>/<name>/
        main.py       entry; defines exactly one entity class
        other.py      siblings, seeded with the same vocabulary
        config.yaml   optional entity config
        data/         NOT detected: models, databases, state

`data/` is excluded from detection everywhere, so runtime writes
never trigger a reload loop and heavyweight payloads (model
checkouts carrying thousands of .py files) never enter the
package. `__pycache__` and dot-prefixed segments are excluded
too; the loaders below also never WRITE bytecode caches, since
their coarse mtime+size key would silently run stale code after
a same-tick edit.

The transactional rule enforced by every consumer:

    build the candidate first; only touch live state once every
    candidate step succeeded.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import itertools
import sys
import time
from importlib.abc import Loader, MetaPathFinder
from pathlib import Path
from types import ModuleType

from .backoff import next_backoff

# Escalating retry schedule for repeated failures of one source:
# the first retry waits 30s, then doubles, and repeated failures
# keep waiting 5 minutes.
BACKOFF_SCHEDULE: tuple[float, ...] = (
    30.0,
    60.0,
    120.0,
    240.0,
    300.0,
)


# ============================================================================
# Fingerprints
# ============================================================================


def file_fingerprint(
    path: Path,
) -> tuple[int, int]:
    stat = path.stat()

    return (
        stat.st_mtime_ns,
        stat.st_size,
    )


def detected_files(
    root: Path,
    *,
    exclude_dirs: tuple[str, ...] = ("data",),
) -> tuple[Path, ...]:
    """
    Enumerate every file belonging to one entity directory.

    Hidden entries (dot-prefixed path segments), `__pycache__`
    and any path segment named in `exclude_dirs` are excluded;
    ordering is stable. Skills pass no `exclude_dirs`.
    """
    if not root.is_dir():
        return ()

    return tuple(
        sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root).parts
            )
            and not any(part in exclude_dirs for part in path.relative_to(root).parts[:-1])
        )
    )


def directory_fingerprint(
    root: Path,
    *,
    exclude_dirs: tuple[str, ...] = ("data",),
) -> tuple[int, int]:
    """
    Aggregate change fingerprint over every detected file.

    A single-file fingerprint would miss edits to siblings,
    config or bundled resources, so the whole directory is
    folded into one value: summed mtimes and sizes plus the
    file count (so additions and removals are detected too).
    An empty or absent directory folds to (0, 0).
    """
    files = detected_files(
        root,
        exclude_dirs=exclude_dirs,
    )

    if not files:
        return (0, 0)

    total_mtime = 0

    total_size = 0

    for path in files:
        try:
            stat = path.stat()

        except OSError:
            continue

        total_mtime += stat.st_mtime_ns

        total_size += stat.st_size

    count = len(files)

    return (
        total_mtime + count,
        total_size + count,
    )


# ============================================================================
# SourceTracker
# ============================================================================


class SourceTracker:
    """
    Change tracking for hot-reloadable sources.

    The key is the source identity -- a file path for MCP
    configs, a folder path for modules and local tools.
    """

    def __init__(self) -> None:
        # Source -> latest fingerprint.
        self.fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        # Source -> load error for the current fingerprint.
        self.errors: dict[
            Path,
            BaseException,
        ] = {}

        # Source -> (monotonic failed_at, attempts) of the
        # current fingerprint, used to schedule retries.
        self.failures: dict[
            Path,
            tuple[float, int],
        ] = {}

    def needs_load(
        self,
        source: Path,
        fingerprint: tuple[int, int],
        *,
        loaded: bool,
    ) -> bool:
        """
        Decide whether a source must be (re)loaded.

        True when the source changed, when it has no recorded
        outcome for this fingerprint yet, or when the backoff
        window since the last failure has elapsed.
        """
        previous = self.fingerprints.get(source)

        if previous != fingerprint:
            return True

        if loaded:
            return False

        if self.errors.get(source) is None:
            return True

        return self._backoff_elapsed(source)

    def _backoff_elapsed(
        self,
        source: Path,
    ) -> bool:
        failure = self.failures.get(source)

        if failure is None:
            return True

        failed_at, attempts = failure

        delay, _ = next_backoff(
            BACKOFF_SCHEDULE,
            attempts - 1,
        )

        return (time.monotonic() - failed_at) >= delay

    def mark_seen(
        self,
        source: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        # A fingerprint change resets the retry schedule; the
        # same fingerprint keeps its attempt count so repeated
        # failures keep backing off.
        if self.fingerprints.get(source) != fingerprint:
            self.failures.pop(source, None)

        self.fingerprints[source] = fingerprint

        self.errors.pop(source, None)

    def mark_failed(
        self,
        source: Path,
        error: BaseException,
    ) -> None:
        self.errors[source] = error

        previous = self.failures.get(source)

        if previous is None:
            self.failures[source] = (
                time.monotonic(),
                1,
            )

        else:
            _, attempts = previous

            self.failures[source] = (
                time.monotonic(),
                attempts + 1,
            )

    def forget(self, source: Path) -> None:
        self.fingerprints.pop(source, None)

        self.errors.pop(source, None)

        self.failures.pop(source, None)

    def known_keys(self) -> set[Path]:
        return set(self.fingerprints)


# ============================================================================
# Folder package loading
# ============================================================================

# Every load generation gets a fresh package root name, so each
# reload builds brand-new class objects in a brand-new namespace
# and eviction is a simple prefix sweep.
_GENERATION = itertools.count()

# Live package root name -> its finder, for eviction.
_FINDERS: dict[
    str,
    _FolderFinder,
] = {}


class _FolderLoader(Loader):
    """
    Execute one sibling file of a folder package.

    The module namespace is seeded with the vocabulary before
    exec, so entity files work without any nan_itself imports;
    an explicit import of the same name simply shadows it.
    Source is compiled fresh on every exec -- never through the
    import system's bytecode cache.
    """

    def __init__(
        self,
        file: Path,
        seeds: dict[str, object],
    ) -> None:
        self.file = file

        self.seeds = seeds

    def create_module(
        self,
        spec: importlib.machinery.ModuleSpec,
    ) -> ModuleType:
        module = ModuleType(spec.name)

        module.__file__ = str(self.file)

        module.__package__ = spec.name.rpartition(".")[0]

        for name, value in self.seeds.items():
            module.__dict__[name] = value

        return module

    def exec_module(
        self,
        module: ModuleType,
    ) -> None:
        source = self.file.read_text(encoding="utf-8")

        code = compile(
            source,
            self.file.as_posix(),
            "exec",
        )

        exec(
            code,
            module.__dict__,
        )


class _FolderFinder(MetaPathFinder):
    """
    Resolve `<pkg>.<sibling>` imports against one folder.

    Only top-level sibling files are importable: `data/` and
    everything nested below is outside the package by design.
    """

    def __init__(
        self,
        pkg_name: str,
        folder: Path,
        seeds: dict[str, object],
    ) -> None:
        self.pkg_name = pkg_name

        self.siblings = {
            path.stem: path
            for path in sorted(folder.glob("*.py"))
            if path.is_file() and not path.stem.startswith("_")
        }

        self.seeds = seeds

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        prefix = self.pkg_name + "."

        if not fullname.startswith(prefix):
            return None

        rest = fullname[len(prefix) :]

        if "." in rest:
            return None

        file = self.siblings.get(rest)

        if file is None:
            return None

        return importlib.util.spec_from_file_location(
            fullname,
            file,
            loader=_FolderLoader(
                file,
                self.seeds,
            ),
        )


def load_folder_package(
    folder: Path,
    *,
    kind: str,
    seeds: dict[str, object],
) -> tuple[str, ModuleType]:
    """
    Import one entity folder as a synthetic package; run main.py.

    Siblings import each other with normal package syntax
    (`from . import helpers`) and are executed on demand by the
    import machinery, seeded with the given vocabulary. The
    package root name is unique per load generation.

    Returns (package_root_name, main_module). On any failure the
    whole package is evicted before the exception propagates.
    """
    folder = folder.resolve()

    main_file = folder / "main.py"

    if not main_file.is_file():
        raise FileNotFoundError(f"{folder} has no main.py")

    pkg_name = f"_nan_pkg_{kind}_{folder.name}_{next(_GENERATION)}"

    pkg = ModuleType(pkg_name)

    pkg.__file__ = str(folder / "__init__.py")

    pkg.__path__ = [str(folder)]

    pkg.__package__ = pkg_name

    sys.modules[pkg_name] = pkg

    finder = _FolderFinder(
        pkg_name,
        folder,
        seeds,
    )

    _FINDERS[pkg_name] = finder

    sys.meta_path.insert(
        0,
        finder,
    )

    try:
        main = importlib.import_module(f"{pkg_name}.main")

    except BaseException:
        evict_package(pkg_name)

        raise

    return (
        pkg_name,
        main,
    )


def evict_package(pkg_name: str | None) -> None:
    """
    Remove one load generation completely.

    Sweeps every sys.modules entry under the package root and
    uninstalls its finder. Safe to call twice or with None.
    """
    if not pkg_name:
        return

    finder = _FINDERS.pop(pkg_name, None)

    if finder is not None:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)

    for name in [
        name for name in sys.modules if name == pkg_name or name.startswith(pkg_name + ".")
    ]:
        sys.modules.pop(name, None)
