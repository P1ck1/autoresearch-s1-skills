"""Pure-Python enforcement gate.

This is a *from-scratch, numpy+scipy-only* task, so the rule is an **allowlist**, not a
blacklist: a submission may import the standard library, `numpy`, `scipy`, and its own local
modules -- nothing else. That closes the whole family of blacklist bypasses (an unlisted
package, a renamed import, `import numpy, numba` on one line) in one move.

Imports are found by **parsing the AST**, not by regex over source text, so:
  * every imported module on a line is seen (`import a, b`; `from x import y`; aliases);
  * `__import__("numba")` and `importlib.import_module("numba")` with a literal name are caught;
  * a forbidden name mentioned in a *docstring or comment* is NOT a false positive (the AST only
    sees real imports), and a genuine local parser named `vcf.py` imported as `import vcf` is
    recognised as local, not as PyVCF.

We also require the entry program to be a Python script and reject native payloads (`.so`,
`.pyd`, compiled archives, ELF binaries) -- the task forbids dropping to another language, and
the image ships no compiler, so a native artefact is prima facie a violation. Data containers are
not executable code: numeric ``.npy``/``.npz`` files and data-only zip/tar/compressed files are
accepted after their members are inspected. Shipping a precomputed numeric table is an ordinary
algorithmic choice, not an external dependency.
"""

from __future__ import annotations

import ast
import bz2
import gzip
import importlib.util
import lzma
import stat
import struct
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

# The whole allowed surface: Python 3.12's standard library plus the two permitted third-party
# packages. The environment provisions Python 3.12 explicitly, so a compatibility table for older
# interpreters would only create a second, drifting definition of the dependency boundary.
_STDLIB = set(sys.stdlib_module_names)
_ALLOWED_THIRD_PARTY = {"numpy", "scipy"}

# Stdlib modules that are otherwise allowlisted but enable the *specific* escapes the task bans:
# ``ctypes``/``_ctypes`` (the prompt rejects ctypes outright) and ``_imp`` load native modules;
# importlib/zipimport/runpy load code from a name or container, and ``marshal`` deserializes raw
# code objects -- all of them "reconstruct an executable payload," which the prompt bans, and none
# is needed for direct NumPy/SciPy PCA. ``subprocess``/``os`` remain source-allowlisted because the
# trusted runtime separately blocks child exec/spawn while preserving Linux fork workers.
#
# The pickle family (``pickle``/``_pickle``/``shelve``) is DELIBERATELY not here. The prompt bans
# reconstructed *executable/native* payloads, not object serialization, and a hand-rolled fork
# worker legitimately pickles NumPy arrays over a pipe (this is what ``multiprocessing`` does
# internally). Its only integrity risk -- a hostile ``__reduce__`` -- cannot exceed the runtime:
# the audit hook (which CPython cannot unregister) still blocks a forbidden import, ``ctypes``,
# ``os``/``subprocess`` exec, and ``code``/``function.__new__`` object construction from inside an
# unpickle exactly as from ordinary code, and a shipped pickle *blob* is still rejected here at file
# level (``_PICKLE_SUFFIXES`` and object-dtype ``.npy``). Denying the module name added nothing the
# runtime does not already enforce and silently zeroed a correct fork-IPC solver -- the same
# static/runtime disagreement that retired the blanket ``exec``/``eval``/``compile`` hit below.
_DENY_STDLIB = {
    "_ctypes", "_imp", "ctypes", "importlib", "marshal", "runpy", "zipimport",
}

_PY_SUFFIXES = {".py", ".pyw"}
_BYTECODE_SUFFIXES = {".pyc", ".pyo"}
_NATIVE_SUFFIXES = {".so", ".pyd", ".dll", ".dylib", ".o", ".a"}
_PICKLE_SUFFIXES = {".pickle", ".pkl"}
_PACKAGE_ARCHIVE_SUFFIXES = {".egg", ".pyz", ".whl"}
_ZIP_SUFFIXES = {".npz", ".zip"}
_COMPRESSED_SUFFIXES = {".bz2", ".gz", ".xz"}
_DYNAMIC_IMPORTERS = {"__import__", "import_module"}

_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_NATIVE_MAGIC = (
    b"\x7fELF", b"MZ", b"\x00asm", b"!<arch>\n",
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",
)
_MAX_ARCHIVE_MEMBERS = 4096
_MAX_NPY_HEADER = 1 << 20
# Archive members are solver-controlled compressed inputs to this trusted scanner.  The on-disk
# submission limit alone is not sufficient: a few kilobytes of DEFLATE data can otherwise make
# the root-owned verifier expand gigabytes while merely deciding whether an archive is data-only.
# Keep these limits comfortably above ordinary lookup tables while bounding both one-member and
# many-member bombs.  The ratio limit is deliberately independent of the byte limits so malformed
# archives with implausible metadata are rejected before a member stream is opened.
_MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_COMPRESSION_RATIO = 512
_ARCHIVE_READ_CHUNK = 1 << 20


class _ArchiveLimitError(Exception):
    """An archive exceeded a trusted-scanner resource bound."""


class _BoundedArchiveReader:
    """Count decompressed bytes and fail before one member can exceed its cap.

    Metadata checks are the cheap first line of defence, but the stream counter is intentional:
    central-directory/header sizes are untrusted too.  Scanner helpers only need ``read`` and use
    bounded requests, so exposing the rest of the archive stream would add unnecessary surface.
    """

    def __init__(self, stream, limit: int):
        self._stream = stream
        self._limit = limit
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = _ARCHIVE_READ_CHUNK
        # Read at most one byte beyond the remaining budget.  This detects overflow without ever
        # asking a decompressor to materialize an attacker-chosen unbounded result.
        size = min(size, self._limit - self.count + 1)
        if size <= 0:
            raise _ArchiveLimitError("archive member exceeds decompressed-byte limit")
        data = self._stream.read(size)
        self.count += len(data)
        if self.count > self._limit:
            raise _ArchiveLimitError("archive member exceeds decompressed-byte limit")
        return data


def _compression_ratio_allowed(uncompressed: int, compressed: int) -> bool:
    if uncompressed < 0 or compressed < 0:
        return False
    if uncompressed == 0:
        return True
    if compressed == 0:
        return False
    return uncompressed <= compressed * _MAX_ARCHIVE_COMPRESSION_RATIO


def _inspect_archive_member(name: str, stream, *, expected_size: int | None,
                            compressed_size: int | None) -> str | None:
    """Inspect and fully drain one member through the decompression byte counter."""
    if expected_size is not None:
        if expected_size > _MAX_ARCHIVE_MEMBER_BYTES:
            return "archive-payload"
        if compressed_size is not None and not _compression_ratio_allowed(
                expected_size, compressed_size):
            return "archive-payload"
    bounded = _BoundedArchiveReader(stream, _MAX_ARCHIVE_MEMBER_BYTES)
    unsafe = _embedded_payload_kind(name, bounded)
    if unsafe is not None:
        return unsafe
    while bounded.read(_ARCHIVE_READ_CHUNK):
        pass
    if expected_size is not None and bounded.count != expected_size:
        return "archive-payload"
    if compressed_size is not None and not _compression_ratio_allowed(
            bounded.count, compressed_size):
        return "archive-payload"
    return None


def _is_python_shebang(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            first = fh.readline(256)
    except OSError:
        return False
    return first.startswith(b"#!") and b"python" in first


def _npy_has_object_dtype(prefix: bytes, stream) -> bool | None:
    """Return whether an NPY stream contains pickle-backed Python objects.

    ``numpy.load(..., allow_pickle=True)`` turns an object array into an encoded Python payload,
    while ordinary numeric arrays are benign helper data. Parse only the bounded public NPY
    header; never deserialize the array itself.
    """
    if not prefix.startswith(b"\x93NUMPY"):
        return None
    if len(prefix) < 10:
        prefix += stream.read(10 - len(prefix))
    if len(prefix) < 10:
        return None
    major = prefix[6]
    if major == 1:
        length_width, encoding = 2, "latin1"
    elif major in (2, 3):
        length_width, encoding = 4, "utf8" if major == 3 else "latin1"
    else:
        return None
    header_start = 8 + length_width
    if len(prefix) < header_start:
        prefix += stream.read(header_start - len(prefix))
    if len(prefix) < header_start:
        return None
    fmt = "<H" if length_width == 2 else "<I"
    header_size = struct.unpack(fmt, prefix[8:header_start])[0]
    if header_size > _MAX_NPY_HEADER:
        return None
    needed = header_start + header_size
    if len(prefix) < needed:
        prefix += stream.read(needed - len(prefix))
    if len(prefix) < needed:
        return None
    try:
        metadata = ast.literal_eval(prefix[header_start:needed].decode(encoding))
        dtype = np.dtype(metadata["descr"])
    except (KeyError, SyntaxError, TypeError, UnicodeDecodeError, ValueError):
        return None
    return bool(dtype.hasobject)


def _embedded_payload_kind(name: str, stream) -> str | None:
    """Inspect one archive/compressed member without materializing its data body."""
    suffix = Path(name).suffix.lower()
    if suffix in _PY_SUFFIXES or suffix in _PICKLE_SUFFIXES:
        return "encoded-python-payload"
    if suffix in _BYTECODE_SUFFIXES:
        return "bytecode-payload"
    if suffix in _NATIVE_SUFFIXES:
        return "native-payload"
    if suffix in (_PACKAGE_ARCHIVE_SUFFIXES | _ZIP_SUFFIXES | _COMPRESSED_SUFFIXES |
                  {".tar", ".tgz"}):
        return "archive-payload"

    try:
        prefix = stream.read(512)
    except (EOFError, OSError, RuntimeError):
        return "archive-payload"
    if prefix.startswith(importlib.util.MAGIC_NUMBER):
        return "bytecode-payload"
    if prefix.startswith(_NATIVE_MAGIC):
        return "native-payload"
    if prefix.startswith(_ZIP_MAGIC) or prefix.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00")):
        return "archive-payload"
    if len(prefix) >= 262 and prefix[257:262] == b"ustar":
        return "archive-payload"
    if prefix.startswith(b"#!") and b"python" in prefix.splitlines()[0].lower():
        return "encoded-python-payload"
    if suffix == ".npy" or prefix.startswith(b"\x93NUMPY"):
        has_object = _npy_has_object_dtype(prefix, stream)
        if has_object is None:
            return "archive-payload"
        if has_object:
            return "encoded-python-payload"
    return None


def _safe_archive_member_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    return (
        bool(normalized)
        and not normalized.startswith("/")
        and ".." not in parts
        and "\x00" not in name
    )


def _zip_payload_kind(path: Path, *, require_npy_members: bool) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                return "archive-payload"
            files = [member for member in members if not member.is_dir()]
            total_size = sum(member.file_size for member in files)
            total_compressed = sum(member.compress_size for member in files)
            if (total_size > _MAX_ARCHIVE_TOTAL_BYTES
                    or not _compression_ratio_allowed(total_size, total_compressed)):
                return "archive-payload"
            for member in members:
                mode = member.external_attr >> 16
                if (mode and stat.S_ISLNK(mode)) or not _safe_archive_member_name(member.filename):
                    return "archive-payload"
                if member.is_dir():
                    continue
                if member.flag_bits & 0x1:  # encrypted: the scanner cannot inspect it
                    return "archive-payload"
                if require_npy_members and Path(member.filename).suffix.lower() != ".npy":
                    return "archive-payload"
                if (member.file_size > _MAX_ARCHIVE_MEMBER_BYTES
                        or not _compression_ratio_allowed(
                            member.file_size, member.compress_size)):
                    return "archive-payload"
                with archive.open(member) as stream:
                    if _inspect_archive_member(
                        member.filename,
                        stream,
                        expected_size=member.file_size,
                        compressed_size=member.compress_size,
                    ) is not None:
                        return "archive-payload"
            return None
    except (_ArchiveLimitError, EOFError, OSError, RuntimeError, zipfile.BadZipFile):
        return "archive-payload"


def _tar_payload_kind(path: Path) -> str | None:
    try:
        with tarfile.open(path, "r:*") as archive:
            total_size = 0
            physical_size = path.stat().st_size
            for index, member in enumerate(archive):
                if index >= _MAX_ARCHIVE_MEMBERS:
                    return "archive-payload"
                if not _safe_archive_member_name(member.name):
                    return "archive-payload"
                if member.isdir():
                    continue
                if not member.isfile():
                    return "archive-payload"
                if member.size > _MAX_ARCHIVE_MEMBER_BYTES:
                    return "archive-payload"
                total_size += member.size
                if total_size > _MAX_ARCHIVE_TOTAL_BYTES:
                    return "archive-payload"
                if not _compression_ratio_allowed(total_size, physical_size):
                    return "archive-payload"
                stream = archive.extractfile(member)
                if stream is None or _inspect_archive_member(
                    member.name,
                    stream,
                    expected_size=member.size,
                    compressed_size=None,
                ) is not None:
                    return "archive-payload"
            # Compressed tar formats do not expose a trustworthy per-member compressed size.  The
            # cumulative declared payload is therefore compared with the physical container
            # before each member is extracted, in addition to the per-member/total byte caps.
            return None
    except (_ArchiveLimitError, EOFError, OSError, tarfile.TarError):
        return "archive-payload"


def _compressed_payload_kind(path: Path, header: bytes) -> str | None:
    inner_name = path.name
    for ending in (".bz2", ".gz", ".xz"):
        if inner_name.lower().endswith(ending):
            inner_name = inner_name[:-len(ending)]
            break
    opener = gzip.open if header.startswith(b"\x1f\x8b") else (
        bz2.open if header.startswith(b"BZh") else lzma.open
    )
    try:
        with opener(path, "rb") as stream:
            unsafe = _inspect_archive_member(
                inner_name,
                stream,
                expected_size=None,
                compressed_size=path.stat().st_size,
            )
            return "archive-payload" if unsafe is not None else None
    except (_ArchiveLimitError, EOFError, OSError, lzma.LZMAError):
        return "archive-payload"


def _opaque_payload_kind(path: Path) -> str | None:
    """Reject executable/code payloads while permitting inspected data containers.

    Extensions alone are not a security boundary: a zipimport archive or native extension can be
    renamed to ``weights.dat``. Conversely, rejecting every ZIP made harmless NumPy ``.npz``
    tables illegal even though the task explicitly permits helper/data files. Container magic is
    therefore inspected and only data-only members are accepted.
    """
    suffix = path.suffix.lower()
    if suffix in _BYTECODE_SUFFIXES:
        return "bytecode-payload"
    if suffix in _NATIVE_SUFFIXES:
        return "native-payload"
    if suffix in _PICKLE_SUFFIXES:
        return "encoded-python-payload"
    if suffix in _PACKAGE_ARCHIVE_SUFFIXES:
        return "archive-payload"
    try:
        with open(path, "rb") as fh:
            header = fh.read(512)
    except OSError:
        return None

    if header.startswith(importlib.util.MAGIC_NUMBER):
        return "bytecode-payload"
    if header.startswith(_NATIVE_MAGIC):
        return "native-payload"
    if suffix in _ZIP_SUFFIXES or header.startswith(_ZIP_MAGIC):
        return _zip_payload_kind(path, require_npy_members=suffix == ".npz")
    tar_named = path.name.lower().endswith((".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz"))
    if tar_named or len(header) >= 262 and header[257:262] == b"ustar":
        return _tar_payload_kind(path)
    if suffix in _COMPRESSED_SUFFIXES or header.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00")):
        return _compressed_payload_kind(path, header)
    if suffix == ".npy" or header.startswith(b"\x93NUMPY"):
        try:
            with open(path, "rb") as stream:
                prefix = stream.read(512)
                has_object = _npy_has_object_dtype(prefix, stream)
        except OSError:
            return "archive-payload"
        if has_object is None:
            return "archive-payload"
        return "encoded-python-payload" if has_object else None
    return None


def _top_level_imports(tree: ast.AST):
    """Yield the top-level module name of every import in an AST, including dynamic imports with
    a literal string argument."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:            # skip relative (local) imports
                yield node.module.split(".")[0]
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name in _DYNAMIC_IMPORTERS and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    yield a0.value.split(".")[0]
                else:
                    # A dynamic import whose module name is NOT a plain string literal --
                    # ``__import__("num" + "ba")``, an f-string, a name built at runtime. The old
                    # code only decoded the literal case and silently let everything else through,
                    # so concatenation trivially evaded the allowlist. A computed import target is
                    # itself the tell: yield a sentinel that is never in ``allowed`` so it is
                    # flagged. (A genuine from-scratch PCA has no reason to compute import names.)
                    yield "<computed-import>"


# NOTE: the blanket "``exec``/``eval``/``compile`` appears in the source" hit was RETIRED.
#
# It contradicted the runtime policy. ``submission_runner`` deliberately permits dynamic Python
# (see its audit hook): the production image ships only NumPy and SciPy, and the import allowlist is
# enforced by an audit hook that CPython cannot unregister, so no amount of runtime codegen reaches a
# library that is not already allowed. After that change the two gates disagreed, and the static one
# was strictly wrong: a submission that metaprograms its own pure-Python helper RAN to completion,
# scored accuracy 1.000 on every dev fold -- and still took reward 0.0, because this scan zeroed it.
# That is the run_019f6281 failure shape exactly (a correct program scoring zero for a reason
# unrelated to its correctness), reintroduced through a different gate, and with an EMPTY stderr, so
# it was even less diagnosable than the runtime refusal it replaced.
#
# It also failed the prompt/verifier test. The prompt forbids encoding or reconstructing
# executable/NATIVE payloads; it says nothing against ordinary Python metaprogramming, and a gate
# may only enforce what the prompt actually states.
#
# The threat this rule was documented to stop -- "turn a data-looking blob back into code" -- is
# covered where it is actually decidable, and none of it depends on spotting an ``exec`` token:
#   * embedded/encoded native payloads: ``_embedded_payload_kind``/``_zip_``/``_tar_``/
#     ``_compressed_``/``_opaque_payload_kind`` + ``_NATIVE_MAGIC``, at file level;
#   * reaching a forbidden library by computing its name: ``<computed-import>`` below, statically,
#     and the runtime import allowlist regardless;
#   * loading native code at runtime: the runner blocks every ``ctypes.*`` and marshal/code-
#     construction audit event, and the image has no compiler.
#
# Verified on a box through the real grader, not argued: a submission doing
# ``exec(compile("import " + "nu" + "mba", ...))`` is still refused with
# ``third-party import is disabled: numba`` and scores 0, while ``cheats/numba_jit`` and
# ``cheats/sklearn_pca`` still hit the static import allowlist and score 0. The margin holds without
# this rule; only honest solves were paying for it.


def scan(submission_dir: str | Path) -> dict:
    submission_dir = Path(submission_dir)
    # Names the submission defines itself (local modules/packages) are always allowed -- but ONLY
    # those actually importable as a TOP-LEVEL name from the submission root (sys.path[0]). The old
    # code used ``rglob`` and added the stem of every ``.py`` anywhere in the tree, so dropping a
    # decoy ``anything/numba.py`` made the scanner treat a top-level ``import numba`` as local while
    # Python still resolved the real third-party ``numba`` at runtime -- a silent false negative.
    # Top-level importable names are: ``.py`` files directly under the root, and directories
    # directly under it that are real packages (contain ``__init__.py``, so they genuinely shadow a
    # same-named installed package). A bare nested file or a non-package dir provides no such name.
    local = {p.stem for p in submission_dir.glob("*.py")}
    local |= {p.name for p in submission_dir.iterdir()
              if p.is_dir() and not p.name.startswith(".") and (p / "__init__.py").exists()}
    allowed = (_STDLIB | _ALLOWED_THIRD_PARTY | local) - _DENY_STDLIB

    hits = []
    py_files = []
    entry = submission_dir / "pca"

    for path in submission_dir.rglob("*"):
        if not path.is_file():
            continue
        # The scan runs on the immutable pre-execution snapshot, so any bytecode present here was
        # shipped by the solver rather than generated by this grade.  Accepting ``__pycache__`` was
        # a real hole: a pyc-only module loaded with SourcelessFileLoader has no AST to inspect.
        payload_kind = _opaque_payload_kind(path)
        if payload_kind is not None:
            hits.append({
                "file": path.name,
                "kind": payload_kind,
                "module": path.suffix or "magic",
            })
            continue
        is_py = path.suffix.lower() in _PY_SUFFIXES or _is_python_shebang(path)
        if not is_py:
            continue
        py_files.append(path)
        try:
            tree = ast.parse(path.read_bytes())
        except (SyntaxError, ValueError):
            # An unparseable "Python" file is suspicious (could hide a non-Python payload behind a
            # python shebang); flag it rather than letting it through unscanned.
            hits.append({"file": path.name, "kind": "unparseable", "module": "?"})
            continue
        for mod in _top_level_imports(tree):
            if mod and mod not in allowed:
                hits.append({"file": path.name, "kind": "import", "module": mod})

    # The entry point must exist and be a Python script (no native/other-language entry).
    if entry.exists() and not (entry.suffix.lower() in _PY_SUFFIXES or _is_python_shebang(entry)
                               or entry in py_files):
        hits.append({"file": "pca", "kind": "non-python-entry", "module": "?"})

    factor = 0.0 if hits else 1.0
    return {"name": "library_scan", "factor": factor, "severity": 1.0 if hits else 0.0,
            "hits": hits[:20], "n_hits": len(hits)}
