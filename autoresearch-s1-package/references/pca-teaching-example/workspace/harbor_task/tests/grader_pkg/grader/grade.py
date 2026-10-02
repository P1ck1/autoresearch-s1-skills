"""Grade a submission on scientific quality, robustness, and bounded resource quality.

Each dataset earns a score in [0, 1]. Scientific agreement dominates. End-to-end runtime is a
small continuous modifier, and private method probes discount the score when the program computes
a nearby but different object. Contract or dependency violations remain zero-credit integrity
failures; ordinary algorithmic weakness retains partial credit.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import errno
import ctypes
import os
import resource
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.release_key import (                                           # noqa: E402
    RELEASE_ID,
    derive_bytes,
    derive_seed,
    key_commitment,
    read_math_key,
)
from reference.full_scan_pca import fit as full_scan_fit          # noqa: E402
from grader.metrics.per_pc import per_pc_report                    # noqa: E402
from grader.metrics.subspace import subspace_accuracy, structure_weights  # noqa: E402
from grader.metrics.structure import (                              # noqa: E402
    population_separation,
    within_group_label_projection,
)
from grader.gates import library_scan                              # noqa: E402


# ---------------------------------------------------------------------------
# Submission execution + score loading
# ---------------------------------------------------------------------------

def _resolve_unpriv() -> tuple[int, int] | None:
    """Return the mandatory unprivileged identity whenever the grader is root.

    Untrusted submissions are never run as root. Their complete source tree is
    staged into the chowned per-invocation sandbox first, so this does not depend
    on the original submission path being traversable. Non-root local/CI runs
    already execute without privilege and return ``None``.
    """
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return None
    import pwd
    try:
        entry = pwd.getpwnam("pcasub")
    except KeyError as error:
        raise RuntimeError("dedicated sandbox account 'pcasub' is not provisioned") from error
    if entry.pw_uid == 0:
        raise RuntimeError("sandbox account 'pcasub' must not be root")
    return entry.pw_uid, entry.pw_gid


_UNPRIV = _resolve_unpriv()
_GZIP_MAGIC = b"\x1f\x8b"
_MAX_SCORES_BYTES = 64 * 1024 * 1024       # legitimate n x k TSVs are KBs--small MBs
_MAX_SCORE_LINE_BYTES = 4 * 1024 * 1024
_MAX_SUBMISSION_BYTES = 64 * 1024 * 1024
_MAX_SUBMISSION_FILES = 4096
_MAX_SUBMISSION_DIRECTORIES = 1024
_MAX_SUBMISSION_DEPTH = 64
_STDERR_TAIL_BYTES = 2000
_MAX_ADDRESS_SPACE_BYTES = 16 * 1024 * 1024 * 1024
_MAX_AGGREGATE_RSS_BYTES = 14 * 1024 * 1024 * 1024
_MAX_WRITABLE_STORAGE_BYTES = 4 * 1024 * 1024 * 1024
# Per-FILE ceiling for anything the submission writes. Deliberately its own constant rather than an
# alias of the aggregate storage cap: the two answer different questions ("how big may ONE file be"
# vs "how much may everything total"), and tests that shrink the aggregate cap to force the storage
# monitor must not also shrink this and trip a different mechanism.
_MAX_SUBMISSION_FILE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_WRITABLE_ENTRIES = 100_000
_MAX_PROCESSES = 128
_MAX_OPEN_FILES = 256
_BWRAP = "/usr/bin/bwrap"
_SETPRIV = "/usr/bin/setpriv"
_CHROOT = "/usr/sbin/chroot"
_RUNTIME = "/opt/hyperfocal/pcabench"
_RUNTIME_PYTHON = f"{_RUNTIME}/bin/python"
_GUARD = "/opt/hyperfocal/pcabench-guard"
_GUARD_RUNNER = f"{_GUARD}/submission_runner.py"
_RESOURCE_WATCHDOG = Path(__file__).resolve().with_name("resource_watchdog.py")
_BWRAP_WORK_ROOT = Path("/opt/hyperfocal/pcabench-work")
_CHROOT_ROOT = Path("/opt/hyperfocal/pcabench-sandbox-root")
_ISOLATION_MODES = {"bwrap", "verifier-container"}
_NORMALIZED_INPUT_TIME_NS = 946684800_000_000_000  # 2000-01-01 UTC
_DEFAULT_GRADING_BUDGET_SECONDS = 8_400
_GRADING_FINALIZATION_RESERVE_SECONDS = 120
_DATASET_SUBMISSION_TIMEOUT_SECONDS = 3_600
_PROBE_SUBMISSION_TIMEOUT_SECONDS = 900
_REFERENCE_TIMEOUT_SECONDS = 7_200
_PROBE_GATE_NAMES = ("hwe_norm", "coverage", "representation_equivalence")
_SUBMISSION_NUMERICAL_THREADS = 8
_CPU_GUARD_HEADROOM_SECONDS = 10
# Work times below this are startup jitter, not algorithmic signal; they floor the ratio so a
# near-instant fit on a tiny fold cannot divide by ~zero.
_WORK_TIME_FLOOR_SECONDS = 0.05


class InvalidSubmission(ValueError):
    """A solver-controlled submission shape that must earn zero, not an infra error."""


class GradingDeadlineExhausted(TimeoutError):
    """The reviewed global grading budget is spent; finalize partial results immediately."""


class GradingBudget:
    """One monotonic wall-clock budget shared by probes, references, and scored datasets.

    Every external invocation receives only the time still available before a protected
    finalization reserve.  This prevents a sequence of individually legal timeouts from running
    into the verifier's outer timeout and erasing results that were already earned.
    """

    def __init__(self, total_seconds: float, *, clock=time.monotonic):
        if not np.isfinite(total_seconds) or total_seconds <= 0:
            raise ValueError("grading time budget must be a positive finite number")
        self.total_seconds = float(total_seconds)
        self._clock = clock
        self.started_at = float(clock())
        self.deadline = self.started_at + self.total_seconds
        # Tiny focused tests and emergency finalization runs remain useful; normal production
        # runs retain the full two-minute JSON/report cleanup reserve.
        self.finalization_reserve = min(
            float(_GRADING_FINALIZATION_RESERVE_SECONDS), self.total_seconds / 2.0,
        )

    def elapsed(self) -> float:
        return max(0.0, float(self._clock()) - self.started_at)

    def work_seconds_left(self) -> float:
        return max(0.0, self.deadline - float(self._clock()) - self.finalization_reserve)

    def exhausted(self) -> bool:
        return self.work_seconds_left() < 1.0

    def invocation_timeout(self, cap_seconds: float) -> int:
        if not np.isfinite(cap_seconds) or cap_seconds <= 0:
            raise ValueError("invocation timeout cap must be a positive finite number")
        available = self.work_seconds_left()
        if available < 1.0:
            raise GradingDeadlineExhausted("global grading deadline exhausted")
        return max(1, min(math.ceil(float(cap_seconds)), math.floor(available)))


_INVALID_TREE_ERRNOS = {
    errno.EACCES,
    errno.ELOOP,
    errno.ENOENT,
    errno.ENOTDIR,
    errno.ENXIO,
    errno.EPERM,
}


def _file_identity(path: Path) -> tuple[str, int, int, int, int, int]:
    """Identity for caches that must invalidate when bytes at a path are replaced or changed."""
    resolved = Path(os.path.realpath(str(path)))
    stat = resolved.stat()
    return (str(resolved), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _content_digest(path: Path) -> str:
    """Streaming content identity for compressed-input cache correctness across coarse mtimes."""
    digest = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_vcf(vcf: Path, workdir: Path) -> Path:
    """Return a seekable plain VCF, decompressing gzip/BGZF once outside timed execution.

    Detection uses file magic rather than a public filename. The opaque cache key includes a
    streaming content digest, so even equal-size rewrites on coarse-timestamp filesystems cannot
    reuse stale decompressed bytes.
    """
    vcf = Path(vcf)
    with open(vcf, "rb") as fh:
        if fh.read(2) != _GZIP_MAGIC:
            return vcf

    cache_dir = workdir / "_vcf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)
    identity = repr((*_file_identity(vcf), _content_digest(vcf))).encode("utf-8")
    token = hashlib.sha256(identity).hexdigest()[:32]
    normalized = cache_dir / f"{token}.vcf"
    if normalized.exists():
        return normalized

    temporary = cache_dir / f".{token}_{secrets.token_hex(8)}.tmp"
    try:
        with gzip.open(vcf, "rb") as source, open(temporary, "xb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)
        os.chmod(temporary, 0o444)
        os.replace(temporary, normalized)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return normalized


def _protect_grader_directory(path: Path, *, sandbox_parent: bool = False) -> None:
    """Root-own grader state; allow traversal only where bwrap must open a staged child."""
    path.mkdir(parents=True, exist_ok=True)
    if _UNPRIV is not None:
        os.chown(path, 0, 0)
        os.chmod(path, 0o711 if sandbox_parent else 0o700)


def _bwrap_work_root() -> Path:
    """Return the provisioned, non-listable parent for host-backed bwrap state."""
    info = _BWRAP_WORK_ROOT.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0
            or info.st_mode & 0o066):
        raise RuntimeError(f"invalid bubblewrap work root: {_BWRAP_WORK_ROOT}")
    return _BWRAP_WORK_ROOT


def _grader_package_root() -> Path | None:
    """The directory on ``sys.path`` that carries the hidden grader/reference/data source -- the
    ``grader_pkg`` PYTHONPATH entry in the packaged Harbor task. It holds ``reference/`` (the EXACT
    full-scan scoring truth) and the metrics/gates. Returns None if it can't be located."""
    try:
        import reference as _ref
    except Exception:                                    # pragma: no cover - import always succeeds here
        return None
    return Path(_ref.__file__).resolve().parent.parent


def _seal_grader_package() -> None:
    """Seal the hidden grader package from the dropped submission uid.

    ``reference/full_scan_pca.py`` (+ ``pca_core``) is the exact truth the submission is scored
    against. If the dropped child can read it, it can skip the task entirely -- e.g.
    ``PYTHONPATH=<grader_pkg> python -m reference.full_scan_pca <vcf> <k> <out>`` shells out (os +
    subprocess are legitimately allowlisted, so the static library scan never sees the import) and
    writes the exact reference scores, acing accuracy and even earning a speed bonus. test.sh locks
    the datasets (0700) and truth sidecars (0600) but ships these source directories world-readable,
    so we make only ``reference/`` and ``grader/`` owner-only.  Sealing the common package parent is
    unsafe in Hyperfocal because that parent can be the environment repository itself, which must
    remain traversable for a later setup/solve cycle. The grader runs as root and ignores the bits;
    the pcasub child is denied traversal into either hidden subtree. No-op for non-root trusted
    dev/CI checks. The static library scan stays as defence-in-depth; this filesystem seal --
    not the scan -- is the confidentiality boundary."""
    if _UNPRIV is None:
        return
    root = _grader_package_root()
    if root is None:
        return
    hidden = (root / "reference", root / "grader")
    if not all(path.is_dir() and not path.is_symlink() for path in hidden):
        raise RuntimeError(f"hidden grader source directories are missing below {root}")
    try:
        for path in hidden:
            os.chown(path, 0, 0)
            os.chmod(path, 0o700)
    except OSError as error:
        raise RuntimeError(f"could not seal hidden grader source below {root}") from error


def _same_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _copy_snapshot_directory(
    source_fd: int, destination: Path, budget: dict[str, int], depth: int = 0,
) -> None:
    """Recursively copy through verified openat descriptors, never through checked paths."""
    if depth > _MAX_SUBMISSION_DEPTH:
        raise InvalidSubmission("submission exceeds the directory-depth limit")
    directory_before = os.fstat(source_fd)
    names = sorted(os.listdir(source_fd))
    for name in names:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            child_fd = os.open(name, flags, dir_fd=source_fd)
        except OSError as error:
            if error.errno in _INVALID_TREE_ERRNOS:
                raise InvalidSubmission(
                    f"submission entry changed, is inaccessible, or is a symlink: {name}"
                ) from error
            raise
        try:
            before = os.fstat(child_fd)
            target = destination / name
            if stat.S_ISDIR(before.st_mode):
                # Honest local testing creates interpreter bytecode beside source helpers.  Never
                # copy that opaque, solver-controlled cache into the execution snapshot: source is
                # rescanned and recompiled under the pinned runtime, while a pyc-only payload
                # cannot cross the boundary.  Open/fstat first so a symlink with this name is still
                # rejected by O_NOFOLLOW rather than silently ignored.
                if name == "__pycache__":
                    continue
                budget["directories"] += 1
                if budget["directories"] > _MAX_SUBMISSION_DIRECTORIES:
                    raise InvalidSubmission("submission exceeds the directory-count limit")
                target.mkdir(mode=0o700)
                _copy_snapshot_directory(child_fd, target, budget, depth + 1)
                target.chmod(0o555)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise InvalidSubmission(f"submission contains a non-regular file: {name}")
            budget["files"] += 1
            if budget["files"] > _MAX_SUBMISSION_FILES:
                raise InvalidSubmission("submission exceeds the file-count limit")
            copied = 0
            with os.fdopen(os.dup(child_fd), "rb", closefd=True) as source, open(target, "xb") as output:
                while chunk := source.read(1024 * 1024):
                    copied += len(chunk)
                    budget["bytes"] += len(chunk)
                    if budget["bytes"] > _MAX_SUBMISSION_BYTES:
                        raise InvalidSubmission("submission exceeds the byte limit")
                    output.write(chunk)
            after = os.fstat(child_fd)
            if copied != before.st_size or not _same_identity(before, after):
                raise InvalidSubmission(f"submission file changed while snapshotting: {name}")
            target.chmod(0o444)
        finally:
            os.close(child_fd)
    directory_after = os.fstat(source_fd)
    if sorted(os.listdir(source_fd)) != names or not _same_identity(directory_before, directory_after):
        raise InvalidSubmission("submission directory changed while snapshotting")


def _stage_submission(submission_dir: Path, destination: Path) -> Path:
    """Create one bounded, immutable, no-follow snapshot of the submitted tree."""
    source_root = Path(submission_dir)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source_root, flags)
    except OSError as error:
        if error.errno in _INVALID_TREE_ERRNOS:
            raise InvalidSubmission("submission root must be a real accessible directory") from error
        raise
    destination.mkdir(mode=0o700)
    try:
        _copy_snapshot_directory(
            source_fd, destination, {"files": 0, "directories": 0, "bytes": 0},
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        os.close(source_fd)
    entry = destination / "pca"
    if not entry.is_file() or entry.is_symlink():
        shutil.rmtree(destination, ignore_errors=True)
        raise InvalidSubmission("submission must contain a regular pca file")
    destination.chmod(0o555)
    return entry


def _submission_cpu_limit_seconds(timeout: float) -> int:
    """Aggregate CPU guard corresponding to the advertised wall allowance.

    Linux ``RLIMIT_CPU`` charges CPU time across a process's threads rather than elapsed wall
    time. Submissions are explicitly given eight numerical threads, so a one-core CPU limit would
    kill a legitimate fully parallel fit after roughly one eighth of its promised wall budget.
    The watchdog remains the authoritative wall timer; this looser limit is defense in depth.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("submission timeout must be positive and finite")
    return math.ceil(timeout * _SUBMISSION_NUMERICAL_THREADS) + _CPU_GUARD_HEADROOM_SECONDS


def _apply_submission_limits(timeout: int):
    """Return a fail-closed pre-exec hook for the untrusted process tree."""
    cpu_seconds = _submission_cpu_limit_seconds(timeout)

    def apply() -> None:
        for resource_id, value in (
            (resource.RLIMIT_AS, _MAX_ADDRESS_SPACE_BYTES),
            # Per-FILE ceiling, NOT the output ceiling. It was _MAX_SCORES_BYTES (64 MiB), which
            # silently contradicted the documented contract: the prompt promises "4 GiB aggregate
            # temporary storage" and scopes 64 MiB to the OUTPUT file, but RLIMIT_FSIZE applies to
            # every file the process writes. A submission that spills a large intermediate to a temp
            # memmap -- the exact memory discipline the biobank ladder is built to reward, and the
            # alternative to holding it against the 14 GiB RSS cap -- died on a short write
            # ("OSError: 16777216 requested and 8388592 written"), losing a whole fold. Measured:
            # that cost run_019f7774 rollout-0 its sample_heavy fold (0.000 at accuracy 0).
            # The OUTPUT stays bounded independently and more gracefully, on read: _copy_bounded_
            # regular and the post-run stat both reject a >64 MiB scores file as "output exceeds size
            # limit" (invalid -> 0). Aggregate temp usage stays bounded by the watchdog's storage
            # monitor at the same 4 GiB. So this ceiling now matches what the prompt actually promises.
            (resource.RLIMIT_FSIZE, _MAX_SUBMISSION_FILE_BYTES),
            (resource.RLIMIT_NPROC, _MAX_PROCESSES),
            (resource.RLIMIT_NOFILE, _MAX_OPEN_FILES),
            (resource.RLIMIT_CPU, cpu_seconds),
            (resource.RLIMIT_CORE, 0),
        ):
            resource.setrlimit(resource_id, (value, value))

    return apply


def _submission_processes(*, include_zombies: bool = False) -> dict[int, str]:
    """Return every process owned by the dedicated submission uid.

    The account is reserved for one untrusted invocation at a time. Process-group cleanup alone is
    insufficient because hostile code can call ``setsid()``; the uid is the kernel identity that a
    capability-free child cannot escape.
    """
    if _UNPRIV is None or not sys.platform.startswith("linux"):
        return {}
    uid, _ = _UNPRIV
    processes: dict[int, str] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdecimal():
            continue
        try:
            info = entry.stat(follow_symlinks=False)
            if info.st_uid != uid:
                continue
            status = Path(entry.path, "status").read_text(errors="replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        state = "?"
        for line in status.splitlines():
            if line.startswith("State:"):
                fields = line.split()
                state = fields[1] if len(fields) > 1 else "?"
                break
        if include_zombies or state != "Z":
            processes[int(entry.name)] = state
    return processes


def _chroot_writable_roots() -> tuple[Path, ...]:
    return (
        _CHROOT_ROOT / "tmp",
        _CHROOT_ROOT / "var/tmp",
        _CHROOT_ROOT / "run/lock",
        _CHROOT_ROOT / "dev/shm",
        _CHROOT_ROOT / "dev/mqueue",
    )


def _assert_no_submission_processes() -> None:
    processes = _submission_processes()
    if processes:
        raise RuntimeError(
            f"submission uid has live processes outside an invocation: {sorted(processes)}"
        )


def _terminate_submission_processes(timeout: float = 5.0) -> None:
    """Freeze then kill every detached process owned by the submission uid.

    We first SIGSTOP the complete uid set and require a stable all-stopped snapshot. That closes
    the fork race: once all members are stopped, none can create a child between enumeration and
    SIGKILL. Repeated enumeration catches children forked before their parent received SIGSTOP.
    Zombies are harmless and excluded; live survivors after the deadline are an infrastructure
    error, never silently carried into another probe or reference run.
    """
    if _UNPRIV is None or not sys.platform.startswith("linux"):
        return
    deadline = time.monotonic() + timeout
    while True:
        processes = _submission_processes()
        if not processes:
            return
        for pid in processes:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        time.sleep(0.01)

        stopped = _submission_processes()
        if stopped and all(state in {"T", "t"} for state in stopped.values()):
            # Re-enumerate after every process is observably stopped. A stable set cannot fork.
            confirmation = _submission_processes()
            if confirmation == stopped:
                for pid in confirmation:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(0.01)

        if time.monotonic() >= deadline:
            survivors = _submission_processes()
            if survivors:
                raise RuntimeError(
                    f"could not terminate submission uid processes: {sorted(survivors)}"
                )
            return


_SYSVIPC_TABLES = {
    "msg": (Path("/proc/sysvipc/msg"), "msqid", "msgctl"),
    "sem": (Path("/proc/sysvipc/sem"), "semid", "semctl"),
    "shm": (Path("/proc/sysvipc/shm"), "shmid", "shmctl"),
}


def _submission_sysvipc() -> dict[str, list[int]]:
    """List System V IPC objects owned or created by the submission uid."""
    if _UNPRIV is None or not sys.platform.startswith("linux"):
        return {}
    uid, _ = _UNPRIV
    found: dict[str, list[int]] = {}
    for kind, (path, id_column, _) in _SYSVIPC_TABLES.items():
        try:
            lines = path.read_text().splitlines()
        except FileNotFoundError:
            continue
        if not lines:
            continue
        header = lines[0].split()
        try:
            id_index = header.index(id_column)
            uid_index = header.index("uid")
            cuid_index = header.index("cuid")
        except ValueError as error:
            raise RuntimeError(f"unrecognized {path} schema") from error
        ids = []
        for line in lines[1:]:
            fields = line.split()
            if len(fields) <= max(id_index, uid_index, cuid_index):
                raise RuntimeError(f"malformed {path} row")
            if int(fields[uid_index]) == uid or int(fields[cuid_index]) == uid:
                ids.append(int(fields[id_index]))
        if ids:
            found[kind] = ids
    return found


def _assert_no_submission_ipc() -> None:
    objects = _submission_sysvipc()
    if objects:
        raise RuntimeError(f"submission uid has persistent System V IPC objects: {objects}")


def _remove_submission_ipc() -> None:
    """Delete every SysV IPC object after all submission processes are frozen and killed."""
    if _UNPRIV is None or not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    for _ in range(4):
        objects = _submission_sysvipc()
        if not objects:
            return
        for kind, ids in objects.items():
            function = getattr(libc, _SYSVIPC_TABLES[kind][2])
            for identifier in ids:
                ctypes.set_errno(0)
                if kind == "sem":
                    result = function(identifier, 0, 0, 0)
                else:
                    result = function(identifier, 0, None)
                if result != 0 and ctypes.get_errno() not in {errno.EIDRM, errno.EINVAL}:
                    raise RuntimeError(
                        f"could not remove submission {kind} object {identifier}: "
                        f"errno {ctypes.get_errno()}"
                    )
    remaining = _submission_sysvipc()
    if remaining:
        raise RuntimeError(f"submission IPC cleanup left objects behind: {remaining}")


def _clear_chroot_state() -> None:
    """Remove writable per-invocation state from the static verifier chroot."""
    if not _CHROOT_ROOT.is_dir():
        raise RuntimeError(f"verifier chroot is missing: {_CHROOT_ROOT}")
    for relative in ("tmp", "var/tmp", "run/lock", "dev/shm", "dev/mqueue"):
        directory = _CHROOT_ROOT / relative
        info = directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0:
            raise RuntimeError(f"invalid verifier chroot state directory: {directory}")
        for child in directory.iterdir():
            child_info = child.lstat()
            if stat.S_ISDIR(child_info.st_mode):
                shutil.rmtree(child)
            else:
                child.unlink()


def _sandbox_argv(
    command: list[str], submission_snapshot: Path, staged_input: Path, staged_output: Path,
) -> list[str]:
    """Build the one production sandbox: root/Linux/bwrap/pcasub with allowlisted mounts."""
    if _UNPRIV is None:
        raise RuntimeError("submission execution requires root and the pcasub account")
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise RuntimeError("submission execution requires root on Linux")
    if not os.path.isfile(_BWRAP) or not os.access(_BWRAP, os.X_OK):
        raise RuntimeError(f"submission execution requires {_BWRAP}")
    if not os.path.isfile(_SETPRIV) or not os.access(_SETPRIV, os.X_OK):
        raise RuntimeError(f"submission execution requires {_SETPRIV}")
    if not os.path.isfile(_RUNTIME_PYTHON):
        raise RuntimeError(f"pinned runtime is missing: {_RUNTIME}")
    if not os.path.isfile(_GUARD_RUNNER):
        raise RuntimeError(f"submission guard is missing: {_GUARD_RUNNER}")
    uid, gid = _UNPRIV
    scratch = staged_input.parent / "tmp"
    shared_memory = scratch / "shm"
    return [
        _SETPRIV,
        "--reuid", str(uid),
        "--regid", str(gid),
        "--clear-groups",
        _BWRAP,
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--dir", "/opt",
        "--dir", "/opt/hyperfocal",
        "--ro-bind", _RUNTIME, _RUNTIME,
        "--ro-bind", _GUARD, _GUARD,
        "--dev", "/dev",
        # Replace bubblewrap's private /dev/shm tmpfs with monitored host-backed storage. The
        # containing /dev mount is remounted read-only below, so this is its only writable child.
        "--bind", str(shared_memory), "/dev/shm",
        # Deliberately do not mount procfs.  A read-only procfs is still an active kernel API:
        # every process can open its own /proc/<pid>/mem read-write and overwrite already
        # executable pages, bypassing the post-import no-PROT_EXEC seccomp policy.  An empty
        # directory preserves the conventional path without exposing process memory, fd aliases,
        # namespace handles, kernel metadata, or dynamically named per-thread ``mem`` files.
        # NumPy/SciPy and fork-only workers use sysconf/affinity and do not require procfs.
        "--dir", "/proc",
        # A host-backed, invocation-private scratch directory lets the trusted parent enforce the
        # same aggregate physical-storage and entry limits as the verifier-container chroot.
        # An anonymous tmpfs would be invisible to that accounting boundary.
        "--bind", str(scratch), "/tmp",
        "--ro-bind", str(submission_snapshot), "/submission",
        # Create the mountpoint directory inside bubblewrap's synthetic root before attaching the
        # two reviewed files.  Binding the host parent read-only first prevents bubblewrap from
        # creating the child mountpoints and fails every submission before its code starts.
        "--dir", "/work",
        "--ro-bind", str(staged_input), "/work/input.vcf",
        "--bind", str(staged_output), "/work/output.tsv",
        "--unshare-user",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--uid", "0",
        "--gid", "0",
        "--cap-drop", "ALL",
        "--die-with-parent",
        "--new-session",
        "--chdir", "/work",
        # The synthetic root and /dev mounts are otherwise owned by namespace uid 0. Remount them
        # read-only so chmod cannot turn an anonymous, unmonitored tmpfs into a storage bypass.
        "--remount-ro", "/",
        "--remount-ro", "/dev",
        "--",
        *command,
    ]


def _verifier_container_argv(command: list[str]) -> list[str]:
    """Chroot, then drop every privilege inside Harbor's no-network verifier container.

    The container is a fresh verifier-only filesystem: the solver never ran in it, `/tests` and
    truth are root-only, and only the immutable submission artifact crosses the boundary. A static
    root-owned chroot contains only the pinned runtime and per-invocation work tree; it exposes no
    grader, datasets, procfs, sysfs, or container-global files. Unlike the Hyperfocal host, an
    ordinary Docker container cannot create a nested user namespace, so this reviewed mode uses
    chroot plus a mandatory uid/capability drop inside the outer container boundary.
    """
    if _UNPRIV is None or not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise RuntimeError("verifier-container execution requires root on Linux")
    if not os.path.isfile(_SETPRIV) or not os.access(_SETPRIV, os.X_OK):
        raise RuntimeError(f"submission execution requires {_SETPRIV}")
    if not os.path.isfile(_CHROOT) or not os.access(_CHROOT, os.X_OK):
        raise RuntimeError(f"submission execution requires {_CHROOT}")
    if not os.path.isfile(_RUNTIME_PYTHON):
        raise RuntimeError(f"pinned runtime is missing: {_RUNTIME}")
    if not os.path.isfile(_GUARD_RUNNER):
        raise RuntimeError(f"submission guard is missing: {_GUARD_RUNNER}")
    root_info = _CHROOT_ROOT.stat()
    if (not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != 0
            or root_info.st_mode & 0o022):
        raise RuntimeError(f"invalid verifier chroot: {_CHROOT_ROOT}")
    if not (_CHROOT_ROOT / _GUARD_RUNNER.lstrip("/")).is_file():
        raise RuntimeError("submission guard is absent from verifier chroot")
    uid, gid = _UNPRIV
    return [
        _CHROOT,
        str(_CHROOT_ROOT),
        _SETPRIV,
        "--reuid", str(uid),
        "--regid", str(gid),
        "--clear-groups",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--no-new-privs",
        *command,
    ]


def _submission_argv(
    command: list[str], submission_snapshot: Path, staged_input: Path, staged_output: Path,
    isolation: str,
) -> list[str]:
    if isolation == "bwrap":
        return _sandbox_argv(command, submission_snapshot, staged_input, staged_output)
    if isolation == "verifier-container":
        return _verifier_container_argv(command)
    raise ValueError(f"unknown isolation mode: {isolation!r}")


def _resource_watchdog_argv(
    command: list[str], child_env: dict[str, str], child_cwd: str,
    storage_roots: tuple[Path, ...], timeout: int,
) -> list[str]:
    """Build the trusted same-uid-parent resource monitor command.

    The watchdog starts as root so it can launch the reviewed chroot/bwrap boundary, then drops to
    ``pcasub`` before releasing that child. Running as the child's same-uid ancestor makes Linux
    procfs ptrace checks authorize ``smaps_rollup``, ``fd``, and ``map_files`` without adding
    ``CAP_SYS_PTRACE`` to the verifier container.
    """
    if _UNPRIV is None:
        raise RuntimeError("submission resource watchdog requires the pcasub account")
    if not _RESOURCE_WATCHDOG.is_file() or _RESOURCE_WATCHDOG.is_symlink():
        raise RuntimeError(f"submission resource watchdog is missing: {_RESOURCE_WATCHDOG}")
    uid, gid = _UNPRIV
    argv = [
        sys.executable,
        "-I",
        str(_RESOURCE_WATCHDOG),
        "--uid", str(uid),
        "--gid", str(gid),
        "--timeout", str(timeout),
        "--cpu-time-limit", str(_submission_cpu_limit_seconds(timeout)),
        "--address-space-limit", str(_MAX_ADDRESS_SPACE_BYTES),
        "--rss-limit", str(_MAX_AGGREGATE_RSS_BYTES),
        "--storage-limit", str(_MAX_WRITABLE_STORAGE_BYTES),
        "--entry-limit", str(_MAX_WRITABLE_ENTRIES),
        "--process-limit", str(_MAX_PROCESSES),
        "--open-file-limit", str(_MAX_OPEN_FILES),
        "--file-size-limit", str(_MAX_SUBMISSION_FILE_BYTES),
        "--child-cwd", child_cwd,
        "--child-env-json", json.dumps(child_env, separators=(",", ":"), sort_keys=True),
    ]
    for root in storage_roots:
        argv.extend(("--storage-root", str(root)))
    argv.extend(("--", *command))
    return argv


def _decode_watchdog_report(payload: bytes, watchdog_pid: int) -> dict:
    """Validate the bounded private watchdog report before trusting any resource result."""
    if len(payload) > 16 * 1024:
        raise RuntimeError("submission resource watchdog report exceeded its bound")
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("submission resource watchdog returned malformed JSON") from error
    expected = {
        "watchdog_pid", "child_returncode", "timed_out", "violation", "monitor_error",
        "peak_pss_bytes", "peak_storage_bytes",
    }
    if not isinstance(report, dict) or set(report) != expected:
        raise RuntimeError("submission resource watchdog returned an invalid schema")
    if report["watchdog_pid"] != watchdog_pid:
        raise RuntimeError("submission resource watchdog report has the wrong process identity")
    if type(report["child_returncode"]) is not int:
        raise RuntimeError("submission resource watchdog returned an invalid child status")
    if type(report["timed_out"]) is not bool:
        raise RuntimeError("submission resource watchdog returned an invalid timeout status")
    for field in ("violation", "monitor_error"):
        if report[field] is not None and not isinstance(report[field], str):
            raise RuntimeError(f"submission resource watchdog returned an invalid {field}")
    for field in ("peak_pss_bytes", "peak_storage_bytes"):
        if type(report[field]) is not int or report[field] < 0:
            raise RuntimeError(f"submission resource watchdog returned an invalid {field}")
    return report


def _assert_resource_monitor_complete(report: dict) -> None:
    """Keep trusted accounting failures distinct from submission resource violations."""
    if report["monitor_error"]:
        raise RuntimeError(f"submission resource monitor failed: {report['monitor_error']}")


def _assert_bwrap_boundary(workdir: Path) -> None:
    """Prove unrelated host files and networking are absent before grading."""
    del workdir
    sentinel = Path("/root") / f".pcabench-sandbox-{secrets.token_hex(12)}"
    sentinel.write_bytes(b"must not be visible")
    sandbox = Path(tempfile.mkdtemp(prefix="_preflight_", dir=str(_bwrap_work_root())))
    try:
        uid, gid = _UNPRIV
        os.chown(sandbox, uid, gid)
        submission = sandbox / "submission"
        submission.mkdir(mode=0o555)
        staged_input = sandbox / "input.vcf"
        staged_input.write_bytes(b"##fileformat=VCFv4.2\n")
        staged_input.chmod(0o444)
        staged_output = sandbox / "output.tsv"
        staged_output.touch(mode=0o600)
        os.chown(staged_output, uid, gid)
        scratch = sandbox / "tmp"
        scratch.mkdir(mode=0o700)
        os.chown(scratch, uid, gid)
        shared_memory = scratch / "shm"
        shared_memory.mkdir(mode=0o700)
        os.chown(shared_memory, uid, gid)
        probe = (
            "import os,socket,sys\n"
            "for path in sys.argv[1:]:\n"
            " try:\n"
            "  os.stat(path); raise SystemExit(20)\n"
            " except OSError:\n"
            "  pass\n"
            "if os.listdir('/proc'): raise SystemExit(21)\n"
            "for path in ('/proc/self/mem','/proc/thread-self/mem','/proc/1/mem'):\n"
            " if os.path.exists(path): raise SystemExit(23)\n"
            "sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
            "sock.settimeout(1)\n"
            "if sock.connect_ex(('1.1.1.1',53)) == 0: raise SystemExit(22)\n"
        )
        command = [_RUNTIME_PYTHON, "-I", "-c", probe, str(sentinel), "/root"]
        result = subprocess.run(
            _sandbox_argv(command, submission, staged_input, staged_output),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={"PATH": f"{_RUNTIME}/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp"},
            timeout=10,
            preexec_fn=_apply_submission_limits(10),
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace")[-1000:]
            raise RuntimeError(
                f"submission sandbox preflight failed ({result.returncode}): {detail}"
            )
    finally:
        sentinel.unlink(missing_ok=True)
        shutil.rmtree(sandbox, ignore_errors=True)


def _assert_verifier_container_boundary(workdir: Path) -> None:
    """Prove isolation, no egress, and immutable read metadata before grading."""
    root = _grader_package_root()
    if root is None:
        raise RuntimeError("hidden grader package could not be located")
    sentinel = Path("/root") / f".pcabench-container-{secrets.token_hex(12)}"
    sentinel.write_bytes(b"must not be visible")
    del workdir
    _assert_no_submission_processes()
    _assert_no_submission_ipc()
    _clear_chroot_state()
    parent = _CHROOT_ROOT / "work"
    _protect_grader_directory(parent, sandbox_parent=True)
    sandbox = Path(tempfile.mkdtemp(prefix="_preflight_", dir=str(parent)))
    immutable_marker = _CHROOT_ROOT / "usr/bin/env"
    marker_atime = immutable_marker.stat().st_atime_ns
    try:
        uid, gid = _UNPRIV
        scratch = sandbox / "tmp"
        scratch.mkdir(mode=0o700)
        os.chown(scratch, uid, gid)
        inside_scratch = "/" + scratch.relative_to(_CHROOT_ROOT).as_posix()
        probe = (
            "import os,socket,sys\n"
            "for path in sys.argv[1:]:\n"
            " try:\n"
            "  os.stat(path); raise SystemExit(20)\n"
            " except OSError:\n"
            "  pass\n"
            "open('/usr/bin/env','rb').read(1)\n"
            "for family,address in ((socket.AF_INET,('1.1.1.1',53)),"
            "(socket.AF_INET6,('2606:4700:4700::1111',53,0,0))):\n"
            " sock=socket.socket(family,socket.SOCK_STREAM); sock.settimeout(1)\n"
            " if sock.connect_ex(address) == 0: raise SystemExit(22)\n"
        )
        command = [
            _RUNTIME_PYTHON, "-I", "-c", probe,
            str(sentinel), str(root / "reference" / "full_scan_pca.py"),
            "/tests", "/opt/hyperfocal/pcabench-data", "/proc", "/sys",
        ]
        result = subprocess.run(
            _verifier_container_argv(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd="/",
            env={"PATH": f"{_RUNTIME}/bin:/usr/bin:/bin", "HOME": inside_scratch,
                 "TMPDIR": inside_scratch},
            timeout=10,
            preexec_fn=_apply_submission_limits(10),
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace")[-1000:]
            raise RuntimeError(
                f"verifier-container preflight failed ({result.returncode}): {detail}"
            )
        if immutable_marker.stat().st_atime_ns != marker_atime:
            raise RuntimeError(
                "verifier-container immutable root updates access times; repeated "
                "submission calls would have a persistent metadata channel"
            )
    finally:
        _terminate_submission_processes()
        _remove_submission_ipc()
        _clear_chroot_state()
        sentinel.unlink(missing_ok=True)
        shutil.rmtree(sandbox, ignore_errors=True)


def _assert_execution_boundary(workdir: Path, isolation: str) -> None:
    if isolation == "bwrap":
        _assert_bwrap_boundary(workdir)
        return
    if isolation == "verifier-container":
        _assert_verifier_container_boundary(workdir)
        return
    raise ValueError(f"unknown isolation mode: {isolation!r}")


def _assert_grader_package_sealed() -> None:
    """Fail the grade CLOSED if the dropped submission uid can still read the hidden reference.

    Runs the check AS that uid -- a forked child under the exact privilege drop the submission
    gets (``_drop_privileges``) -- so it verifies the real OS decision rather than trusting the
    chmod. A readable reference is an evaluation-integrity breach (the submission could exec the
    exact scoring truth for reward ~1), so we raise rather than silently grade. No-op off the
    deployed drop path."""
    if _UNPRIV is None:
        return
    root = _grader_package_root()
    if root is None:
        return
    probes = [
        root / "reference" / "full_scan_pca.py",
        root / "reference" / "pca_core.py",
        root / "grader" / "gates" / "probes.py",
    ]
    hidden = [root / "reference", root / "grader"]
    # Child exits 3 iff it can list a hidden source directory or read any hidden artifact.
    code = ("import os,sys\n"
            "cut=int(sys.argv[1])\n"
            "for directory in sys.argv[2:2+cut]:\n"
            "    try:\n"
            "        os.listdir(directory); sys.exit(3)\n"
            "    except OSError:\n"
            "        pass\n"
            "for path in sys.argv[2+cut:]:\n"
            "    try:\n"
            "        open(path, 'rb').read(1); sys.exit(3)\n"
            "    except OSError:\n"
            "        pass\n"
            "sys.exit(0)\n")
    proc = subprocess.run([sys.executable, "-c", code, str(len(hidden)),
                           *map(str, hidden), *map(str, probes)],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, preexec_fn=_drop_privileges())
    if proc.returncode == 3:
        raise RuntimeError(
            "SECURITY: hidden reference source is readable by the submission uid "
            f"({root}); refusing to grade. _seal_grader_package must run before the privilege drop.")
    if proc.returncode != 0:
        raise RuntimeError(
            "hidden-reference confidentiality preflight did not complete successfully "
            f"(exit {proc.returncode}); refusing to grade"
        )


def _write_private_json(destination: Path, payload: dict) -> None:
    """Create one owner-only result file without following or replacing a path."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(payload, stream, indent=2, default=float)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stage_input(source: str, destination: Path) -> None:
    """Copy one VCF to an opaque, metadata-normalized inode.

    Hardlinks leak source ``st_nlink`` and reference/probe access times to the submission. Linux
    production uses ``copy_file_range`` when both paths share a filesystem and an explicit streamed
    copy when the reviewed data and sandbox live on different mounts; macOS development uses APFS
    clone-copy. Staging is trusted, unscored grader work, so it finishes before the end-to-end
    submission timer begins.
    """
    source_stat = os.stat(source, follow_symlinks=False)
    if not stat.S_ISREG(source_stat.st_mode):
        raise PermissionError("VCF source must be a regular file")
    if _UNPRIV is not None:
        uid, _ = _UNPRIV
        if source_stat.st_uid == uid or source_stat.st_mode & 0o022:
            raise PermissionError("deployed VCF source must not be writable by the submission uid")
        if not source_stat.st_mode & 0o004:
            raise PermissionError("deployed VCF source must be world-readable")

    if sys.platform == "darwin":
        subprocess.run(["/bin/cp", "-c", source, str(destination)], check=True,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    elif sys.platform.startswith("linux"):
        source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o444,
        )
        try:
            remaining = source_stat.st_size
            # Hyperfocal's worker Python and the sealed Debian verifier are both supported current
            # execution contexts, but only the latter exposes CPython's optional wrapper. The
            # streamed path is therefore live platform support, not legacy compatibility.
            if (os.fstat(destination_fd).st_dev == source_stat.st_dev
                    and hasattr(os, "copy_file_range")):
                while remaining:
                    copied = os.copy_file_range(
                        source_fd, destination_fd, min(remaining, 64 << 20),
                    )
                    if copied <= 0:
                        raise OSError(
                            errno.EIO, "copy_file_range ended before the VCF was staged",
                        )
                    remaining -= copied
            else:
                while remaining:
                    chunk = os.read(source_fd, min(remaining, 8 << 20))
                    if not chunk:
                        raise OSError(errno.EIO, "source ended before the VCF was staged")
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_fd, view)
                        if written <= 0:
                            raise OSError(errno.EIO, "destination stopped accepting VCF bytes")
                        view = view[written:]
                    remaining -= len(chunk)
            os.fchmod(destination_fd, 0o444)
        finally:
            os.close(destination_fd)
            os.close(source_fd)
    else:
        raise RuntimeError("input staging requires Linux copy_file_range or macOS clone-copy")

    os.utime(
        destination,
        ns=(_NORMALIZED_INPUT_TIME_NS, _NORMALIZED_INPUT_TIME_NS),
        follow_symlinks=False,
    )
    staged_stat = os.stat(destination, follow_symlinks=False)
    if (not stat.S_ISREG(staged_stat.st_mode)
            or staged_stat.st_ino == source_stat.st_ino
            or staged_stat.st_nlink != 1
            or staged_stat.st_size != source_stat.st_size
            or staged_stat.st_atime_ns != _NORMALIZED_INPUT_TIME_NS
            or staged_stat.st_mtime_ns != _NORMALIZED_INPUT_TIME_NS):
        raise RuntimeError("cloned VCF did not satisfy the opaque metadata contract")


def _copy_bounded_regular(source: Path, destination: Path) -> tuple[bool, str]:
    """Copy a submission output only if it is a no-follow regular file within the byte cap."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        if exc.errno in _INVALID_TREE_ERRNOS:
            return False, f"output is not a readable regular file: {exc}"
        raise
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    try:
        source_stat = os.fstat(fd)
        if not stat.S_ISREG(source_stat.st_mode):
            return False, "output is not a regular file"
        if source_stat.st_size > _MAX_SCORES_BYTES:
            return False, "output exceeds size limit"
        copied = 0
        with os.fdopen(fd, "rb", closefd=False) as source_fh, open(temporary, "xb") as target_fh:
            while True:
                chunk = source_fh.read(min(1 << 20, _MAX_SCORES_BYTES + 1 - copied))
                if not chunk:
                    break
                copied += len(chunk)
                if copied > _MAX_SCORES_BYTES:
                    return False, "output exceeds size limit"
                target_fh.write(chunk)
        os.replace(temporary, destination)
        return True, "ok"
    except OSError as exc:
        if exc.errno in _INVALID_TREE_ERRNOS:
            return False, f"output collection failed: {exc}"
        raise
    finally:
        os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass


def _drop_privileges():
    """preexec_fn that drops the child to the unprivileged uid/gid, or None (no drop).

    Runs in the forked child before exec: shed supplementary groups, then gid, then uid, so a
    submission executes with no privilege to read the grader package, the truth sidecars, or the
    other datasets even though it shares the container. Returns None when we are not root (dev/CI),
    making the whole mechanism a no-op there."""
    if _UNPRIV is None:
        return None
    uid, gid = _UNPRIV

    def _pre():
        try:
            os.setgroups([])
        except (OSError, AttributeError):
            pass
        os.setgid(gid)
        os.setuid(uid)
    return _pre


def run_submission(submission_snapshot: Path, vcf: Path, k: int, out: Path, *,
                   isolation: str, timeout: int = 3600) -> dict:
    """Run the submission's ``pca`` program: ``pca <vcf> <k> <out>``. Returns timing/status.

    The pinned environment interpreter has the only allowed third-party stack (NumPy/SciPy), so
    every submission runs through it regardless of its shebang.

    ISOLATION. Every invocation gets a fresh sandbox directory that holds both the opaque input
    name and output. The submission is handed relative paths inside that sandbox, so:
      * it sees an identical path shape for EVERY run -- ``<sandbox>/<rand>.vcf`` -- whether the
        input is a scored dataset or an identity probe, removing the old ``_staged`` vs
        direct-in-workdir tell that let a fit behave honestly only on probes;
      * ``realpath`` of the staged input is the sandbox path itself, never the real dataset name, so
        the adjacent ``<name>.vcf.truth.json`` sidecar is not reachable by dereferencing;
      * in the deployed UID-separated verifier, the sandbox parent is execute-only and the dataset
        directory is mode 0700, preventing listing or traversal to sibling probes/datasets.
    The caller creates one immutable, fd-verified submission snapshot before any static scan or
    probe. ``isolation`` is mandatory: Hyperfocal uses an allowlisted bubblewrap filesystem and
    fresh namespaces; Harbor uses a fresh no-network verifier container whose grader/truth are
    root-only, then drops uid, groups, capabilities, and no-new-privileges. Both modes expose only
    one pre-created output to the child. The output is copied back only after no-follow type/size
    validation; the process group is killed before collection so descendants cannot race it."""
    if isolation not in _ISOLATION_MODES:
        raise ValueError(f"unknown isolation mode: {isolation!r}")
    _assert_no_submission_processes()
    _assert_no_submission_ipc()
    if isolation == "verifier-container":
        _clear_chroot_state()
    interp = _RUNTIME_PYTHON
    out = Path(os.path.abspath(str(out)))
    _protect_grader_directory(out.parent)
    if isolation == "verifier-container":
        sandbox_parent = _CHROOT_ROOT / "work"
        _protect_grader_directory(sandbox_parent, sandbox_parent=True)
    else:
        sandbox_parent = _bwrap_work_root()
    sandbox = Path(tempfile.mkdtemp(prefix="_sbx_", dir=str(sandbox_parent)))
    try:
        sbx_in = sandbox / f"{secrets.token_hex(12)}.vcf"
        src = os.path.realpath(str(vcf))
        # Dataset staging is trusted grader work. An I/O, permission, or clone/link
        # failure is infrastructure and must bubble to adapter `errored`, not be
        # charged to the solver as a zero-scoring execution.
        _stage_input(src, sbx_in)
        sbx_out = sandbox / f"{secrets.token_hex(12)}.scores.tsv"
        output_fd = os.open(
            sbx_out,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.close(output_fd)
        os.chown(sbx_out, _UNPRIV[0], _UNPRIV[1])
        # The same-uid watchdog must be able to walk this exact invocation tree. The parent above
        # remains root-owned and non-listable, so ownership exposes no sibling probe/dataset.
        os.chown(sandbox, _UNPRIV[0], _UNPRIV[1])
        os.chmod(sandbox, 0o700)
        scratch = sandbox / "tmp"
        scratch.mkdir(mode=0o700)
        os.chown(scratch, _UNPRIV[0], _UNPRIV[1])
        shared_memory = scratch / "shm"
        shared_memory.mkdir(mode=0o700)
        os.chown(shared_memory, _UNPRIV[0], _UNPRIV[1])
        if isolation == "bwrap":
            execution_snapshot = sandbox / "submission"
            shutil.copytree(submission_snapshot, execution_snapshot)
            command = [
                interp, "-I", _GUARD_RUNNER,
                "/submission/pca", "/work/input.vcf", str(k), "/work/output.tsv",
            ]
            child_cwd = "/"
            child_home = "/tmp"
        else:
            chroot_submission = sandbox / "submission"
            shutil.copytree(submission_snapshot, chroot_submission)
            execution_snapshot = submission_snapshot
            inside = "/" + sandbox.relative_to(_CHROOT_ROOT).as_posix()
            command = [
                interp, "-I", _GUARD_RUNNER,
                f"{inside}/submission/pca",
                f"{inside}/{sbx_in.name}",
                str(k),
                f"{inside}/{sbx_out.name}",
            ]
            child_cwd = "/"
            child_home = f"{inside}/tmp"

        env = {
            "PATH": f"{_RUNTIME}/bin:/usr/bin:/bin",
            "HOME": child_home,
            "TMPDIR": child_home,
            "TMP": child_home,
            "TEMP": child_home,
            "OMP_NUM_THREADS": str(_SUBMISSION_NUMERICAL_THREADS),
            "OPENBLAS_NUM_THREADS": str(_SUBMISSION_NUMERICAL_THREADS),
            "MKL_NUM_THREADS": str(_SUBMISSION_NUMERICAL_THREADS),
            "VECLIB_MAXIMUM_THREADS": str(_SUBMISSION_NUMERICAL_THREADS),
        }

        cmd = _submission_argv(
            command, execution_snapshot, sbx_in, sbx_out, isolation,
        )

        t0 = time.perf_counter()
        # Drain stderr concurrently while retaining only a bounded tail. Calling
        # subprocess.run(..., stderr=PIPE) stores the entire hostile stream in
        # grader memory before slicing it and can be used to OOM the verifier.
        stderr_tail = bytearray()

        def drain_stderr(stream):
            while chunk := stream.read(64 * 1024):
                stderr_tail.extend(chunk)
                if len(stderr_tail) > _STDERR_TAIL_BYTES:
                    del stderr_tail[:-_STDERR_TAIL_BYTES]

        storage_roots = (
            (sandbox,)
            if isolation == "bwrap"
            else (*_chroot_writable_roots(), sandbox)
        )
        watchdog_cmd = _resource_watchdog_argv(
            cmd, env, child_cwd, storage_roots, timeout,
        )
        proc = subprocess.Popen(
            watchdog_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "PATH": f"{_RUNTIME}/bin:/usr/bin:/bin",
                "HOME": "/",
                "PYTHONSAFEPATH": "1",
            },
            cwd="/",
            start_new_session=(os.name == "posix"),
        )
        stderr_thread = threading.Thread(
            target=drain_stderr, args=(proc.stderr,), daemon=True,
        )
        stderr_thread.start()
        watchdog_error: str | None = None
        report: dict | None = None
        try:
            proc.wait(timeout=timeout + 15)
            payload = proc.stdout.read(16 * 1024 + 1)
            if proc.returncode != 0:
                watchdog_error = (
                    f"resource watchdog exited unexpectedly ({proc.returncode})"
                )
            else:
                try:
                    report = _decode_watchdog_report(payload, proc.pid)
                except RuntimeError as error:
                    watchdog_error = str(error)
        except subprocess.TimeoutExpired:
            watchdog_error = "resource watchdog exceeded its fail-safe deadline"
        finally:
            # The raw submission child initially shares the watchdog's process group. Kill that
            # group before uid cleanup so a watchdog killed while its reviewed chroot/setpriv
            # command was still root cannot leave an unaccounted process behind. Once setpriv has
            # completed, detached workers are capability-free pcasub processes and the uid sweep
            # below catches them even if they called setsid().
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            _terminate_submission_processes()
            _remove_submission_ipc()
            stderr_thread.join(timeout=5)
            if isolation == "verifier-container":
                _clear_chroot_state()
            if stderr_thread.is_alive():
                raise RuntimeError("submission stderr pipe remained open after process cleanup")
            proc.stdout.close()

        dt = time.perf_counter() - t0
        if report is None:
            report = {
                "child_returncode": 137,
                "timed_out": False,
                "violation": None,
                "monitor_error": watchdog_error or "resource watchdog returned no report",
                "peak_pss_bytes": 0,
                "peak_storage_bytes": 0,
            }
        elif watchdog_error:
            report["monitor_error"] = watchdog_error
        _assert_resource_monitor_complete(report)
        if report["timed_out"]:
            return {
                "seconds": dt,
                "returncode": 124,
                "stderr": "timeout",
                "peak_pss_bytes": report["peak_pss_bytes"],
                "peak_storage_bytes": report["peak_storage_bytes"],
            }
        returncode = report["child_returncode"]
        if report["violation"]:
            returncode = 137
            stderr_tail.extend(str(report["violation"]).encode())
        output_reason = None
        if returncode == 0 and os.path.lexists(sbx_out):
            copied, output_reason = _copy_bounded_regular(sbx_out, out)
            if copied:
                output_reason = None
        stderr = bytes(stderr_tail).decode("utf-8", "ignore")
        if output_reason:
            stderr = (stderr + f"\noutput rejected: {output_reason}")[-2000:]
        return {
            "seconds": dt,
            "returncode": returncode,
            "stderr": stderr,
            "peak_pss_bytes": report["peak_pss_bytes"],
            "peak_storage_bytes": report["peak_storage_bytes"],
        }
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def load_scores(path: Path, sample_ids: list[str], k: int) -> tuple[np.ndarray | None, str]:
    """Load and STRICTLY validate scores.tsv against the output contract, returning
    (scores in VCF-sample order, reason). The contract (instruction.md) is: header
    ``sample_id<TAB>PC1..PCk``, exactly one row per sample in VCF ``#CHROM`` order, unique IDs,
    exactly the requested number of PC columns. Enforcing it closes several score-fabrication
    seams the old lenient loader left open -- extra ``PC`` columns (an overcomplete random block
    that tries to span the reference subspace), duplicate sample rows silently overwriting each
    other, and unknown extra samples padding the file."""
    if not path.exists():
        return None, "no scores.tsv written"
    try:
        output_stat = path.lstat()
        if not stat.S_ISREG(output_stat.st_mode):
            return None, "scores.tsv is not a regular file"
        if output_stat.st_size > _MAX_SCORES_BYTES:
            return None, "scores.tsv too large"
    except FileNotFoundError:
        return None, "no scores.tsv written"
    expected_header = ["sample_id"] + [f"PC{i}" for i in range(1, k + 1)]
    S = np.empty((len(sample_ids), k), dtype=np.float64)
    try:
        with open(path, encoding="utf-8", errors="strict", newline=None) as stream:
            header_line = stream.readline(_MAX_SCORE_LINE_BYTES + 1)
            if not header_line:
                return None, "empty scores.tsv"
            if len(header_line.encode("utf-8")) > _MAX_SCORE_LINE_BYTES:
                return None, "score header exceeds line limit"
            header = header_line.rstrip("\r\n").split("\t")
            if header != expected_header:
                return None, f"header must be {'/'.join(expected_header)}"
            for row_index, expected_sid in enumerate(sample_ids):
                line = stream.readline(_MAX_SCORE_LINE_BYTES + 1)
                if not line:
                    return None, f"missing score row for {expected_sid!r}"
                if len(line.encode("utf-8")) > _MAX_SCORE_LINE_BYTES:
                    return None, "score row exceeds line limit"
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != k + 1:
                    return None, f"expected exactly {k} PC columns, got {len(parts) - 1}"
                if parts[0] != expected_sid:
                    return None, "samples not in VCF order"
                try:
                    S[row_index] = [float(value) for value in parts[1:]]
                except ValueError:
                    return None, f"non-numeric score for {expected_sid!r}"
            if stream.readline(1):
                return None, "extra score rows"
    except UnicodeDecodeError as error:
        return None, f"unreadable: {error}"
    if not np.isfinite(S).all():
        return None, "non-finite scores"
    return S, "ok"


# ---------------------------------------------------------------------------
# Reference cache
# ---------------------------------------------------------------------------

def _time_reference_subprocess(
    vcf: Path, k: int, workdir: Path, *, module: str = "reference.full_scan_pca",
    timeout: int = _REFERENCE_TIMEOUT_SECONDS,
) -> float:
    """Wall-clock of a reference implementation measured the SAME way the submission is measured:
    a fresh subprocess that pays interpreter startup, argv parsing, VCF read, and TSV write.

    ``module`` selects which reference is timed. Two are:
      * ``reference.full_scan_pca`` -- the naive baseline (reads the whole file). Anchors "parity".
      * ``reference.fast_pca``      -- the proven-optimal fast solution. Anchors "best achievable".
    Both anchors are needed because how much speed is *available* is a property of the fold, not a
    universal constant (see ``systems_quality``).

    The old baseline timed ``full_scan_fit`` **in-process**, excluding all of those, then compared
    it against the submission's end-to-end subprocess time -- so the submission was charged startup
    + serialization the reference never paid, an apples-to-oranges runtime comparison. Timing the
    reference through its own ``__main__`` on the same input closes that gap. Both sides read the
    same (by now warm) dataset pages, so neither gets a page-cache handicap. This runs with the
    grader's own environment (``reference`` importable) -- it is the grader's tool, not the
    submission -- and is cached per dataset, so it is measured once regardless of how many
    submissions are graded."""
    _assert_no_submission_processes()
    interp = _RUNTIME_PYTHON
    # Ensure the reference package is importable by ``-m`` in the child (repo root / grader_pkg).
    import reference as _ref
    pkg_parent = str(Path(_ref.__file__).resolve().parent.parent)
    env = {
        "PATH": f"{_RUNTIME}/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONPATH": pkg_parent,
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
    }
    out = workdir / f"_ref_{secrets.token_hex(8)}.tsv"
    descriptor = os.open(
        out,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    os.close(descriptor)
    cmd = [interp, "-m", module, os.path.abspath(str(vcf)), str(k), str(out)]
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, env=env, cwd=pkg_parent, timeout=timeout,
            text=True,
        )
        if completed.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            tail = (completed.stderr or "")[-1000:]
            raise RuntimeError(
                f"timed reference execution failed with code {completed.returncode} "
                f"({module}): {tail}"
            )
    finally:
        try:
            out.unlink()
        except OSError:
            pass
    return time.perf_counter() - t0


_REFERENCE_SHIM = (
    "#!/usr/bin/env python3\n"
    "import os, runpy, sys\n"
    "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
    "runpy.run_module({module!r}, run_name='__main__')\n"
)


def _reference_submission_dir(module: str, destination: Path) -> Path:
    """Materialize a reference implementation as a *submission-shaped* tree.

    The anchors must be timed through the very same execution path as the submission -- see
    ``ReferenceCache`` -- and that path takes a directory containing a ``pca`` entry point. The
    reference package is copied in whole (so relative imports keep working and the anchor is the
    real code, never a copy that can drift out of sync) and ``pca`` is a shim that runs the chosen
    module as ``__main__`` with the submission's own argv.

    This puts reference source inside a sandbox for the duration of an anchor run. That is safe:
    bubblewrap exposes only the invocation's *own* submission directory, anchor runs happen with no
    submission process alive (``run_submission`` asserts it), and the sandbox is destroyed
    afterwards -- so a submission never has a path to these bytes."""
    package = Path(__file__).resolve().parent.parent / "reference"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, destination / "reference",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "fast_submission"))
    entry = destination / "pca"
    entry.write_text(_REFERENCE_SHIM.format(module=module))
    # Normalize modes. ``_seal_grader_package`` makes the real ``reference/`` root-only (0700) so the
    # submission uid can never read the scoring truth in place -- and ``copytree`` faithfully
    # preserves that. Left alone, the anchor's own sandbox tree would be unreadable to the uid that
    # must execute it and unwalkable by the same-uid resource watchdog, which fails the run closed.
    # These bytes are a throwaway copy inside a per-invocation sandbox that no submission can reach,
    # so give them ordinary submission modes; the seal on the original directory is untouched.
    destination.chmod(0o755)
    for path in sorted((destination / "reference").rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    (destination / "reference").chmod(0o755)
    entry.chmod(0o755)
    return destination


_NOOP_SUBMISSION = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "import numpy, scipy.linalg, scipy.linalg.blas, scipy.linalg.lapack   # noqa: F401\n"
    "open(sys.argv[3], 'w').write('sample_id\\tPC1\\n')\n"
)
_NOOP_VCF = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2\n"
    "1\t1\t.\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\n"
)


def measure_execution_overhead(workdir: Path, *, isolation: str, timeout: int = 120) -> float:
    """Measure the fixed cost every timed sandbox invocation pays before PCA work begins.

    The submission and both reference anchors run through ``run_submission`` with the same isolation,
    so interpreter boot, numeric imports, sandbox entry, guard setup, and input staging are common.
    Subtracting their measured common baseline restores resolution on small folds without giving any
    side a different tax.

    Measured once, by running a no-op submission (imports the numeric stack, writes a header, exits)
    through the real execution path on a minimal VCF. This is scoring state: failure is an
    infrastructure error, never permission to silently change the runtime normalization."""
    root = Path(tempfile.mkdtemp(prefix="_overhead_", dir=str(workdir)))
    try:
        submission = root / "submission"
        submission.mkdir(mode=0o755)
        entry = submission / "pca"
        entry.write_text(_NOOP_SUBMISSION)
        entry.chmod(0o755)
        vcf = root / "tiny.vcf"
        vcf.write_text(_NOOP_VCF)
        os.chmod(vcf, 0o444)
        out = root / "out.tsv"
        best: float | None = None
        for _ in range(3):
            run = run_submission(
                submission, vcf, 1, out, isolation=isolation, timeout=timeout,
            )
            if run["returncode"] != 0:
                raise RuntimeError(
                    "execution-overhead calibration failed with code "
                    f"{run['returncode']}: {run['stderr'][-400:]}"
                )
            seconds = float(run["seconds"])
            if not np.isfinite(seconds) or seconds <= 0.0:
                raise RuntimeError(
                    f"execution-overhead calibration returned invalid time {seconds!r}"
                )
            best = seconds if best is None else min(best, seconds)
        if best is None or best > 30.0:
            raise RuntimeError(
                f"execution-overhead calibration is outside its contract: {best!r}s"
            )
        return float(best)
    finally:
        shutil.rmtree(root, ignore_errors=True)


class ReferenceCache:
    """Per-VCF reference facts: the truth subspace plus BOTH speed anchors.

    ``scores``/``weights`` (the truth subspace) are computed in-process. Two wall-clocks are
    measured, and -- crucially -- they are measured by running the reference implementations
    **through the identical sandbox the submission runs in** (``run_submission``):

      * ``seconds``      -- the full-scan baseline: what a correct naive implementation costs.
      * ``fast_seconds`` -- the fast reference: the best speed actually achievable on this fold.

    Timing them as plain grader subprocesses was subtly but badly wrong. The submission pays sandbox
    entry -- setpriv/bwrap, the guard runner's seccomp install and eager numeric preload, staging --
    which measured at ~2 s, while the anchors paid none of it. So the same code scored ~2.4x slower
    as a submission than as an anchor, and no program could reach the ceiling. Subtracting an
    estimated overhead cannot rescue this: recovering ~0.3 s of work from a ~2.3 s measurement
    amplifies any error in the estimate tenfold. Running the anchors through the same path makes the
    overhead genuinely COMMON, so it cancels in the comparison instead of being guessed at, and a
    submission that matches the fast reference's code lands at exactly 1.0.

    Both anchors are submission-independent and cached per dataset, so this costs one extra
    sandboxed run per fold for the whole grading session, not per submission.

    Both anchors are required scoring state. Failure of either is an infrastructure error; treating
    a broken fast anchor as "no achievable speedup" would inflate slow submissions' systems credit."""

    def __init__(self):
        self._store: dict[tuple, dict] = {}

    def get(
        self, vcf: Path, k: int, workdir: Path | None = None, *,
        isolation: str | None = None,
        timeout: int = _REFERENCE_TIMEOUT_SECONDS,
    ) -> dict:
        # Anchor times depend on the execution boundary. A cache shared across local validation and
        # bwrap grading must never reuse plain-subprocess timings for sandboxed submissions.
        key = (*_file_identity(vcf), int(k), isolation)
        if key not in self._store:
            sample_ids, scores, kept, spectrum = full_scan_fit(vcf, k)
            weights = structure_weights(spectrum, scores.shape[1])[:scores.shape[1]]
            if not np.any(weights > 0):
                raise ValueError(f"reference PCA has no structured PCs: {vcf}")
            wd = workdir or Path(tempfile.gettempdir())
            wd.mkdir(parents=True, exist_ok=True)
            secs = self._time_anchor(
                "reference.full_scan_pca", vcf, k, wd, isolation=isolation, timeout=timeout,
            )
            fast_secs = self._time_anchor(
                "reference.fast_pca", vcf, k, wd, isolation=isolation, timeout=timeout,
            )
            self._store[key] = {"sample_ids": sample_ids, "scores": scores,
                                "kept": kept, "seconds": secs, "fast_seconds": fast_secs,
                                "weights": weights, "spectrum": spectrum}
        return self._store[key]

    @staticmethod
    def _time_anchor(module: str, vcf: Path, k: int, workdir: Path, *,
                     isolation: str | None, timeout: int) -> float:
        """Time one reference through the submission's own execution path when we have one.

        ``isolation`` is None only for trusted local/CI callers that never run a sandbox at all; in
        that case both anchors and the submission are plain subprocesses, so a plain subprocess is
        still the apples-to-apples measurement."""
        if isolation is None:
            return _time_reference_subprocess(vcf, k, workdir, module=module, timeout=timeout)
        staging = Path(tempfile.mkdtemp(prefix="_anchor_", dir=str(workdir)))
        try:
            snapshot = _reference_submission_dir(module, staging / "submission")
            out = staging / "anchor.tsv"
            run = run_submission(
                snapshot, vcf, k, out, isolation=isolation, timeout=timeout,
            )
            if run["returncode"] == 124:
                # A timed-out anchor is a grading-BUDGET event, not an infrastructure failure. The
                # plain-subprocess path (`_time_reference_subprocess`) already surfaces a timeout as
                # ``subprocess.TimeoutExpired``; the sandboxed path must too, or the callers'
                # ``except subprocess.TimeoutExpired`` handlers -- which convert an exhausted-budget
                # anchor timeout into ``GradingDeadlineExhausted`` and otherwise into a clearly
                # attributed infra error -- would never fire in production. Left as a bare
                # ``RuntimeError`` it escaped both handlers and erased every already-earned fold when
                # the budget capped a late anchor's timeout low enough to time out.
                raise subprocess.TimeoutExpired(f"reference anchor {module}", timeout)
            if run["returncode"] != 0:
                raise RuntimeError(
                    f"sandboxed reference anchor {module} failed: {run['stderr'][-400:]}"
                )
            seconds = float(run["seconds"])
            if not np.isfinite(seconds) or seconds <= 0.0:
                raise RuntimeError(
                    f"sandboxed reference anchor {module} returned invalid time {seconds!r}"
                )
            return seconds
        finally:
            shutil.rmtree(staging, ignore_errors=True)


# Below this much achievable speedup, the "a full scan earns half credit" anchor stops being
# meaningful -- parity and optimal are the same program -- and the scale hands over to the plain
# slowness rule instead. See ``systems_quality`` for how the two are joined.
_MIN_SCORED_HEADROOM_OCTAVES = 0.3
# With no parity headroom, the historical operational zero is 8x slower than optimal: 3 octaves at
# 1/3 per octave. The final hard clamp is softened below, but this remains the rate at which gross
# slowness is charged. Above the 1.5-octave join, full-scan parity necessarily binds instead: the
# full scan can itself be more than 8x slower than optimal and must still earn 0.5.
_SLOWNESS_DECAY_OCTAVES = 3.0
# Softness of the successful-runtime tail, in systems-quality units. The old piecewise-linear curve
# clipped every successful run beyond its zero crossing to the same value. A smooth positive-part
# approximation with an algebraic tail preserves a materially reportable ordering even far beyond
# that boundary while keeping catastrophically slow work worth very little.
_SYSTEMS_TAIL_SOFTNESS = 0.02


def _soft_positive_tail(raw: float, parity_value: float) -> float:
    """Map the post-parity linear value to a positive asymptotic tail.

    ``0.5 * (x + hypot(x, softness))`` is a smooth positive-part approximation whose negative
    branch decays algebraically rather than exponentially. That distinction is load-bearing: the
    previous softplus gave a 649-second exact solver only ~4e-8 extra reward, which the five-decimal
    dataset/rollup serialization erased back to the same 0.10000 as arbitrarily worse successes.
    Normalising at ``parity_value`` preserves the full-scan anchor exactly; execution failures bypass
    ``systems_quality`` and remain exact zero.
    """
    softness = _SYSTEMS_TAIL_SOFTNESS
    numerator = 0.5 * (raw + math.hypot(raw, softness))
    denominator = 0.5 * (parity_value + math.hypot(parity_value, softness))
    return float(parity_value * numerator / denominator)


def systems_quality(t_sub: float, t_full: float, t_fast: float,
                    overhead: float = 0.0) -> float:
    """Score end-to-end speed against what is actually ACHIEVABLE on this fold, in [0, 1].

    The old metric mapped "8x faster than the full scan" to 1.0. That bakes in an assumption that
    turned out to be false: how much speed is available is a property of the *fold*, not a constant.
    Measured through this grader, the proven-optimal fast reference is only ~3x faster than the full
    scan on the large folds and is *at or below parity* on the small ones (its sampling overhead
    costs more than the I/O it avoids on a 60k-variant file). So "8x faster than a full scan" was
    unreachable everywhere -- the top of the scale could not be earned by ANY correct program, and
    even a perfect solution was capped well below a reward of 1.

    So we anchor on the two references instead:

        full scan  -> 0.5   (parity: a correct naive implementation)
        best achievable -> 1.0   (the fast reference, or the full scan where sampling cannot win)

    and interpolate linearly in log-time between them. Consequences, all intended:

      * A submission that matches the best achievable speed scores 1.0 -- on EVERY fold, whatever
        its size. The ceiling is earnable, so a perfect solution can actually score a perfect reward.
      * Where sampling genuinely wins (big folds), a full scan scores 0.5: you must skip work.
      * Where sampling cannot win (small folds), the full scan IS optimal, so it scores 1.0 -- we do
        not punish a program for failing to achieve a speedup that does not exist.
      * Being slower than optimal decays smoothly; being faster than the reference clips at 1.0.

    Those two anchors used to be implemented as two BRANCHES, switching on whether the measured
    headroom cleared 0.3 octaves. That made the score discontinuous in a noisy measurement. Scored
    through this grader, a full scan jumped from 0.900 to 0.500 -- across the 0.75 mastery bar -- when
    the fold's measured speedup moved from 1.2305x to 1.2315x. The deployed folds sit exactly there:
    ``rare_structure`` measured 0.246 octaves and ``spatial_ld`` 0.360, so two folds offering nearly
    the same real speedup (1.19x vs 1.28x) scored a naive full scan 0.923 (mastery) versus 0.531
    (failure). Worse, the classification is not even a stable property of a fold: ``ill_conditioned``
    measured "fast reference below parity" on one box and 0.405 octaves of real headroom on another,
    which under the branching rule flipped a naive submission between 1.0 and 0.5 based on nothing it
    did. Fold difficulty must live in the metric, not in which side of a threshold the timing noise
    landed on.

    So there is one scale, joined rather than branched. The metric has exactly two calibrations, and
    the scale is the continuous interpolation between them:

      * "a full scan earns half"  -> penalty ``0.5/headroom`` per octave behind optimal;
      * "8x slower is operationally zero" -> penalty ``1/_SLOWNESS_DECAY_OCTAVES`` per octave.

    These are not rival rules. They AGREE exactly at 1.5 octaves of headroom, and the parity rule is
    the binding one above that. They diverge as headroom shrinks, for a reason worth stating: on a
    fold offering almost no speedup, "parity" and "optimal" are the same program, so the parity
    anchor no longer says anything and ``0.5/headroom`` diverges. Simply flooring the headroom is
    therefore NOT the fix -- it pins the scale at its steepest exactly where the fold offers least,
    charging a submission more for being 1.3x slow on a fold with nothing to win than on one with a
    3x prize. Measured, that inversion cost the gold reference its ``variable_width`` mastery
    (0.87 -> 0.40) for a shortfall a large fold would have scored ~0.85.

    So below ``_MIN_SCORED_HEADROOM_OCTAVES`` the parity rule hands over: the penalty rate falls
    linearly from ``0.5/_MIN_SCORED_HEADROOM_OCTAVES`` to the plain slowness rate as the achievable
    speedup goes to zero. Every compatible property survives exactly -- matching optimal is 1.0 on
    every fold, a full scan on a fold with real headroom is still exactly 0.5, and the score is now
    continuous in the measured headroom, so noise near a threshold can no longer move a category
    across the mastery bar.

    THE SCALE MUST NOT SATURATE, which a single straight line could not avoid. Anchoring only on
    parity gives a rate of ``0.5/headroom``, and a straight line from 1.0 through 0.5 at parity hits
    ZERO at ``2 * headroom`` -- which on a low-headroom fold is absurdly close to optimal. Measured
    on the deployed folds, that put the zero at **1.65x** slower than optimal on ``spatial_ld``,
    1.75x on ``ill_conditioned`` and 2.4x on ``admixed``, versus 5.4x on ``scaling`` and 47x on
    ``biobank``. So the documented "8x slower earns nothing" was true only where ``0.5/h == 1/3``,
    i.e. h = 1.5; everywhere below it the scale zeroed early and became a sign bit reading "are you
    within ~2x of optimal?".

    That is not hypothetical: an Opus rollout (run_019f6688) landed **eleven datasets at exactly
    time_quality 0.000 while running only 2-3x slower than the plain full scan**. The env could not
    tell it apart from a program 100x slower, and it scored below the naive full scan for work that
    was merely unoptimized rather than catastrophic.

    So the curve BENDS at parity instead of running straight through it: it is piecewise linear
    through ``(0, 1.0)``, ``(headroom, parity_value)`` and ``(zero_at, 0.0)``. Now BOTH documented
    calibrations hold simultaneously on EVERY fold -- parity is exactly 0.5 wherever headroom is
    real, and 8x behind optimal reaches the small asymptotic tail -- while the whole range remains
    resolvable.

    ``zero_at = max(8x, 2 * headroom)`` locates the former hard zero; the soft tail keeps ordering
    beyond it. Above 1.5 octaves parity must dominate: a full scan can itself be more
    than 8x behind optimal, so assigning that runtime both 0.5 and 0.0 would be impossible.

    This rewards picking the *right strategy per fold* -- sample where it pays, scan where it does
    not -- which is the real engineering judgment the task is about.

    All three times come from the SAME execution path (``ReferenceCache`` runs the anchors through
    the submission's own sandbox), so the fixed cost of entering it -- setpriv/bwrap, the guard
    runner, staging, the numeric import -- is common to all three and cancels. ``overhead`` is that
    shared baseline, subtracted only to stop a ~2 s fixed cost from compressing the ratios on the
    smaller folds; because it is common, an error in it perturbs all three sides alike instead of
    handing one of them an advantage.
    """
    if (not all(np.isfinite(value) for value in (t_sub, t_full, t_fast, overhead))
            or min(t_sub, t_full, t_fast) <= 0.0 or overhead < 0.0):
        raise ValueError("systems-quality times must be finite and positive; overhead must be nonnegative")
    floor = _WORK_TIME_FLOOR_SECONDS
    w_sub = max(t_sub - overhead, floor)
    w_full = max(t_full - overhead, floor)
    w_fast = max(t_fast - overhead, floor)
    # The best speed anyone has been shown to reach here. On folds where the fast reference loses to
    # a plain scan, the plain scan is the thing to match.
    w_best = min(w_fast, w_full)
    headroom = math.log2(w_full / w_best)          # octaves of speedup this fold actually offers
    behind = max(math.log2(w_sub / w_best), 0.0)   # octaves the submission is behind optimal
    # SEGMENT 1, optimal -> parity. Unchanged: the rate is anchored on parity where headroom is
    # real, and hands over continuously to the plain slowness rate as headroom vanishes, so a
    # submission is always charged in proportion to the shortfall it actually has.
    slowness_rate = 1.0 / _SLOWNESS_DECAY_OCTAVES
    if headroom >= _MIN_SCORED_HEADROOM_OCTAVES:
        rate = 0.5 / headroom
    else:
        parity_rate = 0.5 / _MIN_SCORED_HEADROOM_OCTAVES
        blend = headroom / _MIN_SCORED_HEADROOM_OCTAVES          # 1 at the handover, 0 at no headroom
        rate = slowness_rate + (parity_rate - slowness_rate) * blend
    if behind <= headroom:
        return float(np.clip(1.0 - rate * behind, 0.0, 1.0))

    # SEGMENT 2, parity -> zero. This bend is what stops the scale saturating: continuing segment 1
    # straight would hit zero at 2*headroom, i.e. 1.65x slower than optimal on `spatial_ld`.
    parity_value = max(0.0, 1.0 - rate * headroom)
    zero_at = max(_SLOWNESS_DECAY_OCTAVES, 2.0 * headroom)
    span = zero_at - headroom
    raw = parity_value * (zero_at - behind) / span
    return float(np.clip(_soft_positive_tail(raw, parity_value), 0.0, 1.0))


# What a correct program earns for correctness ALONE, at any speed whatsoever.
#
# This term is paid unconditionally, so it must be sized by exactly one question: what is the least
# that keeps "correct but slow" distinguishable from "wrong"? That distinction is worth preserving --
# a scientifically sound fit that is too slow is a different, more informative failure than a fast
# wrong answer, and collapsing both to 0 would throw that signal away.
#
# Anything ABOVE that minimum is paid for systems work the submission did not do. It was 0.30, and
# 0.30 bought a lot of nothing: even when `systems_quality` reaches ZERO on the systems axis, such
# a submission still banked 30% of the reward. Measured through the real
# grader, a correct-but-naive full scan -- which skips sampling, parallelism and memory discipline
# entirely -- scored 0.6841 while failing 16 of 18 categories. The first line of the task is "Build a
# FAST population-genetics PCA"; a program that has not built a fast PCA should not bank two thirds
# of the score for it.
#
# 0.10 is small but unmistakably nonzero, so it still separates correct-from-wrong, and it cannot
# compete with genuine optimization: the climb from full-scan parity to the achievable ceiling is now
# worth ~0.45 rather than ~0.35, which is the gradient the task is actually about.
_CORRECTNESS_FLOOR = 0.10

# Accuracy band over which the systems term is UNLOCKED.
#
# Multiplying by `accuracy` alone was not enough to make speed worthless to a broken PCA. It scaled
# the systems payout down, but it never switched it off: at accuracy 0.5 a maximally optimized
# program still banked 0.50 -- five times what a scientifically perfect but slow fit earned. That is
# backwards. A PCA that does not recover the population structure has not computed the thing being
# timed, so its wall time measures nothing; paying it for speed pays for a fast wrong answer.
#
# So the systems term is gated by a ramp on accuracy rather than a plain product. Below
# `_SYSTEMS_UNLOCK_MIN` the speed term is worth exactly zero and no amount of optimization moves the
# score; above `_SYSTEMS_UNLOCK_FULL` it is paid in full.
#
# The anchors come from measured behaviour, not taste. Genuine implementations saturate at accuracy
# 1.000 on essentially every fold, and `scripts/validate.py` already encodes 0.90 as the worst any
# real PCA may score (FN_MIN) -- so full unlock at 0.90 leaves every genuine solve untouched and the
# ramp bites only submissions that are actually wrong. Impostors land far below: the MAF-filter cheat
# scored 0.005 on `rare_structure`. The band between is a ramp, not a cliff, so a submission near the
# boundary loses a little rather than falling off an edge.
_SYSTEMS_UNLOCK_MIN = 0.75
_SYSTEMS_UNLOCK_FULL = 0.90


def systems_unlock(accuracy: float) -> float:
    """How much of the systems term a fit of this accuracy has earned the right to be paid."""
    accuracy = float(np.clip(accuracy, 0.0, 1.0))
    span = _SYSTEMS_UNLOCK_FULL - _SYSTEMS_UNLOCK_MIN
    return float(np.clip((accuracy - _SYSTEMS_UNLOCK_MIN) / span, 0.0, 1.0))


def dataset_score(accuracy: float, runtime_quality: float, gate_product: float) -> float:
    """Bounded score: scientific accuracy UNLOCKS a demanding systems-quality term.

    Correctness is the key, not a coefficient. It pays a small flat amount on its own
    (`_CORRECTNESS_FLOOR`), and it unlocks the systems term -- which is the large majority of a
    perfect run (0.90 of it) and the thing the task is actually about. A broken PCA cannot buy any
    of that with speed: below `_SYSTEMS_UNLOCK_MIN` its systems term is zeroed, capping it at
    `accuracy * _CORRECTNESS_FLOOR` < the floor a correct-but-slow fit earns. Wrong-and-fast is
    therefore strictly worse than right-and-slow, at every speed.
    """
    accuracy = float(np.clip(accuracy, 0.0, 1.0))
    runtime_quality = float(np.clip(runtime_quality, 0.0, 1.0))
    gate_product = float(np.clip(gate_product, 0.0, 1.0))
    earned_systems = systems_unlock(accuracy) * runtime_quality
    return accuracy * (_CORRECTNESS_FLOOR
                       + (1.0 - _CORRECTNESS_FLOOR) * earned_systems) * gate_product


_METHOD_FACTOR_FLOORS = {
    # These are scientific/model weaknesses, not integrity failures.  A completely wrong
    # method result should be strongly discounted while retaining a small learning signal.
    "hwe_norm": 0.15,
    "coverage": 0.15,
    # Parser representation fragility is narrower than computing the wrong PCA object.
    "representation_equivalence": 0.35,
}
_REPRESENTATION_CATEGORIES = frozenset({"messy", "variable_width"})


def effective_gate_product(gates: dict[str, dict], category: str) -> float:
    """Combine integrity gates and scientific method factors.

    Dependency/output violations remain hard zeroes.  Method probes have explicit nonzero
    floors so ordinary algorithmic shortcomings receive bounded partial credit.
    """
    product = 1.0
    for name, gate in gates.items():
        # Representation fragility is a distinct parser capability, not evidence that the
        # submission computed the wrong PCA on a clean input.  Its full discount belongs on the
        # scored parser-stress categories; globally multiplying it into every unrelated category
        # would erase valid scientific work and destroy useful partial credit.
        if name == "representation_equivalence" and category not in _REPRESENTATION_CATEGORIES:
            continue
        factor = float(np.clip(gate["factor"], 0.0, 1.0))
        factor = max(_METHOD_FACTOR_FLOORS.get(name, 0.0), factor)
        product *= factor
    return float(np.clip(product, 0.0, 1.0))


def _gate_report(gate: dict) -> dict:
    """Keep a gate's actionable evidence while normalising its verdict fields.

    Gate implementations already return bounded, JSON-safe diagnostics such as rejected imports,
    invalid-output reasons, probe agreement, and per-repetition severities.  Dropping those fields
    here made a fired gate opaque even though the platform reporter explicitly surfaces them.
    """
    report = dict(gate)
    report["factor"] = round(float(gate["factor"]), 4)
    report["severity"] = round(float(gate.get("severity", 0.0)), 4)
    return report


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------

def grade_one_dataset(submission_dir: Path, vcf: Path, truth: dict, k: int,
                      cache: ReferenceCache, workdir: Path,
                      lib: dict | None = None,
                      probe_gates: dict | None = None,
                      probe_runtime_share: dict[str, float] | None = None,
                      *, isolation: str, budget: GradingBudget | None = None,
                      execution_overhead: float = 0.0) -> dict:
    source_vcf = vcf
    vcf = _normalize_vcf(vcf, workdir)
    reference_timeout = (budget.invocation_timeout(_REFERENCE_TIMEOUT_SECONDS)
                         if budget is not None else _REFERENCE_TIMEOUT_SECONDS)
    try:
        ref = cache.get(vcf, k, workdir, isolation=isolation, timeout=reference_timeout)
    except subprocess.TimeoutExpired as error:
        if budget is not None and budget.exhausted():
            raise GradingDeadlineExhausted(
                "global grading deadline exhausted during reference measurement"
            ) from error
        raise RuntimeError(f"trusted reference exceeded {reference_timeout}s") from error
    sample_ids = ref["sample_ids"]

    # Gate: forbidden libraries (static, cheap; computed once per submission by the caller).
    if lib is None:
        lib = library_scan.scan(submission_dir)

    # Run the submission and time it. ``run_submission`` stages the input into an opaque,
    # sidecar-free sandbox path identical in shape to a probe's, so the submission can neither
    # reach the truth-label sidecar nor tell a scored dataset from a probe by its path. The
    # OUTPUT name is a random token too -- the old ``<dataset-stem>.scores.tsv`` handed the real
    # dataset name straight back to the submission, from which it could derive the sibling
    # ``data/<stem>.vcf.truth.json`` and fabricate a label-separating score matrix. The grader
    # tracks the dataset name itself (``vcf.stem`` below); the submission never sees it.
    out = workdir / f"{secrets.token_hex(12)}.scores.tsv"
    if out.exists():
        out.unlink()
    submission_timeout = (budget.invocation_timeout(_DATASET_SUBMISSION_TIMEOUT_SECONDS)
                          if budget is not None else _DATASET_SUBMISSION_TIMEOUT_SECONDS)
    run = run_submission(
        submission_dir, vcf, k, out, isolation=isolation, timeout=submission_timeout,
    )

    S, reason = load_scores(out, sample_ids, k)
    validity = _validity(S, reason, run, k)

    if S is None or not np.isfinite(S).all():
        accuracy = 0.0
        acc_detail = {"accuracy": 0.0, "reason": reason}
    else:
        acc_detail = subspace_accuracy(S, ref["scores"], ref["weights"])
        acc_detail["pc_resolution"] = per_pc_report(S, ref["scores"], ref["spectrum"])
        accuracy = acc_detail["accuracy"]

    # Population-structure recovery vs known labels -- reported for every labelled dataset,
    # and the headline biology check on real data. Measured "within tolerance" by comparing
    # the submission's separation to the full-scan reference's on the same labels.
    structure = None
    labels = truth.get("sample_pop")
    if S is not None and np.isfinite(S).all() and labels is not None \
            and len(set(x for x in labels if x >= 0)) >= 2:
        lab = np.asarray(labels)
        sub_sep = population_separation(S, lab)
        ref_sep = population_separation(ref["scores"], lab)
        super_ratio = (sub_sep["label_projection"] / ref_sep["label_projection"]
                       if ref_sep["label_projection"] > 1e-8 else 1.0)
        super_ratio = float(np.clip(super_ratio, 0.0, 1.0))
        ratio = super_ratio
        hierarchy = None
        nested_labels = truth.get("sample_subpop")
        if nested_labels is not None:
            nested = np.asarray(nested_labels)
            if nested.shape != lab.shape:
                raise ValueError("private nested population labels do not align with samples")
            sub_within = within_group_label_projection(S, lab, nested)
            ref_within = within_group_label_projection(ref["scores"], lab, nested)
            if ref_within < 0.05:
                raise ValueError(
                    "reviewed nested population signal is too weak for stable scoring"
                )
            within_ratio = sub_within / ref_within
            within_ratio = float(np.clip(within_ratio, 0.0, 1.0))
            hierarchy_weight = float(truth["spec"].get("hierarchy_weight", 0.45))
            if not np.isfinite(hierarchy_weight) or not 0.0 <= hierarchy_weight <= 0.7:
                raise ValueError("invalid private hierarchy_weight")
            ratio = (1.0 - hierarchy_weight) * super_ratio + hierarchy_weight * within_ratio
            hierarchy = {
                "submission_within_projection": round(sub_within, 6),
                "reference_within_projection": round(ref_within, 6),
                "within_projection_ratio_vs_reference": round(within_ratio, 4),
                "hierarchy_weight": hierarchy_weight,
            }
        structure_weight = float(truth["spec"].get("structure_weight", 0.15))
        if not np.isfinite(structure_weight) or not 0.0 <= structure_weight <= 0.4:
            raise ValueError("invalid private structure_weight")
        subspace_only = accuracy
        accuracy = (1.0 - structure_weight) * subspace_only + structure_weight * ratio
        acc_detail["subspace_accuracy"] = subspace_only
        acc_detail["label_projection_ratio"] = ratio
        acc_detail["structure_weight"] = structure_weight
        structure = {"submission": sub_sep, "reference": ref_sep,
                     "superpopulation_projection_ratio_vs_reference": round(super_ratio, 4),
                     "combined_projection_ratio_vs_reference": round(ratio, 4),
                     "hierarchy": hierarchy}

    overhead = probe_runtime_share or {"submission_seconds": 0.0, "reference_seconds": 0.0}
    scored_submission_seconds = run["seconds"] + overhead["submission_seconds"]
    scored_reference_seconds = ref["seconds"] + overhead["reference_seconds"]
    scored_fast_reference_seconds = ref["fast_seconds"] + overhead["reference_seconds"]
    # Score speed against what is ACHIEVABLE on this fold (full scan = parity, fast reference = the
    # earnable ceiling). All three times come from the same sandboxed path, so the fixed entry cost
    # is common and cancels; a submission matching the fast reference's code reaches exactly 1.0.
    # A failed invocation did not perform a usable timed computation.  Its short wall time is a
    # failure diagnostic, not systems quality; reporting it as 1.0 made policy exits look faster
    # than the reference even though validity correctly zeroed their reward.
    runtime_quality = (
        systems_quality(
            scored_submission_seconds, scored_reference_seconds, scored_fast_reference_seconds,
            overhead=execution_overhead,
        )
        if run["returncode"] == 0 else 0.0
    )

    gates = {"library_scan": lib, "validity": validity}
    # Object-identity probes are run once per submission. HWE/coverage describe the fitted
    # scientific object globally; representation equivalence is applied only to the two
    # parser-stress categories by ``effective_gate_product``.
    if probe_gates:
        gates.update(probe_gates)

    gate_product = effective_gate_product(gates, truth["spec"]["category"])

    reward = dataset_score(accuracy, runtime_quality, gate_product)
    return {
        "dataset": source_vcf.stem,
        "category": truth["spec"]["category"],
        "k": k,
        "weight": truth["spec"].get("weight", 1.0),
        # Preserve the actual scalar used by rollup. Readable console lines format these values,
        # but rounding the data object here destroyed ordering in the successful-runtime tail.
        "reward": float(reward),
        "accuracy": float(accuracy),
        "time_quality": float(runtime_quality),
        "gate_product": float(gate_product),
        "sub_seconds": float(scored_submission_seconds),
        "ref_seconds": float(scored_reference_seconds),
        "primary_sub_seconds": float(run["seconds"]),
        "primary_ref_seconds": float(ref["seconds"]),
        # The earnable ceiling on this fold: the fast reference's end-to-end time. A submission that
        # matches it scores systems_quality 1.0.
        #
        # Reported on the SAME BASIS as `sub_seconds` and `ref_seconds` -- i.e. the scored time, with
        # the probe share included -- so that
        #
        #     systems_quality(sub_seconds, ref_seconds, fast_ref_seconds,
        #                     overhead=execution_overhead_seconds)
        #
        # reproduces `time_quality` exactly. It previously reported the RAW anchor while its two
        # neighbours were scored, which silently made the speed score unreproducible from the
        # artifact: a reader recomputing it got systematically LOW answers (0.305 against a reported
        # 0.594 on high_rank) and could not tell whether the grader or their own arithmetic was
        # wrong. The comment here claimed the score was "auditable" while the mixed bases guaranteed
        # it was not. `test_reported_timings_reproduce_time_quality` now enforces the property this
        # comment asserts.
        "fast_ref_seconds": float(scored_fast_reference_seconds),
        "execution_overhead_seconds": float(execution_overhead),
        "probe_runtime_share": {
            "submission_seconds": float(overhead["submission_seconds"]),
            "reference_seconds": float(overhead["reference_seconds"]),
        },
        "ref_kept": ref["kept"],
        "accuracy_detail": acc_detail,
        "structure": structure,
        "gates": {name: _gate_report(gate) for name, gate in gates.items()},
        "run": {
            "returncode": run["returncode"],
            "stderr_tail": run["stderr"][-400:],
            # The watchdog polls only the scored primary invocation. Probe invocations are
            # accounted for in runtime but intentionally do not contribute to these diagnostics.
            "sampled_primary_invocation_peak_pss_bytes": run["peak_pss_bytes"],
            "sampled_primary_invocation_peak_storage_bytes": run["peak_storage_bytes"],
        },
    }


def _validity(S, reason, run, k) -> dict:
    if run["returncode"] != 0:
        return {"name": "validity", "factor": 0.0, "severity": 1.0,
                "reason": f"nonzero exit {run['returncode']}"}
    if S is None:
        return {"name": "validity", "factor": 0.0, "severity": 1.0, "reason": reason}
    if not np.isfinite(S).all():
        return {"name": "validity", "factor": 0.0, "severity": 1.0,
                "reason": "non-finite scores"}
    if S.shape[1] != k:
        return {"name": "validity", "factor": 0.0, "severity": 1.0,
                "reason": f"expected exactly {k} PCs, got {S.shape[1]}"}
    return {"name": "validity", "factor": 1.0, "severity": 0.0, "reason": "ok"}


# ---------------------------------------------------------------------------
# Roll-up across datasets
# ---------------------------------------------------------------------------

def rollup(per_dataset: list[dict]) -> dict:
    cats: dict[str, list[dict]] = {}
    for d in per_dataset:
        cats.setdefault(d["category"], []).append(d)
    cat_scores = {}
    cat_weights = {}
    for cat, ds in cats.items():
        w = np.array([d["weight"] for d in ds], dtype=float)
        r = np.array([d["reward"] for d in ds], dtype=float)
        cat_scores[cat] = float((w * r).sum() / w.sum()) if w.sum() > 0 else 0.0
        cat_weights[cat] = float(w.mean()) if w.size else 0.0
    total_category_weight = sum(cat_weights.values())
    benchmark = (sum(cat_scores[cat] * cat_weights[cat] for cat in cat_scores)
                 / total_category_weight if total_category_weight > 0 else 0.0)
    # These are the actual RL scalars, not presentation strings. Formatting belongs in the
    # reporting layer; rounding here can collapse distinct successful programs into one reward.
    return {"benchmark": float(benchmark),
            "category_scores": cat_scores}


def _invalid_submission_result(data_dir: Path, reason: str, *, isolation: str) -> dict:
    """Build the complete zero-score contract for a rejected submission snapshot.

    Reading and validating the generated dataset contracts remains a trusted grader
    operation. Missing/malformed data therefore still raises and is reported as an
    infrastructure error by the adapter; only ``InvalidSubmission`` reaches here.
    """
    per_dataset = []
    categories = set()
    for truth_path in sorted(Path(data_dir).glob("*.truth.json")):
        vcf = Path(str(truth_path).removesuffix(".truth.json"))
        if not vcf.is_file():
            raise RuntimeError(f"generated dataset is missing for {truth_path.name}")
        truth = json.loads(truth_path.read_text())
        spec = truth["spec"]
        category = spec["category"]
        dataset_k = spec.get("k")
        weight = float(spec.get("weight", 1.0))
        if (not isinstance(category, str) or not category
                or not isinstance(dataset_k, int) or isinstance(dataset_k, bool) or dataset_k < 1
                or not np.isfinite(weight) or weight <= 0):
            raise RuntimeError(f"invalid generated truth contract: {truth_path.name}")
        categories.add(category)
        per_dataset.append({
            "dataset": vcf.stem,
            "category": category,
            "k": dataset_k,
            "weight": weight,
            "reward": 0.0,
        })
    if not per_dataset:
        raise RuntimeError("generated grading suite is empty")
    zeros = {category: 0.0 for category in sorted(categories)}
    return {
        "submission_status": "failed",
        "submission_error": reason,
        "reward": 0.0,
        "reward_detail": {
            "execution_sandbox": isolation,
            "category_scores": zeros,
            "per_dataset": per_dataset,
        },
    }


def _deadline_dataset_result(
    vcf: Path, truth: dict, k: int, lib: dict,
    probe_gates: dict | None, probe_runtime_share: dict[str, float] | None,
    *, probes_enabled: bool,
) -> dict:
    """A schema-complete zero for work not started before the global deadline.

    Completed datasets retain their measured scores.  Only unexecuted work receives zero, keeping
    the aggregate denominator fixed and preventing either an outer-timeout score erasure or an
    accidental reward boost from grading only an easy prefix. Resource fields remain ``None``
    because no primary invocation existed to measure.
    """
    reason = "global grading deadline exhausted before this dataset"
    overhead = probe_runtime_share or {
        "submission_seconds": 0.0,
        "reference_seconds": 0.0,
    }
    gates = {
        "library_scan": lib,
        "validity": {
            "name": "validity",
            "factor": 0.0,
            "severity": 1.0,
            "reason": reason,
        },
    }
    if probes_enabled and float(lib.get("factor", 0.0)) > 0:
        for name in _PROBE_GATE_NAMES:
            gates[name] = (probe_gates or {}).get(name, {
                "name": name,
                "factor": 0.0,
                "severity": 1.0,
                "reason": "probe not completed before global grading deadline",
            })
    return {
        "dataset": vcf.stem,
        "category": truth["spec"]["category"],
        "k": k,
        "weight": truth["spec"].get("weight", 1.0),
        "reward": 0.0,
        "accuracy": 0.0,
        "time_quality": 0.0,
        "gate_product": 0.0,
        "sub_seconds": float(overhead["submission_seconds"]),
        "ref_seconds": float(overhead["reference_seconds"]),
        "primary_sub_seconds": 0.0,
        "primary_ref_seconds": 0.0,
        # No primary/reference anchor ran for a deadline row, so publishing a numeric anchor would
        # invent a measurement. The harness requires explicit nulls for this one exceptional row.
        "fast_ref_seconds": None,
        "execution_overhead_seconds": None,
        "probe_runtime_share": {
            "submission_seconds": float(overhead["submission_seconds"]),
            "reference_seconds": float(overhead["reference_seconds"]),
        },
        "ref_kept": 0,
        "accuracy_detail": {"accuracy": 0.0, "reason": reason},
        "structure": None,
        "gates": {name: _gate_report(gate) for name, gate in gates.items()},
        "run": {
            "returncode": 124,
            "stderr_tail": reason,
            # This primary invocation never ran, so zero would falsely claim a measurement.
            "sampled_primary_invocation_peak_pss_bytes": None,
            "sampled_primary_invocation_peak_storage_bytes": None,
        },
        "deadline_exhausted": True,
    }


def _make_run_one(
    submission_dir: Path, workdir: Path, timings: list[dict[str, float]], *, isolation: str,
    budget: GradingBudget | None = None,
):
    def run_one(vcf: Path, k: int, out: Path, sample_ids):
        if out.exists():
            out.unlink()
        reference_timeout = (budget.invocation_timeout(_REFERENCE_TIMEOUT_SECONDS)
                             if budget is not None else _REFERENCE_TIMEOUT_SECONDS)
        try:
            # Same path as the probe submission below -- see ReferenceCache for why mixing a plain
            # subprocess clock with a sandboxed one silently biases the comparison.
            reference_seconds = ReferenceCache._time_anchor(
                "reference.full_scan_pca", vcf, k, workdir,
                isolation=isolation, timeout=reference_timeout,
            )
        except subprocess.TimeoutExpired as error:
            if budget is not None and budget.exhausted():
                raise GradingDeadlineExhausted(
                    "global grading deadline exhausted during probe reference measurement"
                ) from error
            raise RuntimeError(f"trusted probe reference exceeded {reference_timeout}s") from error
        submission_timeout = (budget.invocation_timeout(_PROBE_SUBMISSION_TIMEOUT_SECONDS)
                              if budget is not None else _PROBE_SUBMISSION_TIMEOUT_SECONDS)
        r = run_submission(
            submission_dir, vcf, k, out, isolation=isolation, timeout=submission_timeout,
        )
        timings.append({
            "submission_seconds": float(r["seconds"]),
            "reference_seconds": float(reference_seconds),
        })
        if r["returncode"] != 0:
            return False, None
        S, _ = load_scores(out, sample_ids, k)
        if S is None or not np.isfinite(S).all():
            return False, None
        return True, S
    return run_one


def _probe_carrier_from_truth(truth: dict):
    """Extract the exact invocation and byte-layout profile of one truth sidecar."""
    from grader.gates.probes import ProbeCarrier

    spec = truth.get("spec", {})
    n_samples = truth.get("n_samples")
    n_records = truth.get("n_variants", spec.get("n_records"))
    k = spec.get("k")
    values = (n_samples, n_records, k)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in values):
        raise RuntimeError("truth sidecar lacks a complete integer (n_samples, n_records, k) carrier")
    representation = spec.get("representation")
    if spec.get("source") is not None:
        layout = "observed"
    elif representation == "variable_width":
        layout = "synthetic_variable_width"
    elif representation == "haploid_gt":
        layout = "synthetic_haploid"
    elif representation == "plain_gt" and float(spec.get("messy_frac", 0.0)) > 0.0:
        layout = "synthetic_messy"
    elif representation == "plain_gt":
        layout = "synthetic_plain"
    else:
        raise RuntimeError("truth sidecar has an unreviewed VCF representation profile")
    carrier = ProbeCarrier(*(int(value) for value in values), layout)
    if (carrier.n_samples < 2 or carrier.n_records < 1
            or carrier.k < 1 or carrier.k >= carrier.n_samples):
        raise RuntimeError(f"truth sidecar contains an invalid probe carrier: {carrier}")
    return carrier


def _balanced_grade_case_order(grade_cases: list[tuple], math_key: bytes) -> list[tuple]:
    """Return a private, release-stable round-robin order across categories.

    Deadline partial credit keeps the full denominator, but a fixed alphabetical prefix would
    still privilege whichever categories happen to sort first.  Private ranking removes that
    semantic ordering signal; round-robin scheduling ensures every category gets one opportunity
    before any category receives a second.  Only execution order changes--the cases and weights
    used by the rollup are untouched.
    """
    grouped: dict[str, list[tuple]] = {}
    for case in grade_cases:
        vcf, truth, dataset_k = case
        category = truth["spec"]["category"]
        identity = f"{category}/{vcf.stem}/{dataset_k}"
        ranked = (derive_bytes(math_key, f"grade-order/case/{identity}"), case)
        grouped.setdefault(category, []).append(ranked)

    categories = sorted(
        grouped,
        key=lambda category: derive_bytes(math_key, f"grade-order/category/{category}"),
    )
    queues = {
        category: [case for _, case in sorted(grouped[category], key=lambda item: item[0])]
        for category in categories
    }
    ordered: list[tuple] = []
    while any(queues.values()):
        for category in categories:
            if queues[category]:
                ordered.append(queues[category].pop(0))
    if len(ordered) != len(grade_cases):
        raise RuntimeError("private category-balanced ordering lost a grading case")
    return ordered


def run_probe_gates(submission_dir: Path, active_carriers, workdir: Path,
                    math_key: bytes, *, isolation: str,
                    budget: GradingBudget | None = None) -> tuple[dict, dict[str, float]]:
    """Run the object-identity probe gates once for a submission.

    Fixed release draws keep repeat evaluation stable. Every probe reuses a complete active
    scored invocation signature -- sample count, record count, and that carrier's actual ``k`` --
    plus opaque input/output paths. Both submission and full-scan reference runtime are measured
    so the carrier work can be amortized into the systems score."""
    from grader.gates.probes import (
        coverage_gate,
        hwe_norm_gate,
        representation_equivalence_gate,
        select_probe_carriers,
    )
    timings: list[dict[str, float]] = []
    run_one = _make_run_one(
        submission_dir, workdir, timings, isolation=isolation, budget=budget,
    )
    representation_key = secrets.token_bytes(32)
    carriers = select_probe_carriers(active_carriers, math_key)
    gates = {}
    deadline_exhausted = False
    probe_calls = (
        ("hwe_norm", hwe_norm_gate, "probe/hwe"),
        ("coverage", coverage_gate, "probe/coverage"),
        ("representation_equivalence", representation_equivalence_gate,
         "probe/representation"),
    )
    for name, gate_fn, seed_domain in probe_calls:
        try:
            kwargs = {"representation_key": representation_key}
            gates[name] = gate_fn(
                run_one, workdir, carriers[name], derive_seed(math_key, seed_domain), **kwargs,
            )
        except GradingDeadlineExhausted:
            deadline_exhausted = True
            break
    for name in _PROBE_GATE_NAMES:
        gates.setdefault(name, {
            "name": name,
            "factor": 0.0,
            "severity": 1.0,
            "reason": "probe not completed before global grading deadline",
        })
    runtime = {
        "submission_seconds": sum(item["submission_seconds"] for item in timings),
        "reference_seconds": sum(item["reference_seconds"] for item in timings),
        "invocations": len(timings),
        "deadline_exhausted": deadline_exhausted,
    }
    return gates, runtime


def grade_suite(submission_dir: Path, data_dir: Path, workdir: Path,
                *, isolation: str, math_key: bytes, probes: bool = True,
                cache: ReferenceCache | None = None,
                time_budget_seconds: float = _DEFAULT_GRADING_BUDGET_SECONDS) -> dict:
    # The full-scan reference is submission-independent, so a caller grading many submissions
    # (e.g. the validation harness) can pass a shared cache to avoid recomputing it.
    if cache is None:
        cache = ReferenceCache()
    budget = GradingBudget(time_budget_seconds)
    if _UNPRIV is None:
        raise RuntimeError("grading must run as root with an unprivileged child uid")
    if isolation not in _ISOLATION_MODES:
        raise ValueError(f"unknown isolation mode: {isolation!r}")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("grading requires Linux")
    if isolation == "bwrap" and not os.path.isfile(_BWRAP):
        raise RuntimeError(f"bwrap isolation requires {_BWRAP}")
    if os.path.abspath(sys.executable) != _RUNTIME_PYTHON:
        raise RuntimeError(f"grading requires {_RUNTIME_PYTHON}")
    # Seal the hidden reference/grader source from the dropped submission uid BEFORE running it,
    # then verify the seal AS that uid and fail closed if the exact scoring truth is still readable.
    _seal_grader_package()
    _assert_grader_package_sealed()
    _protect_grader_directory(workdir)
    _protect_grader_directory(data_dir)
    _assert_execution_boundary(workdir, isolation)
    # The fixed cost of entering the execution path, measured once. Every timed run -- submission
    # AND both reference anchors -- pays it, so it is common and merely compresses the ratios; we
    # subtract it to restore resolution on the smaller folds. Calibration failure is an
    # infrastructure error because silently using raw times would change the scoring scale.
    execution_overhead = measure_execution_overhead(workdir, isolation=isolation)
    snapshot = workdir / f"_submission_snapshot_{secrets.token_hex(12)}"
    try:
        try:
            _stage_submission(submission_dir, snapshot)
        except InvalidSubmission as error:
            return _invalid_submission_result(data_dir, str(error), isolation=isolation)
        lib = library_scan.scan(snapshot)
        commitment = key_commitment(math_key)
        grade_cases = []
        active_carriers = []
        for truth_path in sorted(data_dir.glob("*.truth.json")):
            vcf = Path(str(truth_path).removesuffix(".truth.json"))
            if not vcf.is_file():
                raise RuntimeError(f"generated dataset is missing for {truth_path.name}")
            truth = json.loads(truth_path.read_text())
            spec = truth.get("spec", {})
            if (spec.get("math_release") != RELEASE_ID
                    or spec.get("math_key_commitment") != commitment):
                raise RuntimeError(
                    f"dataset was not constructed with the active private release key: "
                    f"{truth_path.name}"
                )
            dataset_k = spec.get("k")
            n_samples = truth.get("n_samples")
            if not isinstance(dataset_k, int) or isinstance(dataset_k, bool) or dataset_k < 1:
                raise RuntimeError(f"invalid per-dataset k in {truth_path.name}")
            if isinstance(n_samples, int) and dataset_k >= n_samples:
                raise RuntimeError(f"per-dataset k must be below n_samples in {truth_path.name}")
            grade_cases.append((vcf, truth, dataset_k))
            active_carriers.append(_probe_carrier_from_truth(truth))
        if not grade_cases:
            raise RuntimeError("generated grading suite is empty")
        grade_cases = _balanced_grade_case_order(grade_cases, math_key)

        probe_gates = None
        probe_runtime_share = None
        probe_deadline = False
        if probes and lib["factor"] > 0:
            if budget.exhausted():
                probe_gates = {
                    name: {
                        "name": name,
                        "factor": 0.0,
                        "severity": 1.0,
                        "reason": "probe not completed before global grading deadline",
                    }
                    for name in _PROBE_GATE_NAMES
                }
                probe_runtime = {
                    "submission_seconds": 0.0,
                    "reference_seconds": 0.0,
                    "invocations": 0,
                    "deadline_exhausted": True,
                }
            else:
                probe_gates, probe_runtime = run_probe_gates(
                    snapshot, active_carriers, workdir, math_key, isolation=isolation,
                    budget=budget,
                )
            probe_deadline = bool(probe_runtime["deadline_exhausted"])
            probe_runtime_share = {
                "submission_seconds": probe_runtime["submission_seconds"] / len(grade_cases),
                "reference_seconds": probe_runtime["reference_seconds"] / len(grade_cases),
            }
        per_dataset = []
        evaluated_datasets = 0
        deadline_exhausted = probe_deadline
        for case_index, (vcf, truth, dataset_k) in enumerate(grade_cases):
            if budget.exhausted():
                deadline_exhausted = True
                remaining = grade_cases[case_index:]
                per_dataset.extend(
                    _deadline_dataset_result(
                        pending_vcf, pending_truth, pending_k, lib,
                        probe_gates, probe_runtime_share, probes_enabled=probes,
                    )
                    for pending_vcf, pending_truth, pending_k in remaining
                )
                break
            try:
                res = grade_one_dataset(
                    snapshot, vcf, truth, dataset_k, cache, workdir,
                    lib=lib, probe_gates=probe_gates,
                    probe_runtime_share=probe_runtime_share,
                    isolation=isolation, budget=budget,
                    execution_overhead=execution_overhead,
                )
            except GradingDeadlineExhausted:
                deadline_exhausted = True
                remaining = grade_cases[case_index:]
                per_dataset.extend(
                    _deadline_dataset_result(
                        pending_vcf, pending_truth, pending_k, lib,
                        probe_gates, probe_runtime_share, probes_enabled=probes,
                    )
                    for pending_vcf, pending_truth, pending_k in remaining
                )
                break
            per_dataset.append(res)
            evaluated_datasets += 1
            print(f"  {res['dataset']:20s} reward={res['reward']:.3f} "
                  f"acc={res['accuracy']:.3f} time={res['time_quality']:.2f} "
                  f"gates={res['gate_product']:.2f} "
                  f"({res['sub_seconds']:.2f}s vs ref {res['ref_seconds']:.2f}s)",
                  file=sys.stderr)
        if len(per_dataset) != len(grade_cases):
            raise RuntimeError("global-deadline finalization produced an incomplete dataset report")
        summary = rollup(per_dataset)
        return {"submission_status": "completed", "reward": summary["benchmark"], "reward_detail": {
            "execution_sandbox": isolation,
            "category_scores": summary["category_scores"],
            "per_dataset": per_dataset,
            "grading_budget": {
                "budget_seconds": round(budget.total_seconds, 3),
                "elapsed_seconds": round(budget.elapsed(), 3),
                "deadline_exhausted": deadline_exhausted,
                "evaluated_datasets": evaluated_datasets,
                "total_datasets": len(grade_cases),
            },
        }}
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_dir")
    ap.add_argument("--data-dir", default="data/generated")
    ap.add_argument("--workdir", default="/tmp/pcabench_work")
    ap.add_argument("--out", default=None)
    ap.add_argument("--math-key-file", required=True)
    ap.add_argument("--isolation", choices=sorted(_ISOLATION_MODES), required=True)
    ap.add_argument(
        "--time-budget-seconds", type=float,
        default=_DEFAULT_GRADING_BUDGET_SECONDS,
        help=argparse.SUPPRESS,
    )
    a = ap.parse_args(argv)
    math_key = read_math_key(a.math_key_file)
    result = grade_suite(
        Path(a.submission_dir), Path(a.data_dir), Path(a.workdir),
        isolation=a.isolation, math_key=math_key,
        time_budget_seconds=a.time_budget_seconds,
    )
    print(json.dumps(result["reward_detail"]["category_scores"], indent=2))
    print(f"BENCHMARK REWARD: {result['reward']:.4f}")
    if a.out:
        _write_private_json(Path(a.out), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
