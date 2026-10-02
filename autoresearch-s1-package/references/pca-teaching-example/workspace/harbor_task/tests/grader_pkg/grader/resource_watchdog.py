"""Trusted same-uid parent for submission resource enforcement.

Ordinary verifier containers intentionally do not grant ``CAP_SYS_PTRACE``.  Container root
therefore cannot reliably read a dropped child's ``smaps_rollup``, ``fd``, or ``map_files`` on
all Docker/user-namespace configurations.  This helper starts the reviewed sandbox command while
privileged, drops itself to the dedicated submission uid, and then monitors its descendants as
their same-uid ancestor.  Linux ptrace access checks consequently permit the required procfs
reads without giving the verifier or the submission another capability.

The helper's stdout is a private report pipe owned by the root grader.  The untrusted child gets
``/dev/null`` for stdout and cannot write a forged report.  A missing/malformed report, an
unreadable procfs surface, or a killed helper is treated as a closed resource boundary by the
root caller.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import resource
import signal
import stat
import sys
import time
from pathlib import Path


class ResourceInspectionError(RuntimeError):
    """The watchdog lost a required accounting surface."""


_FD_PERMISSION_RETRY_DELAYS = (0.0, 0.01, 0.04, 0.10)


def _is_transient_process_error(error: BaseException) -> bool:
    return isinstance(error, (FileNotFoundError, ProcessLookupError)) or (
        isinstance(error, OSError) and error.errno in {errno.ENOENT, errno.ESRCH}
    )


def _is_permission_error(error: BaseException) -> bool:
    return isinstance(error, PermissionError) or (
        isinstance(error, OSError) and error.errno in {errno.EACCES, errno.EPERM}
    )


def _process_state(pid: int) -> str | None:
    """Return one procfs process state, or ``None`` when the process disappeared.

    A task can exit between two procfs reads.  That transition is ordinary lifecycle churn, not
    a lost accounting surface.  Permission failures and malformed live metadata remain fatal.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text(errors="replace")
    except OSError as error:
        if _is_transient_process_error(error):
            return None
        raise ResourceInspectionError(
            f"cannot inspect submission process {pid}: {error}"
        ) from error
    for line in status.splitlines():
        if line.startswith("State:"):
            fields = line.split()
            if len(fields) > 1:
                return fields[1]
            break
    raise ResourceInspectionError(f"submission process {pid} has no valid state")


def _uid_processes(
    uid: int, *, exclude: frozenset[int] = frozenset(), include_zombies: bool = False,
) -> dict[int, str]:
    """Return all live processes owned by ``uid``, failing closed on hidden metadata."""
    processes: dict[int, str] = {}
    try:
        entries = os.scandir("/proc")
    except OSError as error:
        raise ResourceInspectionError(f"cannot enumerate procfs: {error}") from error
    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            if pid in exclude:
                continue
            try:
                info = entry.stat(follow_symlinks=False)
                if info.st_uid != uid:
                    continue
            except OSError as error:
                if _is_transient_process_error(error):
                    continue
                raise ResourceInspectionError(
                    f"cannot inspect submission process {pid}: {error}"
                ) from error
            state = _process_state(pid)
            if state is None:
                continue
            if include_zombies or state != "Z":
                processes[pid] = state
    return processes


def _aggregate_pss_bytes(pids: dict[int, str]) -> int:
    """Read aggregate proportional RSS, rejecting any live process hidden from the parent."""
    total = 0
    for pid in pids:
        for attempt in range(2):
            try:
                lines = Path(f"/proc/{pid}/smaps_rollup").read_text(
                    errors="replace",
                ).splitlines()
            except OSError as error:
                if _is_transient_process_error(error):
                    break
                raise ResourceInspectionError(
                    f"cannot read submission smaps_rollup for {pid}: {error}"
                ) from error
            pss_line = next((line for line in lines if line.startswith("Pss:")), None)
            if pss_line is not None:
                try:
                    total += int(pss_line.split()[1]) * 1024
                except (IndexError, ValueError) as error:
                    raise ResourceInspectionError(
                        f"malformed submission smaps_rollup for {pid}"
                    ) from error
                break

            # Linux can expose an empty smaps_rollup while a task sheds its mm and becomes a
            # zombie between the uid scan and this read. Recheck lifecycle state before deciding
            # that the accounting surface was malformed. A still-live task gets one retry and
            # then fails closed; an exited task has no resident memory left to account.
            state = _process_state(pid)
            if state is None or state == "Z":
                break
            if attempt == 0:
                time.sleep(0)
                continue
            raise ResourceInspectionError(
                f"submission smaps_rollup for live process {pid} has no Pss total"
            )
    return total


def _retry_submission_fd_access(
    pid: int, action: str, operation, *, deadline: float, retry_delays,
):
    """Run one required ``/proc/<pid>/fd`` operation across an exec transition.

    Bubblewrap enters a user namespace and changes its visible credentials immediately before
    exec. Linux can return EACCES for its ``fd`` directory during that short transition even to
    the same-uid parent watchdog. Treating the first denial as permanent converts scheduler timing
    into an infrastructure failure. We retry for a fixed 150 ms total, rechecking task lifecycle
    after every denial. An exited/zombie task has no open storage left to account; a task that is
    still live and unreadable after the bound remains a closed resource boundary.
    """
    while True:
        try:
            return operation()
        except OSError as error:
            if _is_transient_process_error(error):
                return None
            if not _is_permission_error(error):
                raise ResourceInspectionError(
                    f"cannot {action} for {pid}: {error}"
                ) from error

            state = _process_state(pid)
            if state is None or state == "Z":
                return None
            now = time.monotonic()
            try:
                delay = next(retry_delays)
            except StopIteration:
                delay = None
            if delay is None or now >= deadline:
                raise ResourceInspectionError(
                    f"cannot {action} for live process {pid}: {error}"
                ) from error
            time.sleep(min(delay, deadline - now))


def _submission_fd_stats(pid: int) -> list[os.stat_result]:
    """Snapshot regular-file candidates from one task's required fd accounting surface."""
    deadline = time.monotonic() + sum(_FD_PERMISSION_RETRY_DELAYS)
    retry_delays = iter(_FD_PERMISSION_RETRY_DELAYS)

    def list_names() -> tuple[str, ...]:
        with os.scandir(f"/proc/{pid}/fd") as descriptors:
            return tuple(descriptor.name for descriptor in descriptors)

    names = _retry_submission_fd_access(
        pid, "inspect submission fd", list_names,
        deadline=deadline, retry_delays=retry_delays,
    )
    if names is None:
        return []

    results: list[os.stat_result] = []
    for name in names:
        info = _retry_submission_fd_access(
            pid,
            "stat submission fd entry",
            lambda name=name: os.stat(f"/proc/{pid}/fd/{name}", follow_symlinks=True),
            deadline=deadline,
            retry_delays=retry_delays,
        )
        if info is not None:
            results.append(info)
    return results


def _deleted_mapped_extents(
    pids: set[int], seen_inodes: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Return conservative logical extents for capability-hidden deleted file mappings."""
    mapped_extents: dict[tuple[int, int], int] = {}
    for pid in pids:
        try:
            lines = Path(f"/proc/{pid}/maps").read_text(errors="replace").splitlines()
        except OSError as error:
            if _is_transient_process_error(error):
                continue
            raise ResourceInspectionError(
                f"cannot read submission maps for {pid}: {error}"
            ) from error
        for line in lines:
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].endswith(" (deleted)"):
                continue
            try:
                start_text, end_text = fields[0].split("-", 1)
                offset = int(fields[2], 16)
                major_text, minor_text = fields[3].split(":", 1)
                inode = int(fields[4])
                identity = (
                    os.makedev(int(major_text, 16), int(minor_text, 16)),
                    inode,
                )
                extent = offset + int(end_text, 16) - int(start_text, 16)
            except (ValueError, OSError) as error:
                raise ResourceInspectionError(
                    f"malformed deleted mapping for submission process {pid}"
                ) from error
            if inode == 0 or identity in seen_inodes:
                continue
            mapped_extents[identity] = max(mapped_extents.get(identity, 0), extent)
    return mapped_extents


def _storage_usage(
    roots: tuple[Path, ...], uid: int, pids: dict[int, str],
    *, storage_limit: int, entry_limit: int,
) -> tuple[int, int]:
    """Count uid-owned blocks, conservatively bounding capability-hidden mapped extents."""
    blocks = 0
    entries = 0
    seen_inodes: set[tuple[int, int]] = set()

    def account(info: os.stat_result) -> None:
        nonlocal blocks
        if info.st_uid != uid:
            return
        identity = (info.st_dev, info.st_ino)
        if identity in seen_inodes:
            return
        seen_inodes.add(identity)
        blocks += info.st_blocks * 512

    def walk_error(error: OSError) -> None:
        if _is_transient_process_error(error):
            return
        raise ResourceInspectionError(
            f"cannot walk submission storage: {error}"
        ) from error

    for root in roots:
        try:
            root_info = root.stat(follow_symlinks=False)
        except OSError as error:
            raise ResourceInspectionError(
                f"submission storage root is unavailable ({root}): {error}"
            ) from error
        if not stat.S_ISDIR(root_info.st_mode):
            raise ResourceInspectionError(f"submission storage root is not a directory: {root}")
        for directory, dirnames, filenames in os.walk(
            root, followlinks=False, onerror=walk_error,
        ):
            for name in [*dirnames, *filenames]:
                path = Path(directory, name)
                try:
                    info = os.lstat(path)
                except OSError as error:
                    if _is_transient_process_error(error):
                        continue
                    raise ResourceInspectionError(
                        f"cannot inspect submission storage entry {path}: {error}"
                    ) from error
                if info.st_uid != uid:
                    continue
                entries += 1
                account(info)
                if entries > entry_limit or blocks > storage_limit:
                    return blocks, entries

    maps_fallback_pids: set[int] = set()
    for pid in pids:
        # ``fd`` catches ordinary open-unlinked files exactly. Unlike map_files it is a required
        # exact surface, so a bounded exec-transition retry still fails closed when a live process
        # remains unreadable.
        for info in _submission_fd_stats(pid):
            if stat.S_ISREG(info.st_mode):
                account(info)
                if blocks > storage_limit:
                    return blocks, entries

        # ``map_files`` can provide exact inode/block accounting after the descriptor is closed,
        # but some Docker kernels require CAP_CHECKPOINT_RESTORE for its magic links even when the
        # reader is the same-uid parent. In that configuration the reviewed ``maps`` fallback below
        # accounts every deleted mapping's logical extent and remains fail-closed without a
        # capability.
        try:
            descriptors = os.scandir(f"/proc/{pid}/map_files")
        except OSError as error:
            if _is_transient_process_error(error):
                continue
            if isinstance(error, PermissionError):
                maps_fallback_pids.add(pid)
                continue
            raise ResourceInspectionError(
                f"cannot inspect submission map_files for {pid}: {error}"
            ) from error
        with descriptors:
            for descriptor in descriptors:
                try:
                    info = descriptor.stat(follow_symlinks=True)
                except OSError as error:
                    if _is_transient_process_error(error):
                        continue
                    if isinstance(error, PermissionError):
                        # Continue trying other entries: a kernel may expose submission-owned maps
                        # while withholding immutable runtime mappings. Any inaccessible deleted
                        # mapping is still covered by the maps-based extent pass.
                        maps_fallback_pids.add(pid)
                        continue
                    raise ResourceInspectionError(
                        f"cannot stat submission map_files entry for {pid}: {error}"
                    ) from error
                if stat.S_ISREG(info.st_mode):
                    account(info)
                    if blocks > storage_limit:
                        return blocks, entries

    # Capability-independent fallback for MAP_SHARED -> unlink -> close(fd). A deleted file with
    # no descriptor remains identifiable in `/proc/<pid>/maps` by device+inode. The kernel does
    # not expose st_blocks there, so count its maximum mapped logical extent. This is deliberately
    # conservative for sparse mappings: the storage boundary must remain enforceable when exact
    # map_files metadata is unavailable, and a solver cannot unlink any immutable external file
    # through the reviewed sandbox mounts.
    mapped_extents = _deleted_mapped_extents(maps_fallback_pids, seen_inodes)
    for identity, extent in mapped_extents.items():
        seen_inodes.add(identity)
        blocks += extent
        if blocks > storage_limit:
            return blocks, entries
    return blocks, entries


def _terminate_uid_processes(
    uid: int, *, exclude: frozenset[int], timeout: float = 5.0,
) -> None:
    """Freeze and kill the complete uid set while excluding this trusted watchdog."""
    deadline = time.monotonic() + timeout
    while True:
        processes = _uid_processes(uid, exclude=exclude)
        if not processes:
            return
        for pid in processes:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                raise ResourceInspectionError(
                    f"cannot stop submission process {pid}: {error}"
                ) from error
        time.sleep(0.01)
        stopped = _uid_processes(uid, exclude=exclude)
        if stopped and all(state in {"T", "t"} for state in stopped.values()):
            confirmation = _uid_processes(uid, exclude=exclude)
            if confirmation == stopped:
                for pid in confirmation:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError as error:
                        raise ResourceInspectionError(
                            f"cannot kill submission process {pid}: {error}"
                        ) from error
                time.sleep(0.01)
        if time.monotonic() >= deadline:
            survivors = _uid_processes(uid, exclude=exclude)
            if survivors:
                raise ResourceInspectionError(
                    f"submission processes survived cleanup: {sorted(survivors)}"
                )
            return


def _wait_status_code(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    raise ResourceInspectionError(f"unexpected child wait status: {status}")


def _apply_limits(args: argparse.Namespace) -> None:
    # The same-uid watchdog consumes one uid process slot but is excluded from submission
    # accounting, so allow one additional kernel slot and enforce the exact submission count in
    # the monitor below.
    for resource_id, value in (
        (resource.RLIMIT_AS, args.address_space_limit),
        (resource.RLIMIT_FSIZE, args.file_size_limit),
        (resource.RLIMIT_NPROC, args.process_limit + 1),
        (resource.RLIMIT_NOFILE, args.open_file_limit),
        (resource.RLIMIT_CPU, args.cpu_time_limit),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(resource_id, (value, value))


def _spawn_submission(args: argparse.Namespace) -> int:
    """Fork a gated child, drop the parent, then release the reviewed root launch command."""
    gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - exercised by integration tests
        try:
            os.close(gate_write)
            null = os.open("/dev/null", os.O_RDWR | os.O_CLOEXEC)
            try:
                os.dup2(null, 0)
                os.dup2(null, 1)
            finally:
                if null > 2:
                    os.close(null)
            _apply_limits(args)
            os.chdir(args.child_cwd)
            if os.read(gate_read, 1) != b"1":
                raise RuntimeError("resource watchdog launch gate closed")
            os.close(gate_read)
            os.execvpe(args.command[0], args.command, json.loads(args.child_env_json))
        except BaseException as error:
            try:
                os.write(2, f"resource watchdog could not exec submission: {error}\n".encode())
            finally:
                os._exit(127)

    os.close(gate_read)
    try:
        os.setgroups([])
    except (AttributeError, OSError):
        pass
    os.setgid(args.gid)
    os.setuid(args.uid)
    if (
        os.getresuid() != (args.uid, args.uid, args.uid)
        or os.getresgid() != (args.gid, args.gid, args.gid)
        or os.getgroups()
    ):
        raise ResourceInspectionError("resource watchdog did not drop to the submission identity")
    os.write(gate_write, b"1")
    os.close(gate_write)
    return pid


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--cpu-time-limit", type=int, required=True)
    parser.add_argument("--address-space-limit", type=int, required=True)
    parser.add_argument("--rss-limit", type=int, required=True)
    parser.add_argument("--storage-limit", type=int, required=True)
    parser.add_argument("--entry-limit", type=int, required=True)
    parser.add_argument("--process-limit", type=int, required=True)
    parser.add_argument("--open-file-limit", type=int, required=True)
    # Per-FILE ceiling (RLIMIT_FSIZE), not the scores-output ceiling: the output is bounded
    # independently on read by the grader. Sized to the documented temp-storage budget so a
    # legitimate large intermediate spill is not killed mid-write.
    parser.add_argument("--file-size-limit", type=int, required=True)
    parser.add_argument("--child-cwd", required=True)
    parser.add_argument("--child-env-json", required=True)
    parser.add_argument("--storage-root", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing submission command")
    if not args.storage_root:
        parser.error("at least one storage root is required")
    if args.cpu_time_limit <= 0:
        parser.error("CPU time limit must be positive")
    return args


def _run(args: argparse.Namespace) -> dict[str, int | str | bool | None]:
    watchdog_pid = os.getpid()
    child_pid = _spawn_submission(args)
    exclude = frozenset({watchdog_pid})
    roots = tuple(Path(path) for path in args.storage_root)
    deadline = time.monotonic() + args.timeout
    peak_pss = 0
    peak_storage = 0
    violation: str | None = None
    monitor_error: str | None = None
    timed_out = False
    child_status: int | None = None

    try:
        while True:
            waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
            if waited_pid == child_pid:
                child_status = status
                # Capture named and open/mapped state from detached workers once more before
                # freezing them. The main child itself has already been reaped.
                processes = _uid_processes(args.uid, exclude=exclude)
                pss = _aggregate_pss_bytes(processes)
                peak_pss = max(peak_pss, pss)
                storage, entries = _storage_usage(
                    roots, args.uid, processes,
                    storage_limit=args.storage_limit,
                    entry_limit=args.entry_limit,
                )
                peak_storage = max(peak_storage, storage)
                if pss > args.rss_limit:
                    violation = "aggregate resident-memory limit exceeded"
                elif storage > args.storage_limit or entries > args.entry_limit:
                    violation = "aggregate temporary-storage limit exceeded"
                break

            processes = _uid_processes(args.uid, exclude=exclude)
            if len(processes) > args.process_limit:
                violation = "aggregate process limit exceeded"
                break
            pss = _aggregate_pss_bytes(processes)
            peak_pss = max(peak_pss, pss)
            if pss > args.rss_limit:
                violation = "aggregate resident-memory limit exceeded"
                break
            storage, entries = _storage_usage(
                roots, args.uid, processes,
                storage_limit=args.storage_limit,
                entry_limit=args.entry_limit,
            )
            peak_storage = max(peak_storage, storage)
            if storage > args.storage_limit or entries > args.entry_limit:
                violation = "aggregate temporary-storage limit exceeded"
                break

            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.20)
    except BaseException as error:
        monitor_error = f"{type(error).__name__}: {error}"

    try:
        _terminate_uid_processes(args.uid, exclude=exclude)
    except BaseException as error:
        detail = f"{type(error).__name__}: {error}"
        monitor_error = f"{monitor_error}; {detail}" if monitor_error else detail

    if child_status is None:
        try:
            waited_pid, child_status = os.waitpid(child_pid, 0)
            if waited_pid != child_pid:
                raise ResourceInspectionError("waitpid returned the wrong submission child")
        except ChildProcessError:
            child_status = None
        except BaseException as error:
            detail = f"{type(error).__name__}: {error}"
            monitor_error = f"{monitor_error}; {detail}" if monitor_error else detail

    return {
        "watchdog_pid": watchdog_pid,
        "child_returncode": _wait_status_code(child_status) if child_status is not None else 137,
        "timed_out": timed_out,
        "violation": violation,
        "monitor_error": monitor_error,
        "peak_pss_bytes": peak_pss,
        "peak_storage_bytes": peak_storage,
    }


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        report = _run(args)
    except BaseException as error:
        report = {
            "watchdog_pid": os.getpid(),
            "child_returncode": 137,
            "timed_out": False,
            "violation": None,
            "monitor_error": f"{type(error).__name__}: {error}",
            "peak_pss_bytes": 0,
            "peak_storage_bytes": 0,
        }
    sys.stdout.write(json.dumps(report, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
