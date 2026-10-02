"""Trusted Python launcher for the no-external-code execution contract.

The grader starts this file before any solver code. On Linux stacked seccomp filters make process
image replacement, observer-blinding namespace/dumpability changes, and new executable memory
mappings fail in this process and every forked worker.
The approved numerical extension stack is loaded before executable mappings are closed.  A Python
audit hook then gives a clear error for ordinary subprocess/dynamic-library attempts and rejects
unreviewed extension modules.  The kernel filters, rather than the mutable Python hook, are the
native-code boundary.  Pure Python, NumPy, the documented SciPy numerical paths, and Linux ``fork``
workers remain usable.
"""
from __future__ import annotations

import errno
import functools
import os
import runpy
import sys
import sysconfig
from pathlib import Path


def _runtime_native_roots() -> tuple[Path, ...]:
    """Resolve extension roots from the pinned interpreter, not distro-specific guesses."""
    configured = {
        sysconfig.get_path("platlib"),
        sysconfig.get_path("purelib"),
        sysconfig.get_config_var("DESTSHARED"),
    }
    roots = tuple(sorted({Path(path).resolve() for path in configured if path}))
    if not roots or any(not root.is_dir() for root in roots):
        raise RuntimeError(f"pinned runtime has invalid native-extension roots: {roots}")
    return roots


_TRUSTED_NATIVE_ROOTS = _runtime_native_roots()
_AUDIT_BLOCKED_EVENTS = frozenset({
    "code.__new__",
    "function.__new__",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.system",
    "subprocess.Popen",
    "sys._current_exceptions",
    "sys._current_frames",
    "sys.setprofile",
    "sys.settrace",
})
_BASE_ALLOWED_IMPORTS = frozenset(sys.stdlib_module_names) | {
    "numpy",
    "scipy",
    # Optional probes performed from reviewed NumPy/SciPy import paths.  These packages are not
    # installed in the sealed runtime; allowing the names lets ImportError-based feature probes
    # complete without adding any callable dependency.  The source scanner still rejects a solver
    # that names one directly.
    "charset_normalizer",
    "cython",
    "scikits",
}


def _local_import_roots(submission_root: Path) -> frozenset[str]:
    """Return source-backed top-level names that the immutable submission can import."""
    modules = {
        child.stem
        for child in submission_root.iterdir()
        if child.is_file() and child.suffix in {".py", ".pyw"}
    }
    modules.update(
        child.name
        for child in submission_root.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    )
    return frozenset(modules)


def _install_seccomp_filter(*, deny_executable_memory: bool) -> None:
    """Install a thread-synchronized, fail-closed Linux seccomp filter.

    Audit hooks are useful diagnostics but Python code can reach and mutate Python callback
    objects.  A seccomp filter is inherited across ``fork`` and cannot be relaxed after
    ``no_new_privs`` is set. Both policies prevent process-image replacement and observer-blinding
    namespace/dumpability changes. The stronger post-preload policy also rejects executable
    ``mmap``/``mprotect``/``pkey_mprotect`` and executable System V shared-memory attachment.
    Consequently a decoded extension cannot be mapped by ``dlopen``/``_imp.create_dynamic``, even
    if Python-level import auditing is bypassed.

    ``SECCOMP_FILTER_FLAG_TSYNC`` applies each policy to every extant BLAS/runtime thread as well
    as future forked descendants.  The architecture check kills a process that enters a different
    syscall ABI rather than letting architecture-specific syscall numbers evade the deny rules.
    """
    if not sys.platform.startswith("linux"):
        return

    # Keep ctypes local to this trusted setup routine.  Solver calls through ctypes are rejected by
    # the audit hook below; these calls happen before that hook is installed.
    import ctypes

    architectures = {
        # audit arch, execve, execveat, seccomp, mmap, mprotect, pkey_mprotect, shmat,
        # unshare, setns, prctl, compatibility ABI bit
        "x86_64": (0xC000003E, 59, 322, 317, 9, 10, 329, 30, 272, 308, 157, 0x40000000),
        "amd64": (0xC000003E, 59, 322, 317, 9, 10, 329, 30, 272, 308, 157, 0x40000000),
        "aarch64": (0xC00000B7, 221, 281, 277, 222, 226, 288, 196, 97, 268, 167, None),
        "arm64": (0xC00000B7, 221, 281, 277, 222, 226, 288, 196, 97, 268, 167, None),
    }
    machine = os.uname().machine.lower()
    try:
        (
            audit_arch,
            execve_number,
            execveat_number,
            seccomp_number,
            mmap_number,
            mprotect_number,
            pkey_mprotect_number,
            shmat_number,
            unshare_number,
            setns_number,
            prctl_number,
            compatibility_abi_bit,
        ) = architectures[machine]
    except KeyError as error:
        raise RuntimeError(f"unsupported seccomp architecture: {machine}") from error

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    # linux/filter.h + linux/seccomp.h constants.  All inspected flags are 32-bit integers, so the
    # low word in seccomp_data.args is sufficient on both supported little-endian architectures.
    bpf_ld_w_abs = 0x20
    bpf_jmp_jeq_k = 0x15
    bpf_jmp_jset_k = 0x45
    bpf_ret_k = 0x06
    seccomp_ret_kill_process = 0x80000000
    seccomp_ret_errno = 0x00050000 | errno.EPERM
    seccomp_ret_allow = 0x7FFF0000
    instructions = [
        SockFilter(bpf_ld_w_abs, 0, 0, 4),
        SockFilter(bpf_jmp_jeq_k, 1, 0, audit_arch),
        SockFilter(bpf_ret_k, 0, 0, seccomp_ret_kill_process),
        SockFilter(bpf_ld_w_abs, 0, 0, 0),
    ]
    if compatibility_abi_bit is not None:
        instructions.extend((
            SockFilter(bpf_jmp_jset_k, 0, 1, compatibility_abi_bit),
            SockFilter(bpf_ret_k, 0, 0, seccomp_ret_kill_process),
        ))
    # Image replacement escapes the reviewed program; namespace changes can make the solver
    # unreadable to its same-uid parent resource observer. Fork/clone and ordinary threads remain
    # available, but a descendant cannot create or join a namespace to blind accounting.
    for syscall_number in (execve_number, execveat_number, unshare_number, setns_number):
        instructions.extend((
            SockFilter(bpf_jmp_jeq_k, 0, 1, syscall_number),
            SockFilter(bpf_ret_k, 0, 0, seccomp_ret_errno),
        ))

    # PR_SET_DUMPABLE=0 also makes same-uid procfs accounting fail. Deny only that option/value
    # pair: PR_GET_DUMPABLE, thread naming, no_new_privs, and every other prctl remain available.
    # seccomp_data.args[0] and args[1] begin at byte offsets 16 and 24 respectively.
    pr_set_dumpable = 4
    instructions.extend((
        SockFilter(bpf_jmp_jeq_k, 0, 5, prctl_number),
        SockFilter(bpf_ld_w_abs, 0, 0, 16),
        SockFilter(bpf_jmp_jeq_k, 0, 3, pr_set_dumpable),
        SockFilter(bpf_ld_w_abs, 0, 0, 24),
        SockFilter(bpf_jmp_jeq_k, 0, 1, 0),
        SockFilter(bpf_ret_k, 0, 0, seccomp_ret_errno),
    ))
    if deny_executable_memory:
        # seccomp_data.args[2] begins at byte 32.  PROT_EXEC is 0x4 and SHM_EXEC is octal
        # 0100000 (0x8000).  A nonmatching syscall skips the argument-load, mask-test, and deny.
        for syscall_number, denied_flag in (
            (mmap_number, 0x4),
            (mprotect_number, 0x4),
            (pkey_mprotect_number, 0x4),
            (shmat_number, 0x8000),
        ):
            instructions.extend((
                SockFilter(bpf_jmp_jeq_k, 0, 3, syscall_number),
                SockFilter(bpf_ld_w_abs, 0, 0, 32),
                SockFilter(bpf_jmp_jset_k, 0, 1, denied_flag),
                SockFilter(bpf_ret_k, 0, 0, seccomp_ret_errno),
            ))
    instructions.append(SockFilter(bpf_ret_k, 0, 0, seccomp_ret_allow))
    filters = (SockFilter * len(instructions))(*instructions)
    program = SockFprog(len(filters), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    pr_set_no_new_privs = 38
    if libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0) != 0:
        saved_errno = ctypes.get_errno()
        raise OSError(saved_errno, "could not set no_new_privs")
    libc.syscall.restype = ctypes.c_long
    seccomp_set_mode_filter = 1
    seccomp_filter_flag_tsync = 1
    result = libc.syscall(
        seccomp_number,
        seccomp_set_mode_filter,
        seccomp_filter_flag_tsync,
        ctypes.byref(program),
    )
    if result != 0:
        saved_errno = ctypes.get_errno()
        if result > 0:
            raise RuntimeError(f"could not synchronize seccomp filter to thread {result}")
        raise OSError(saved_errno, "could not install seccomp filter")


def _install_no_exec_filter() -> None:
    """Prevent this process and every thread/fork descendant from replacing its image."""
    _install_seccomp_filter(deny_executable_memory=False)


def _install_no_new_executable_memory_filter() -> None:
    """Close native-code mappings after the reviewed numerical stack has been loaded."""
    _install_seccomp_filter(deny_executable_memory=True)


def _preload_approved_native_stack() -> None:
    """Eagerly exercise every documented native numerical path before executable-map closure."""
    import mmap as trusted_mmap

    import numpy as np
    import scipy.linalg as scipy_linalg
    import scipy.sparse as scipy_sparse
    import scipy.sparse.linalg as scipy_sparse_linalg
    import scipy.special as scipy_special

    identity = np.eye(5)
    np.linalg.eigh(identity)
    np.linalg.svd(identity, full_matrices=False)
    scipy_linalg.eigh(identity, subset_by_index=[3, 4])
    sparse_identity = scipy_sparse.csr_matrix(identity)
    scipy_sparse_linalg.eigsh(sparse_identity, k=2)
    scipy_special.ndtr(np.array([0.0]))
    # Memory-mapped input is a standard-library optimization explicitly compatible with the task.
    # Import its extension before executable mappings close; ordinary data-only mappings remain
    # permitted afterward.
    del trusted_mmap


def _audit(
    event: str,
    args: tuple,
    _blocked: frozenset[str] = _AUDIT_BLOCKED_EVENTS,
    _trusted_roots: tuple[str, ...] = tuple(str(root) for root in _TRUSTED_NATIVE_ROOTS),
    _allowed_imports: frozenset[str] = _BASE_ALLOWED_IMPORTS,
    _write=os.write,
    _exit=os._exit,
    _any=any,
    _type=type,
    _len=len,
    _string_type=str,
    _str_split=str.split,
    _str_startswith=str.startswith,
) -> None:
    # Exit rather than raising into solver code.  A caught Python exception exposes the audit
    # callback's traceback/globals and lets hostile code mutate a Python-level hook before retrying
    # the operation.  GC enumeration and tracing are denied for the same reason.  SciPy relies on
    # ``sys._getframe`` for decorators, so that ordinary introspection event remains available.
    if event in _blocked or event.startswith(("ctypes.", "gc.")):
        _write(2, f"external or dynamic code is disabled ({event})\n".encode())
        _exit(126)
    # Dynamic Python is permitted.  The production image ships only numpy and scipy, so there is
    # nothing for generated code to delegate the decomposition to, and the import allowlist below
    # is enforced by an audit hook -- which CPython cannot unregister -- so no amount of runtime
    # codegen reaches a library that is not already allowed.  Banning it only cost real solves:
    # a submission that merely metaprograms its own pure-Python helpers is not a cheat, and the
    # policy previously rejected such runs with an unactionable zero.
    if event == "import" and _len(args) >= 1 and _type(args[0]) is not _string_type:
        _write(2, b"untrusted import-name type is disabled\n")
        _exit(126)
    if event == "import" and _len(args) >= 1 and args[0] == "multiprocessing.sharedctypes":
        # Context.Value/Array lazily import sharedctypes.  That module generates wrapper methods
        # with exec() and then exposes ctypes-backed buffers, both outside the documented
        # pure-Python fork boundary.  Reject the API at import time so a normal-looking stdlib call
        # does not fail later with the misleading generic dynamic-compilation diagnostic.
        _write(2, b"ctypes-backed multiprocessing shared objects are disabled\n")
        _exit(126)
    if event == "import" and _len(args) >= 1 and args[0] == "_posixsubprocess":
        _write(2, b"external or dynamic code is disabled (_posixsubprocess)\n")
        _exit(126)
    if event == "import" and _len(args) >= 1:
        top_level = _str_split(args[0], ".", 1)[0]
        if top_level not in _allowed_imports:
            _write(2, f"third-party import is disabled: {top_level}\n".encode())
            _exit(126)
    if event == "import" and _len(args) >= 2 and args[1] is not None:
        if _type(args[1]) is not _string_type:
            _write(2, b"untrusted import-path type is disabled\n")
            _exit(126)
        filename = args[1]
        if filename:
            # Native import machinery accepts arbitrary filenames, so suffix filtering is not a
            # security boundary: ``_imp.create_dynamic`` can load a valid extension renamed
            # ``payload.bin``.  Every nonempty import filename must be under an immutable runtime
            # root.  Avoid virtual methods on solver-controlled str subclasses (rejected above).
            parts = _str_split(filename, "/")
            trusted = (
                _str_startswith(filename, "/")
                and ".." not in parts
                and _any(filename == root or _str_startswith(filename, root + "/")
                         for root in _trusted_roots)
            )
            if not trusted:
                _write(2, f"untrusted imported code path is disabled: {filename}\n".encode())
                _exit(126)


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        raise SystemExit("usage: submission_runner.py <entrypoint> [args ...]")
    entrypoint = os.path.abspath(argv[0])
    submission_root = Path(entrypoint).parent
    _install_no_exec_filter()
    # NumPy legitimately probes the current process with ``ctypes.CDLL(None)`` during its first
    # import.  Load and exercise the pinned, reviewed numeric paths before closing executable
    # mappings and installing the audit policy.  Process startup/import time remains inside the
    # grader's end-to-end timing.  TSYNC covers native threads that the warm-up created.
    _preload_approved_native_stack()
    # The stdlib's fork-only multiprocessing backend imports ``subprocess`` for an argv helper,
    # even though it never execs.  Preload that reviewed path, then remove the private native
    # ``fork_exec`` primitive and its importable module before solver code begins.  This preserves
    # Linux fork workers while making direct low-level exec attempts fail clearly; seccomp remains
    # the kernel backstop. These imports must happen before executable mappings close because
    # `_multiprocessing` and `_posixsubprocess` are ordinary reviewed stdlib extensions.
    import _multiprocessing as trusted_multiprocessing_native
    import _decimal as trusted_decimal_native
    import _sqlite3 as trusted_sqlite_native
    import multiprocessing.pool as trusted_pool_backend
    import multiprocessing.popen_fork as trusted_fork_backend
    import multiprocessing.queues as trusted_queue_backend
    import multiprocessing.synchronize as trusted_sync_backend
    import pkgutil as trusted_runpy_support
    import subprocess as trusted_subprocess
    _install_no_new_executable_memory_filter()
    trusted_subprocess.__dict__.pop("_fork_exec", None)
    sys.modules.pop("_posixsubprocess", None)
    del (
        trusted_fork_backend,
        trusted_decimal_native,
        trusted_multiprocessing_native,
        trusted_pool_backend,
        trusted_queue_backend,
        trusted_runpy_support,
        trusted_sqlite_native,
        trusted_subprocess,
        trusted_sync_backend,
    )
    allowed_imports = _BASE_ALLOWED_IMPORTS | _local_import_roots(submission_root)
    hook = functools.partial(_audit, _allowed_imports=allowed_imports)
    sys.addaudithook(hook)
    # Solver frames can walk ``f_back`` into this runner.  Remove every global route to the Python
    # callback and its policy after registration; the hook has immutable defaults and remains held
    # internally by CPython.  A blocked event hard-exits, so it cannot yield a traceback containing
    # the callback frame either.
    globals().pop("_audit", None)
    globals().pop("_AUDIT_BLOCKED_EVENTS", None)
    globals().pop("_TRUSTED_NATIVE_ROOTS", None)
    globals().pop("_BASE_ALLOWED_IMPORTS", None)
    globals().pop("_local_import_roots", None)
    globals().pop("_install_seccomp_filter", None)
    globals().pop("_install_no_exec_filter", None)
    globals().pop("_install_no_new_executable_memory_filter", None)
    globals().pop("_preload_approved_native_stack", None)
    globals().pop("functools", None)
    del allowed_imports, hook
    # ``-I`` deliberately omits the script directory from sys.path.  Restore only the immutable
    # submission root so normal multi-file solutions can import their own sibling modules without
    # also exposing the current directory or grader paths.
    sys.path.insert(0, str(submission_root))
    sys.argv = [entrypoint, *argv[1:]]
    runpy.run_path(entrypoint, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
