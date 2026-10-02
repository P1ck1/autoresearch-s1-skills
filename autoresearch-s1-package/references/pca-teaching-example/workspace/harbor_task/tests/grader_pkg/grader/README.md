# Grader

Turns one submission run into a bounded score on a single dataset:

```
score = accuracy
      * (0.10 + 0.90 * systems_unlock(accuracy) * time_quality)
      * ∏ method_factor
```

- **accuracy** — principal-angle subspace agreement vs the full-scan anchor (`metrics/subspace.py`),
  capped to the top-`k` PCs so a submission can't inflate its score by returning extra columns.
- **`systems_unlock(accuracy)`** — correctness is the *key* to the systems term, not a coefficient on
  it. A PCA that does not recover the population structure has not computed the thing being timed, so
  its wall time measures nothing and buys it nothing: below accuracy 0.75 the speed term is worth
  exactly zero, ramping to fully paid at 0.90. Genuine implementations saturate at accuracy 1.000 and
  `scripts/validate.py` already pins 0.90 as the worst a real PCA may score, so the ramp is invisible
  to honest submissions and only bites wrong ones. The property it buys: **wrong-and-fast is strictly
  worse than right-and-slow at every speed** — a broken PCA is capped at `accuracy * 0.10`, below the
  0.10 a correct-but-slow fit earns for correctness alone.
- **systems quality** (reported as `time_quality`) — end-to-end speed scored against what is
  actually *achievable on this fold*, not against a fixed multiple of the full scan. Two anchors are
  measured per dataset: the **full scan** (a correct naive implementation → 0.5) and the **fast
  reference** (the best speed anyone has been shown to reach → 1.0), interpolated linearly in
  log-time. Consequences: a submission that matches the best achievable speed scores 1.0 on *every*
  fold whatever its size, so a flawless solution can actually earn a reward of 1; where sampling
  genuinely wins, a full scan sits at 0.5 and you must skip work to beat it; and where sampling
  *cannot* win (small folds, where a plain scan is already optimal) the full scan scores 1.0 — a
  program is never punished for failing to achieve a speedup that does not exist. This rewards
  choosing the right strategy per fold. Accuracy remains a hard multiplicative ceiling.

  The submission and both anchors run through the identical sandbox. The same measured fixed
  baseline—interpreter boot, numeric imports, sandbox entry, guard setup, and staging—is subtracted
  from all three, so work is compared with work and a program running the fast reference's exact
  code reaches exactly 1.0.
  Past the ordinary parity-to-slow region, a small asymptotic tail keeps different successful but
  catastrophic runtimes ordered instead of clipping all of them to the same value. The former hard
  zero is worth only ~0.035 systems quality and rapidly decays, so correctness-floor semantics are
  preserved while a real multi-fold engineering improvement remains visible. Failed or timed-out
  executions still receive exactly zero systems quality.
  Both speed anchors and the overhead calibration are required grader state. If any cannot execute,
  grading errors as infrastructure; it never substitutes a slower anchor or silently changes scale.
- **method factors** — continuous anti-shortcut evidence in `gates/`. A genuine PCA reads near 1;
  nearby but different objects are heavily discounted. Scientific factors have explicit nonzero
  floors; only dependency or output-contract integrity failures can hard-zero a real result.

Gates:

| gate | catches |
|------|---------|
| `library_scan` | non-numpy/scipy deps, encoded Python/bytecode/native payloads (including code hidden in archives or behind renamed magic), non-Python entry (AST allowlist). Inspected numeric/data-only archives such as `.npz` are permitted. |
| `hwe_norm`      | raw-covariance / non-Patterson standardisation (variance-weighted axis probe) |
| `coverage`      | too-few-variants sampling that misses subtle structure |
| `representation_equivalence` | sample/record order, allele polarity, case, phase, or FORMAT-sensitive fits |
| `validity`      | malformed, incomplete, or non-finite output |

Only dependency and output-contract violations affect the rollout validity status. Scientific
weakness in an identity probe lowers the continuous score without relabelling the evaluation as
an execution failure or erasing every bit of partial credit. The HWE and coverage factors floor at
0.15; the narrower representation-equivalence factor floors at 0.35 and is scoped to the
`messy` and `variable_width` parser-stress categories. A parser defect therefore loses the
capability it failed without multiplying away correct clean-input PCA work.

Probe shapes are drawn from the scored shape support, and staged inputs are distinct
inodes with normalized metadata. The verifier-container preflight fails closed if its immutable
layer updates access times. These properties prevent file size, sample count, link count, or prior
reference reads from becoming a cheap probe/category oracle.
Both submission and full-scan carrier runtimes are measured and amortized over scored folds; a
method cannot buy perfect gate factors with an unscored slow path.

## Isolation boundary

Production grading has two explicit, reviewed isolation contracts and no automatic fallback.
The Hyperfocal adapter selects `bwrap`: it requires root plus Linux bubblewrap and constructs an
allowlisted filesystem with fresh user, PID, IPC, UTS, and network namespaces. The child sees the
pinned runtime and guard, an immutable submission snapshot, one opaque input, one pre-created
output, a private host-backed scratch directory, and an empty conventional `/proc`
directory—not the host filesystem or
network. Procfs is intentionally absent: even a read-only mount exposes writable
`/proc/<pid>/task/<tid>/mem` handles for the process itself, which could overwrite an existing
executable mapping after the no-new-executable-memory policy is installed.

The packaged Harbor task selects `verifier-container`. Harbor first moves only the immutable
submission artifact into a fresh verifier image that the solver never occupied, and the task's
outer container has networking disabled. Inside that boundary, each invocation enters a static,
root-owned chroot containing only the pinned interpreter/runtime, ordinary system libraries,
device nodes, the trusted launcher, and its per-invocation work tree. It has no `/tests`, hidden
grader/reference package, datasets, `/proc`, or `/sys`. The child then drops uid, gid,
supplementary groups, every inheritable/ambient/bounding capability, and enables
no-new-privileges. Nested bubblewrap is intentionally not attempted inside the ordinary Docker
verifier because user namespaces are not available there.

Both paths start the trusted `submission_runner.py` before loading solver code. It first installs a
thread-synchronized seccomp filter returning `EPERM` for `execve`, `execveat`, `unshare`, `setns`,
and only the observer-blinding `prctl(PR_SET_DUMPABLE, 0)` pair. Other `prctl` operations, threads,
and `fork` remain available. It eagerly exercises the reviewed NumPy/SciPy numerical paths, then
stacks a second thread-synchronized policy denying new executable `mmap`, executable
`mprotect`/`pkey_mprotect`, and executable System V attachment. This prevents a decoded or renamed
extension from being mapped even if Python-level auditing is bypassed, including from an
already-running native thread or forked descendant. A fail-closed
Python audit hook separately rejects spawn/system calls, dynamic `ctypes` access, GC/tracing routes
to the hook, imports outside stdlib/NumPy/SciPy/local source, and native extensions outside the
immutable reviewed Python, NumPy, and SciPy roots. Shipped
`__pycache__` directories are removed from the immutable snapshot, so local testing is harmless
without allowing opaque bytecode to replace inspected helpers. This does not turn the task into a
single-file restriction: the launcher adds the immutable submission directory to `sys.path`, so
sibling pure-Python modules, data-only memory maps, and ordinary Linux `fork` workers remain
available. Ctypes-backed multiprocessing `Value`/`Array` objects are outside that boundary even
when created from a fork context. Fork workers cannot replace their process image or create new
executable mappings.

Both contracts copy a bounded regular-file submission tree once through no-follow descriptors,
freeze that snapshot before dependency scanning, use it for every probe and scored dataset, run
the program as the dedicated unprivileged uid, expose only an opaque input and one pre-created
writable output, pass a minimal credential-free environment, and apply fail-closed memory, file,
process, descriptor, CPU, and wall limits. Each contract fails closed if its selected boundary is
unavailable; an unconfined mode does not exist.
Solver-controlled tree rejection (missing `pca`, links/special files, size
limits, or a concurrent mutation) emits a complete zero-reward `failed`
contract. Trusted runtime, sandbox, dataset, or malformed-result failures remain
adapter `errored` results rather than being misreported as model failures.

Before grading, a mode-specific preflight proves that the dropped uid cannot read an outside
sentinel or hidden reference. It also verifies that the selected network boundary has no route.
Every invocation starts with an empty dedicated-uid process and System V IPC set. On completion or
timeout, cleanup freezes and kills *all* processes owned by that uid—not only the original process
group—then removes its message queues, semaphores, and shared-memory objects. Harbor additionally
clears every writable chroot directory (`tmp`, `var/tmp`, `run/lock`, `dev/shm`, and
`dev/mqueue`) before and after each invocation. Detached sessions and writable state therefore
cannot communicate between probes or scored datasets.

- **Probe secrecy.** `gates/probes.py` writes probe VCFs to opaque random paths and reads back the
  submission's scores. This assumes the submission cannot enumerate the grader's working directory,
  read the truth anchor, or tamper with probe files mid-run. Under linux-user isolation the
  submission has no read access to the grader's tree; without it, the probes are guessable.
- **Stable private-release mathematics.** Synthetic scored/probe draws and the observed
  cohort/record selection are derived through separate domains of one root-private release key,
  so an identical submission receives the same eigengaps and carrier difficulty without making
  the fixtures reconstructible from the repository. Each provision only
  re-keys opaque aliases and the sample-column permutation, which changes representation but not
  the PCA object. Kernel isolation, opaque paths,
  scored-shape carriers, and inaccessible evaluator code—not noisy synthetic redraws—block phase
  recognition.
- **Resource bounds.** Wall-clock (the speed term) and memory are only meaningful if the submission
  cannot leave descendants behind or exhaust verifier memory with logs. Each invocation has its
  process group killed, followed by UID-wide process and IPC cleanup, while stderr is drained and
  only a fixed diagnostic tail is retained. A root monitor sums proportional RSS across the uid and
  physical blocks in writable state so forked workers cannot multiply per-process limits. Reported
  diagnostics are polling-based sampled peaks for the primary scored invocation only; probe peaks
  are excluded, and a primary invocation skipped at the suite deadline is reported as unmeasured.

`scripts/validate.py` explicitly selects the Hyperfocal `bwrap` path. The Harbor entrypoint
explicitly selects `verifier-container`. There is no unconfined grader API or CLI mode.
