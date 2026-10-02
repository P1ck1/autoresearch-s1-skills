# Agent Public/Dev smoke evidence

`final_smoke.stdout.json`, `final_smoke.stderr.txt` and
`final_smoke.execution.json` are directly saved outputs and execution metadata
from one actual final smoke run with `--repeats 3`. The metadata contains the
exact command argv, UTC timestamps, exit code, runtime versions and source
hashes. Empty stderr is an actual empty stream, not a missing log.

The unmodified public starter was evaluated against the same frozen baseline.
All three cases passed, the command exited 0, and the score was
`0.05065513417874165`. This self-comparison value reflects noisy end-to-end
timings; it is not a method improvement or a normalized Hidden score.

The runtime was Python 3.12.14 / NumPy 2.3.5. SciPy was unavailable and not used
by this smoke. The pinned Docker runtime (Python 3.12.8 / NumPy 2.2.6 /
SciPy 1.15.3) was not built or executed. There is no formal Hidden, security,
resource-limit, G03, training-seed or long-duration validation in these files.

## Earlier tool-trace-only observations

The following is a retrospective summary of outputs shown in this chat's tool
trace. Original stdout/stderr files and exact per-attempt timestamps were not
saved at that time. This section is not reconstructed raw logging.

1. The first attempted command was system `python3` running
   `workspace/harbor_task/environment/public_eval/grader.py --submission
   workspace/harbor_task/environment/starter` (paths relative to this package
   shown for readability; actual command was launched from the surrounding
   authoring workspace). It exited 1 with JSON status `evaluator_error`, score
   null. Its frozen-baseline child failed with `ModuleNotFoundError: No module
   named 'numpy'`. Inspection showed system Python 3.9.6's NumPy 2.0.2 and
   SciPy 1.13.1 were installed in the user site, which the child deliberately
   disabled via `PYTHONNOUSERSITE=1`. No package was installed to bypass this.
2. A following one-repeat command using the bundled Python 3.12.14 / NumPy
   2.3.5 exited 0. All three case similarities were 1.0 and the reported Dev
   score was `0.38416920225556045`. This preceded a small robustness edit to
   the Dev parser/scale normalization; it is not the final raw log above.
3. A local verification script reported AST parsing and all fixture hashes
   passing; a hand-constructed six-sample VCF matched an independently formed
   NumPy SVD with similarity `0.9999999999999031`. A malformed output header
   produced `invalid_output` / null score / exit 1; zero-rank PC output and a
   missing pca entry were rejected. This verification script was ephemeral,
   so its complete source/individual subprocess logs were not retained.
4. A subsequent small robustness check confirmed the public subspace metric
   remains 1.0 for equivalent columns scaled by `1e300` and `1e-200` (tolerance
   1e-10). Its output was a single PASS line in the tool trace, not a saved log.

See the outer authoring report `agent_adaptation_report.md` for the adaptation
scope and explicitly unverified checks. Only the three `final_smoke.*` files
are raw persisted runtime evidence from this directory.
