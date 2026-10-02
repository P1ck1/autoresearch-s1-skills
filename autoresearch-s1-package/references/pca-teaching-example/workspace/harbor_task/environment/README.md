# Public PCA development environment

This teaching variant supplies a runnable full-scan baseline and small public
fixtures. The baseline reads every eligible VCF site, mean-imputes missing
genotypes, applies `(x - 2*p) / sqrt(2*p*(1-p))`, and computes exact sample PCA.
It is an ordinary executable candidate, not an optimized answer.

The candidate entry is `/workspace/solution/pca`. Invoke it with:

```sh
python /workspace/solution/pca <vcf_path> <k> <out_path>
```

The baseline supports plain or gzip VCF, GT-first FORMAT fields, diploid and
pseudo-diploidized haploid calls. It skips ineligible or uninformative variants.
Output is a TSV with `sample_id`, `PC1`, ..., `PCk`, in VCF sample order, written
directly into the supplied output file. Helpers must remain inside `solution/`.
Only the standard library, NumPy and SciPy are available for candidate code.

Run this after each method change, in the same Agent environment:

```sh
python /workspace/public_eval/grader.py --submission /workspace/solution
python /workspace/public_eval/grader.py --submission /workspace/solution --repeats 3
```

The JSON `score` is the raw, maximize-oriented **Public/Dev proxy**, not a
normalized final score. For each of three public cases, let `t_B` and `t_C` be
the median end-to-end wall times of the frozen baseline and candidate over
`--repeats` runs. Process startup, VCF reading and output writing are included.
The default is one run; the same repeat count is used for both programs.

`score = mean_cases(log2(t_B / t_C))`.

All output schemas, sample orders and finite values must be valid. For each
run, centered and column-normalized score matrices produce orthonormal bases
`Q_C` and `Q_B`; the subspace similarity is
`||Q_C.T @ Q_B||_F^2 / k`. The quality gate requires full column rank and
similarity at least **0.98 on every case and repetition**. On gate or execution
failure, `score` is `null`, `status` identifies the failure and the command
exits nonzero. Successful scores are not clipped; negative values mean slower
than the contemporaneous public baseline. Each invocation times out at 30 s.

Fixtures, public seeds and SHA-256 hashes are recorded in
`public_assets/dev/manifest.json`. Their independent public generator is
`public_eval/generate_dev.py`. Committed fixture bytes are the versioned input;
regenerating with another NumPy version can change bytes and requires a new
manifest. Do not overwrite frozen fixtures during Agent runs.

These tiny fixtures test the workflow and some parsing/numerical behavior.
Timings include startup and cache/order noise (baseline runs before candidate).
They do not establish long-run optimization difficulty, G03 significance,
large-input speed, generalization, or compliance with the complete final
execution/library rules. Public scoring is not a security sandbox.

After the Agent exits, Harbor copies the declared final candidate to an
independent verifier for final Hidden evaluation. The native Hidden reward,
cases, constraints and runtime comparisons are separate from this proxy;
the two numbers must not be treated as equal or substituted for each other.
Do not use Hidden results as feedback inside the current Agent iteration.

`/workspace/starter`, `/workspace/public_eval`, `/workspace/public_assets` and
this README are root-owned and frozen in the image. Work in `/workspace/solution`.
The runtime output directory `/expert_evidence` is initially empty and contains
only this run's new notes; it is not the complete platform evidence package.
