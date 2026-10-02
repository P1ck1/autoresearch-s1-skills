# Fast Population-Structure PCA

## Goal

Given raw multi-sample VCF files, implement a fast HWE-normalized PCA program that outputs the leading sample principal components. Preserve the mathematical object and output contract while improving end-to-end runtime and resource use. Submit your own runnable implementation in /workspace/solution/pca and its necessary helper files.

## Task Setting

The Agent environment is CPU-only: 8 CPUs and 16 GiB memory, Python 3.12.8, NumPy 2.2.6 and SciPy 1.15.3, with no runtime network or package installation. A runnable full-scan Starter is frozen in /workspace/starter and copied to /workspace/solution initially. There is no model training or pretrained checkpoint.

Three small public development VCFs, their public seeds and SHA-256 hashes are frozen under /workspace/public_assets/dev/. They support iterative debugging and timing feedback; they do not cover the sizes and diversity of final testing. Final test data and the trusted evaluator are outside the Agent environment. The final evaluator runs after this Agent run ends.

A VCF is plain text: metadata lines starting with `##`, one `#CHROM` header line naming the
samples, then one line per variant. Records are sorted by genomic position within each
chromosome. Genotypes are in the per-sample columns; the `GT` sub-field (always first when
present) is the genotype, `|` phased or `/` unphased, `.` missing. A realistic head:

```
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1>
#CHROM  POS     ID   REF  ALT   QUAL  FILTER  INFO   FORMAT   sample_A  sample_B  sample_C  sample_D
chr1    10177   .    A    AC    100   PASS    .      GT       0|0      0|1      1|1      0|0
chr1    11008   .    C    G     100   PASS    .      GT       0|0      ./.      0|1      1|1
chr1    13110   .    G    A     100   PASS    AC=3   GT:DP    0/0:31   0/1:22   1/1:18   0/1:27
chr1    16145   .    T    G,C   100   PASS    .      GT       0|0      1|2      0|1      2|2
```

Real data is messy: some records are indels (`A`->`AC`), symbolic/structural (`<DEL>`),
multiallelic (`T`->`G,C`), or have missing calls (`./.`); the FORMAT may carry extra
sub-fields (`GT:DP:GQ`). Your program must read the genotypes out of this correctly and not
crash on the messiness.

Inputs may be much larger than available memory. End-to-end runtime includes reading the VCF,
computing the scores, and writing the result.

## Objective and Metrics

For each biallelic SNV with alt-allele frequency `p`, a genotype dosage `x in {0,1,2}` is
standardised as

    z = (x - 2p) / sqrt(2 p (1 - p))

with missing calls imputed to the column mean. The requested output is the leading **sample score
subspace** of the standardised genotype matrix. The `sqrt(2p(1-p))` denominator is required --
it is what makes this a population-genetics PCA rather than a plain covariance PCA.

Consider every biallelic SNV whose REF and ALT are each exactly one of A/C/G/T,
case-insensitively, eligible. Skip
multiallelic records (comma in ALT), symbolic/structural ALTs (`<DEL>`, breakends), indels, and
monomorphic sites. A diploid call contains two alleles from `{0,1}` separated by `/` or `|`.
A haploid `0` or `1` call is pseudo-diploid dosage `0` or `2`. Missing, partial, out-of-range,
or malformed calls are missing and are mean-imputed for an otherwise eligible site.

**Do not apply a *selective* filter on allele frequency or linkage.** Do **not** apply a
minor-allele-frequency (MAF) cutoff, LD pruning, a missingness threshold, or an HWE-departure
test -- those are common in ancestry pipelines but they bias *which* variants enter the fit and
change the result into a *different* object, which is not what this tool computes. Rare variants
belong in the fit: the `sqrt(2p(1-p))` standardisation already weights every site correctly.

Exact and approximate numerical methods are permitted. Any approximation must remain faithful to
the defined all-eligible-marker decomposition and must not systematically exclude markers by
allele frequency, linkage, missingness, record width, chromosome, or genomic region.

For iterative development, run:

```bash
python /workspace/public_eval/grader.py --submission /workspace/solution --repeats 3
```

This returns the raw public proxy mean_log2_dev_speedup, which is maximized. For each case, t_B and t_C are the median wall-clock times of the frozen full-scan Starter and candidate over the same number of repeats. The score is mean_cases(log2(t_B/t_C)). Startup, input reading and output writing are included. Scores are not clipped; negative values mean slower than the public baseline. Timings fluctuate, so compare methods with the same repeats and environment rather than one lucky observation.

Every output must have correct sample order, exactly k finite PC columns and full rank. The public quality gate requires principal-subspace similarity at least 0.98 for every case and repetition, computed as ||Q_C.T @ Q_B||_F^2 / k from centered, column-normalized matrices. Failure has a separate status and score=null. Each public invocation has a 30-second timeout. This public score is a development proxy, not the final reward; it neither predicts the final ranking perfectly nor replaces final constraints.

The final objective is the source task's native_reward, maximized in [0,1]. Per-dataset reward is accuracy × (0.10 + 0.90 × systems_unlock(accuracy) × time_quality) × method_factors. The accuracy unlock ramps from 0 at 0.75 accuracy to 1 at 0.90. Accuracy combines appropriate subspace/structure checks; time_quality compares end-to-end work to trusted timing anchors under the same execution conditions; method factors test the defined mathematical object. Dataset results are weighted within categories, then categories are aggregated with their mean dataset weights. Private evaluation fixtures, reference implementations and per-case final results are not available to the Agent. The public proxy and final reward are different numbers and must never be substituted for each other.

## Allowed Scope

- Modify /workspace/solution/pca and add necessary pure-Python helpers or data under /workspace/solution, subject to the bounds below.
- Read the public Starter, public evaluator and public development assets. These are frozen and must not be modified.
- Use NumPy/SciPy and the Python standard library within the from-scratch rules. Method implementations may change; the task is not restricted to tuning a few scalar parameters.
- Write new run notes and the current best code only to the dedicated empty output location supplied by the Harness; never read another run's notes or a complete expert evidence package.

## Hard Boundaries

This is a **pure-Python** task (Python 3.12). Implement the PCA yourself using only the general
numeric stack: **`numpy`** and **`scipy`** (`scipy.linalg`, `scipy.sparse`, `scipy.special`, and
`numpy.linalg.eigh`/`svd` for the eigendecomposition). That is the whole allowed dependency set.

What is **banned**:

- **any other third-party package** -- in particular no JIT/compiler or native-extension
  toolkits (`numba`, `cython`, `cffi`, `ctypes`-loaded native code, `pybind11`, `torch`, `jax`,
  ...), and no shelling out to a compiler or to another language runtime. Write the hot loops in
  numpy/Python;
- encoded or reconstructed executable/native payloads and child `exec`/`spawn` calls. Parallel
  pure-Python workers may use the Linux `fork` start method, but the submitted program must not
  start another executable. Fork workers must remain on pure-Python/NumPy/SciPy paths;
- foreign-function interfaces: `ctypes` is rejected outright -- not merely ctypes-loaded native
  code -- and so is any standard-library API implemented through it, including the ctypes-backed
  multiprocessing shared objects (`Value`/`Array`/`sharedctypes`);
- genotype/VCF ingestion via a genomics library (`pysam`, `cyvcf2`, `hail`, `sgkit`,
  `scikit-allel`, `pandas-plink`, `bed-reader`, `plinkio`, `pgenlib`, ...), or shelling out to
  `plink`/`plink2`/`bcftools`/`gcta`/`flashpca`/`vcftools`;
- the PCA/decomposition itself via a stats/ML library (`sklearn`'s `PCA`/`TruncatedSVD`,
  `statsmodels`, `scikit-allel`'s `pca`, `dask-ml`, ...).

The speed has to come from the algorithm and from numpy/BLAS, not from dropping to another language.

Keep the submission tree ordinary and self-contained: `pca` and any helper/data
files must be regular files under `/workspace/solution`. Symbolic links, device/FIFO
entries, more than 4096 files, or more than 64 MiB of submitted files are
rejected before execution.

The runtime copies that tree once into an immutable private snapshot. Each
invocation is limited to 16 GiB of address space per process, 14 GiB aggregate
resident memory, 4 GiB of aggregate temporary storage, 128 processes,
256 open files, and 64 MiB per output file. The wall-clock limit is one hour per
invocation; calls on smaller inputs may use a 15-minute limit. Only the named
output and an isolated temporary filesystem are writable.

- Work fully offline: no web, no installing packages; use the pre-installed toolchain.
- Confine file reads and writes to the paths you are given. You may inspect the supplied VCF's
  metadata and size, and ordinary process CPU-count/affinity information. Do not inspect private
  runtime files, sibling paths, Git state, or unrelated filesystem contents.
- Ship whatever helper modules or data files you like under `/workspace/solution`; that whole tree is
  available to your program. **While `pca` runs, the only paths it can WRITE are `<out_path>` and
  its temporary directory (`TMPDIR`, an isolated scratch filesystem).** Its own directory is
  read-only at that point, so put runtime scratch in `TMPDIR` -- not beside your program, and not
  beside the output.
- Keep a valid, runnable `pca` in place.


The public development evaluator is a convenience tool, not a security sandbox. Final execution is in a separate verifier with immutable candidate staging, file/user/process restrictions and trusted score output. The Agent must not request final hidden evaluations during iteration, inspect private assets, alter the evaluator or fabricate metrics. Final contract/dependency failures retain the native source task's failed/zero-reward behavior; development failures use their own explicit status and null score.

## Submission Instructions

Provide one program:

- `/workspace/solution/pca`

Invoked as:

    pca <vcf_path> <k> <out_path>

- `<vcf_path>`: a multi-sample VCF (plain `.vcf`; may be a large, unindexed text file).
- `<k>`: number of principal components to output.
- `<out_path>`: where to write the scores TSV.

`pca` is run as a Python 3.12 script under the runtime interpreter with `numpy` and `scipy`;
there is no build/compile step. Because the entry file is named `pca` (no `.py` extension), start
it with a shebang line containing `python` (e.g. `#!/usr/bin/env python3`) so it is recognised as
a Python program.

`pca` writes `<out_path>`, a TSV with a header row:

    sample_id<TAB>PC1<TAB>PC2<TAB>...<TAB>PCk

exactly one row per sample, **in the order the samples appear in the VCF `#CHROM` header
line**, with the sample IDs from that line in the `sample_id` column. Always emit exactly `k`
finite PC columns; any other shape is invalid.

Write `<out_path>` **in place**. That exact file already exists and is the only writable entry in
its directory: you cannot create a sibling next to it, so the usual write-a-temp-file-then-rename
idiom fails there. Use your own scratch space if you want to stage the bytes, then write them into
`<out_path>` itself.

The program must run non-interactively, read no network, and exit 0 on success. The caller
supplies `1 <= k < number_of_samples`.

Inputs contain at least `k` identifiable, positive-variance sample directions.

The only transferred artifact is /workspace/solution. It must be self-contained; no dependencies on Agent-only /tmp files, your output notes, or the public evaluator are allowed during final invocation. Keep the final program and all helpers together. This task has no model checkpoint. Do not submit scores, logs or a copy of the grader as a solution.

## Workflow & Iteration

Run the unchanged Starter with public evaluation first. Then repeat: implement a method change → run the same Public/Dev protocol → read metrics and errors → retain the best valid method or revert → choose the next direction. Keep the eight trajectory fields round, policy_name, method_summary, status, score, failure_reason, retained_best and time when recording iterations; score means the actual public proxy returned by the evaluator, not a claimed final result.

Before ending, restore the best valid method discovered under the same public protocol to /workspace/solution and rerun public evaluation. A best-code copy in an output folder is not enough: the transferred directory itself must contain that candidate. Harbor transfers this final directory only after the Agent stops; the independent verifier then evaluates it without launching another Coding Agent.

## Completion Criteria

- /workspace/solution/pca and all required helpers are present, regular files and runnable under the program contract.
- The selected candidate has valid, finite public feedback and passed the public quality gate; its source is the candidate actually in /workspace/solution.
- Frozen files and private assets have not been accessed or changed, and results have not been fabricated.
- The final program does not depend on an Agent-only environment path outside the declared runtime and the provided input/output/scratch paths.
- Final acceptance remains the independent verifier's result; successful public development is not a claim of final acceptance.
