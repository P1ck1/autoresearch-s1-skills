"""Full-scan reference PCA -- the accuracy target and runtime comparator.

This is what a competent scientist writes without the systems tricks: read the entire
VCF, decode every clean biallelic SNV with a vectorised stride parse, Patterson-normalise,
and solve the smaller exact orientation (sample Gram normally, marker-space SVD when samples
outnumber markers). It uses *all* the data, so its top-k
subspace is the ground truth every submission is scored against; and its wall-clock is the
speed baseline a fast submission is measured against.

Usage:  python -m reference.full_scan_pca <vcf_path> <k> <out_scores.tsv>
"""

from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import numpy as np

from .pca_core import _standardize_kept, gram_to_scores


def _open(path: Path):
    with open(path, "rb") as fh:
        compressed = fh.read(2) == b"\x1f\x8b"
    if compressed:
        return gzip.open(path, "rb")
    return open(path, "rb")


def read_samples(path: Path) -> list[str]:
    with _open(path) as fh:
        for raw in fh:
            if raw.startswith(b"#CHROM"):
                cols = raw.rstrip(b"\r\n").split(b"\t")
                return [s.decode() for s in cols[9:]]
    raise ValueError("no #CHROM header line found")


def fit(path: Path, k: int, chunk: int = 20000):
    """Return sample IDs, scores, kept-variant count, and the full truth spectrum.

    Decodes with the reference's FORMAT-aware line decoder (bare GT fast path + GT:DP:GQ
    general path), so the truth anchor accepts exactly the records a correct FORMAT-aware
    submission accepts -- the two can never disagree because of a decode gap."""
    from .pca_core import _decode_lines                     # shared FORMAT-aware decode (non-oracle)
    sample_ids = read_samples(path)
    n = len(sample_ids)
    if n < 2 or k < 1 or k >= n:
        raise ValueError("k must satisfy 1 <= k < number of samples")
    gram: np.ndarray | None = None
    standardized_blocks: list[np.ndarray] = []
    counter = {"kept": 0}
    lines: list[bytes] = []

    def _flush():
        nonlocal gram
        mat = _decode_lines(lines, n)
        if mat is not None:
            standardized = _standardize_kept(mat)
            if standardized is not None:
                counter["kept"] += standardized.shape[0]
                if gram is None:
                    standardized_blocks.append(standardized)
                    if counter["kept"] > n:
                        # Once informative markers outnumber samples, the sample Gram is the
                        # smaller exact orientation. Fold the short undecided prefix into it and
                        # stream every later block without retaining the full design.
                        gram = np.zeros((n, n), dtype=np.float64)
                        for block in standardized_blocks:
                            gram += block.T @ block
                        standardized_blocks.clear()
                else:
                    gram += standardized.T @ standardized
        lines.clear()

    with _open(path) as fh:
        for raw in fh:
            if raw[:1] == b"#":
                continue
            line = raw.rstrip(b"\n").rstrip(b"\r")
            if line:
                lines.append(line)
                if len(lines) >= chunk:
                    _flush()
    if lines:
        _flush()

    if gram is None:
        if counter["kept"] < 1:
            raise ValueError("PCA requires at least one informative variant")
        standardized = np.vstack(standardized_blocks)
        # For markers x samples Z, the right singular vectors are the sample-PC directions and
        # singular_value/sqrt(n_markers) is the score scale. This is algebraically identical to
        # eigendecomposing Z.T @ Z / n_markers, but avoids materializing an n_samples-squared
        # matrix in the sample-heavy regime.
        _, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
        rank_floor = float(singular_values[0]) * np.sqrt(
            32.0 * np.finfo(standardized.dtype).eps
        )
        if singular_values.size < k or float(singular_values[k - 1]) <= rank_floor:
            raise ValueError(f"standardized design has rank below requested {k} PCs")
        scaled = singular_values / np.sqrt(counter["kept"])
        scores = right[:k].T * scaled[:k][None, :]
        spectrum = np.zeros(n, dtype=np.float64)
        spectrum[:scaled.size] = np.square(scaled)
        for component in range(scores.shape[1]):
            column = scores[:, component]
            if column[np.argmax(np.abs(column))] < 0:
                scores[:, component] = -column
    else:
        scores, _, spectrum = gram_to_scores(gram, counter["kept"], k)
    return sample_ids, scores, counter["kept"], spectrum


def write_scores(sample_ids, scores, out_path: Path):
    k = scores.shape[1]
    with open(out_path, "w") as fh:
        fh.write("sample_id\t" + "\t".join(f"PC{i+1}" for i in range(k)) + "\n")
        for sid, row in zip(sample_ids, scores):
            fh.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")


def main(argv):
    if len(argv) != 3:
        print("usage: full_scan_pca <vcf> <k> <out.tsv>", file=sys.stderr)
        return 2
    vcf, k, out = Path(argv[0]), int(argv[1]), Path(argv[2])
    t0 = time.time()
    sample_ids, scores, kept, _ = fit(vcf, k)
    write_scores(sample_ids, scores, out)
    dt = time.time() - t0
    print(f"full_scan: {len(sample_ids)} samples x {kept} variants -> {scores.shape[1]} PC "
          f"in {dt:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
