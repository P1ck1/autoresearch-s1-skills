"""Shared VCF representation primitives for private data and identity carriers.

Keeping these byte-level choices in one place prevents identity carriers from drifting into a
distinct file format that a submission could recognise without computing the requested object.
The helpers alter representation only; callers remain responsible for the mathematical draw.
"""

from __future__ import annotations

import numpy as np


# Kept separate so representation-equivalence probes can contrast conventional missing syntax
# with prefix-ambiguous malformed syntax while the scored generator exercises the full union.
CANONICAL_MISSING_GT_ENCODINGS = (b".", b"./.", b".|.")
MALFORMED_MISSING_GT_ENCODINGS = (
    b"0/.", b"./1", b"2/2", b"0/", b"1/1/1",
)
ZERO_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS = (b"0x1", b"00")
ONE_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS = (b"1x0", b"11", b"10/0")
PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS = (
    ZERO_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS
    + ONE_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS
)
MISSING_GT_ENCODINGS = (
    CANONICAL_MISSING_GT_ENCODINGS
    + MALFORMED_MISSING_GT_ENCODINGS
    + PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS
)
VARIABLE_FORMATS = (b"GT", b"GT:DP", b"GT:DP:GQ", b"GT:AD:DP:GQ")


def variable_gt_fields(
    gt_codes: np.ndarray,
    missing: np.ndarray | None,
    style: int,
    rng: np.random.Generator,
    missing_encodings: tuple[bytes, ...],
    missing_encoding_indices: np.ndarray | None,
) -> tuple[bytes, bytes]:
    """Encode one dosage row using the selected variable-width missing-call catalogue."""
    if style < 0 or style >= len(VARIABLE_FORMATS):
        raise ValueError(f"unknown variable-width FORMAT style: {style}")
    tokens: list[bytes] = []
    depths = rng.integers(4, 100, size=gt_codes.size)
    qualities = rng.integers(1, 100, size=gt_codes.size)
    phased = rng.random(gt_codes.size) < 0.55
    reverse_het = rng.random(gt_codes.size) < 0.5

    if missing_encoding_indices is not None:
        missing_encoding_indices = np.asarray(missing_encoding_indices)
        if missing_encoding_indices.shape != gt_codes.shape:
            raise ValueError("missing-encoding indices must match the dosage row")
        if (np.any(missing_encoding_indices < 0)
                or np.any(missing_encoding_indices >= len(missing_encodings))):
            raise ValueError("missing-encoding index is outside the selected catalogue")

    for col, raw_code in enumerate(gt_codes):
        is_missing = bool(missing[col]) if missing is not None else False
        if is_missing:
            if missing_encoding_indices is None:
                encoding_index = int(rng.integers(0, len(missing_encodings)))
            else:
                encoding_index = int(missing_encoding_indices[col])
            gt = missing_encodings[encoding_index]
            ad = b".,."
        else:
            code = int(raw_code)
            sep = b"|" if phased[col] else b"/"
            if code == 0:
                gt = b"0" + sep + b"0"
                ad = f"{int(depths[col])},0".encode()
            elif code == 1:
                alleles = (b"1", b"0") if reverse_het[col] else (b"0", b"1")
                gt = alleles[0] + sep + alleles[1]
                alt_depth = int(depths[col]) // 2
                ad = f"{int(depths[col]) - alt_depth},{alt_depth}".encode()
            elif code == 2:
                gt = b"1" + sep + b"1"
                ad = f"0,{int(depths[col])}".encode()
            else:
                raise ValueError(f"invalid simulated dosage: {code}")
            if code != 1 and rng.random() < 0.35:
                gt = b"0" if code == 0 else b"1"

        if style == 0:
            token = gt
        elif style == 1:
            token = gt + b":" + (b"." if is_missing else str(int(depths[col])).encode())
        elif style == 2:
            token = b":".join([
                gt,
                b"." if is_missing else str(int(depths[col])).encode(),
                b"." if is_missing else str(int(qualities[col])).encode(),
            ])
        else:
            token = b":".join([
                gt,
                ad,
                b"." if is_missing else str(int(depths[col])).encode(),
                b"." if is_missing else str(int(qualities[col])).encode(),
            ])
        tokens.append(token)

    return VARIABLE_FORMATS[style], b"\t".join(tokens)


def variable_info(rng: np.random.Generator) -> bytes:
    """Return scored-like INFO width variation independent of genotype and structure."""
    padding = b"x" * int(rng.integers(0, 193))
    return b";".join([
        b"DP=" + str(int(rng.integers(100, 1_000_000))).encode(),
        b"ZZ=" + padding,
    ])


def randomized_positions(
    n_variants: int, n_chroms: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw scored-like chromosome counts and increasing irregular positions."""
    chrom = np.sort(rng.integers(1, n_chroms + 1, size=n_variants)).astype(np.int32)
    pos = np.empty(n_variants, dtype=np.int64)
    for chromosome in range(1, n_chroms + 1):
        indices = np.flatnonzero(chrom == chromosome)
        if indices.size == 0:
            continue
        gaps = rng.integers(50, 4000, size=indices.size)
        pos[indices] = 10_000 + np.cumsum(gaps)
    return chrom, pos
