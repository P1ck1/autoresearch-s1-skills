"""Shared PCA math for the reference implementations.

This is the *definition* of the object the benchmark is about: a Patterson-Price-Reich
(2006) HWE-normalised principal component analysis of genotype dosages.

For each biallelic SNV j with alt-allele frequency p_j, a genotype dosage x_ij in {0,1,2}
is standardised as

    z_ij = (x_ij - 2 p_j) / sqrt(2 p_j (1 - p_j))

with missing calls set to the column mean (0 after centring). The sample-by-sample Gram
matrix G = Z Z^T (n_samples x n_samples) has top-k eigenvectors which, scaled by
sqrt(eigenvalue), are the PC scores. In the usual genomic regime n_samples << n_variants,
working through the small n x n Gram is exact and much cheaper than an SVD of the full Z.
The full-scan reference switches to the algebraically equivalent marker-space SVD for the
separate sample-heavy stress regime.

The HWE denominator sqrt(2p(1-p)) is load-bearing, not cosmetic: it is what makes this a
*population-genetics* PCA rather than a plain covariance PCA, and the identity gate probes
for exactly it.
"""

from __future__ import annotations

import numpy as np


# Byte-level GT decoding constants.
_ZERO = ord("0")
_ONE = ord("1")
_TAB = ord("\t")
_COLON = ord(":")
_SLASH = ord("/")
_PIPE = ord("|")
_DIPLOID_LUT = np.full(1 << 16, -1, dtype=np.int8)
_DIPLOID_LUT[(_ZERO << 8) | _ZERO] = 0
_DIPLOID_LUT[(_ZERO << 8) | _ONE] = 1
_DIPLOID_LUT[(_ONE << 8) | _ZERO] = 1
_DIPLOID_LUT[(_ONE << 8) | _ONE] = 2


def is_clean_biallelic_snv(ref: bytes, alt: bytes) -> bool:
    """True iff REF and ALT are single ACGT bases (drops multiallelic + symbolic SV)."""
    return (len(ref) == 1 and len(alt) == 1
            and ref in b"ACGTacgt" and alt in b"ACGTacgt")


def _decode_general(sb: bytes, n_samples: int) -> np.ndarray | None:
    """Decode leading diploid GT tokens, treating unsupported individual calls as missing.

    Field boundaries are found in one C pass and valid calls are gathered by fancy indexing.
    A malformed sample count rejects the record; bare missing, polyploid, multiallelic, or
    otherwise unsupported *calls* become -1 without discarding valid peers. Haploid 0/1 calls
    are pseudo-diploidized to 0/2 so sex-chromosome and mitochondrial variants remain usable.

    This decoder (and ``_decode_lines`` below) is the *shared* FORMAT-aware VCF decode used by the
    full-scan truth anchor. It deliberately lives in this non-oracle module -- NOT in the fast
    reference -- so the packaged task tree can ship the truth/decoder without shipping the fast
    solver a submission could exec. ``reference/fast_pca.py`` keeps its own self-contained copy.
    """
    if n_samples < 1:
        return None
    u = np.frombuffer(sb, dtype=np.uint8)
    tabs = np.flatnonzero(u == _TAB)
    if tabs.size != n_samples - 1:
        return None
    starts = np.empty(n_samples, dtype=np.intp)
    ends = np.empty(n_samples, dtype=np.intp)
    starts[0] = 0
    if n_samples > 1:
        starts[1:] = tabs + 1
        ends[:-1] = tabs
    ends[-1] = u.size
    widths = ends - starts
    dose = np.full(n_samples, -1, dtype=np.int8)

    haploid = np.flatnonzero(widths >= 1)
    if haploid.size:
        pos = starts[haploid]
        allele = u[pos]
        term_pos = pos + 1
        field_ends = ends[haploid]
        good_term = term_pos == field_ends
        has_suffix = term_pos < field_ends
        good_term[has_suffix] = u[term_pos[has_suffix]] == _COLON
        called = ((allele == _ZERO) | (allele == _ONE)) & good_term
        dose[haploid[called]] = 2 * (allele[called] == _ONE).astype(np.int8)

    diploid = np.flatnonzero(widths >= 3)
    if diploid.size:
        pos = starts[diploid]
        a1 = u[pos]
        sep = u[pos + 1]
        a2 = u[pos + 2]
        term_pos = pos + 3
        field_ends = ends[diploid]
        good_term = term_pos == field_ends
        has_suffix = term_pos < field_ends
        good_term[has_suffix] = u[term_pos[has_suffix]] == _COLON
        called = (((a1 == _ZERO) | (a1 == _ONE))
                  & ((a2 == _ZERO) | (a2 == _ONE))
                  & ((sep == _SLASH) | (sep == _PIPE))
                  & good_term)
        targets = diploid[called]
        dose[targets] = ((a1[called] == _ONE).astype(np.int8)
                         + (a2[called] == _ONE).astype(np.int8))
    return dose


def _decode_lines(lines, n_samples: int):
    """Decode an iterable of raw VCF record lines into a (n_kept x n_samples) int8 dosage
    matrix (missing = -1), or None. GT-only rows take a validated batched fast path; richer or
    irregular GT rows use the general leading-token decoder. Invalid calls become missing while
    valid calls in the same record remain usable. Shared truth-side decoder.
    """
    if n_samples < 1:
        return None
    expected = 4 * n_samples - 1
    fields = []
    general = []
    for line in lines:
        if not line or line[:1] == b"#":
            continue
        if line[-1:] == b"\r":
            line = line[:-1]
        parts = line.split(b"\t", 9)
        if len(parts) < 10 or not is_clean_biallelic_snv(parts[3], parts[4]):
            continue
        fmt = parts[8]
        if fmt != b"GT" and not fmt.startswith(b"GT:"):
            continue
        sb = parts[9]
        if fmt == b"GT" and len(sb) == expected:        # validate layouts together below
            fields.append(sb)
        else:
            row = _decode_general(sb, n_samples)
            if row is not None:
                general.append(row)
    mats = []
    if fields:
        arr = np.frombuffer(b"".join(fields), dtype=np.uint8).reshape(len(fields), expected)
        tab_slots = arr[:, 3::4]
        layout_ok = None if np.all(tab_slots == _TAB) else np.all(tab_slots == _TAB, axis=1)
        if layout_ok is None or layout_ok.any():
            good = arr if layout_ok is None else arr[layout_ok]
            a1 = good[:, 0::4]
            sep = good[:, 1::4]
            a2 = good[:, 2::4]
            keys = (a1.astype(np.uint16) << 8) | a2
            dose = _DIPLOID_LUT[keys]
            dose[(sep != _SLASH) & (sep != _PIPE)] = -1
            mats.append(dose)
        if layout_ok is not None:
            for index in np.flatnonzero(~layout_ok):
                row = _decode_general(fields[int(index)], n_samples)
                if row is not None:
                    general.append(row)
    if general:
        mats.append(np.asarray(general, dtype=np.int8))
    if not mats:
        return None
    return mats[0] if len(mats) == 1 else np.vstack(mats)


def _standardize_kept(dose_block: np.ndarray) -> np.ndarray | None:
    """Return informative Patterson-standardized rows in float64 for the truth anchor.

    The integer retention rule and minor-allele orientation exactly match the fast reference;
    float64 arithmetic keeps the full-scan Gram a high-precision accuracy target.
    """
    missing = dose_block < 0
    n_missing = missing.sum(axis=1)
    n_valid = dose_block.shape[1] - n_missing
    isum = dose_block.sum(axis=1, dtype=np.int64) + n_missing
    isumsq = np.square(dose_block).sum(axis=1, dtype=np.int64) - n_missing
    total_alleles = 2 * n_valid
    minor_count = np.minimum(isum, total_alleles - isum)
    informative = n_valid * isumsq > isum * isum
    keep = (n_valid > 1) & (minor_count > 0) & informative
    if not keep.any():
        return None
    Z = dose_block[keep].astype(np.float64)
    flip = isum[keep] > n_valid[keep]
    Z[flip] = 2.0 - Z[flip]
    p = minor_count[keep].astype(np.float64) / (2.0 * n_valid[keep])
    Z -= (2.0 * p)[:, None]
    Z[missing[keep]] = 0.0
    Z /= np.sqrt(2.0 * p * (1.0 - p))[:, None]
    return Z


def standardize_into_gram(dose_block: np.ndarray, gram: np.ndarray,
                          counter: dict) -> None:
    """Patterson-standardize informative rows and accumulate their sample Gram in place."""
    Z = _standardize_kept(dose_block)
    if Z is None:
        return
    gram += Z.T @ Z
    counter["kept"] += Z.shape[0]


def gram_to_scores(gram: np.ndarray, n_variants_kept: int, k: int):
    """Top-k scores plus the complete descending eigenvalue spectrum of the truth Gram."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram matrix must be square")
    n = gram.shape[0]
    k = int(k)
    if n < 2 or k < 1 or k >= n:
        raise ValueError("k must satisfy 1 <= k < number of samples")
    if n_variants_kept < 1:
        raise ValueError("PCA requires at least one informative variant")
    G = gram / n_variants_kept
    # Symmetric eigendecomposition, ascending eigenvalues.
    all_evals, evecs = np.linalg.eigh(G)
    raw_spectrum = all_evals[::-1]
    rank_floor = abs(float(raw_spectrum[0])) * (32.0 * np.finfo(G.dtype).eps)
    if float(raw_spectrum[k - 1]) <= rank_floor:
        raise ValueError(f"standardized design has rank below requested {k} PCs")
    spectrum = np.clip(raw_spectrum, 0.0, None)
    evals = spectrum[:k]
    evecs = evecs[:, ::-1][:, :k]
    scores = evecs * np.sqrt(evals)[None, :]
    # Canonical sign per component.
    for c in range(scores.shape[1]):
        col = scores[:, c]
        if col[np.argmax(np.abs(col))] < 0:
            scores[:, c] = -col
    return scores, evals, spectrum
