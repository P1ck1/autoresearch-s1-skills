"""Subspace-agreement accuracy metric.

A PCA is only defined up to rotation, reflection and scaling within its subspace, so we
cannot compare score matrices entrywise. What is invariant -- and what actually matters for
ancestry -- is whether the submission's score space *spans the same directions over samples*
as the reference. For each reference PC c (a unit direction over samples) we measure how much
of it lives inside the submission's score subspace:

    a_c = || Q_sub^T r_c ||   in [0, 1]

the cosine of the principal angle between direction r_c and span(submission). Because a_c
depends only on the submission's *span*, it is invariant to any rotation/reflection/scaling
of the submission within its own subspace -- a legitimate PCA basis is never penalised.

AGGREGATION -- why a geometric mean over the *structured* PCs, not an eigenvalue-weighted
arithmetic mean.  Ancestry structure lives on a handful of PCs standing far above the
Marchenko-Pastur noise bulk (see ``structure_weights``); the bulk PCs are arbitrary
directions no subsample can reproduce and must not be graded. But the structured PCs are
wildly unequal in eigenvalue -- on real 1000G chr22 the leading axis (~45) dwarfs a genuine
sub-continental axis (~3). An eigenvalue-weighted *arithmetic* mean is therefore dominated by
PC1: a submission can silently DROP a real structured axis (a_c -> ~0) and still score ~0.99
because PC1's weight swamps the loss. That is exactly backwards -- losing a real continental
axis should cost real reward.

So we (a) select the structured PCs using the *same* floor as ``structure_weights`` (weight
> 0), ignoring the noise bulk entirely, and (b) aggregate their agreements with an *equal-
weight geometric mean* rather than an eigenvalue-weighted arithmetic one. What matters is
*whether a PC is structured*, not how large its eigenvalue happens to be, so every structured
axis is weighted equally. The geometric mean makes each structured axis a multiplicative
factor: recover them all and the product is ~1; drop any single one (a_c -> 0) and the whole
raw score collapses toward 0. It is the smooth relative of ``min_c a_c`` ("you must recover
every structured axis") without the brittleness of a hard minimum.

We then affine-normalise so a *random* submission scores ~0 and a subspace that fully
contains the reference directions scores 1:

    raw      = geomean_{c structured} a_c
    accuracy = clip( (raw - chance) / (1 - chance), 0, 1 )

where ``chance`` is the exact expected projection norm for a random k_sub-dimensional subspace
of the centred sample space. A random submission therefore maps to ~0 and a faithful anchor to 1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


_MIN_NOISE_BULK = 64
_TW_SAFETY = 5.0


def _johnstone_upper_edge(bulk: np.ndarray) -> float:
    """Finite-spectrum upper noise edge in Johnstone's Tracy-Widom scaling.

    ``var / mean**2`` estimates the effective aspect ratio, including LD-driven widening. The
    asymptotic MP edge is only the centre of the largest-noise-eigenvalue distribution: using it
    as a hard cutoff promotes ordinary finite-sample fluctuations to equally weighted ancestry
    PCs. Five Tracy-Widom scale units give those fluctuations the required safety margin."""
    mean = float(bulk.mean())
    aspect = max(float(bulk.var()) / max(mean * mean, 1e-24), 0.0)
    if aspect <= np.finfo(float).eps:
        return mean

    p = float(bulk.size)
    effective_markers = p / aspect
    n_term = effective_markers - 0.5
    p_term = p - 0.5
    if n_term <= 0.0 or p_term <= 0.0:
        raise ValueError("PCA noise bulk has an invalid effective aspect ratio")
    root_n = np.sqrt(n_term)
    root_p = np.sqrt(p_term)
    centre = mean * (root_n + root_p) ** 2 / effective_markers
    scale = (mean * (root_n + root_p) / effective_markers
             * (1.0 / root_n + 1.0 / root_p) ** (1.0 / 3.0))
    return float(centre + _TW_SAFETY * scale)


def _noise_bulk_edge(positive: np.ndarray) -> float:
    """Fit one self-consistent, finite-spectrum noise edge.

    Initialize from the largest guaranteed noise tail under the model that structured spikes are
    a strict minority. This prevents any minority of arbitrarily large spikes from inflating its
    own cutoff. The Johnstone edge is then refit against the complete positive spectrum until
    membership is stable, restoring reserved directions that are ordinary noise. At least 64
    eigenvalues must identify noise; anything smaller is statistically underdetermined."""
    floor = max(_MIN_NOISE_BULK, (positive.size + 1) // 2)
    bulk = np.asarray(positive[-floor:], dtype=float).copy()

    for _ in range(positive.size):
        edge = _johnstone_upper_edge(bulk)
        candidate = positive[positive <= edge * (1.0 + 1e-12)]
        if candidate.size < floor:
            raise ValueError("structured axes leave too little identifiable PCA noise bulk")
        if candidate.size == bulk.size:
            return edge
        bulk = candidate
    raise RuntimeError("finite PCA noise-edge estimation did not converge")


def structure_weights(evals: np.ndarray, n_scored: int) -> np.ndarray:
    """Excess over a self-consistent finite-spectrum PCA noise edge.

    ``evals`` is the complete descending truth spectrum, including the centred zero mode. Genuine
    outliers retain positive weight while finite-sample noise-edge fluctuations receive exactly
    zero. ``n_scored`` is the number of leading axes the caller will score and reserves enough
    additional dimensions to identify noise. Short spectra are rejected because they do not
    identify a noise model accurately enough; a well-populated spectrum with no significant
    outlier returns zero weights and is rejected by the reference cache as structureless."""
    evals = np.asarray(evals, dtype=float)
    n_scored = int(n_scored)
    if n_scored < 1:
        raise ValueError("n_scored must be positive")
    if evals.ndim != 1 or evals.size < _MIN_NOISE_BULK + n_scored + 1:
        raise ValueError("structure detection requires a complete PCA spectrum with at least "
                         f"{_MIN_NOISE_BULK} noise dimensions beyond the scored axes")
    if not np.isfinite(evals).all() or np.any(evals < -1e-10):
        raise ValueError("invalid PCA eigenvalue spectrum")
    evals = np.clip(evals, 0.0, None)
    scale = max(float(evals.max()), 1e-12)
    if np.any(evals[:-1] + scale * 1e-12 < evals[1:]):
        raise ValueError("PCA eigenvalue spectrum must be descending")
    positive = evals[evals > scale * 1e-12]
    if positive.size < _MIN_NOISE_BULK + n_scored:
        raise ValueError("too few positive eigenvalues beyond the scored axes to identify "
                         "the PCA noise bulk")

    edge = _noise_bulk_edge(positive)
    return np.clip(evals - edge * (1.0 + 1e-12), 0.0, None)


def _random_projection_mean(k: int, dimension: int) -> float:
    """E[sqrt(Beta(k/2, (dimension-k)/2))], the exact random-subspace chance level."""
    if k <= 0:
        return 0.0
    if k >= dimension:
        return 1.0
    return float(np.exp(
        gammaln((k + 1) / 2) + gammaln(dimension / 2)
        - gammaln(k / 2) - gammaln((dimension + 1) / 2)
    ))


def _centred_unit_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and unit-normalise columns without an absolute magnitude threshold.

    PCA score columns have arbitrary nonzero scale. Computing a raw norm first can underflow on a
    tiny but valid column, while comparing that norm with a fixed epsilon makes the metric change
    when an otherwise identical solver merely changes output units. Divide by each column's maximum
    absolute entry before taking its norm; this is stable for every finite representable scale.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("score matrix must be two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("score matrix must be finite")
    # Scale before centring: summing a finite column near float64's upper edge can overflow while
    # computing its mean. Per-column division preserves the centred direction exactly up to normal
    # floating-point rounding and makes both the mean and subsequent norm bounded.
    raw_maxima = np.max(np.abs(values), axis=0, initial=0.0)
    raw_nonzero = raw_maxima > 0.0
    scaled = np.zeros_like(values)
    if np.any(raw_nonzero):
        scaled[:, raw_nonzero] = values[:, raw_nonzero] / raw_maxima[raw_nonzero][None, :]
    centred = scaled - scaled.mean(axis=0, keepdims=True)
    centred_maxima = np.max(np.abs(centred), axis=0, initial=0.0)
    valid = centred_maxima > 0.0
    unit = np.zeros_like(values)
    if np.any(valid):
        stable = centred[:, valid] / centred_maxima[valid][None, :]
        norms = np.linalg.norm(stable, axis=0)
        numerically_nonzero = norms > 0.0
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices[~numerically_nonzero]] = False
        if np.any(numerically_nonzero):
            kept_indices = valid_indices[numerically_nonzero]
            unit[:, kept_indices] = (
                stable[:, numerically_nonzero] / norms[numerically_nonzero][None, :]
            )
    return unit, valid


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    """Column-orthonormal basis invariant to every finite nonzero column rescaling."""
    unit, valid = _centred_unit_columns(matrix)
    if not np.any(valid):
        return np.zeros((unit.shape[0], 0), dtype=np.float64)
    u, singular, _ = np.linalg.svd(unit[:, valid], full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros((unit.shape[0], 0), dtype=np.float64)
    tolerance = singular[0] * max(unit.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(singular > tolerance))
    return u[:, :rank]


def subspace_accuracy(sub_scores: np.ndarray, ref_scores: np.ndarray,
                      ref_weights: np.ndarray) -> dict:
    """Equal-weight geometric agreement over empirically structured reference PCs.

    Returns a dict with the normalised ``accuracy`` plus diagnostics (per-PC agreement,
    raw score, chance floor).
    """
    n = ref_scores.shape[0]
    if sub_scores.shape[0] != n:
        raise ValueError(f"sample count mismatch: sub {sub_scores.shape[0]} vs ref {n}")

    # Use ONLY the submission's leading k columns (k = number of reference PCs). Otherwise a
    # cheat could emit a rank-(n-1) block of random PCs whose span contains every reference
    # direction and score ~1 without ever computing a PCA. Capping the subspace at k also keeps
    # the chance floor sqrt(k/(n-1)) well below 1, so the normalisation is never singular.
    k_ref = ref_scores.shape[1]
    sub_scores = np.asarray(sub_scores)[:, :k_ref]

    Qsub = _orthonormal_basis(sub_scores)
    k_sub = Qsub.shape[1]
    if k_sub == 0:
        return {"accuracy": 0.0, "raw": 0.0, "chance": 0.0, "per_pc": [],
                "reason": "submission subspace is empty/degenerate"}

    Rc = ref_scores - ref_scores.mean(axis=0, keepdims=True)
    rnorms = np.linalg.norm(Rc, axis=0)
    valid = rnorms > 1e-12
    Rc = Rc[:, valid] / rnorms[valid][None, :]         # unit reference directions

    # a_c = cosine of principal angle between reference direction c and submission subspace.
    proj = Qsub.T @ Rc                                  # (k_sub, n_ref)
    a = np.linalg.norm(proj, axis=0)                    # (n_ref,)
    a = np.clip(a, 0, 1)

    # The full-spectrum upper-edge fit gives noise PCs exactly zero weight. Structured axes are
    # then treated equally so a dominant PC1 cannot mask a dropped weak-but-real ancestry axis.
    weights = np.asarray(ref_weights, dtype=float)
    if weights.shape != (ref_scores.shape[1],):
        # A genuine grader-internal contract violation (weights are computed from the same
        # reference the scores come from), not submission data -- fail loudly rather than mask it.
        raise ValueError("reference weights must match the reported reference PCs")
    structured = weights[valid] > 0
    if not structured.any():
        raise ValueError("reference PCA has no structured PCs in the requested range")

    a_struct = a[structured]
    # Equal-weight geometric mean of the structured agreements. Clip away from 0 only to keep
    # the log finite; a genuinely dropped axis (a_c ~ 0) still drags the product toward 0.
    logs = np.log(np.clip(a_struct, 1e-6, 1.0))
    raw = float(np.exp(logs.mean()))

    chance = _random_projection_mean(min(k_sub, n - 1), n - 1)
    if chance == 1.0:
        accuracy = float(raw >= 1.0 - 1e-10)
    else:
        accuracy = float(np.clip((raw - chance) / (1 - chance), 0.0, 1.0))
    return {"accuracy": accuracy, "raw": raw, "chance": chance,
            "per_pc": [round(float(x), 4) for x in a],
            "structured": [bool(x) for x in structured]}
