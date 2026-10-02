"""Per-principal-component accuracy metric -- DIAGNOSTIC ONLY, never a reward term.

The aggregate subspace metric (``subspace.subspace_accuracy``) asks a *set* question: do the
submission's scores, taken together, span the same directions over samples as the reference?
That is the right question for the reward, and its equal-weight geometric mean over structured
PCs already collapses toward 0 when any single structured axis is genuinely *dropped* from the
span. What that aggregate deliberately does NOT distinguish -- and must not, on pain of a false
negative -- is whether a recovered reference direction lands on the submission's matching column
or is merely present somewhere in its span after a within-subspace rotation. On real 1000G chr22
we saw submissions whose per-PC *strict* agreement decayed with PC index

    per-PC |corr| vs anchor:  PC1 0.998  PC2 0.994  PC3 0.975  PC4 0.946  PC5 0.878  PC6 0.086

This module measures that decay directly, PC by PC, as an auditing signal for "how deep, axis by
axis, did the submission actually go". It is emitted into ``accuracy_detail`` for inspection and
is intentionally NOT wired into the score: the strict per-axis measure penalises legitimate
rotations of a near-degenerate plane (a valid PCA basis), so grading on it would deny credit to a
scientifically-correct submission. Keep it as a diagnostic; do not feed it to the reward.

Two complementary agreement definitions are provided, because a PCA is only defined up to
rotation/reflection/scaling within a degenerate (equal-eigenvalue) subspace, and the two
definitions disagree in exactly the informative way:

* ``per_pc_agreement`` -- the *strict* one. For each reference PC k it is
      a_k = | corr( centred ref PC_k , centred sub PC_k ) |
  the absolute Pearson correlation between the reference's k-th score direction and the
  submission's k-th score direction, over samples. Sign-robust (PCA sign is arbitrary) but
  NOT rotation-robust: if the submission resolves the same 2-D plane as ref PC4/PC5 but with
  the axes rotated 45 degrees, this reports ~0.7 on each, correctly flagging that the
  submission did not recover *those individual axes*. This is the measure that exposes the
  higher-PC degradation above, and the one ``resolved_pc_count`` uses.

* ``per_pc_subspace_agreement`` -- the *rotation-robust* one. For each reference PC k it is
      a_k = || Q_sub[:, :k]^T r_k ||   in [0, 1]
  the cosine of the principal angle between reference direction r_k and the span of the
  submission's leading k score columns (Q_sub is a column-orthonormal basis of those columns,
  centred). A within-subspace rotation among the first k submission PCs leaves that span
  unchanged, so it is not penalised. This mirrors ``subspace.subspace_accuracy`` restricted to
  the nested leading-k blocks, and answers "is ref direction k already captured by the
  submission's top-k axes?" rather than "did column k land on it exactly?".

Use the strict measure to detect per-axis mis-estimation; use the subspace measure when a
within-subspace rotation should be forgiven (e.g. near-degenerate eigenvalues where the split
of a plane into two axes is genuinely arbitrary).
"""

from __future__ import annotations

import numpy as np

from .subspace import _centred_unit_columns, _orthonormal_basis, structure_weights


def _prepare(sub_scores: np.ndarray, ref_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate sample counts and clip/pad the submission to the reference width k = ref cols.

    We compare a *fixed* k = number of reference PCs. If the submission has more columns we
    keep only the leading k -- same anti-cheat reasoning as ``subspace.subspace_accuracy``: a
    rank-(n-1) block of random columns would otherwise let a cheat contain every reference
    direction without ever computing a PCA. If it has fewer, the missing PCs are padded with
    zero columns, which the centring/degeneracy guards below score as agreement 0.
    """
    ref_scores = np.asarray(ref_scores, dtype=float)
    sub_scores = np.asarray(sub_scores, dtype=float)
    n = ref_scores.shape[0]
    if sub_scores.shape[0] != n:
        raise ValueError(f"sample count mismatch: sub {sub_scores.shape[0]} vs ref {n}")
    k_ref = ref_scores.shape[1]
    if sub_scores.shape[1] >= k_ref:
        sub_scores = sub_scores[:, :k_ref]
    else:
        pad = np.zeros((n, k_ref - sub_scores.shape[1]), dtype=float)
        sub_scores = np.concatenate([sub_scores, pad], axis=1)
    return sub_scores, ref_scores


def per_pc_agreement(sub_scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """Strict per-PC agreement: |corr(centred ref PC_k, centred sub PC_k)| for each ref PC k.

    Returns an array of length k = ref width in [0, 1]. Sign-robust (absolute value absorbs the
    arbitrary eigenvector sign) but deliberately *not* rotation-robust -- it scores whether the
    submission's k-th axis lands on the reference's k-th axis. A degenerate/zero column on
    either side (no variance over samples) yields 0 for that PC rather than a divide-by-zero.
    """
    sub_scores, ref_scores = _prepare(sub_scores, ref_scores)
    Rc, ref_valid = _centred_unit_columns(ref_scores)
    Sc, sub_valid = _centred_unit_columns(sub_scores)
    a = np.abs(np.sum(Rc * Sc, axis=0))
    a[~(ref_valid & sub_valid)] = 0.0
    return np.clip(a, 0.0, 1.0)


def per_pc_subspace_agreement(sub_scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """Rotation-robust per-PC agreement: ||Q_sub[:, :k]^T r_k|| for each ref PC k.

    ``r_k`` is the centred, unit-normalised reference direction for PC k; ``Q_sub[:, :k]`` is a
    column-orthonormal basis of the submission's leading k score columns (centred, with
    degenerate columns dropped exactly as in ``subspace._orthonormal_basis``). The result, the
    cosine of the principal angle between r_k and that leading-k span, lies in [0, 1] and is
    unchanged by any rotation/reflection *within* the submission's top-k subspace -- so a
    faithful PCA that merely splits a near-degenerate plane differently is not penalised.

    The submission is capped at k = ref width overall (via ``_prepare``); for reference PC k
    only the first k of those columns are used, so the leading-k spans are nested exactly as
    the reference PCs are ordered.
    """
    sub_scores, ref_scores = _prepare(sub_scores, ref_scores)
    k_ref = ref_scores.shape[1]
    Rc, ref_valid = _centred_unit_columns(ref_scores)
    a = np.zeros(k_ref, dtype=float)
    for k in range(1, k_ref + 1):
        if not ref_valid[k - 1]:                # reference PC k has no variance -> undefined -> 0
            continue
        r = Rc[:, k - 1]
        Q = _orthonormal_basis(sub_scores[:, :k])
        if Q.shape[1] == 0:
            continue
        a[k - 1] = float(np.linalg.norm(Q.T @ r))
    return np.clip(a, 0.0, 1.0)


def _structured_mask(k_ref: int, ref_evals: np.ndarray) -> np.ndarray:
    """Boolean mask (length k_ref) of which reference PCs sit above the noise floor.

    The noise edge is fit on the complete truth spectrum, then restricted to the reported PCs.
    """
    w = structure_weights(np.asarray(ref_evals, dtype=float), k_ref)[:k_ref]
    mask = np.zeros(k_ref, dtype=bool)
    mask[: w.shape[0]] = w > 0
    return mask


def resolved_pc_count(sub_scores: np.ndarray, ref_scores: np.ndarray,
                      ref_evals: np.ndarray, threshold: float = 0.9) -> int:
    """Number of *structured* reference PCs the submission recovers at agreement >= threshold.

    Recovery is judged by the strict ``per_pc_agreement`` (per-axis, not per-subspace): a PC is
    resolved only when the submission's own k-th axis correlates with the reference's k-th axis
    at least ``threshold``. If ``ref_evals`` is supplied, only PCs above the noise floor (see
    ``_structured_mask`` / ``subspace.structure_weights``) are eligible -- grading a submission
    on reference noise axes, which even a re-run of the anchor could not reproduce, would be
    unfair. This is the headline number for "how deep does the submission's PCA actually go".
    """
    a = per_pc_agreement(sub_scores, ref_scores)
    mask = _structured_mask(ref_scores.shape[1], ref_evals)
    return int(np.sum((a >= threshold) & mask))


def per_pc_report(sub_scores: np.ndarray, ref_scores: np.ndarray,
                  ref_evals: np.ndarray) -> dict:
    """Diagnostic bundle for logging: both per-PC agreement vectors plus resolution summary.

    Returns:
      ``per_pc``            -- strict |corr| per reference PC.
      ``per_pc_subspace``   -- rotation-robust leading-k subspace agreement per reference PC.
      ``resolved``          -- ``resolved_pc_count`` at the default threshold.
      ``n_structured``      -- number of reference PCs above the noise floor (all, if no evals).
      ``weakest_structured_pc`` -- (index, value) of the structured PC with the lowest strict
                               agreement -- the first place the submission's coverage breaks
                               down. ``None`` when there are no structured PCs.
    """
    a = per_pc_agreement(sub_scores, ref_scores)
    a_sub = per_pc_subspace_agreement(sub_scores, ref_scores)
    mask = _structured_mask(ref_scores.shape[1], ref_evals)
    resolved = int(np.sum((a >= 0.9) & mask))
    n_structured = int(mask.sum())
    if n_structured > 0:
        struct_idx = np.flatnonzero(mask)
        worst = struct_idx[int(np.argmin(a[mask]))]
        weakest = (int(worst), float(a[worst]))
    else:
        weakest = None
    return {
        "per_pc": [round(float(x), 4) for x in a],
        "per_pc_subspace": [round(float(x), 4) for x in a_sub],
        "resolved": resolved,
        "n_structured": n_structured,
        "weakest_structured_pc": weakest,
    }
