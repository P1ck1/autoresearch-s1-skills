"""Population-structure recovery metric for datasets with known discrete ancestry labels.

Independent of the subspace-vs-reference accuracy: this asks the biology question directly --
does the submission's PCA contain the known population contrasts? We summarise that three ways:

* ``label_projection`` -- fraction of the centred one-hot label subspace captured by the score
  subspace. This is the scored quantity. It is invariant to every nonsingular rotation, sign, or
  rescaling of the score columns and cannot be harmed by additional orthogonal score directions;

* ``ncc_accuracy`` -- leave-one-out nearest-centroid classification accuracy in the top-few PC
  space (how well the PCs alone recover the labels);
* ``mean_eta2`` -- mean one-way ANOVA eta^2 of the labels across the leading PCs (fraction of
  PC variance explained by population membership).

For real 1000 Genomes data the super-populations are strongly separable, so a genuine PCA
scores high; a degenerate or wrong-object fit does not. The grader compares the submission's
recovery to the full-scan reference's recovery so the check is "within tolerance of what the
data actually supports", never an absolute threshold the data may not meet.
"""
from __future__ import annotations

import numpy as np

from .subspace import _orthonormal_basis


def label_projection(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the fraction of identifiable label contrast energy in ``span(scores)``."""
    labels = np.asarray(labels)
    valid = labels >= 0
    score_basis = _orthonormal_basis(np.asarray(scores)[valid])
    classes = np.unique(labels[valid])
    if score_basis.shape[1] == 0 or classes.size < 2:
        return 0.0
    one_hot = np.column_stack([(labels[valid] == value).astype(float) for value in classes])
    label_basis = _orthonormal_basis(one_hot)
    if label_basis.shape[1] == 0:
        return 0.0
    captured = np.linalg.norm(score_basis.T @ label_basis, ord="fro") ** 2
    return float(np.clip(captured / label_basis.shape[1], 0.0, 1.0))


def within_group_label_projection(
    scores: np.ndarray,
    group_labels: np.ndarray,
    nested_labels: np.ndarray,
) -> float:
    """Recover nested contrasts after conditioning on each parent group."""
    scores = np.asarray(scores)
    group_labels = np.asarray(group_labels)
    nested_labels = np.asarray(nested_labels)
    if scores.shape[0] != group_labels.size or group_labels.shape != nested_labels.shape:
        raise ValueError("hierarchical labels must align with score rows")
    captured = 0.0
    total_rank = 0
    for group in np.unique(group_labels[group_labels >= 0]):
        members = (group_labels == group) & (nested_labels >= 0)
        contrast_rank = max(0, np.unique(nested_labels[members]).size - 1)
        if contrast_rank == 0:
            continue
        captured += contrast_rank * label_projection(scores[members], nested_labels[members])
        total_rank += contrast_rank
    if total_rank == 0:
        return 0.0
    return float(np.clip(captured / total_rank, 0.0, 1.0))


def population_separation(scores: np.ndarray, labels: np.ndarray, n_lead: int = 6) -> dict:
    labels = np.asarray(labels)
    valid = labels >= 0
    S = scores[valid]
    y = labels[valid]
    if S.shape[0] == 0 or len(np.unique(y)) < 2:
        return {"label_projection": 0.0, "ncc_accuracy": 0.0, "mean_eta2": 0.0,
                "n": int(valid.sum())}
    # Use an orthonormal representation of the complete submitted subspace for diagnostics too.
    # This removes arbitrary output-column scale and rotation from the nearest-centroid distances.
    basis = _orthonormal_basis(S)
    if basis.shape[1] == 0:
        return {"label_projection": 0.0, "ncc_accuracy": 0.0, "mean_eta2": 0.0,
                "n": int(valid.sum()), "n_classes": int(len(np.unique(y)))}
    kk = min(n_lead, basis.shape[1])
    Sx = basis[:, :kk]
    classes = np.unique(y)

    # Leave-one-out nearest-centroid accuracy (cheap closed form: recompute centroid without i).
    correct = 0
    counts = {c: (y == c).sum() for c in classes}
    sums = {c: Sx[y == c].sum(axis=0) for c in classes}
    for i in range(Sx.shape[0]):
        ci = y[i]
        best_c, best_d = None, np.inf
        for c in classes:
            if c == ci:
                if counts[c] <= 1:
                    cent = sums[c]
                else:
                    cent = (sums[c] - Sx[i]) / (counts[c] - 1)
            else:
                cent = sums[c] / counts[c]
            d = np.sum((Sx[i] - cent) ** 2)
            if d < best_d:
                best_d, best_c = d, c
        correct += int(best_c == ci)
    ncc = correct / Sx.shape[0]

    # mean eta^2 across leading PCs.
    eta2s = []
    for c in range(kk):
        x = Sx[:, c]
        gm = x.mean()
        ssb = sum((y == cl).sum() * (x[y == cl].mean() - gm) ** 2 for cl in classes)
        sst = ((x - gm) ** 2).sum()
        eta2s.append(ssb / sst if sst > 0 else 0.0)
    return {"label_projection": round(label_projection(S, y), 6),
            "ncc_accuracy": round(float(ncc), 4), "mean_eta2": round(float(np.mean(eta2s)), 4),
            "n": int(valid.sum()), "n_classes": int(len(classes))}
