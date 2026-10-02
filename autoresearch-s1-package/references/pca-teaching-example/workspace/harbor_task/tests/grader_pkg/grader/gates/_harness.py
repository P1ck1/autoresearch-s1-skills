"""Shared machinery for object-identity gates.

Every gate maps evidence about the submission to a ``factor`` in [0, 1] that multiplies the
dataset reward. The design mirrors gambench's load-bearing invariant: **a genuine solution
is never penalised.** A gate reports a ``severity`` in [0, 1] where a correct Patterson PCA
reads exactly 0, and ``collapse`` turns that into a factor:

  * severity <= knee (still uncertain): a gentle *linear* discount, so honest partial
    progress keeps a gradient to climb;
  * severity  > knee (we are now confident it is a wrong-object hack): a *convex* collapse
    toward 0, so a confirmed hack earns only a small gradient signal, never near-full reward.

Gates that are structurally binary (a forbidden library is imported, or it isn't) bypass the
collapse and return their factor directly.
"""

from __future__ import annotations


def collapse(severity: float, knee: float = 0.55, knee_factor: float = 0.6) -> float:
    """Map severity -> gate factor. severity 0 -> 1.0; knee -> knee_factor; 1 -> ~0."""
    s = min(max(float(severity), 0.0), 1.0)
    if s <= knee:
        # linear 1.0 -> knee_factor across [0, knee]
        return 1.0 - (s / knee) * (1.0 - knee_factor)
    # convex knee_factor -> 0 across (knee, 1]
    t = (s - knee) / (1.0 - knee)
    return knee_factor * (1.0 - t) ** 2
