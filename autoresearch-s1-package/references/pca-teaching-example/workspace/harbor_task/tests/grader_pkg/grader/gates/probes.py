"""Object-identity gates driven by probe datasets.

A gate feeds the submission a scored-scale VCF with known, discriminating structure, then
inspects the returned scores. Each probe reuses a complete invocation and serialization profile
loaded from an active scored sidecar: sample and record counts, requested rank, FORMAT-width
family, opaque IDs, chromosome layout, and newline policy. Mathematical draws are fixed for the
benchmark release so repeat score variance comes from the solver, not the eval.

Three gates live here:

* ``hwe_norm`` -- is the fit an *HWE-normalised* (Patterson) PCA, or a plain covariance PCA?
  The probe makes those objects disagree at the requested top-k subspace boundary. Severity is
  based on scale/rotation-invariant capture of the two competing directions.

* ``coverage`` -- did the fit use enough effective markers? The probe has subtle structure that
  needs thousands of variants to resolve. A wide-scan (even a subsampled one) resolves it; a
  hard-capped "only read 200 SNPs" hack cannot. Severity is how poorly the submission recovers
  the subtle axes that adequate coverage makes recoverable.

* ``representation_equivalence`` -- does the fit recover the same scientific object from two
  equivalent, scored-like mixed-width VCF serialisations? The pair independently permutes samples
  and records, allele polarity, case, phase, haploid spellings, missing-call spellings, and INFO /
  FORMAT widths. Each result is compared with the exact Patterson reference after restoring
  canonical sample order. Severity measures both the agreement gap and absolute fidelity, with
  dead-zones for honest approximation variability.

All gates read severity 0 for a genuine fit (dead-zoned) so the load-bearing invariant --
a correct PCA is never rejected -- holds.
"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Literal, NamedTuple, Sequence

import numpy as np

from grader.gates._harness import collapse
from grader.metrics.subspace import subspace_accuracy, structure_weights
from data.vcf_serialization import (
    CANONICAL_MISSING_GT_ENCODINGS,
    ONE_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS,
    PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS,
    ZERO_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS,
    randomized_positions,
    variable_gt_fields,
    variable_info,
)
from reference.pca_core import gram_to_scores, standardize_into_gram


# ---------------------------------------------------------------------------
# Probe genotype construction + VCF writing
# ---------------------------------------------------------------------------

_GT = np.array([b"0/0", b"0/1", b"1/1"], dtype="S3")
_BASES = np.array([b"A", b"C", b"G", b"T"], dtype="S1")

ProbeLayout = Literal[
    "synthetic_plain", "synthetic_messy", "synthetic_variable_width", "synthetic_haploid",
    "observed",
]
_PROBE_LAYOUTS = frozenset({
    "synthetic_plain", "synthetic_messy", "synthetic_variable_width", "synthetic_haploid",
    "observed",
})


class ProbeCarrier(NamedTuple):
    """One complete invocation and byte-layout profile copied from a scored dataset.

    Sample count or record count alone is not sufficient camouflage: a shortcut can fingerprint
    the complete ``(n_samples, n_records, k)`` tuple before choosing whether to compute the
    requested object. Matching only that tuple is also insufficient when the matched scored file
    has a different FORMAT/layout family. Carriers are therefore constructed from loaded private
    truth sidecars and include the reviewed serialization family.
    """

    n_samples: int
    n_records: int
    k: int
    layout: ProbeLayout


# Odd counts >= 3 so np.median genuinely rejects an outlier draw rather than averaging two (or, at
# 1, doing nothing). Each rep runs the submission once on a small synthetic probe VCF, so the added
# cost is a few seconds per submission; the payoff is that one unlucky release draw can no longer
# pull a genuine fit's method factor off 1.0 (the gate is anchored to per-run references, but the
# median over 3 makes the seed-robustness claim real -- especially for representation, which had no
# cross-draw smoothing at all).
_PROBE_REPETITIONS = {
    "hwe_norm": 3,
    "coverage": 3,
    "representation_equivalence": 3,
}
_MAX_DENSE_PROBE_SAMPLES = 1024


def _validated_carriers(carriers: Sequence[ProbeCarrier]) -> tuple[ProbeCarrier, ...]:
    validated: list[ProbeCarrier] = []
    for value in carriers:
        try:
            carrier = ProbeCarrier(*value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "probe carriers must include n_samples, n_records, k, and layout"
            ) \
                from error
        integer_fields = carrier[:3]
        if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
               for item in integer_fields):
            raise ValueError("probe carrier fields must be integers")
        if not isinstance(carrier.layout, str) or carrier.layout not in _PROBE_LAYOUTS:
            raise ValueError(f"unknown probe carrier layout: {carrier.layout!r}")
        carrier = ProbeCarrier(*(int(item) for item in integer_fields), carrier.layout)
        if (carrier.n_samples < 2 or carrier.n_records < 1
                or carrier.k < 1 or carrier.k >= carrier.n_samples):
            raise ValueError(f"invalid probe carrier: {carrier}")
        validated.append(carrier)
    if not validated:
        raise ValueError("at least one active scored carrier is required")
    return tuple(validated)


def _walsh_carrier_compatible(carrier: ProbeCarrier) -> bool:
    count = carrier.k + 1
    period = 1 << max(1, count.bit_length())
    return count < carrier.n_samples and carrier.n_samples % period == 0


def _carrier_rank(math_key: bytes, domain: str, carrier: ProbeCarrier) -> bytes:
    message = (f"probe-carrier-v1\0{domain}\0{carrier.n_samples}\0"
               f"{carrier.n_records}\0{carrier.k}\0{carrier.layout}").encode()
    return hmac.digest(math_key, message, "sha256")


def select_probe_carriers(
    active_carriers: Sequence[ProbeCarrier], math_key: bytes,
) -> dict[str, tuple[ProbeCarrier, ...]]:
    """Choose release-stable carriers from the suite that is actually being scored.

    Every chosen probe exactly reuses a complete active invocation signature, including that
    dataset's requested rank. Selection is private-keyed and preferentially spends unused ranks,
    so one run exercises multiple ``k`` values instead of advertising every probe through a
    single rank. The dense probe oracle is intentionally bounded; the sample-heavy scored arm is
    still graded normally but is not duplicated as hidden probe overhead.
    """
    if len(math_key) != 32:
        raise ValueError("math_key must contain exactly 32 bytes")
    carriers = tuple(dict.fromkeys(_validated_carriers(active_carriers)))
    ordinary = tuple(
        carrier for carrier in carriers
        if carrier.n_samples <= _MAX_DENSE_PROBE_SAMPLES
    )
    if not ordinary:
        raise ValueError("active suite has no carrier suitable for dense probe scoring")
    plain = tuple(carrier for carrier in ordinary if carrier.layout == "synthetic_plain")
    variable = tuple(
        carrier for carrier in ordinary if carrier.layout == "synthetic_variable_width"
    )
    pools = {
        "hwe_norm": tuple(filter(_walsh_carrier_compatible, plain)),
        "coverage": plain,
        "representation_equivalence": variable,
    }
    if not pools["hwe_norm"]:
        raise ValueError("active suite has no Walsh-compatible HWE probe carrier")
    if not pools["coverage"]:
        raise ValueError("active suite has no plain synthetic coverage probe carrier")
    if not pools["representation_equivalence"]:
        raise ValueError("active suite has no variable-width representation probe carrier")

    used_ranks: set[int] = set()
    selected: dict[str, tuple[ProbeCarrier, ...]] = {}
    for gate, count in _PROBE_REPETITIONS.items():
        chosen: list[ProbeCarrier] = []
        for slot in range(count):
            available = [carrier for carrier in pools[gate] if carrier not in chosen]
            if not available:
                available = list(pools[gate])
            unseen_rank = [carrier for carrier in available if carrier.k not in used_ranks]
            if unseen_rank:
                available = unseen_rank
            domain = f"{gate}/{slot}"
            choice = min(available, key=lambda carrier: _carrier_rank(math_key, domain, carrier))
            chosen.append(choice)
            used_ranks.add(choice.k)
        selected[gate] = tuple(chosen)

    attainable = min(
        len({carrier.k for carrier in ordinary}),
        sum(_PROBE_REPETITIONS.values()),
    )
    required_distinct = min(3, attainable)
    if len(used_ranks) < required_distinct:
        raise RuntimeError("private carrier selection did not span enough active ranks")
    return selected


def _write_common_header(fh, sample_ids: list[str], line_ending: bytes = b"\n") -> None:
    """Use the same generic metadata envelope as hidden scored synthetic VCFs."""
    lines = [
        b"##fileformat=VCFv4.2",
        b"##source=pcabench_generate",
        b'##FILTER=<ID=PASS,Description="All filters passed">',
        b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        b'##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
        b'##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
        b'##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
        b'##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">',
        b'##INFO=<ID=ZZ,Number=1,Type=String,Description="Opaque width padding">',
        b'##ALT=<ID=DEL,Description="Deletion">',
    ]
    lines.extend(f"##contig=<ID=chr{chromosome}>".encode() for chromosome in range(1, 23))
    fixed = [b"#CHROM", b"POS", b"ID", b"REF", b"ALT", b"QUAL", b"FILTER",
             b"INFO", b"FORMAT"]
    lines.append(b"\t".join(fixed + [sample.encode() for sample in sample_ids]))
    fh.write(line_ending.join(lines) + line_ending)


def _ordered_layout(order: np.ndarray, rng) -> list[tuple[int, int, int]]:
    """Assign a row permutation the scored synthetic chromosome/position distribution."""
    chromosome, position = randomized_positions(order.size, 22, rng)
    return [
        (int(row), int(chrom), int(pos))
        for row, chrom, pos in zip(order, chromosome, position, strict=True)
    ]


def _write_plain_probe_vcf(
    path: Path, G: np.ndarray, sample_ids: list[str], rng, carrier: ProbeCarrier,
) -> None:
    """Write a carrier-matched plain synthetic VCF with no probe-only FORMAT switch."""
    n, m = G.shape
    if carrier.layout != "synthetic_plain":
        raise ValueError("plain probe writer requires a synthetic_plain carrier")
    if (n, m) != (carrier.n_records, carrier.n_samples) or len(sample_ids) != m:
        raise ValueError("plain probe matrix does not match its active carrier")
    with open(path, "wb") as fh:
        _write_common_header(fh, sample_ids)
        bases = [b"A", b"C", b"G", b"T"]
        lines = []
        for r, chromosome, pos in _ordered_layout(rng.permutation(n), rng):
            ref = bases[rng.integers(4)]
            alt = bases[(bases.index(ref) + 1 + rng.integers(3)) % 4]
            row = _GT[G[r]].tolist()
            gt = b"\t".join(row)
            fixed = b"\t".join([f"chr{chromosome}".encode(), str(pos).encode(), b".",
                                ref, alt, b".", b"PASS", b".", b"GT"])
            lines.append(fixed + b"\t" + gt)
            if len(lines) >= 5000:
                fh.write(b"\n".join(lines) + b"\n")
                lines = []
        if lines:
            fh.write(b"\n".join(lines) + b"\n")


def _rand_sample_ids(m: int, rng) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    while len(sample_ids) < m:
        candidate = "p" + rng.bytes(10).hex()
        if candidate not in seen:
            seen.add(candidate)
            sample_ids.append(candidate)
    return sample_ids


def _keyed_representation(
    G: np.ndarray, representation_key: bytes, domain: str,
) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Apply only exact Patterson-PCA symmetries under a runtime-private key.

    The mathematical draw and record order remain release-stable.  A fresh sample permutation,
    aliases, and per-marker allele polarity make public byte fixtures and precomputed score rows
    unusable without changing the Gram spectrum or an honest implementation's difficulty.
    """
    if len(representation_key) != 32:
        raise ValueError("representation_key must contain exactly 32 bytes")
    seed = int.from_bytes(
        hmac.digest(representation_key, b"probe-v4\0" + domain.encode(), "sha256")[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    sample_order = rng.permutation(G.shape[1])
    keyed = G[:, sample_order].copy()
    polarity_flip = rng.random(G.shape[0]) < 0.5
    flipped = keyed[polarity_flip]
    keyed[polarity_flip] = np.where(flipped >= 0, 2 - flipped, flipped)
    return keyed, sample_order, rng


def _patterson_pca(G: np.ndarray, k: int):
    n_samp = G.shape[1]
    gram = np.zeros((n_samp, n_samp))
    counter = {"kept": 0}
    standardize_into_gram(G.astype(np.int8), gram, counter)
    scores, _, spectrum = gram_to_scores(gram, counter["kept"], k)
    return scores, spectrum


def _rawcov_scores(G: np.ndarray, k: int):
    X = G.astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    Gr = (X.T @ X) / max(X.shape[0], 1)
    ev, V = np.linalg.eigh(Gr)
    idx = np.argsort(ev)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(ev[idx], 0, None))[None, :]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _axis_capture(S: np.ndarray, axis: np.ndarray) -> float:
    """Fraction of one direction captured by the submitted score subspace.

    The QR basis makes this invariant to every nonsingular rotation, sign change, or rescaling of
    the emitted columns, exactly matching the score-subspace output contract.
    """
    u = _unit(axis - axis.mean())
    Sc = S - S.mean(axis=0, keepdims=True)
    basis, triangular = np.linalg.qr(Sc, mode="reduced")
    if triangular.size == 0:
        return 0.0
    diagonal = np.abs(np.diag(triangular))
    tolerance = diagonal.max(initial=0.0) * max(Sc.shape) * np.finfo(float).eps
    rank = int(np.sum(diagonal > tolerance))
    if rank == 0:
        return 0.0
    return float(np.clip(np.sum((basis[:, :rank].T @ u) ** 2), 0.0, 1.0))


_PROBE_TOKEN = "abcdefghijkmnpqrstuvwxyz0123456789"


def _probe_paths(workdir: Path, rng) -> tuple[Path, Path]:
    """Opaque random file stem so a submission cannot recognise a probe from its name."""
    tok = "".join(rng.choice(list(_PROBE_TOKEN), size=12))
    return workdir / f"{tok}.vcf", workdir / f"{tok}.scores.tsv"


def _write_representation_pair(path_a: Path, path_b: Path, G: np.ndarray,
                               sample_ids: list[str], rng, carrier: ProbeCarrier,
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Write two scientifically equivalent VCF serialisations of ``G``.

    Returns the canonical-sample indices represented by the row order of each VCF. The caller
    uses these permutations to restore score rows before comparing them with the common exact
    reference. Both records and samples are independently shuffled. Both arms use the mixed
    variable-width distribution of their matched scored carrier while independently choosing
    valid spellings that must not alter Patterson PCA.
    """
    n, m = G.shape
    if len(sample_ids) != m:
        raise ValueError("sample_ids must match genotype columns")
    if n < 2 or m < 2 or np.any((G < -1) | (G > 2)):
        raise ValueError("representation probe requires dosages in {-1,0,1,2}")
    if carrier.layout != "synthetic_variable_width":
        raise ValueError("representation writer requires a synthetic_variable_width carrier")
    if (n, m) != (carrier.n_records, carrier.n_samples):
        raise ValueError("representation matrix does not match its active carrier")

    sample_order_a = rng.permutation(m)
    sample_order_b = rng.permutation(m)
    variant_order_a = rng.permutation(n)
    variant_order_b = rng.permutation(n)

    # This balanced sample contrast is drawn from the runtime-private representation RNG in
    # production. It carries NO genotype information: every affected cell is semantically missing.
    # Its only purpose is to make a parser that reads the first byte of a malformed token invent a
    # coherent 0-vs-2 sample axis instead of harmless isotropic noise. A whole-token parser ignores
    # every malformed spelling, so both arms remain exactly the same Patterson object.
    malformed_contrast = np.zeros(m, dtype=np.bool_)
    malformed_contrast[rng.permutation(m)[:m // 2]] = True

    def write_one(path: Path, sample_order: np.ndarray, variant_order: np.ndarray,
                  missing_encodings: tuple[bytes, ...], *,
                  prefix_contrast: np.ndarray | None) -> None:
        ref_idx = rng.integers(0, 4, size=n)
        alt_idx = (ref_idx + rng.integers(1, 4, size=n)) % 4
        flip = rng.random(n) < 0.5
        flip[0], flip[1] = True, False
        with open(path, "wb") as fh:
            _write_common_header(
                fh, [sample_ids[i] for i in sample_order], line_ending=b"\r\n")
            lines: list[bytes] = []
            for original, chromosome, position in _ordered_layout(variant_order, rng):
                dosages = G[original]
                if flip[original]:
                    dosages = np.where(dosages >= 0, 2 - dosages, -1)
                    ri, ai = alt_idx[original], ref_idx[original]
                else:
                    ri, ai = ref_idx[original], alt_idx[original]
                ref, alt = _BASES[ri], _BASES[ai]
                if rng.random() < 0.5:
                    ref = ref.lower()
                if rng.random() < 0.5:
                    alt = alt.lower()
                ordered = dosages[sample_order]
                missing = ordered < 0
                missing_encoding_indices = None
                if prefix_contrast is not None:
                    ordered_contrast = prefix_contrast[sample_order]
                    missing_encoding_indices = np.empty(m, dtype=np.int8)
                    zero = ~ordered_contrast
                    missing_encoding_indices[zero] = rng.integers(
                        0,
                        len(ZERO_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS),
                        size=int(np.sum(zero)),
                    )
                    missing_encoding_indices[ordered_contrast] = (
                        len(ZERO_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS)
                        + rng.integers(
                            0,
                            len(ONE_PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS),
                            size=int(np.sum(ordered_contrast)),
                        )
                    )
                fmt, calls = variable_gt_fields(
                    np.clip(ordered, 0, 2), missing, int(original % 4), rng,
                    missing_encodings, missing_encoding_indices,
                )
                fixed = [
                    f"chr{chromosome}".encode(), str(position).encode(), b".", ref, alt,
                    b".", b"PASS", variable_info(rng), fmt,
                ]
                lines.append(b"\t".join(fixed) + b"\t" + calls)
                if len(lines) >= 2000:
                    fh.write(b"\r\n".join(lines) + b"\r\n")
                    lines.clear()
            if lines:
                fh.write(b"\r\n".join(lines) + b"\r\n")
            fh.flush()
            fh.seek(-2, 2)
            fh.truncate()

    # The arms carry the same mathematical missing mask. One uses conventional missing syntax;
    # the other uses only malformed/out-of-range spellings that the task contract also defines as
    # missing. A parser that accepts a malformed prefix as a haploid call therefore disagrees with
    # itself across two scientifically identical inputs instead of making the same diluted error
    # in both arms and hiding inside the representation dead-zone.
    write_one(
        path_a, sample_order_a, variant_order_a, CANONICAL_MISSING_GT_ENCODINGS,
        prefix_contrast=None,
    )
    write_one(
        path_b, sample_order_b, variant_order_b, PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS,
        prefix_contrast=malformed_contrast,
    )

    return sample_order_a, sample_order_b


# ---------------------------------------------------------------------------
# hwe_norm gate
# ---------------------------------------------------------------------------

def _walsh_axes(m: int, count: int, rng) -> np.ndarray:
    """Return shuffled, balanced, exactly orthogonal +/-1 sample contrasts."""
    period = 1 << max(1, count.bit_length())
    if count >= m or m % period:
        raise ValueError(
            "Walsh design requires sample count divisible by its contrast period"
        )
    rows = np.arange(m, dtype=np.uint32)
    axes = np.empty((count, m), dtype=np.int8)
    for column in range(1, count + 1):
        values = rows & np.uint32(column)
        parity = np.zeros(m, dtype=np.uint8)
        while np.any(values):
            parity ^= (values & 1).astype(np.uint8)
            values >>= 1
        axes[column - 1] = np.where(parity == 0, -1, 1)
    return axes[:, rng.permutation(m)]


def _draw_axis_markers(rng, axis: np.ndarray, count: int,
                       low: float, high: float, delta: float) -> np.ndarray:
    base = rng.uniform(low, high, size=count)
    frequencies = np.clip(base[:, None] + delta * axis[None, :], 0.001, 0.999)
    return rng.binomial(2, frequencies).astype(np.int8)


def _build_hwe_probe(rng, carrier: ProbeCarrier):
    """Make Patterson and raw covariance disagree at the top-k subspace boundary.

    ``k-1`` strong shared Walsh contrasts enter both objects. A rare-variant contrast wins the
    final Patterson slot while a common-variant contrast wins the final raw-covariance slot.
    Subspace capture of those competing directions is therefore discriminating without relying
    on eigenvalue-scaled output columns.
    """
    m, target_variants, k = carrier.n_samples, carrier.n_records, carrier.k
    if not _walsh_carrier_compatible(carrier):
        raise ValueError(f"HWE probe carrier is not Walsh-compatible: {carrier}")
    axes = _walsh_axes(m, k + 1, rng)
    shared_axes = axes[:max(0, k - 1)]
    common_axis = axes[k - 1]
    rare_axis = axes[k]
    base_common = int(rng.integers(1500, 2000))
    base_rare = int(rng.integers(9000, 11000))
    base_total = len(shared_axes) * 800 + base_common + base_rare
    scale = target_variants / base_total
    n_shared = max(1, int(round(800 * scale)))
    n_common = max(1, int(round(base_common * scale)))
    n_rare = target_variants - len(shared_axes) * n_shared - n_common
    if n_rare < 1:
        raise RuntimeError("HWE carrier allocation produced no rare markers")
    blocks = [
        _draw_axis_markers(rng, axis, n_shared, 0.40, 0.60, 0.30)
        for axis in shared_axes
    ]
    blocks.append(_draw_axis_markers(rng, common_axis, n_common, 0.40, 0.60, 0.12))
    rare_base = rng.uniform(0.02, 0.04, size=n_rare)
    rare_frequency = np.where(
        rare_axis[None, :] > 0,
        np.minimum(rare_base[:, None] * 3.5, 0.20),
        rare_base[:, None],
    )
    blocks.append(rng.binomial(2, rare_frequency).astype(np.int8))
    G = np.concatenate(blocks, axis=0)
    order = rng.permutation(G.shape[0])
    return G[order], common_axis, rare_axis


def _hwe_severity_once(
    run_one, workdir: Path, carrier: ProbeCarrier, seed: int, representation_key: bytes,
):
    rng = np.random.default_rng(seed)
    G, common_axis, rare_axis = _build_hwe_probe(rng, carrier)
    k = carrier.k
    G, sample_order, representation_rng = _keyed_representation(
        G, representation_key, f"hwe-{seed}",
    )
    common_axis = common_axis[sample_order]
    rare_axis = rare_axis[sample_order]
    m = G.shape[1]
    sample_ids = _rand_sample_ids(m, representation_rng)
    vcf, out = _probe_paths(workdir, rng)
    _write_plain_probe_vcf(vcf, G, sample_ids, rng, carrier)
    ok, S = run_one(vcf, k, out, sample_ids)
    if not ok or S is None:
        # Fail-CLOSED: a submission that cannot produce a valid PCA of a valid VCF (or refuses a
        # recognised probe) is treated as maximally wrong, not waved through.
        return 1.0, 0.0, 0.0, 0.0

    def _boundary_preference(scores):
        return _axis_capture(scores, rare_axis) - _axis_capture(scores, common_axis)

    a_sub = _boundary_preference(S)
    # Anchor to both exact objects on this draw. The statistic depends only on the emitted span,
    # not eigenvector ordering or eigenvalue scaling.
    pat_ref = _boundary_preference(_patterson_pca(G, k)[0])
    raw_ref = _boundary_preference(_rawcov_scores(G, k))
    gap = pat_ref - raw_ref
    if gap < 0.50:
        # Degenerate draw where Patterson and raw-cov are indistinguishable on this probe -> we
        # cannot tell the objects apart, so do not fire (fail-open only when non-discriminative).
        return 0.0, a_sub, pat_ref, raw_ref
    # t in [0, 1+]: 0 at Patterson and 1 at raw covariance. Exact and well-sampled Patterson fits
    # sit extremely close to zero; the dead-zone only absorbs numerical/sampling variation.
    t = max(0.0, (pat_ref - a_sub) / gap)
    dead_zone = 0.15
    sev = float(np.clip((t - dead_zone) / (1.0 - dead_zone), 0.0, 1.0))
    return sev, a_sub, pat_ref, raw_ref


def hwe_norm_gate(
    run_one, workdir: Path, carriers: Sequence[ProbeCarrier], seed: int, *,
    representation_key: bytes,
) -> dict:
    carriers = _validated_carriers(carriers)
    sevs, observed, pats, raws = [], [], [], []
    for r, carrier in enumerate(carriers):
        sev, value, pat_ref, raw_ref = _hwe_severity_once(
            run_one, workdir, carrier, seed + 101 + 17 * r, representation_key)
        sevs.append(sev)
        observed.append(value)
        pats.append(pat_ref)
        raws.append(raw_ref)
    severity = float(np.median(sevs))
    return {"name": "hwe_norm", "factor": collapse(severity), "severity": severity,
            "boundary_preference": round(float(np.median(observed)), 3),
            "patterson_anchor": round(float(np.median(pats)), 3),
            "rawcov_anchor": round(float(np.median(raws)), 3),
            "severities": [round(s, 3) for s in sevs]}


# ---------------------------------------------------------------------------
# coverage gate
# ---------------------------------------------------------------------------

def _build_coverage_probe(rng, carrier: ProbeCarrier):
    """Subtle structure resolvable only with many markers at the carrier's exact signature."""
    m, n_variants = carrier.n_samples, carrier.n_records
    n_pops = int(rng.integers(3, 5))
    fst = float(rng.uniform(0.0008, 0.0014))
    p_anc = np.clip(rng.beta(0.7, 0.7, size=n_variants), 0.05, 0.95)
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    pk = np.clip(np.stack([rng.beta(a, b) for _ in range(n_pops)]), 1e-6, 1 - 1e-6)
    pop = rng.integers(0, n_pops, size=m)
    G = np.empty((n_variants, m), dtype=np.int8)
    for kk in range(n_pops):
        cols = np.where(pop == kk)[0]
        if cols.size:
            G[:, cols] = rng.binomial(2, pk[kk][:, None], size=(n_variants, cols.size))
    return G


def _coverage_severity_once(
    run_one, workdir: Path, carrier: ProbeCarrier, seed: int, representation_key: bytes,
):
    rng = np.random.default_rng(seed)
    G = _build_coverage_probe(rng, carrier)
    k = carrier.k
    G, _, representation_rng = _keyed_representation(
        G, representation_key, f"coverage-{seed}",
    )
    m = G.shape[1]
    sample_ids = _rand_sample_ids(m, representation_rng)
    vcf, out = _probe_paths(workdir, rng)
    _write_plain_probe_vcf(vcf, G, sample_ids, rng, carrier)
    # Reference: Patterson full-scan of the probe resolves the subtle axes.
    ref, spectrum = _patterson_pca(G, k)
    w = structure_weights(spectrum, ref.shape[1])[:ref.shape[1]]
    if not np.any(w > 0):
        # Bad-luck draw (~1%): at Fst as low as 0.0022 the full-scan reference's leading
        # eigenvalue can itself sit at the Marchenko-Pastur edge, so no structured axis is
        # resolved and there is nothing to measure undercoverage against. Skip this rep --
        # subspace_accuracy would (correctly) raise, and scoring a genuine submission over
        # pure noise directions would false-fire the gate. The other reps carry the median.
        return None
    ok, S = run_one(vcf, k, out, sample_ids)
    if not ok or S is None:
        return 1.0, 0.0                                # fail-CLOSED (see hwe gate)
    agree = subspace_accuracy(S, ref, w)["accuracy"]
    sev = float(np.clip(1.0 - agree / 0.85, 0.0, 1.0))
    return sev, agree


def coverage_gate(
    run_one, workdir: Path, carriers: Sequence[ProbeCarrier], seed: int, *,
    representation_key: bytes,
) -> dict:
    carriers = _validated_carriers(carriers)
    sevs, agrees = [], []
    for r, carrier in enumerate(carriers):
        res = _coverage_severity_once(
            run_one, workdir, carrier, seed + 202 + 19 * r, representation_key,
        )
        if res is None:                                # degenerate probe draw -- skip (see above)
            continue
        sev, agree = res
        sevs.append(sev)
        agrees.append(agree)
    if not sevs:
        # Every draw degenerated: no informative probe, so we cannot assess
        # coverage. Fail-open -- the probe is grader-generated and submission-independent, so a
        # degenerate draw is our bad luck, not an exploit, and must not penalise a genuine fit.
        return {"name": "coverage", "factor": 1.0, "severity": 0.0,
                "subtle_agreement": None, "severities": [],
                "note": "no informative probe draw"}
    severity = float(np.median(sevs))
    return {"name": "coverage", "factor": collapse(severity), "severity": severity,
            "subtle_agreement": round(float(np.median(agrees)), 3),
            "severities": [round(s, 3) for s in sevs]}


# ---------------------------------------------------------------------------
# representation-equivalence gate
# ---------------------------------------------------------------------------

_REPRESENTATION_DEAD_ZONE = 0.08
_REPRESENTATION_FULL_GAP = 0.55
_REPRESENTATION_FIDELITY_DEAD_ZONE = 0.85
_REPRESENTATION_FIDELITY_FLOOR = 0.25


def _build_representation_probe(rng, carrier: ProbeCarrier) -> np.ndarray:
    """A strongly structured population mixture for serialisation-equivalence checks.

    Strong, replicated structure keeps the exact reference's outlier axes well clear of its
    empirical noise edge. Sample count reserves the 64 noise dimensions required by the shared
    structure detector even when a hidden dataset asks for a relatively large ``k``.
    """
    m, n_variants, k = carrier.n_samples, carrier.n_records, carrier.k
    if m < int(k) + 80:
        raise RuntimeError("reviewed probe shape cannot support the requested rank")
    # k reported directions require at least k genuine structured axes. With fewer populations,
    # a malformed-token parser could add one fake axis without displacing any real top-k direction,
    # making detection depend on a lucky population-count draw. k+1 populations guarantee k real
    # contrasts on every repetition; the reviewed shape already reserves k+80 samples.
    n_pops = int(k) + 1
    pop = np.arange(m) % n_pops
    pop = pop[rng.permutation(m)]
    fst = float(rng.uniform(0.012, 0.020))
    p_anc = np.clip(rng.beta(0.7, 0.7, size=n_variants), 0.03, 0.97)
    a = p_anc * (1.0 - fst) / fst
    b = (1.0 - p_anc) * (1.0 - fst) / fst
    frequencies = np.clip(
        np.stack([rng.beta(a, b) for _ in range(n_pops)]), 1e-6, 1.0 - 1e-6)
    G = np.empty((n_variants, m), dtype=np.int8)
    for group in range(n_pops):
        columns = np.flatnonzero(pop == group)
        G[:, columns] = rng.binomial(
            2, frequencies[group, :, None], size=(n_variants, columns.size))
    return G


def _representation_severity(accuracy_a: float, accuracy_b: float) -> float:
    """Continuous representation sensitivity and fidelity with honest-fit dead-zones.

    The gap arm catches sensitivity to a valid serialization change. The absolute arm prevents
    two equally broken parses from passing merely because both miss the same object. Strong
    planted structure makes the fidelity dead-zone conservative for reviewed approximate fits.
    """
    gap = abs(float(accuracy_a) - float(accuracy_b))
    gap_severity = float(np.clip(
        (gap - _REPRESENTATION_DEAD_ZONE)
        / (_REPRESENTATION_FULL_GAP - _REPRESENTATION_DEAD_ZONE), 0.0, 1.0))
    fidelity = min(float(accuracy_a), float(accuracy_b))
    fidelity_severity = float(np.clip(
        (_REPRESENTATION_FIDELITY_DEAD_ZONE - fidelity)
        / (_REPRESENTATION_FIDELITY_DEAD_ZONE - _REPRESENTATION_FIDELITY_FLOOR),
        0.0,
        1.0,
    ))
    return max(gap_severity, fidelity_severity)


def _representation_severity_once(
    run_one, workdir: Path, carrier: ProbeCarrier, seed: int, representation_key: bytes,
):
    rng = np.random.default_rng(seed)
    G = _build_representation_probe(rng, carrier)
    k = carrier.k
    # Missingness is independent of population, record order, and representation. Both arms
    # encode this exact mask differently, and the truth anchor mean-imputes it after centring.
    # This is a parser-identity probe, not an estimate of cohort missingness. A deliberately dense
    # 60--70% mask makes the private malformed-prefix contrast displace a true top component for a
    # prefix-only parser, while a correct mean-imputing implementation sees exactly the same object
    # in both arms and remains unaffected. Lower rates produced only a scientifically negligible
    # perturbation and therefore (correctly) stayed inside the gate's 0.08 safety dead-zone.
    missing_rate = float(rng.uniform(0.60, 0.70))
    for start in range(0, G.shape[0], 4096):
        block = G[start:start + 4096]
        missing = rng.random(block.shape) < missing_rate
        block[missing] = -1
    G, _, representation_rng = _keyed_representation(
        G, representation_key, f"representation-{seed}",
    )
    sample_ids = _rand_sample_ids(G.shape[1], representation_rng)
    ref, spectrum = _patterson_pca(G, k)
    weights = structure_weights(spectrum, ref.shape[1])[:ref.shape[1]]
    if not np.any(weights > 0):
        # Bad-luck draw: the probe's own leading eigenvalue can sit at the Marchenko-Pastur
        # edge, leaving no structured axis to measure representation fidelity against. Skip this
        # rep exactly as the coverage gate does -- subspace_accuracy would (correctly) raise on
        # pure noise directions, and scoring a genuine submission over them would false-fire the
        # gate. Signalling a skip (not raising) keeps a single unlucky release draw from crashing
        # the entire grade; the other reps carry the median, or the gate fails open below.
        return None

    expected_shape = (G.shape[1], k)
    vcf_a, out_a = _probe_paths(workdir, rng)
    vcf_b, out_b = _probe_paths(workdir, rng)
    order_a, order_b = _write_representation_pair(
        vcf_a, vcf_b, G, sample_ids, rng, carrier)
    ok_a, scores_a = run_one(vcf_a, k, out_a, [sample_ids[i] for i in order_a])
    ok_b, scores_b = run_one(vcf_b, k, out_b, [sample_ids[i] for i in order_b])
    if (not ok_a or scores_a is None or scores_a.shape != expected_shape
            or not ok_b or scores_b is None or scores_b.shape != expected_shape):
        return 1.0, 0.0, 0.0
    canonical_a = np.empty_like(scores_a)
    canonical_b = np.empty_like(scores_b)
    canonical_a[order_a] = scores_a
    canonical_b[order_b] = scores_b
    accuracy_a = subspace_accuracy(canonical_a, ref, weights)["accuracy"]
    accuracy_b = subspace_accuracy(canonical_b, ref, weights)["accuracy"]
    return _representation_severity(accuracy_a, accuracy_b), accuracy_a, accuracy_b


def representation_equivalence_gate(
    run_one, workdir: Path, carriers: Sequence[ProbeCarrier], seed: int, *,
    representation_key: bytes,
) -> dict:
    carriers = _validated_carriers(carriers)
    severities, accuracies_a, accuracies_b, gaps = [], [], [], []
    for rep, carrier in enumerate(carriers):
        result = _representation_severity_once(
            run_one, workdir, carrier, seed + 303 + 23 * rep, representation_key)
        if result is None:                             # degenerate probe draw -- skip (see above)
            continue
        severity, accuracy_a, accuracy_b = result
        severities.append(severity)
        accuracies_a.append(accuracy_a)
        accuracies_b.append(accuracy_b)
        gaps.append(abs(accuracy_a - accuracy_b))
    if not severities:
        # Every draw degenerated: the probe is grader-generated and submission-independent, so a
        # structureless draw is our bad luck, not an exploit, and must not penalise a genuine fit.
        # Fail open, exactly as the coverage gate does when all its draws degenerate.
        return {"name": "representation_equivalence", "factor": 1.0, "severity": 0.0,
                "reference_agreement_a": None, "reference_agreement_b": None,
                "agreement_gap": None, "severities": [],
                "note": "no informative probe draw"}
    severity = float(np.median(severities))
    median_a = float(np.median(accuracies_a))
    median_b = float(np.median(accuracies_b))
    return {
        "name": "representation_equivalence",
        "factor": collapse(severity),
        "severity": severity,
        "reference_agreement_a": round(median_a, 3),
        "reference_agreement_b": round(median_b, 3),
        "agreement_gap": round(float(np.median(gaps)), 3),
        "severities": [round(value, 3) for value in severities],
    }
