"""Synthetic multi-sample VCF generator with PLANTED population structure.

The whole benchmark rests on knowing the truth. We generate genotypes under the
Balding-Nichols model, which gives us three things the grader needs:

  1. per-sample ancestry (population labels, or admixture proportions), so we know
     what structure a correct PCA *must* recover;
  2. a controllable difficulty knob (Fst): large Fst -> continental-scale structure
     that a few thousand markers resolve; small Fst -> subtle within-continent
     structure that needs many markers (this is what separates a genuine wide-scan
     PCA from a 200-SNP undercoverage hack);
  3. a big, unindexed, plain-text VCF on disk, so the I/O cost is real and the
     systems-level tricks (mmap + random byte-offset sampling + parallel pread)
     actually pay off.

Balding-Nichols: an ancestral allele frequency p is drawn per variant; each
population k draws its own frequency p_k ~ Beta(p(1-F)/F, (1-p)(1-F)/F) with
F = Fst; a sample in population k has genotype ~ Binomial(2, p_k). Larger F ->
populations drift further apart -> stronger structure.

We also inject "messy" records (multiallelic, symbolic SV ALTs, missing calls) at a
controlled rate so the robustness category has teeth: a genuine implementation must
skip/handle them without crashing or corrupting allele-frequency estimates.

Truth is written next to the VCF as ``<vcf-path>.truth.json``. The VCF itself is written
with numpy-vectorised formatting (no per-cell Python loop) so multi-hundred-MB files
generate in seconds.
"""

from __future__ import annotations

import argparse
import gzip
import hmac
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from data.vcf_serialization import (
    MISSING_GT_ENCODINGS,
    randomized_positions,
    variable_gt_fields,
    variable_info,
)

from data.release_key import RELEASE_ID, derive_seed, key_commitment, read_math_key


# ---------------------------------------------------------------------------
# Genotype simulation (Balding-Nichols)
# ---------------------------------------------------------------------------

SimulationRegime = Literal[
    "balding_nichols",
    "haploid_structure",
    "high_rank",
    "ill_conditioned",
    "spectral_selection",
    "spatial_ld",
    "rare_structure",
]
VCFRepresentation = Literal["plain_gt", "variable_width", "haploid_gt"]
@dataclass
class DatasetSpec:
    name: str
    category: str          # scoring rollup category
    n_samples: int
    n_variants: int
    n_pops: int
    fst: float
    admixed: bool = False
    messy_frac: float = 0.0     # fraction of records that are multiallelic/SV/missing
    missing_frac: float = 0.0   # per-genotype missingness on normal records
    n_chroms: int = 22
    seed: int = 0
    weight: float = 1.0         # dataset weight in its category (harder = heavier)
    k: int = 10                 # requested score-subspace rank for this dataset
    regime: SimulationRegime = "balding_nichols"
    representation: VCFRepresentation = "plain_gt"


def _bn_pop_freqs(
    rng: np.random.Generator,
    p_anc: np.ndarray,
    fst: float,
    n_pops: int,
) -> np.ndarray:
    """Draw per-population allele frequencies. Returns (n_pops, n_variants)."""
    if fst <= 0:
        return np.repeat(p_anc[None, :], n_pops, axis=0)
    a = p_anc * (1.0 - fst) / fst
    b = (1.0 - p_anc) * (1.0 - fst) / fst
    # Beta(a, b) per population, independent draws.
    out = np.empty((n_pops, p_anc.size), dtype=np.float64)
    for k in range(n_pops):
        out[k] = rng.beta(a, b)
    return np.clip(out, 1e-6, 1 - 1e-6)


def _balanced_population_labels(
    rng: np.random.Generator,
    n_samples: int,
    n_pops: int,
) -> np.ndarray:
    """Return shuffled labels with every population represented almost equally."""
    if n_pops < 2:
        raise ValueError("structured simulations require at least two populations")
    if n_samples < n_pops:
        raise ValueError("structured simulations require at least one sample per population")
    labels = np.arange(n_samples, dtype=np.int32) % n_pops
    rng.shuffle(labels)
    return labels


def _draw_population_genotypes(
    rng: np.random.Generator,
    pop_freqs: np.ndarray,
    sample_pop: np.ndarray,
) -> np.ndarray:
    """Draw a variants-by-samples dosage matrix for hard population assignments."""
    n_pops, n_variants = pop_freqs.shape
    G = np.empty((n_variants, sample_pop.size), dtype=np.int8)
    for pop in range(n_pops):
        cols = np.flatnonzero(sample_pop == pop)
        if cols.size == 0:
            raise ValueError(f"population {pop} has no samples")
        block = rng.binomial(2, pop_freqs[pop][:, None], size=(n_variants, cols.size))
        G[:, cols] = block.astype(np.int8)
    return G


def _simulate_haploid_structure(spec: DatasetSpec, rng: np.random.Generator):
    """Plant population contrasts using genuinely haploid calls.

    The public contract maps VCF haploid alleles to pseudo-diploid dosages 0 and 2. Keeping the
    mathematical matrix in that dosage convention lets the ordinary Patterson reference remain
    the only scoring truth while making a parser that drops unseparated calls fail on a real,
    identifiable PCA direction.
    """
    if spec.n_pops - 1 < spec.k:
        raise ValueError("haploid simulation requires n_pops - 1 >= k")
    p_anc = np.clip(rng.beta(0.7, 0.7, size=spec.n_variants), 0.02, 0.98)
    pop_freqs = _bn_pop_freqs(rng, p_anc, spec.fst, spec.n_pops)
    sample_pop = _balanced_population_labels(rng, spec.n_samples, spec.n_pops)
    G = np.empty((spec.n_variants, spec.n_samples), dtype=np.int8)
    for pop in range(spec.n_pops):
        columns = np.flatnonzero(sample_pop == pop)
        G[:, columns] = 2 * rng.binomial(
            1, pop_freqs[pop][:, None], size=(spec.n_variants, columns.size)
        ).astype(np.int8)
    return G, sample_pop, pop_freqs, p_anc, None


def _simulate_high_rank(spec: DatasetSpec, rng: np.random.Generator):
    """Plant ``n_pops - 1`` genuine population-contrast axes.

    Ordinary synthetic cohorts use only a handful of populations. This regime deliberately
    balances many populations so every requested high-order axis is represented, while retaining
    the same Balding-Nichols genotype model and Patterson-PCA contract as the rest of the suite.
    """
    if spec.n_pops - 1 < spec.k:
        raise ValueError("high-rank simulation requires n_pops - 1 >= k")
    p_anc = np.clip(rng.beta(0.7, 0.7, size=spec.n_variants), 0.02, 0.98)
    pop_freqs = _bn_pop_freqs(rng, p_anc, spec.fst, spec.n_pops)
    sample_pop = _balanced_population_labels(rng, spec.n_samples, spec.n_pops)
    G = _draw_population_genotypes(rng, pop_freqs, sample_pop)
    return G, sample_pop, pop_freqs, p_anc, None


def _spectral_axis_strengths(spec: DatasetSpec) -> np.ndarray:
    """Return a descending population-contrast spectrum with a boundary gap at ``k``.

    This regime differs from the equal-strength high-rank arm: it contains genuine structure on
    both sides of the requested rank.  The first ``k`` contrasts decay gradually, then the
    remaining contrasts continue at a lower, still-identifiable level.  A solver must therefore
    select the leading subspace rather than returning an arbitrary collection of structured axes.
    """
    n_axes = spec.n_pops - 1
    if n_axes < spec.k + 4:
        raise ValueError(
            "spectral-selection simulation requires at least four structured axes beyond k"
        )
    if not 0.0 < spec.fst <= 0.4:
        raise ValueError("spectral-selection strength must satisfy 0 < fst <= 0.4")
    leading = np.linspace(spec.fst, 0.75 * spec.fst, spec.k)
    trailing = np.linspace(0.55 * spec.fst, 0.45 * spec.fst, n_axes - spec.k)
    return np.concatenate([leading, trailing])


def _simulate_spectral_selection(spec: DatasetSpec, rng: np.random.Generator):
    """Plant more identifiable population axes than requested, with a clear rank-``k`` gap.

    Markers are shuffled across orthogonal population contrasts, so file position carries no
    ordering clue.  Axis-specific frequency effects control the spectrum while ordinary diploid
    binomial draws preserve the same Patterson-PCA object and finite noise bulk used elsewhere.
    """
    strengths = _spectral_axis_strengths(spec)
    sample_pop = _balanced_population_labels(rng, spec.n_samples, spec.n_pops)
    contrasts = _helmert_contrasts(spec.n_pops)

    marker_axis = np.arange(spec.n_variants, dtype=np.int32) % strengths.size
    rng.shuffle(marker_axis)
    ancestral = rng.uniform(0.40, 0.60, size=spec.n_variants)
    polarity = rng.choice(np.array([-1.0, 1.0]), size=spec.n_variants)
    effects = polarity * strengths[marker_axis]
    pop_freqs = np.clip(
        ancestral[None, :] + contrasts[:, marker_axis] * effects[None, :],
        0.01,
        0.99,
    )
    G = _draw_population_genotypes(rng, pop_freqs, sample_pop)
    p_anc = pop_freqs.mean(axis=0)
    return G, sample_pop, pop_freqs, p_anc, None


def _observed_population_frequencies(
    G: np.ndarray,
    sample_pop: np.ndarray,
    n_pops: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed cohort-wide and per-population alternate-allele frequencies."""
    p_anc = G.mean(axis=1, dtype=np.float64) / 2.0
    pop_freqs = np.empty((n_pops, G.shape[0]), dtype=np.float64)
    for pop in range(n_pops):
        cols = np.flatnonzero(sample_pop == pop)
        if cols.size == 0:
            raise ValueError(f"population {pop} has no samples")
        pop_freqs[pop] = G[:, cols].mean(axis=1, dtype=np.float64) / 2.0
    return pop_freqs, p_anc


def _simulate_ill_conditioned(spec: DatasetSpec, rng: np.random.Generator):
    """Build a structured matrix with exact duplicates and pathological-frequency rows.

    The planted prototype markers carry at least ``k`` population axes. Exact copies make the
    feature matrix rank-deficient, one-call mutations create nearly duplicate rows, singleton /
    doubleton sites exercise the HWE denominator near its finite-sample extremes, and fixed rows
    must be discarded as monomorphic. Row order is shuffled so none of these classes forms a
    recognizable region in the resulting VCF.
    """
    if spec.n_pops - 1 < spec.k:
        raise ValueError("ill-conditioned simulation requires n_pops - 1 >= k")
    if spec.n_variants < 10 * spec.n_pops:
        raise ValueError("ill-conditioned simulation needs at least 10 markers per population")

    sample_pop = _balanced_population_labels(rng, spec.n_samples, spec.n_pops)
    n_monomorphic = max(2, int(round(0.10 * spec.n_variants)))
    n_near_fixed = max(2, int(round(0.20 * spec.n_variants)))
    n_prototypes = max(4 * spec.n_pops, int(round(0.20 * spec.n_variants)))
    n_related = spec.n_variants - n_monomorphic - n_near_fixed - n_prototypes
    if n_related < 2:
        raise ValueError("ill-conditioned simulation has too few duplicate markers")
    n_exact_duplicates = n_related // 2
    n_near_duplicates = n_related - n_exact_duplicates

    prototype_p = np.clip(rng.beta(0.7, 0.7, size=n_prototypes), 0.02, 0.98)
    prototype_freqs = _bn_pop_freqs(rng, prototype_p, spec.fst, spec.n_pops)
    prototypes = _draw_population_genotypes(rng, prototype_freqs, sample_pop)

    exact_source = rng.integers(0, n_prototypes, size=n_exact_duplicates)
    exact_duplicates = prototypes[exact_source].copy()

    near_source = rng.integers(0, n_prototypes, size=n_near_duplicates)
    near_duplicates = prototypes[near_source].copy()
    changed_rows = np.arange(n_near_duplicates)
    changed_cols = rng.integers(0, spec.n_samples, size=n_near_duplicates)
    old = near_duplicates[changed_rows, changed_cols]
    heterozygote_replacement = rng.choice(np.array([0, 2], dtype=np.int8), size=n_near_duplicates)
    replacement = np.where(old == 0, 1, np.where(old == 2, 1, heterozygote_replacement))
    near_duplicates[changed_rows, changed_cols] = replacement.astype(np.int8)

    near_fixed = np.zeros((n_near_fixed, spec.n_samples), dtype=np.int8)
    fixed_rows = np.arange(n_near_fixed)
    carriers = rng.integers(0, spec.n_samples, size=n_near_fixed)
    near_fixed[fixed_rows, carriers] = rng.choice(
        np.array([1, 2], dtype=np.int8), size=n_near_fixed, p=[0.8, 0.2]
    )
    inverted = rng.random(n_near_fixed) < 0.5
    near_fixed[inverted] = 2 - near_fixed[inverted]

    monomorphic = np.zeros((n_monomorphic, spec.n_samples), dtype=np.int8)
    monomorphic[n_monomorphic // 2:] = 2

    G = np.concatenate(
        [prototypes, exact_duplicates, near_duplicates, near_fixed, monomorphic], axis=0
    )
    G = G[rng.permutation(G.shape[0])]
    pop_freqs, p_anc = _observed_population_frequencies(G, sample_pop, spec.n_pops)
    return G, sample_pop, pop_freqs, p_anc, None


def _helmert_contrasts(n_pops: int) -> np.ndarray:
    """Return an orthonormal basis for population contrasts.

    Column ``j`` contrasts population ``j + 1`` against the preceding populations. The basis is
    deterministic, centered, and treats every population as part of the global target.
    """
    out = np.zeros((n_pops, n_pops - 1), dtype=np.float64)
    for j in range(n_pops - 1):
        scale = np.sqrt((j + 1) * (j + 2))
        out[:j + 1, j] = 1.0 / scale
        out[j + 1, j] = -(j + 1) / scale
    return out


def _simulate_spatial_ld(spec: DatasetSpec, rng: np.random.Generator):
    """Plant complementary population axes in long, strongly linked genomic blocks.

    Each contiguous block carries one population contrast and contains descendants of a small
    set of genotype prototypes. Across the whole VCF every contrast receives the same number of
    blocks, so the exact all-marker Patterson PCA has the requested global target. A sampler that
    reads only a few contiguous regions instead sees only a subset of those axes.
    """
    n_axes = spec.n_pops - 1
    if n_axes < spec.k:
        raise ValueError("spatial-LD simulation requires n_pops - 1 >= k")
    if spec.n_variants < 40 * n_axes:
        raise ValueError("spatial-LD simulation needs at least 40 markers per contrast")

    sample_pop = _balanced_population_labels(rng, spec.n_samples, spec.n_pops)
    contrasts = _helmert_contrasts(spec.n_pops)
    n_blocks = 4 * n_axes
    block_sizes = np.full(n_blocks, spec.n_variants // n_blocks, dtype=np.int64)
    block_sizes[:spec.n_variants % n_blocks] += 1
    block_axes = np.tile(np.arange(n_axes, dtype=np.int32), 4)
    rng.shuffle(block_axes)

    blocks: list[np.ndarray] = []
    for block_size, axis_idx in zip(block_sizes, block_axes, strict=True):
        axis = contrasts[:, axis_idx].copy()
        axis /= np.max(np.abs(axis))
        n_prototypes = max(8, min(32, int(block_size) // 60))
        prototype_rows = np.empty((n_prototypes, spec.n_samples), dtype=np.int8)
        prototype_freqs = np.empty((n_prototypes, spec.n_pops), dtype=np.float64)
        for prototype in range(n_prototypes):
            base = rng.uniform(0.20, 0.80)
            direction = -1.0 if rng.random() < 0.5 else 1.0
            freq = np.clip(base + direction * 0.24 * axis, 0.01, 0.99)
            prototype_freqs[prototype] = freq
            for pop in range(spec.n_pops):
                cols = np.flatnonzero(sample_pop == pop)
                prototype_rows[prototype, cols] = rng.binomial(2, freq[pop], size=cols.size)

        source = rng.integers(0, n_prototypes, size=int(block_size))
        block = prototype_rows[source].copy()
        # Sparse within-block mutation avoids making the challenge an exact-duplicate fixture while
        # retaining strong local LD. Mutations are redrawn from the same local contrast model.
        mutate = rng.random(block.shape) < 0.015
        for pop in range(spec.n_pops):
            cols = np.flatnonzero(sample_pop == pop)
            if cols.size == 0:
                continue
            local_mask = mutate[:, cols]
            if not np.any(local_mask):
                continue
            redraw_p = prototype_freqs[source, pop][:, None]
            redraw = rng.binomial(2, np.broadcast_to(redraw_p, local_mask.shape))
            target = block[:, cols]
            target[local_mask] = redraw[local_mask]
            block[:, cols] = target
        blocks.append(block)

    G = np.concatenate(blocks, axis=0)
    pop_freqs, p_anc = _observed_population_frequencies(G, sample_pop, spec.n_pops)
    return G, sample_pop, pop_freqs, p_anc, None


def _simulate_rare_structure(spec: DatasetSpec, rng: np.random.Generator):
    """Combine a population-neutral common backbone with structured MAC 1--20 sites.

    Rare sites are exact low-count observations rather than frequencies clipped away from zero.
    Their carrier populations are balanced, so retaining the rare tail contributes every planted
    population contrast to the all-marker Patterson PCA. Common sites provide realistic eligible
    background variation but deliberately carry no population contrast: otherwise an illegal MAF
    filter can discard the rare tail and recover the same target from a redundant common-marker
    copy, defeating the purpose of this distinct regime.
    """
    n_axes = spec.n_pops - 1
    if n_axes < spec.k:
        raise ValueError("rare-structure simulation requires n_pops - 1 >= k")
    if spec.n_samples // spec.n_pops < 20:
        raise ValueError("rare-structure simulation needs at least 20 samples per population")

    sample_pop = _balanced_population_labels(rng, spec.n_samples, spec.n_pops)
    n_rare = int(round(0.65 * spec.n_variants))
    n_common = spec.n_variants - n_rare

    common_p = np.clip(rng.beta(0.7, 0.7, size=n_common), 0.05, 0.95)
    common_freqs = np.repeat(common_p[None, :], spec.n_pops, axis=0)
    common = _draw_population_genotypes(rng, common_freqs, sample_pop)

    rare = np.zeros((n_rare, spec.n_samples), dtype=np.int8)
    carrier_pops = np.arange(n_rare, dtype=np.int32) % spec.n_pops
    rng.shuffle(carrier_pops)
    # Singletons are load-bearing rather than incidental numerical noise.  A common shortcut is
    # to discard MAC <= 1 even though the task's all-eligible-marker object includes them.
    mac = rng.integers(1, 21, size=n_rare)
    for row, (pop, count) in enumerate(zip(carrier_pops, mac, strict=True)):
        eligible = np.flatnonzero(sample_pop == pop)
        carriers = rng.choice(eligible, size=int(count), replace=False)
        rare[row, carriers] = 1

    G = np.concatenate([common, rare], axis=0)
    G = G[rng.permutation(G.shape[0])]
    pop_freqs, p_anc = _observed_population_frequencies(G, sample_pop, spec.n_pops)
    return G, sample_pop, pop_freqs, p_anc, None


def simulate_genotypes(spec: DatasetSpec):
    """Return (G, sample_pop, pop_freqs, p_anc, admix).

    G is (n_variants, n_samples) int8 in {0,1,2}; -1 marks missing (filled later).
    """
    rng = np.random.default_rng(spec.seed)
    n, m = spec.n_variants, spec.n_samples

    if spec.regime == "haploid_structure":
        return _simulate_haploid_structure(spec, rng)
    if spec.regime == "high_rank":
        return _simulate_high_rank(spec, rng)
    if spec.regime == "ill_conditioned":
        return _simulate_ill_conditioned(spec, rng)
    if spec.regime == "spectral_selection":
        return _simulate_spectral_selection(spec, rng)
    if spec.regime == "spatial_ld":
        return _simulate_spatial_ld(spec, rng)
    if spec.regime == "rare_structure":
        return _simulate_rare_structure(spec, rng)
    if spec.regime != "balding_nichols":
        raise ValueError(f"unknown simulation regime: {spec.regime}")

    # Ancestral frequencies: Beta(0.5, 0.5)-ish spectrum favours informative sites,
    # then clip away the monomorphic tails.
    p_anc = np.clip(rng.beta(0.7, 0.7, size=n), 0.02, 0.98)
    pop_freqs = _bn_pop_freqs(rng, p_anc, spec.fst, spec.n_pops)   # (K, n)

    if spec.admixed:
        # Each sample gets Dirichlet admixture proportions over the K populations.
        alpha = np.full(spec.n_pops, 0.3)
        admix = rng.dirichlet(alpha, size=m)                       # (m, K)
        sample_pop = np.argmax(admix, axis=1).astype(np.int32)     # dominant pop label
        # Per-sample effective frequency is a mixture: freq[i, variant] = admix[i] @ pop_freqs
        # Draw genotypes column-by-column to bound memory on big datasets.
        G = np.empty((n, m), dtype=np.int8)
        for i in range(m):
            f_i = admix[i] @ pop_freqs                              # (n,)
            G[:, i] = rng.binomial(2, f_i).astype(np.int8)
    else:
        admix = None
        sample_pop = rng.integers(0, spec.n_pops, size=m).astype(np.int32)
        # Ordinary regimes intentionally allow the random population counts to vary.
        G = np.empty((n, m), dtype=np.int8)
        for k in range(spec.n_pops):
            cols = np.where(sample_pop == k)[0]
            if cols.size == 0:
                continue
            block = rng.binomial(2, pop_freqs[k][:, None], size=(n, cols.size))
            G[:, cols] = block.astype(np.int8)

    return G, sample_pop, pop_freqs, p_anc, admix


# ---------------------------------------------------------------------------
# VCF serialisation (vectorised)
# ---------------------------------------------------------------------------

_GT_STR = np.array([b"0/0", b"0/1", b"1/1"], dtype="S3")
def _clean_record(
    row_index: int,
    chrom: int,
    position: int,
    ref: bytes,
    alt: bytes,
    gt_codes: np.ndarray,
    missing: np.ndarray | None,
    representation: VCFRepresentation,
    rng: np.random.Generator,
) -> bytes:
    tab = b"\t"
    if representation == "plain_gt":
        row = _GT_STR[np.clip(gt_codes, 0, 2)].copy()
        if missing is not None:
            row[missing] = b"./."
        gt_field = tab.join(row.tolist())
        info = b"."
        format_field = b"GT"
    elif representation == "variable_width":
        style = row_index % 4
        format_field, gt_field = variable_gt_fields(
            gt_codes, missing, style, rng, MISSING_GT_ENCODINGS, None,
        )
        # Width varies independently of genotype, chromosome, frequency, and planted structure.
        info = variable_info(rng)
        if rng.random() < 0.5:
            ref = ref.lower()
        if rng.random() < 0.5:
            alt = alt.lower()
    elif representation == "haploid_gt":
        if np.any((gt_codes != 0) & (gt_codes != 2)):
            raise ValueError("haploid representation requires pseudo-diploid dosages 0 or 2")
        row = np.where(gt_codes == 0, b"0", b"1").astype("S1")
        if missing is not None:
            row[missing] = b"."
        gt_field = tab.join(row.tolist())
        info = b"."
        format_field = b"GT"
    else:
        raise ValueError(f"unknown VCF representation: {representation}")

    fixed = tab.join([
        _chrom_name(chrom), str(position).encode(), b".", ref, alt,
        b".", b"PASS", info, format_field,
    ])
    return fixed + tab + gt_field


def write_vcf(
    spec: DatasetSpec,
    path: Path,
    *,
    representation_key: bytes,
    math_commitment: str,
) -> dict:
    if len(representation_key) != 32:
        raise ValueError("representation_key must contain exactly 32 bytes")
    t0 = time.time()
    G, sample_pop, pop_freqs, p_anc, admix = simulate_genotypes(spec)
    n, m = G.shape
    rng = np.random.default_rng(spec.seed + 777)
    representation_seed = int.from_bytes(
        hmac.digest(
            representation_key,
            b"synthetic-v4\0" + spec.name.encode(),
            "sha256",
        )[:8],
        "little",
    )
    representation_rng = np.random.default_rng(representation_seed)

    # Randomize only exact PCA symmetries. Sample permutation/aliases and per-marker allele
    # polarity change the raw fixture and defeat byte/ID hardcoding, but preserve Z Z^T up to the
    # corresponding sample-row permutation. The planted genotypes and eigengaps remain fixed.
    sample_order = representation_rng.permutation(m)
    G = G[:, sample_order]
    sample_pop = sample_pop[sample_order]
    if admix is not None:
        admix = admix[sample_order]

    chrom, pos = randomized_positions(n, spec.n_chroms, rng)
    bases = np.array([b"A", b"C", b"G", b"T"])
    ref = rng.choice(bases, size=n)
    alt = rng.choice(bases, size=n)
    # ensure ref != alt
    clash = ref == alt
    alt[clash] = bases[(np.searchsorted(bases, alt[clash]) + 1) % 4]
    polarity_flip = representation_rng.random(n) < 0.5
    original_ref = ref[polarity_flip].copy()
    ref[polarity_flip] = alt[polarity_flip]
    alt[polarity_flip] = original_ref
    flipped = G[polarity_flip]
    G[polarity_flip] = np.where(flipped >= 0, 2 - flipped, flipped)

    # Decide messy records.
    n_messy = int(round(spec.messy_frac * n))
    messy_idx = set(rng.choice(n, size=n_messy, replace=False).tolist()) if n_messy else set()

    sample_ids = [
        "p" + hmac.digest(
            representation_key,
            b"synthetic-sample\0" + spec.name.encode() + b"\0" + str(int(index)).encode(),
            "sha256",
        ).hex()[:20]
        for index in sample_order
    ]
    if len(set(sample_ids)) != m:
        raise RuntimeError("synthetic sample aliases collided")

    # Missingness mask on normal records.
    if spec.missing_frac > 0:
        # Draw missingness in canonical sample order, then apply the same representation
        # permutation as the genotypes. Otherwise a new alias/order key would move missing calls
        # between biological samples and silently change the scored PCA object.
        miss_mask = (rng.random((n, m)) < spec.missing_frac)[:, sample_order]
    else:
        miss_mask = None

    opener = gzip.open if path.suffix == ".gz" else open
    header = _vcf_header(sample_ids, spec)
    line_ending = b"\r\n" if spec.representation == "variable_width" else b"\n"
    if line_ending != b"\n":
        header = header.replace(b"\n", line_ending)
    kept_clean = 0
    with opener(path, "wb") as fh:
        fh.write(header)
        # Write in row-chunks; format each chunk with numpy string ops.
        CHUNK = 20000
        newline = line_ending
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            lines = []
            for r in range(start, end):
                gt_codes = G[r]
                if r in messy_idx:
                    lines.append(_messy_record(r, chrom[r], pos[r], ref[r], alt[r], gt_codes, rng))
                else:
                    missing = miss_mask[r] if miss_mask is not None else None
                    lines.append(_clean_record(
                        r, int(chrom[r]), int(pos[r]), bytes(ref[r]), bytes(alt[r]),
                        gt_codes, missing, spec.representation, rng,
                    ))
                    kept_clean += 1
            is_last_chunk = end == n
            fh.write(newline.join(lines))
            # The rich parser-stress fold deliberately exercises a valid final record without a
            # trailing newline. Other folds keep the conventional separator.
            if not is_last_chunk or spec.representation != "variable_width":
                fh.write(newline)

    dt = time.time() - t0
    private_spec = asdict(spec)
    private_spec.pop("seed")
    private_spec["math_release"] = RELEASE_ID
    private_spec["math_key_commitment"] = math_commitment
    truth = {
        "spec": private_spec,
        "n_samples": m,
        "n_variants": n,
        "n_clean_biallelic_snv": kept_clean,
        "sample_ids": sample_ids,
        "sample_pop": sample_pop.tolist(),
        "admix": admix.tolist() if admix is not None else None,
        "generate_seconds": round(dt, 3),
    }
    # Sidecar name = ``<full vcf path>.truth.json`` ALWAYS (append, don't substitute), so it is the
    # exact inverse of the grader's ``str(truth_path).replace(".truth.json", "")`` reconstruction
    # for both ``foo.vcf`` and ``foo.vcf.gz``. The old ``.vcf.gz`` special-case wrote ``foo.truth
    # .json`` -> reconstructed to ``foo`` (nonexistent) -> the gzipped dataset was silently skipped.
    truth_path = Path(str(path) + ".truth.json")
    truth_path.write_text(json.dumps(truth))
    return {"vcf": str(path), "truth": str(truth_path), "seconds": dt,
            "size_mb": round(path.stat().st_size / 1e6, 1), "clean_snv": kept_clean}


def _chrom_name(c: int) -> bytes:
    return f"chr{c}".encode()


def _vcf_header(sample_ids, spec: DatasetSpec) -> bytes:
    # NO identifying metadata. The header used to carry ``name={spec.name};category={spec.category}``
    # -- handing the submission the real dataset stem (-> the sibling ``<stem>.vcf.truth.json``
    # sidecar) and its category directly in the file it is asked to read. It is stripped to a bare
    # generic source tag. Contigs are declared as a fixed chr1..chr22 superset (not ``spec.n_chroms``)
    # so the number of populations/chromosomes doesn't leak through the header either.
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
    for c in range(1, 23):
        lines.append(f"##contig=<ID=chr{c}>".encode())
    cols = [b"#CHROM", b"POS", b"ID", b"REF", b"ALT", b"QUAL", b"FILTER", b"INFO", b"FORMAT"]
    cols += [s.encode() for s in sample_ids]
    lines.append(b"\t".join(cols))
    return b"\n".join(lines) + b"\n"


def _messy_record(r, c, pos, ref, alt, gt_codes, rng) -> bytes:
    """Emit a record a naive parser would mishandle: multiallelic, symbolic SV, or
    all-missing. A genuine implementation must not let these corrupt the PCA."""
    kind = rng.integers(0, 3)
    tab = b"\t"
    m = gt_codes.size
    if kind == 0:  # multiallelic: two ALTs, GT indices can be 0/1/2
        alt2 = b"%s,%s" % (bytes(alt), bytes(rng.choice([b"A", b"C", b"G", b"T"])))
        codes = np.clip(gt_codes, 0, 2)
        # encode as 0/0,0/1,1/1,1/2 sometimes to look genuinely multiallelic
        strs = np.array([b"0/0", b"0/1", b"1/2"], dtype="S3")[codes]
        gt = tab.join(strs.tolist())
        fixed = tab.join([_chrom_name(c), str(pos).encode(), b".", bytes(ref), alt2,
                          b".", b"PASS", b".", b"GT"])
        return fixed + tab + gt
    elif kind == 1:  # symbolic SV ALT
        gt = tab.join(np.array([b"0/0", b"0/1", b"1/1"], dtype="S3")[np.clip(gt_codes, 0, 2)].tolist())
        fixed = tab.join([_chrom_name(c), str(pos).encode(), b".", bytes(ref), b"<DEL>",
                          b".", b"PASS", b"SVTYPE=DEL", b"GT"])
        return fixed + tab + gt
    else:  # all-missing
        gt = tab.join([b"./."] * m)
        fixed = tab.join([_chrom_name(c), str(pos).encode(), b".", bytes(ref), bytes(alt),
                          b".", b"PASS", b".", b"GT"])
        return fixed + tab + gt


# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

def _keyed_specs(specs: list[DatasetSpec], math_key: bytes, suite: str) -> list[DatasetSpec]:
    """Assign every mathematical draw a private, domain-separated release seed."""
    return [
        replace(spec, seed=derive_seed(math_key, f"synthetic/{suite}/{spec.name}"))
        for spec in specs
    ]


def dev_specs(math_key: bytes) -> list[DatasetSpec]:
    """Small, fast datasets for local development / CI."""
    return _keyed_specs([
        # 320 is divisible by every Walsh period induced by the reviewed dev ranks. Keeping the
        # development catalogue probe-compatible makes scripts/validate.py exercise the real
        # method gates instead of failing before any submission is measured.
        DatasetSpec("dev_continental", "continental", 320, 40000, 5, 0.08, seed=1, k=5),
        DatasetSpec("dev_subtle", "subtle", 320, 60000, 3, 0.006, seed=2, weight=1.5, k=8),
        DatasetSpec("dev_admixed", "admixed", 320, 40000, 4, 0.05, admixed=True, seed=3, k=3),
        DatasetSpec("dev_messy", "messy", 320, 40000, 5, 0.06, seed=4,
                    messy_frac=0.15, missing_frac=0.03, k=10,
                    representation="variable_width"),
        DatasetSpec("dev_variable_width", "variable_width", 320, 30000, 6, 0.055,
                    seed=5, missing_frac=0.03, k=5,
                    representation="variable_width"),
    ], math_key, "dev")


def big_specs(math_key: bytes) -> list[DatasetSpec]:
    """Large datasets where I/O dominates -- this is where the systems tricks pay off and
    the speed axis becomes real. Sized for a ~1 GB+ plain-text VCF."""
    return _keyed_specs([
        DatasetSpec("big_continental", "continental", 800, 400000, 6, 0.07, seed=11, k=12),
        DatasetSpec("big_subtle", "subtle", 800, 500000, 3, 0.005, seed=12, weight=1.5, k=16),
    ], math_key, "big")


def grade_specs(math_key: bytes) -> list[DatasetSpec]:
    """The hidden grading suite: balanced categories, a couple of large datasets so the speed axis
    is real.

    One private release key fixes the mathematics across submissions while keeping it impossible
    to reconstruct from this public generator. Fresh representation keys independently randomize
    exact PCA symmetries without adding score noise."""
    high_k = 36
    very_high_k = 64

    return _keyed_specs([
        # Marker-heavy (500k, ~1 GB) so the achievable optimum must SUBSAMPLE: at 120k the reference
        # read the whole file (only the parallel-pipeline speedup separated it from a full scan), but
        # at 500k it reads ~50% and gains ~5x headroom while still recovering all 5 continental axes at
        # |acc| 0.9999 (measured on r5.2xlarge) -- strong structure (fst 0.06 >= 0.05) and low rank
        # (k=5 <= 8) put it squarely in the regime where subsampling provably preserves every axis, so
        # it passes test_every_grade_fold_has_a_fair_fast_reference_speed_ceiling. A full-scanning
        # frontier is now capped near time_quality ~0.15 here (was ~0.25) while a demigod that
        # subsamples still reaches 1.0 -- the same fair-and-hard lever as io_wide, on a second shape.
        DatasetSpec("grade_continental_a", "continental", 500, 500000, 6, 0.06, weight=1.8, k=5),
        DatasetSpec("grade_continental_b", "continental", 400, 250000, 5, 0.08, weight=1.8, k=12),
        DatasetSpec("grade_subtle_a", "subtle", 500, 150000, 3, 0.005, weight=1.8, k=8),
        DatasetSpec("grade_subtle_b", "subtle", 400, 200000, 4, 0.004, weight=1.8, k=16),
        DatasetSpec("grade_admixed", "admixed", 500, 120000, 4, 0.05, admixed=True, k=3),
        DatasetSpec("grade_messy", "messy", 500, 120000, 5, 0.06,
                    messy_frac=0.12, missing_frac=0.03, k=10),
        DatasetSpec("grade_scaling", "scaling", 800, 300000, 6, 0.06, weight=1.8, k=6),
        DatasetSpec(
            "grade_high_rank", "high_rank", 392, 80000, high_k + 1, 0.10,
            weight=0.9, k=high_k, regime="high_rank",
        ),
        # A second, materially deeper requested subspace is intentionally separate from the
        # ordinary high-rank arm. It catches fixed-width output, hard-coded ten-PC paths, and
        # algorithms whose cost or accuracy collapses only after dozens of structured axes,
        # without requiring any particular eigensolver or signed eigenvector convention.
        DatasetSpec(
            "grade_very_high_rank", "very_high_rank", 576, 90000,
            very_high_k + 1, 0.12, weight=1.4, k=very_high_k,
            regime="high_rank",
        ),
        DatasetSpec(
            "grade_ill_conditioned", "ill_conditioned", 320, 60000, 17, 0.08,
            weight=0.7, k=16, regime="ill_conditioned",
        ),
        DatasetSpec(
            "grade_spectral_selection", "spectral_selection", 456, 70000, 19, 0.40,
            weight=1.0, k=12, regime="spectral_selection",
        ),
        DatasetSpec(
            "grade_spatial_ld", "spatial_ld", 384, 64000, 9, 0.0,
            weight=0.9, k=8, regime="spatial_ld",
        ),
        DatasetSpec(
            "grade_rare_structure", "rare_structure", 360, 60000, 8, 0.018,
            weight=0.9, k=7, regime="rare_structure",
        ),
        # variable_width has little speed headroom, but remains a full-weight parser capability arm:
        # the final malformed-prefix probe makes the exact faulty rollout fail mastery (factor 0.706)
        # while gold stays at 1.0. Haploid is also near a full scan's optimum and showed only modest
        # frontier variance, so its continuous-score weight is slightly below the 1.0 default.
        DatasetSpec(
            "grade_variable_width", "variable_width", 360, 50000, 6, 0.055,
            weight=0.6, k=5, missing_frac=0.03,
            representation="variable_width",
        ),
        DatasetSpec(
            "grade_haploid", "haploid", 320, 30000, 9, 0.08,
            weight=0.4, k=8, regime="haploid_structure",
            representation="haploid_gt",
        ),
        # Deliberately marker-heavy (1.5M SNVs, ~625 MB): far above the fast reference's ~200k
        # marker budget, so the achievable optimum READS ONLY ~13% OF THE FILE and the systems
        # ceiling demands genuine subsampling, not just a parallel full scan. Measured on r5.2xlarge:
        # full scan ~6 s vs fast ~0.65 s (~9x headroom) while fast still recovers all 8 structured
        # axes at |acc| 0.999 -- so a full-scanning frontier (Opus never subsamples) is capped near
        # parity here while a demigod that subsamples reaches 1.0. k=8 keeps it low-rank, so the
        # structured axes provably survive uniform subsampling (test_every_grade_fold_has_a_fair_
        # fast_reference_speed_ceiling); the larger marker count also lifts this fold's structure
        # spike well clear of the noise edge, retiring its previously thin detection margin.
        DatasetSpec(
            "grade_io_wide", "io_wide", 96, 1_500_000, 9, 0.0,
            weight=2.5, k=8, regime="spatial_ld",
        ),
        # Adapted from the useful shape axis in efficient_pca's Tall benchmark, but with a
        # biological high-rank Patterson target rather than iid dense numbers. This makes an
        # always-materialize-n_samples^2 implementation measurably weaker without dictating an
        # implementation or hard-failing a correct sample-Gram fit.
        DatasetSpec(
            "grade_sample_heavy", "sample_heavy", 4096, 1600, 9, 0.10,
            weight=2.5, k=8, regime="high_rank",
        ),
        # Neither side of this design is asymptotically negligible. The arm exercises the
        # algorithmic crossover between sample-space and marker-space methods while scoring only
        # the common Patterson subspace, so either correct regime choice remains valid.
        DatasetSpec(
            "grade_crossover", "crossover", 2048, 2048, 33, 0.11,
            weight=0.8, k=32, regime="high_rank",
        ),
        # Biobank regime: a SCALE LADDER, far more samples than markers, where materializing the
        # sample Gram (N x N) grows out of memory as N rises. The marker-space decomposition of the
        # same Patterson object needs ~1 GB at every rung, so the *right* answer sails through all
        # three; a sample-space answer is admitted or killed by the 14 GB resident cap according to
        # how disciplined it is. That turns memory from one all-or-nothing cliff into a continuous
        # axis -- credit grows with how far a submission's chosen intermediates scale:
        #
        #   rung        n_samples   float64 N^2 Gram   float32 N^2 Gram   who passes
        #   biobank_a      24000        4.6 GB            2.3 GB          any sample Gram (f32/f64) + marker
        #   biobank_b      50000       20.0 GB           10.0 GB          only float32 sample Gram + marker
        #   biobank        64000       32.8 GB           16.4 GB          only marker space
        #
        # So a float64-Gram fit banks ~1/3 of the ladder, a float32-Gram fit ~2/3, and the
        # marker-space fit 3/3 -- a graded reward for scale discipline instead of a single 0.
        # Structure stays comfortably recoverable at every rung because detectability scales as
        # n_variants * fst / n_pops (33 here) and is independent of n_samples: the signal spike and
        # the noise edge both grow linearly in the sample count, so their ratio does not move. Each
        # VCF is small (24k-64k samples x 2000 markers, ~0.2-0.5 GB), so the ladder is cheap to
        # generate and grade. Nothing is a trick -- an agent that reasons about the shape of its own
        # intermediates picks marker space and passes every rung.
        DatasetSpec(
            "grade_biobank_a", "biobank_a", 24000, 2000, 6, 0.10,
            weight=0.8, k=5, regime="high_rank",
        ),
        DatasetSpec(
            "grade_biobank_b", "biobank_b", 50000, 2000, 6, 0.10,
            weight=1.6, k=5, regime="high_rank",
        ),
        DatasetSpec(
            "grade_biobank", "biobank", 64000, 2000, 6, 0.10,
            weight=2.5, k=5, regime="high_rank",
        ),
    ], math_key, "grade")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/generated")
    ap.add_argument("--suite", choices=["dev", "big", "grade"], default="dev")
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--math-key-file", required=True)
    ap.add_argument("--representation-key-file", required=True)
    args = ap.parse_args()
    math_key = read_math_key(args.math_key_file)
    representation_key = Path(args.representation_key_file).read_bytes()
    if len(representation_key) != 32:
        raise ValueError("representation key file must contain exactly 32 bytes")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {"dev": dev_specs, "big": big_specs, "grade": grade_specs}[args.suite](math_key)
    commitment = key_commitment(math_key)
    for spec in specs:
        ext = ".vcf.gz" if args.gzip else ".vcf"
        path = out / (spec.name + ext)
        info = write_vcf(
            spec,
            path,
            representation_key=representation_key,
            math_commitment=commitment,
        )
        print(f"{spec.name:20s} {info['size_mb']:8.1f} MB  {info['clean_snv']:>8} clean SNV  "
              f"{info['seconds']:.2f}s")


if __name__ == "__main__":
    main()
