"""Build the reviewed real-data arm from the official phase-3 chromosome VCF.

The source callset is too large for an exact per-submission reference fit at its
full 2,504-sample width. Provisioning therefore creates one versioned, balanced
cohort and a genome-spanning record sample from the pinned source.
Retained GT calls remain the real chromosome data. Public identifiers, source
metadata, positions, and auxiliary FORMAT fields are replaced with a generic
PCA-equivalent representation; a per-build representation key privately permutes
and aliases sample columns so a submission cannot join the observed fold to a
published population panel.  The representation key never changes the mathematical
dataset, which keeps repeated evaluations deterministic.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path

from data.release_key import RELEASE_ID, derive_bytes, key_commitment, read_math_key


SUPERPOPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
DERIVATION_VERSION = "observed-chr22-v5"


@dataclass(frozen=True)
class PanelLabel:
    population: str
    superpopulation: str


def _open(path: str):
    return gzip.open(path, "rt", newline="") if path.endswith((".gz", ".bgz")) \
        else open(path, "rt", newline="")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


def read_vcf_samples(vcf: str) -> list[str]:
    with _open(vcf) as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                return line.rstrip("\r\n").split("\t")[9:]
    raise ValueError("no #CHROM header")


def read_panel(panel: str) -> dict[str, PanelLabel]:
    """Return the reviewed population hierarchy for every official sample."""
    mapping: dict[str, PanelLabel] = {}
    with open(panel, newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        try:
            sample_index = header.index("sample")
            subpopulation_index = header.index("pop")
            population_index = header.index("super_pop")
        except ValueError as error:
            raise ValueError("panel must name sample and super_pop columns") from error
        for line in fh:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) <= max(sample_index, subpopulation_index, population_index):
                raise ValueError("malformed panel row")
            sample = fields[sample_index]
            subpopulation = fields[subpopulation_index]
            superpop = fields[population_index]
            if not subpopulation:
                raise ValueError("empty panel population")
            if superpop not in SUPERPOPS:
                raise ValueError(f"unexpected super-population {superpop!r}")
            if sample in mapping:
                raise ValueError(f"duplicate panel sample {sample!r}")
            mapping[sample] = PanelLabel(subpopulation, superpop)
    population_parents: dict[str, str] = {}
    for label in mapping.values():
        previous = population_parents.setdefault(label.population, label.superpopulation)
        if previous != label.superpopulation:
            raise ValueError(f"population {label.population!r} spans super-populations")
    return mapping


def _private_rank(key: bytes, domain: bytes, value: str) -> bytes:
    if len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    return hmac.digest(key, domain + b"\0" + value.encode(), "sha256")


def select_balanced_samples(samples: list[str], panel: dict[str, PanelLabel],
                            per_superpop: int, cohort_key: bytes,
                            representation_key: bytes) -> list[str]:
    """Select a stable hierarchy-balanced cohort, then privately permute its columns."""
    if per_superpop < 2:
        raise ValueError("per_superpop must be at least two")
    missing = [sample for sample in samples if sample not in panel]
    if missing:
        raise ValueError(
            f"{len(missing)} VCF samples are absent from the reviewed panel: {missing[:3]}"
        )
    selected: set[str] = set()
    for superpop in SUPERPOPS:
        populations = sorted({
            panel[sample].population for sample in samples
            if panel[sample].superpopulation == superpop
        })
        if not populations:
            raise ValueError(f"no reviewed populations for {superpop}")
        if per_superpop < len(populations):
            raise ValueError(f"cohort is too small to represent every {superpop} population")
        base, remainder = divmod(per_superpop, len(populations))
        candidates = [
            sample for sample in samples if panel[sample].superpopulation == superpop
        ]
        if len(candidates) < per_superpop:
            raise ValueError(f"not enough {superpop} samples for the reviewed cohort")
        for index, population in enumerate(populations):
            quota = base + int(index < remainder)
            population_candidates = [
                sample for sample in candidates if panel[sample].population == population
            ]
            if len(population_candidates) < quota:
                raise ValueError(
                    f"not enough {population} samples for the balanced {superpop} cohort"
                )
            ranked = sorted(
                population_candidates,
                key=lambda sample: _private_rank(
                    cohort_key,
                    f"cohort\0{superpop}\0{population}".encode(),
                    sample,
                ),
            )
            selected.update(ranked[:quota])
    return sorted(
        selected,
        key=lambda sample: _private_rank(representation_key, b"column-order", sample),
    )


def _record_selected(line: str, modulus: int, record_key: bytes) -> bool:
    """Deterministically sample records across the whole chromosome.

    The decision depends only on the five identifying VCF columns, not line
    width, FORMAT payload, allele frequency, or genomic region.
    """
    key_fields = line.split("\t", 5)
    if len(key_fields) < 5:
        raise ValueError("malformed VCF record")
    digest = hashlib.blake2s(
        "\t".join(key_fields[:5]).encode(), key=record_key, digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % modulus == 0


def _private_header(sample_aliases: list[str]) -> str:
    lines = [
        "##fileformat=VCFv4.2",
        "##source=pcabench_generate",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    ]
    lines.extend(f"##contig=<ID=chr{chromosome}>" for chromosome in range(1, 23))
    fixed = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    lines.append("\t".join(fixed + sample_aliases))
    return "\n".join(lines) + "\n"


def write_reviewed_subset(source_vcf: str, panel_path: str, output_vcf: str, *,
                          per_superpop: int, record_modulus: int,
                          cohort_key: bytes, record_key: bytes,
                          representation_key: bytes,
                          minimum_records: int = 50_000) -> tuple[dict, dict[str, PanelLabel]]:
    """Stream a stable balanced subset through a privately keyed representation."""
    if record_modulus < 2:
        raise ValueError("record_modulus must be at least two")
    if minimum_records < 1:
        raise ValueError("minimum_records must be positive")
    panel = read_panel(panel_path)
    source_samples = read_vcf_samples(source_vcf)
    selected_samples = select_balanced_samples(
        source_samples, panel, per_superpop, cohort_key, representation_key,
    )
    source_columns = {sample: index for index, sample in enumerate(source_samples)}
    selected_columns = [source_columns[sample] for sample in selected_samples]
    aliases = {
        sample: "p" + _private_rank(
            representation_key, b"sample-alias", sample,
        ).hex()[:20]
        for sample in selected_samples
    }
    if len(set(aliases.values())) != len(aliases):
        raise RuntimeError("private sample aliases collided")
    alias_labels = {aliases[sample]: panel[sample] for sample in selected_samples}

    output = Path(output_vcf)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    retained = 0
    try:
        with _open(source_vcf) as source, open(temporary, "wt", newline="\n") as target:
            header_seen = False
            for line in source:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    fields = line.rstrip("\r\n").split("\t")
                    if fields[9:] != source_samples:
                        raise ValueError("VCF sample header changed while reading")
                    target.write(_private_header([aliases[sample] for sample in selected_samples]))
                    header_seen = True
                    continue
                if not header_seen:
                    raise ValueError("VCF record precedes #CHROM header")
                record = line.rstrip("\r\n")
                if not record or not _record_selected(record, record_modulus, record_key):
                    continue
                fields = record.split("\t")
                if len(fields) != 9 + len(source_samples):
                    raise ValueError("VCF record has the wrong sample-column count")
                calls = [fields[9 + index].split(":", 1)[0] for index in selected_columns]
                # PCA ignores genomic coordinates, IDs, INFO, and non-GT FORMAT payloads. Replace
                # those public fingerprints while retaining a valid, sorted chr22 VCF and every
                # selected real genotype call exactly.
                opaque_position = 10_000 + retained * 10
                fixed = [
                    "chr22", str(opaque_position), ".", fields[3], fields[4],
                    ".", "PASS", ".", "GT",
                ]
                target.write("\t".join(fixed + calls) + "\n")
                retained += 1
        if retained < minimum_records:
            raise ValueError(f"reviewed record sample is unexpectedly small: {retained}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hashlib.sha256()
    with open(output, "rb") as fh:
        while chunk := fh.read(8 << 20):
            digest.update(chunk)
    metadata = {
        "sample_ids": [aliases[sample] for sample in selected_samples],
        "records": retained,
        "bytes": output.stat().st_size,
        "derived_sha256": digest.hexdigest(),
    }
    return metadata, alias_labels


def build_truth(vcf: str, sample_labels: dict[str, PanelLabel], *, name: str, category: str,
                weight: float, k: int, source_sha256: str,
                math_commitment: str,
                derived_sha256: str | None = None,
                derived_bytes: int | None = None,
                n_records: int | None = None) -> dict:
    _validate_sha256(source_sha256)
    if derived_sha256 is not None:
        _validate_sha256(derived_sha256)
    samples = read_vcf_samples(vcf)
    missing = [sample for sample in samples if sample not in sample_labels]
    if missing:
        raise ValueError(
            f"{len(missing)} VCF samples are absent from the reviewed panel: {missing[:3]}"
        )
    if set(sample_labels) != set(samples):
        raise ValueError("reviewed label map contains samples absent from the VCF")
    superpops = sorted({sample_labels[sample].superpopulation for sample in samples})
    if tuple(superpops) != SUPERPOPS:
        raise ValueError(f"unexpected super-population catalog: {superpops}")
    code = {superpop: index for index, superpop in enumerate(superpops)}
    sample_pop = [code[sample_labels[sample].superpopulation] for sample in samples]
    populations = sorted({sample_labels[sample].population for sample in samples})
    population_code = {population: index for index, population in enumerate(populations)}
    sample_subpop = [population_code[sample_labels[sample].population] for sample in samples]
    spec = {
        "name": name,
        "category": category,
        "weight": weight,
        "k": k,
        "structure_weight": 0.30,
        "hierarchy_weight": 0.45,
        "source": "1000genomes_phase3_grch37",
        "source_sha256": source_sha256,
        "derivation_version": DERIVATION_VERSION,
        "math_release": RELEASE_ID,
        "math_key_commitment": math_commitment,
    }
    if derived_sha256 is not None:
        spec["derived_sha256"] = derived_sha256
    if derived_bytes is not None:
        if derived_bytes < 1:
            raise ValueError("derived_bytes must be positive")
        spec["derived_bytes"] = derived_bytes
    if n_records is not None:
        spec["n_records"] = n_records
    return {
        "spec": spec,
        "n_samples": len(samples),
        "sample_ids": samples,
        "sample_pop": sample_pop,
        "sample_subpop": sample_subpop,
        "superpop_names": superpops,
        "population_names": populations,
        "admix": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_vcf")
    parser.add_argument("panel")
    parser.add_argument("output_vcf")
    parser.add_argument("--name", default="observed_population_structure")
    parser.add_argument("--category", default="observed")
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--samples-per-superpop", type=int, default=100)
    parser.add_argument("--record-modulus", type=int, default=7)
    parser.add_argument("--math-key-file", required=True)
    parser.add_argument("--representation-key-file", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    if args.k < 1:
        raise ValueError("k must be positive")
    _validate_sha256(args.source_sha256)
    math_key = read_math_key(args.math_key_file)
    representation_key = Path(args.representation_key_file).read_bytes()
    if len(representation_key) != 32:
        raise ValueError("representation key file must contain exactly 32 bytes")
    cohort_key = derive_bytes(
        math_key, f"real/cohort/{DERIVATION_VERSION}/{args.source_sha256}",
    )
    record_key = derive_bytes(
        math_key, f"real/records/{DERIVATION_VERSION}/{args.source_sha256}",
    )
    result, sample_labels = write_reviewed_subset(
        args.source_vcf,
        args.panel,
        args.output_vcf,
        per_superpop=args.samples_per_superpop,
        record_modulus=args.record_modulus,
        cohort_key=cohort_key,
        record_key=record_key,
        representation_key=representation_key,
    )
    truth = build_truth(
        args.output_vcf,
        sample_labels,
        name=args.name,
        category=args.category,
        weight=args.weight,
        k=args.k,
        source_sha256=args.source_sha256,
        math_commitment=key_commitment(math_key),
        derived_sha256=result["derived_sha256"],
        derived_bytes=result["bytes"],
        n_records=result["records"],
    )
    truth_path = Path(args.output_vcf + ".truth.json")
    truth_path.write_text(json.dumps(truth))
    counts = {
        superpop: truth["sample_pop"].count(index)
        for index, superpop in enumerate(truth["superpop_names"])
    }
    print(
        f"{args.name}: {truth['n_samples']} samples, {result['records']} records, "
        f"{result['bytes']} bytes, super-populations {counts} -> {truth_path}"
    )


if __name__ == "__main__":
    main()
