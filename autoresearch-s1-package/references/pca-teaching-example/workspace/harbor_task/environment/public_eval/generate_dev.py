#!/usr/bin/env python3
"""Reproduce the small, deliberately public Dev fixtures; no private inputs are used."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CASES = [
    ("structured", 2026092901, 64, 768, 3),
    ("mixed_calls", 2026092902, 80, 960, 4),
    ("sample_heavy", 2026092903, 96, 40, 3),
]


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, seed, samples, variants, k in CASES:
        rng = np.random.default_rng(seed)
        groups = np.arange(samples) % 4
        global_p = rng.uniform(0.08, 0.85, (variants, 1))
        group_shift = rng.normal(0.0, 0.16, (variants, 4))
        probabilities = np.clip(global_p + group_shift[:, groups], 0.02, 0.98)
        dosage = rng.binomial(2, probabilities)
        missing = rng.random(dosage.shape) < 0.02
        ids = [f"dev_{name}_{i:03d}" for i in range(samples)]
        lines = [
            "##fileformat=VCFv4.2",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(ids),
        ]
        for j in range(variants):
            extra = name == "mixed_calls" and j % 3 == 0
            calls = []
            for i in range(samples):
                value = int(dosage[j, i])
                gt = ("0/0", "0/1", "1/1")[value]
                if (j + i) % 4 == 0:
                    gt = gt.replace("/", "|")
                if missing[j, i]:
                    gt = "./."
                elif name == "mixed_calls" and (j + 3 * i) % 71 == 0:
                    gt = "1" if value else "0"
                elif name == "mixed_calls" and (j + i) % 137 == 0:
                    gt = "0/."
                calls.append(gt + (":23:40" if extra else ""))
            fmt = "GT:DP:GQ" if extra else "GT"
            lines.append(f"1\t{j+1}\t.\tA\tG\t.\tPASS\t.\t{fmt}\t" + "\t".join(calls))
        # Public negative parsing examples: these records must not change the fit.
        for index, (ref, alt, gt) in enumerate([
            ("A", "AC", "0/1"), ("A", "G,T", "1/2"),
            ("A", "<DEL>", "0/1"), ("C", "T", "0/0"),
        ]):
            lines.append(f"1\t{variants+index+1}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join([gt]*samples))
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        filename = f"{name}.vcf"
        (destination / filename).write_bytes(payload)
        entries.append({
            "id": name, "path": filename, "public_seed": seed,
            "samples": samples, "generated_markers": variants,
            "extra_ineligible_or_monomorphic_records": 4, "k": k,
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = {
        "schema_version": "pca-public-dev.v1", "visibility": "public",
        "purpose": "Small workflow/accuracy fixtures; not Hidden data or research evidence.",
        "generator": "public_eval/generate_dev.py", "rng": "numpy.random.default_rng/PCG64",
        "generation_numpy_version": np.__version__,
        "cases": entries,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "public_assets" / "dev")
