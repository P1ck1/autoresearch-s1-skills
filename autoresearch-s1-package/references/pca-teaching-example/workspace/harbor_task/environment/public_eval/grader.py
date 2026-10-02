#!/usr/bin/env python3
"""Transparent local Public/Dev feedback. This is not the final Hidden verifier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACCURACY_THRESHOLD = 0.98
TIMEOUT_SECONDS = 30
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class DevFailure(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def read_scores(path: Path, expected_ids: list[str], k: int) -> np.ndarray:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_OUTPUT_BYTES:
        raise DevFailure("invalid_output", "Expected a regular output TSV of at most 64 MiB.")
    try:
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
    except csv.Error as exc:
        raise DevFailure("invalid_output", "Output is not a valid bounded-field TSV.") from exc
    if not rows or rows[0] != ["sample_id"] + [f"PC{i+1}" for i in range(k)]:
        raise DevFailure("invalid_output", "Header must be sample_id, PC1, ..., PCk (tab-separated).")
    if len(rows) != len(expected_ids) + 1:
        raise DevFailure("invalid_output", "Output must contain exactly one row per input sample.")
    if any(len(row) != k + 1 for row in rows[1:]):
        raise DevFailure("invalid_output", "Each output row must contain exactly k PC values.")
    if [row[0] for row in rows[1:]] != expected_ids:
        raise DevFailure("invalid_output", "Sample IDs/order must match the VCF header.")
    try:
        scores = np.asarray([[float(x) for x in row[1:]] for row in rows[1:]], dtype=np.float64)
    except ValueError as exc:
        raise DevFailure("invalid_output", "PC values must be numeric.") from exc
    if not np.isfinite(scores).all():
        raise DevFailure("invalid_output", "PC values must be finite.")
    return scores


def subspace_similarity(actual: np.ndarray, expected: np.ndarray) -> float:
    k = actual.shape[1]
    bases = []
    for values in (actual, expected):
        # Centering removes offsets; column normalization respects arbitrary PC scale.
        max_abs = np.max(np.abs(values), axis=0)
        if np.any(max_abs == 0):
            raise DevFailure("quality_gate_failed", "Output has a zero-variance PC.")
        normalized = values / max_abs
        centered = normalized - normalized.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(centered, axis=0)
        if np.any(scale <= np.finfo(np.float64).tiny):
            raise DevFailure("quality_gate_failed", "Output has a zero-variance PC.")
        left, singular, _ = np.linalg.svd(centered / scale, full_matrices=False)
        if singular[-1] <= singular[0] * 1e-10:
            raise DevFailure("quality_gate_failed", "Output PC columns are rank-deficient.")
        bases.append(left[:, :k])
    # Mean squared cosine of principal angles, invariant to sign/rotation/scale.
    value = np.linalg.norm(bases[0].T @ bases[1], "fro") ** 2 / k
    return float(min(1.0, max(0.0, value)))


def invoke(program: Path, vcf: Path, k: int, output: Path, scratch: Path) -> float:
    output.write_text("")  # The CLI must overwrite the exact existing output path.
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(scratch), "TMPDIR": str(scratch),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    start = time.perf_counter()
    with (scratch / "process.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", str(program), str(vcf), str(k), str(output)],
            cwd=program.parent, env=env, stdout=log, stderr=log, start_new_session=True,
        )
        timed_out = threading.Event()

        def terminate() -> None:
            if process.poll() is None:
                timed_out.set()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        timer = threading.Timer(TIMEOUT_SECONDS, terminate)
        timer.daemon = True
        timer.start()
        try:
            code = process.wait()
        finally:
            timer.cancel()
        if timed_out.is_set():
            raise DevFailure("timeout", f"Public Dev invocation exceeded {TIMEOUT_SECONDS}s.")
        elapsed = time.perf_counter() - start
        if code != 0:
            log.seek(0)
            detail = log.read(2000)
            raise DevFailure("execution_error", f"pca exited {code}; diagnostic: {detail}")
    return elapsed


def evaluate(submission: Path, repeats: int) -> dict:
    candidate = submission.resolve() / "pca"
    baseline = ROOT / "starter" / "pca"
    if not candidate.is_file() or candidate.is_symlink():
        raise DevFailure("invalid_submission", "Submission must contain a regular Python entry file named pca.")
    with candidate.open("rb") as handle:
        first_line = handle.readline(1024)
    if not first_line.startswith(b"#!") or b"python" not in first_line.lower():
        raise DevFailure("invalid_submission", "The pca entry must start with a Python shebang.")
    asset_root = ROOT / "public_assets" / "dev"
    manifest = json.loads((asset_root / "manifest.json").read_text())
    results = []
    for case in manifest["cases"]:
        vcf = asset_root / case["path"]
        if hashlib.sha256(vcf.read_bytes()).hexdigest() != case["sha256"]:
            raise DevFailure("evaluator_error", f"Frozen public fixture hash differs: {case['id']}.")
        with vcf.open() as handle:
            expected_ids = next(line.rstrip("\r\n").split("\t")[9:] for line in handle if line.startswith("#CHROM\t"))
        baseline_times, candidate_times, similarities = [], [], []
        with tempfile.TemporaryDirectory(prefix="pca-dev-") as temp:
            temp_root = Path(temp)
            for repetition in range(repeats):
                baseline_dir = temp_root / f"baseline-{repetition}"
                candidate_dir = temp_root / f"candidate-{repetition}"
                baseline_dir.mkdir()
                candidate_dir.mkdir()
                expected_file = baseline_dir / "scores.tsv"
                actual_file = candidate_dir / "scores.tsv"
                try:
                    baseline_times.append(invoke(baseline, vcf, case["k"], expected_file, baseline_dir))
                    expected = read_scores(expected_file, expected_ids, case["k"])
                except DevFailure as exc:
                    raise DevFailure("evaluator_error", f"Frozen baseline failed: {exc}") from exc
                candidate_times.append(invoke(candidate, vcf, case["k"], actual_file, candidate_dir))
                actual = read_scores(actual_file, expected_ids, case["k"])
                similarities.append(subspace_similarity(actual, expected))
        t_baseline = statistics.median(baseline_times)
        t_candidate = statistics.median(candidate_times)
        results.append({
            "case": case["id"], "public_seed": case["public_seed"],
            "fixture_sha256": case["sha256"], "k": case["k"],
            "min_subspace_similarity": min(similarities),
            "quality_gate_passed": min(similarities) >= ACCURACY_THRESHOLD,
            "baseline_seconds": baseline_times, "candidate_seconds": candidate_times,
            "median_speedup": t_baseline / t_candidate,
            "log2_speedup": math.log2(t_baseline / t_candidate),
        })
    passed = all(case["quality_gate_passed"] for case in results)
    return {
        "protocol": "pca-public-dev.v1", "split": "public_dev",
        "status": "ok" if passed else "quality_gate_failed",
        "score": statistics.mean(case["log2_speedup"] for case in results) if passed else None,
        "score_name": "mean_log2_dev_speedup", "direction": "maximize",
        "quality_gate": {"min_subspace_similarity": ACCURACY_THRESHOLD, "all_cases_passed": passed},
        "repeats": repeats, "cases": results,
        "scope": "Public timing/accuracy proxy only; not Hidden reward, security acceptance, or G03 evidence.",
        "timing_note": "Baseline then candidate; process startup and warm-cache/order effects are included. Small-fixture timings fluctuate.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("/workspace/solution"))
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    try:
        result = evaluate(args.submission, args.repeats)
    except DevFailure as exc:
        result = {"protocol": "pca-public-dev.v1", "split": "public_dev", "status": exc.status, "score": None, "error": str(exc)}
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        result = {"protocol": "pca-public-dev.v1", "split": "public_dev", "status": "evaluator_error", "score": None, "error": str(exc)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
