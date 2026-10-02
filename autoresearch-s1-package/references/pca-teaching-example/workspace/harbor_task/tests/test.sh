#!/bin/bash
# Harbor verifier entrypoint. Generates the hidden grading suite, grades /workspace/solution
# through the real grading path, and writes reward.json + reward_detail.json.
#
# The submission is run privilege-separated by the grader; here we only orchestrate. The
# datasets are (re)generated from fixed seeds at grade time so nothing about them ships to the
# agent, and the byte layout the submission must exploit is real.
set -euo pipefail

VERIFIER_STARTED_EPOCH="$(date +%s)"
VERIFIER_OUTER_TIMEOUT_SECONDS=10800
POST_GRADE_RESERVE_SECONDS=300
MAX_GRADER_BUDGET_SECONDS=8400

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="${TESTS_DIR}/grader_pkg"
WORK="$(mktemp -d /tmp/pcabench_grade.XXXXXX)"
DATA="${WORK}/data"
OUT="/logs/verifier"
trap 'rm -rf -- "${WORK}"' EXIT
mkdir -p "${DATA}" "${OUT}"
rm -f "${OUT}/reward.json" "${OUT}/reward.txt" "${OUT}/reward_detail.json"

# Run the grader from the root-owned tests dir, NOT the agent-writable WORKDIR (/workspace/solution).
# With ``python -m grader.grade``/``-m data.generate`` the interpreter searches the CURRENT
# DIRECTORY (sys.path[0]) BEFORE PYTHONPATH, so an agent that plants ``grader/grade.py`` or
# ``data/generate.py`` in the cwd during solve would hijack the grader / dataset generation and run
# it AS ROOT (the verifier is root here, before it drops the submission uid). ``cd`` moves cwd to a
# root-owned dir and ``PYTHONSAFEPATH=1`` drops the cwd entry from sys.path entirely, so the two
# ``-m`` calls resolve only from PYTHONPATH (the sealed grader_pkg). Belt and suspenders.
cd "${TESTS_DIR}"
export PYTHONSAFEPATH=1
export PYTHONPATH="${PKG}"
PY="/opt/hyperfocal/pcabench/bin/python"
MATH_KEY="/opt/hyperfocal/pcabench-secrets/release-v3.math-key"

# The verifier container itself is the outer filesystem/network boundary. Close every shared
# world-writable directory before the first dropped child starts, and make the imported artifact
# root-traversable only. The grader copies it once into its own immutable snapshot; the submission
# must not use its original artifact or a global tmpfs as a cross-invocation state channel.
chmod 0755 /tmp /var/tmp /run/lock /dev/shm /dev/mqueue
"${PY}" - <<'PY'
import os

path = "/workspace/solution"
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
descriptor = os.open(path, flags)
try:
    os.fchown(descriptor, 0, 0)
    os.fchmod(descriptor, 0o500)
finally:
    os.close(descriptor)
PY

echo "[test.sh] generating grading suite..."
REPRESENTATION_KEY="${WORK}/synthetic-representation.key"
"${PY}" - "${REPRESENTATION_KEY}" <<'PY'
import secrets
import sys

with open(sys.argv[1], "xb") as destination:
    destination.write(secrets.token_bytes(32))
PY
"${PY}" -m data.generate --suite grade --out-dir "${DATA}" \
    --math-key-file "${MATH_KEY}" \
    --representation-key-file "${REPRESENTATION_KEY}" || {
    echo "[test.sh] dataset generation failed"; exit 1; }
rm -f "${REPRESENTATION_KEY}"

# The separate verifier image privately bakes the checksum-pinned official chromosome arm. Stage
# it beside the synthetic suite with the same truth-sidecar contract used by Hyperfocal.
REAL_VCF=/opt/hyperfocal/pcabench-data/chr22.vcf
REAL_TRUTH=${REAL_VCF}.truth.json
test -f "${REAL_VCF}" -a -f "${REAL_TRUTH}" || {
    echo "[test.sh] observed chromosome bundle missing"; exit 1; }
if [ "$(stat -c %d "${REAL_VCF}")" = "$(stat -c %d "${DATA}")" ]; then
    ln "${REAL_VCF}" "${DATA}/grade_observed.vcf"
else
    install -m 0444 -o root -g root "${REAL_VCF}" "${DATA}/grade_observed.vcf"
fi
install -m 0400 -o root -g root "${REAL_TRUTH}" "${DATA}/grade_observed.vcf.truth.json"

"${PY}" - "${DATA}" <<'PY'
import hashlib
import json
import pathlib
import sys

data = pathlib.Path(sys.argv[1])
truth_paths = sorted(data.glob("*.truth.json"))
assert len(truth_paths) == 22, len(truth_paths)
truths = [json.load(open(path)) for path in truth_paths]
assert {truth["spec"]["category"] for truth in truths} == {
    "admixed", "biobank", "biobank_a", "biobank_b", "continental", "crossover", "haploid", "high_rank",
    "ill_conditioned", "io_wide", "messy",
    "observed", "rare_structure", "sample_heavy", "scaling", "spatial_ld", "subtle",
    "spectral_selection", "variable_width", "very_high_rank",
}
observed = next(truth for truth in truths if truth["spec"]["category"] == "observed")
vcf = data / "grade_observed.vcf"
digest = hashlib.sha256()
with vcf.open("rb") as source:
    while chunk := source.read(8 << 20):
        digest.update(chunk)
assert observed["spec"]["derived_sha256"] == digest.hexdigest()
assert observed["spec"]["derived_bytes"] == vcf.stat().st_size
assert observed["spec"]["n_records"] >= 50_000
assert observed["n_samples"] == 500
assert all(len(sample) == 21 and sample.startswith("p") for sample in observed["sample_ids"])
assert not any(sample.startswith(("HG", "NA")) for sample in observed["sample_ids"])
assert len(observed["population_names"]) >= 20
assert len(observed["sample_ids"]) == len(observed["sample_pop"]) == len(observed["sample_subpop"]) == 500
PY

# Truth sidecars hold the ground-truth population labels. The grader (root) reads them; the
# submission is always dropped to an unprivileged uid and must not. Hide the dataset
# directory itself; VCF inodes stay world-readable only so the trusted grader can stage a
# metadata-normalized, opaque input inode for the dropped child.
chmod 700 "${DATA}"
find "${DATA}" -name '*.truth.json' -exec chmod 600 {} + 2>/dev/null || true

# Seal the hidden grader/reference source (the EXACT scoring truth in grader_pkg/reference) from
# the submission uid BEFORE grading drops privileges. Otherwise a submission could run
# `PYTHONPATH=${PKG} python -m reference.full_scan_pca ...` -- importing only allowlisted os/subprocess,
# so the static library scan never fires -- and ace accuracy with the reference itself. Only
# meaningful as root (where the submission is dropped to an unprivileged uid); grade.py re-seals and
# then verifies the seal AS that uid, failing closed if the reference is still readable.
if [ "$(id -u)" = "0" ]; then
    chown -R 0:0 "${PKG}" 2>/dev/null || true
    chmod 700 "${PKG}"
fi

echo "[test.sh] grading /workspace/solution ..."
ELAPSED_SECONDS=$(( $(date +%s) - VERIFIER_STARTED_EPOCH ))
GRADER_BUDGET_SECONDS=$((
    VERIFIER_OUTER_TIMEOUT_SECONDS - ELAPSED_SECONDS - POST_GRADE_RESERVE_SECONDS
))
if [ "${GRADER_BUDGET_SECONDS}" -gt "${MAX_GRADER_BUDGET_SECONDS}" ]; then
    GRADER_BUDGET_SECONDS="${MAX_GRADER_BUDGET_SECONDS}"
fi
# Even after unexpectedly slow trusted data generation, invoke the grader with only the wall time
# that remains. A one-second budget zero-fills immediately; extending it beyond the verifier's
# real remainder could let the outer timeout erase the reward artifact we are protecting.
if [ "${GRADER_BUDGET_SECONDS}" -lt 1 ]; then
    GRADER_BUDGET_SECONDS=1
fi
"${PY}" -m grader.grade /workspace/solution \
    --data-dir "${DATA}" \
    --workdir "${WORK}/run" \
    --out "${OUT}/reward_detail.json" \
    --math-key-file "${MATH_KEY}" \
    --isolation verifier-container \
    --time-budget-seconds "${GRADER_BUDGET_SECONDS}"

# grade.py writes the full detail; extract the scalar reward for reward.json.
"${PY}" - "${OUT}/reward_detail.json" "${OUT}/reward.json" <<'PY'
import json
import math
import os
from pathlib import Path
import sys
import tempfile

with open(sys.argv[1]) as source:
    detail = json.load(source)
reward = detail["reward"]
if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
    raise ValueError("reward must be a finite JSON number")

output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=".reward.", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w") as destination:
        json.dump({"reward": reward}, destination, allow_nan=False)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, output)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
print(f"[test.sh] reward = {reward:.4f}")
PY
