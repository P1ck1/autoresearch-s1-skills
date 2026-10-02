# Private Harbor verifier image. Unlike the solver image, this image contains the sealed grader,
# sandbox tooling, and checksum-pinned observed chromosome arm. Harbor builds it from tests/ and
# only copies the completed /workspace/solution artifact into it after the agent exits.
FROM python:3.12.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/hyperfocal/pcabench \
    && /opt/hyperfocal/pcabench/bin/pip install --no-cache-dir \
        "numpy==2.2.6" \
        "scipy==1.15.3"

RUN useradd --system --no-create-home --shell /usr/sbin/nologin pcasub

# Build the static filesystem seen by each untrusted invocation. It contains only the pinned
# runtime and ordinary system libraries: no grader, truth, datasets, procfs, sysfs, or verifier
# paths. chroot needs no nested namespace capability and works inside Harbor's ordinary Docker
# verifier; the child then loses every uid/group/capability via setpriv.
RUN set -eux; \
    root=/opt/hyperfocal/pcabench-sandbox-root; \
    install -d -m 0555 -o root -g root "$root"; \
    cp -a /usr "$root/usr"; \
    install -d -m 0555 -o root -g root "$root/opt" "$root/opt/hyperfocal"; \
    cp -a /opt/hyperfocal/pcabench "$root/opt/hyperfocal/pcabench"; \
    ln -s usr/bin "$root/bin"; \
    ln -s usr/lib "$root/lib"; \
    ln -s usr/lib64 "$root/lib64"; \
    ln -s usr/sbin "$root/sbin"; \
    install -d -m 0555 -o root -g root "$root/etc" "$root/dev" "$root/run" "$root/var"; \
    cp -a /etc/ld.so.cache /etc/passwd /etc/group /etc/nsswitch.conf "$root/etc/"; \
    install -d -m 1777 -o root -g root \
      "$root/tmp" "$root/var/tmp" "$root/run/lock" "$root/dev/shm" "$root/dev/mqueue"; \
    install -d -m 0711 -o root -g root "$root/work"; \
    mknod -m 0666 "$root/dev/null" c 1 3; \
    mknod -m 0666 "$root/dev/zero" c 1 5; \
    mknod -m 0666 "$root/dev/random" c 1 8; \
    mknod -m 0666 "$root/dev/urandom" c 1 9

# Keep the large checksum-pinned download/derivation layer independent of routine grader edits.
# Only the reviewed derivation code and one private benchmark-release key participate in this
# cache key. The key exists only in the sealed verifier image, never the separate agent image.
COPY grader_pkg/data/prepare_1kg.py grader_pkg/data/release_key.py /opt/hyperfocal/private-data/data/
COPY grader_pkg/grader/private/release-v3.math-key /opt/hyperfocal/pcabench-secrets/release-v3.math-key
RUN chown -R root:root /opt/hyperfocal/private-data /opt/hyperfocal/pcabench-secrets \
    && chmod 0700 /opt/hyperfocal/private-data /opt/hyperfocal/pcabench-secrets \
    && chmod 0400 /opt/hyperfocal/pcabench-secrets/release-v3.math-key

# Derive a deterministic balanced 500-sample, genome-spanning cohort directly from the official
# phase-3 chr22 source. The compressed source and panel are checksum-pinned and removed after the
# reviewed VCF/truth pair is built, keeping the runtime image leaner.
RUN set -eux; \
    install -d -m 0700 -o root -g root /opt/hyperfocal/pcabench-data; \
    /opt/hyperfocal/pcabench/bin/python -c \
      'import secrets; open("/tmp/observed.key", "xb").write(secrets.token_bytes(32))'; \
    curl --proto '=https' --tlsv1.2 -fL --retry 5 --retry-all-errors \
      -o /tmp/chr22.vcf.gz \
      https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz; \
    test "$(stat -c %s /tmp/chr22.vcf.gz)" -eq 205612353; \
    echo 'a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5  /tmp/chr22.vcf.gz' | sha256sum -c -; \
    curl --proto '=https' --tlsv1.2 -fL --retry 5 --retry-all-errors \
      -o /tmp/1kg.panel \
      https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel; \
    echo 'b4023dc6ee2d62ee89c8d4d347db4d348e65518d66d346574cdae7a4bbd76858  /tmp/1kg.panel' | sha256sum -c -; \
    PYTHONPATH=/opt/hyperfocal/private-data \
    /opt/hyperfocal/pcabench/bin/python /opt/hyperfocal/private-data/data/prepare_1kg.py \
      /tmp/chr22.vcf.gz /tmp/1kg.panel /opt/hyperfocal/pcabench-data/chr22.vcf \
      --name observed_population_structure --category observed --weight 2.0 --k 8 \
      --samples-per-superpop 100 --record-modulus 7 \
      --math-key-file /opt/hyperfocal/pcabench-secrets/release-v3.math-key \
      --representation-key-file /tmp/observed.key \
      --source-sha256 a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5; \
    rm -f /tmp/chr22.vcf.gz /tmp/1kg.panel /tmp/observed.key; \
    chown root:root /opt/hyperfocal/pcabench-data/chr22.vcf*; \
    chmod 0444 /opt/hyperfocal/pcabench-data/chr22.vcf; \
    chmod 0400 /opt/hyperfocal/pcabench-data/chr22.vcf.truth.json

RUN /opt/hyperfocal/pcabench/bin/python -c \
  'import hashlib,json,pathlib; v=pathlib.Path("/opt/hyperfocal/pcabench-data/chr22.vcf"); t=json.load(open(str(v)+".truth.json")); key=bytes.fromhex(open("/opt/hyperfocal/pcabench-secrets/release-v3.math-key").read().strip()); commitment=hashlib.sha256(b"pcabench-eval-v3\0"+key).hexdigest(); assert 100_000_000 <= v.stat().st_size <= 2_000_000_000; assert t["spec"]["category"] == "observed" and t["spec"]["k"] == 8 and t["spec"]["derivation_version"] == "observed-chr22-v5" and t["spec"]["math_release"] == "pcabench-eval-v3" and t["spec"]["math_key_commitment"] == commitment and t["spec"]["derived_bytes"] == v.stat().st_size and t["spec"]["n_records"] >= 50_000; assert t["n_samples"] == len(t["sample_ids"]) == len(t["sample_pop"]) == len(t["sample_subpop"]) == 500; assert all(len(s) == 21 and s.startswith("p") for s in t["sample_ids"]); assert not any(s.startswith(("HG", "NA")) for s in t["sample_ids"]); assert len(t["population_names"]) >= 20'

COPY . /tests
RUN chown -R root:root /tests \
    && chmod -R go-rwx /tests \
    && chmod 0700 /tests/test.sh \
    && install -d -m 0555 -o root -g root /opt/hyperfocal/pcabench-guard \
    && install -m 0444 -o root -g root \
      /tests/grader_pkg/grader/submission_runner.py \
      /opt/hyperfocal/pcabench-guard/submission_runner.py \
    && install -d -m 0555 -o root -g root \
      /opt/hyperfocal/pcabench-sandbox-root/opt/hyperfocal/pcabench-guard \
    && install -m 0444 -o root -g root \
      /tests/grader_pkg/grader/submission_runner.py \
      /opt/hyperfocal/pcabench-sandbox-root/opt/hyperfocal/pcabench-guard/submission_runner.py \
    && rm -rf /opt/hyperfocal/private-data

# Each invocation receives a fresh pcasub-owned scratch directory. Keep the global temporary
# directory root-only for writes so it cannot become a cross-invocation communication channel.
RUN chmod 0755 /tmp

RUN mkdir -p /workspace/solution /logs/verifier /logs/agent
WORKDIR /workspace/solution

CMD ["/bin/bash"]
