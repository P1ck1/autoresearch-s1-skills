FROM python:3.12.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/hyperfocal/pcabench/bin:${PATH}"

COPY requirements.txt /tmp/pca-requirements.txt
RUN python -m venv /opt/hyperfocal/pcabench \
    && /opt/hyperfocal/pcabench/bin/pip install --only-binary=:all: -r /tmp/pca-requirements.txt \
    && rm /tmp/pca-requirements.txt \
    && groupadd --gid 1000 agent \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash agent

WORKDIR /workspace
COPY starter/ /workspace/starter/
COPY public_eval/ /workspace/public_eval/
COPY public_assets/ /workspace/public_assets/
COPY README.md /workspace/PUBLIC_DEV_README.md
RUN cp -R /workspace/starter /workspace/solution \
    && chown -R root:root /workspace/starter /workspace/public_eval /workspace/public_assets \
    && chmod -R a-w /workspace/starter /workspace/public_eval /workspace/public_assets \
    && chmod 755 /workspace/starter/pca \
    && chown -R agent:agent /workspace/solution \
    && chmod -R u+rwX /workspace/solution \
    && chmod 755 /workspace/solution/pca \
    && mkdir -p /expert_evidence /logs/agent \
    && chown agent:agent /expert_evidence /logs/agent \
    && chmod 644 /workspace/PUBLIC_DEV_README.md

# Runtime iteration uses the same public environment; final Hidden is separate.
USER agent
CMD ["/bin/bash"]
