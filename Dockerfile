ARG BASE_IMAGE=pytorch/pytorch@sha256:ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126
FROM ${BASE_IMAGE}
COPY runtime/requirements.txt /opt/w045/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/w045/requirements.txt
COPY runtime/app /opt/w045/app
COPY runtime/entrypoint_rg.py /opt/w045/entrypoint_rg.py
RUN python -c "import hashlib; p='/opt/w045/app/evaluation_v2.py'; assert hashlib.sha256(open(p,'rb').read()).hexdigest()=='b4af91a1c0eab7cc5935da5956905576daabe67b226bbac6eeb5537e9a9beb37', 'Obtain the matching official evaluation_v2.py first'"
RUN groupadd --gid 10000 worker && useradd --uid 10053 --gid 10000 --create-home worker \
 && mkdir -p /input /output /weights /checkpoint \
 && chown -R 10053:10000 /opt/w045 /input /output /weights /checkpoint
WORKDIR /opt/w045
USER 10053:10000
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/huggingface \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
    FORITHMUS_WEIGHTS_DIR=/weights
ENTRYPOINT ["python", "/opt/w045/entrypoint_rg.py"]
