FROM mambaorg/micromamba:1.5.8

WORKDIR /project

COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY docs ./docs
COPY tests ./tests

RUN python -m pip install -e .

ENV PYTHONPATH=/project/src
CMD ["python", "-m", "mffiresale.pipeline", "--help"]
