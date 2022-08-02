# syntax=docker/dockerfile:1

FROM quay.io/fenicsproject/stable:latest

USER root

# ENV OPENBLAS_NUM_THREADS=8

COPY ./statFEM_analysis statFEM_analysis/
COPY setup.cfg .
COPY pyproject.toml .
COPY setup.py .

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir joblib POT seaborn nbdev tqdm numba

RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab

EXPOSE 8888/tcp
ENV SHELL /bin/bash

RUN pip3 install .
