# syntax=docker/dockerfile:1

FROM quay.io/fenicsproject/stable:latest

USER root

# ENV OPENBLAS_NUM_THREADS=8

COPY . code/

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir joblib POT seaborn nbdev tqdm numba

RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab

EXPOSE 8888/tcp
ENV SHELL /bin/bash

WORKDIR code/
RUN pip3 install .