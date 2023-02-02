# syntax=docker/dockerfile:1

FROM quay.io/fenicsproject/stable:latest

USER root

# ENV OPENBLAS_NUM_THREADS=8

COPY ./src src/
COPY setup.cfg .
COPY pyproject.toml .
COPY setup.py .

# Copy requirements.txt file to the container
COPY requirements.txt .

# upgrade pip
RUN pip3 install --upgrade pip

# Install the required packages
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install --no-cache-dir joblib POT seaborn nbdev tqdm numba

# RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab

EXPOSE 8888/tcp
ENV SHELL /bin/bash

RUN pip3 install .
