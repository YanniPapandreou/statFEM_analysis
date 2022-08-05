# Getting started
> Code for the numerical experiments demonstrating our error analysis of the statFEM method.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yannipapandreou.github.io/statFEM_analysis/)

## Overview

This repo contains code accompanying our error analysis of the Statistical Finite Element Method (StatFEM) as described in the paper by Girolami et al. [1]. The code accompanies [2].

The folder `scripts` contains the scripts for the numerical experiments from our paper {cite}`papandreou2021theoretical`.

## Installation

The code can be run in a Docker container.

Prerequisites: Docker must be installed and set up following [these instructions](https://docs.docker.com/get-started/).

### Steps:
- Clone the repo.
- Navigate to the repo directory: `cd statFEM_analysis`
- Build the Docker image: `docker build .`
- Docker will `build` the container using the instructions in the `Dockerfile`. After the build is complete Docker will output a hash, e.g.:
  ```bash
  Successfully built 10c79a08651f
  ```
- Use this to `tag` your container for future use:
  ```bash
  docker tag 10c79 quay.io/my-user/my-docker-image
  ```
- Run the Docker container with the repo directory mounted in :bash:`/home/fenics/shared`:
  ```bash
  docker run -ti --name my-name -w /home/fenics -v $(pwd):/home/fenics/shared -p 8888:8888 quay.io/my-user/my-docker-image
  ```

## Running the code

Once the Docker image is built and running, you will be in a Docker container running Ubuntu. The recommended way for running code is to either:

- Run scripts using `python3 run script.py`
- Launch a Jupyter lab session using:
  ```bash
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
  ```
  - The Jupyter lab session can then be accessed by opening `http://localhost:8888/lab` in your browser and pasting the token written in the Docker container.


## References:

 [1] Mark Girolami, Eky Febrianto, Ge Yin, and Fehmi Cirak. The
    statistical finite element method (statFEM) for coherent synthesis
    of observation data and model predictions. *Computer Methods in
    Applied Mechanics and Engineering*, Volume 375, 2021, 113533,
    https://doi.org/10.1016/j.cma.2020.113533.

[2] Yanni Papandreou, Jon Cockayne, Mark Girolami, and Andrew B. Duncan. "Theoretical Guarantees for the Statistical Finite Element Method." arXiv preprint arXiv:2111.07691 (2021), https://arxiv.org/abs/2111.07691.
