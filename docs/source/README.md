# Getting started
> Code for the numerical experiments demonstrating our error analysis of the statFEM method.

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yannipapandreou.github.io/statFEM/) -->

## Overview

This repo contains code accompanying our error analysis of the Statistical Finite Element Method (StatFEM) {cite}`Girolami2021`. The code accompanies {cite}`papandreou2021theoretical`.

## Installation

The code can be run in a Docker container.

Prerequisites: Docker must be installed and set up following [these instructions](https://docs.docker.com/get-started/).

### Recommended method (Docker image):

#### Steps:
- Clone the repo.
- Navigate to the repo directory: `cd statFEM_analysis`
- A Docker image is available [online](https://hub.docker.com/r/yannipapandreou/statfem_analysis). To use run the following command, remembering to replace `my-name` with your desired image/container name. 
```bash
docker run -ti --name my-name -w /home/fenics -v $(pwd):/home/fenics/shared -p 8888:8888 yannipapandreou/statfem_analysis
```

#### Alternative method (Building Docker container locally) :

#### Steps:
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

Once the Docker image is pulled/built and running, you will be in a Docker container running Ubuntu. The recommended way for running code is to either:

- Run scripts using `python3 run script.py`
- Launch a Jupyter lab session using:
  ```bash
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
  ```
  - The Jupyter lab session can then be accessed by opening `http://localhost:8888/lab` in your browser and pasting the token written in the Docker container.
