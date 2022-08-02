# statFEM
> Code for the numerical experiments demonstrating our error analysis of the statFEM method.


[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yannipapandreou.github.io/statFEM/)

# Overview

This repo contains code accompanying our error analysis of the Statistical Finite Element Method (StatFEM) as described in the paper by Girolami et al. [1]. The code accompanies [2].

`oneDim` contains code needed for our 1-D example while `twoDim` contains code needed for our 2-D example. `maxDist` contains required code for our 1-D max example. The other notebooks all contain the relevant code for each experiment.

# Installation/Running

We utilise docker to run the code.

Prerequisites: Docker must be installed and set up following [these instructions](https://docs.docker.com/get-started/).

## Steps:
- Clone the repo.
- Navigate to repo directory: `cd statFEM`
- Navigate to docker subdirectory: `cd docker`
- Build the docker image: `docker build .`
- Docker will `build` the container using the instructions in the `Dockerfile`. After the build is complete Docker will output a hash, e.g.:
  ```bash
  Successfully built 10c79a08651f
  ```
- Use this to `tag` your container for future use:
  ```bash
  docker tag 10c79a quay.io/my-user/my-docker-image
  ```
<!-- - Run the following command to launch a Jupyterlab session:
  ```bash
  docker run --name statfem-lab -w /home/fenics -v $(pwd):/home/fenics/shared -t -i -p 127.0.0.1:8888:8888 quay.io/my-user/my-docker-image
  ```
- _**TODO: Explain how to open JupyterLab**_ -->
- Navigate to top level of repo directory: `cd ..`
- Run the following command to launch a Jupyterlab session with the repo directory mounted in `/home/fenics/shared`:
```bash
docker run --name my-name -w /home/fenics -v $(pwd):/home/fenics/shared -p 8888:8888 quay.io/my-user/my-docker-image
```

# References:

 [1] Mark Girolami, Eky Febrianto, Ge Yin, and Fehmi Cirak. The
    statistical finite element method (statFEM) for coherent synthesis
    of observation data and model predictions. *Computer Methods in
    Applied Mechanics and Engineering*, Volume 375, 2021, 113533,
    https://doi.org/10.1016/j.cma.2020.113533.

[2] Yanni Papandreou, Jon Cockayne, Mark Girolami, and Andrew B. Duncan. "Theoretical Guarantees for the Statistical Finite Element Method." arXiv preprint arXiv:2111.07691 (2021), https://arxiv.org/abs/2111.07691.
