.. role:: bash(code)
   :language: bash

=====
Usage
=====

Installation
============

The code can be run in a Docker container.

Prerequisites: Docker must be installed and set up following these `instructions <https://docs.docker.com/get-started/>`_.

Steps:
------

* Clone the repo.
* Navigate to the repo directory: :bash:`cd statFEM_analysis`
* Build the Docker image: :bash:`docker build .`
* Docker will build the container using the instructions in the :bash:`Dockerfile`. After the build is complete Docker will output a hash, e.g.:
  
  .. code-block:: console

     Successfully built 10c79a08651h

* Use this to tag your container for future use:

  .. code-block:: bash

     docker tag 10c79 quay.io/my-user/my-docker-image

* Run the Docker container with the repo directory mounted in :bash:`/home/fenics/shared`:

  .. code-block:: bash

     docker run -ti --name my-name -w /home/fenics -v $(pwd):/home/fenics/shared -p 8888:8888 quay.io/my-user/my-docker-image

Running the code
================

Once the Docker image is built and running, you will be in a Docker container running Ubuntu. The recommended way for running code is to either:

* Run scripts using :bash:`python3 run script.py`
* Launch a Jupyter lab session using:

  .. code-block:: bash

     jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

  * The Jupyter lab session can then be accessed by opening ``http://localhost:8888/lab`` in your browser and pasting the token written in the Docker container.
