# ConcVAE: Conceptual Representation Learning

Code for ConcVAE models

## Setup
We developed and tested this code in the environment below.
- Ubuntu 20.04.4
- NVIDIA GeForce RTX 2080 x1
- Python 3.9.5 with venv
- PyTorch 1.10.1+cu102
- 32GB of RAM

1. Install python & venv
    ```bash
    $ sudo apt update
    $ sudo apt upgrade
    $ sudo apt install python3.9 python3.9-dev python3.9-venv
    ```
2. Create an environment
    ```bash
    $ python -m venv .env
    $ source .env/bin/activate
    ```
3. Install the dependencies
    We are using some third-party libraries of PyTorch.
    ```bash
    $ pip install -U pip
    $ pip install wheel
    $ pip install -r requirements.txt
    ```
4. Try conceptual representation learning
    ```bash
    $ python train.py settings/concvae_celeba.yaml
    $ python train.py settings/concvae_getchu.yaml
    $ python train.py settings/concvae_mnist.yaml
    ```
    Some datasets requires to be manually downloaded.