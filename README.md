# ConcVAE: Conceptual Representation Learning

This repository contains the official implementation of the ConcVAE models proposed in our paper "ConcVAE: Conceptual Representation Learning" (IEEE TNNLS, in press).

Our ConcVAE models, based on variational autoencoders (VAEs), are designed to learn data representations consisting of variables paired with verbal concepts. Each verbal concept includes a pair of antonyms, which represent the negative and positive directions of the representation variables.

这个代码库包含了我们在论文《ConcVAE: 概念表征学习》（IEEE TNNLS，待刊）中提出的ConcVAE模型的官方实现。

我们的ConcVAE模型基于变分自编码器（VAE）设计，旨在学习由与语言概念配对的变量组成的数据表征。每个语言概念都包含一对反义词，分别表示表征变量的负方向和正方向。

## BibTeX Citation

Please cite our paper if you use this code.

**Once the paper is officially published, this BibTeX citation will be updated.**

```bibtex
@article{Togo2024,
    author  = {Togo, Ren and Nakagawa, Nao and Ogawa, Takahiro and Haseyama, Miki},
    title   = {Conc{VAE}: Conceptual Representation Learning},
    journal = {{IEEE} Transactions on Neural Networks and Learning Systems},
    year    = {2024},
    volume  = {},
    number  = {},
    pages   = {in press}
}
```

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
   $ sudo apt install python3.9 python3.9-dev python3.9-venv python3-tk
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
