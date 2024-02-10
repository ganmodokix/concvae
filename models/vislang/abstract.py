from typing import Optional, List, Dict, Tuple

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class VisionLanguageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

    @property
    def embed_dim(self) -> int:
        """
        returns # of dims. of embedded vectors
        """
        raise NotImplementedError()

    @property
    def word_dim(self) -> int:
        """
        returns # of dims. of word embeddings
        """

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        encode images x with size (batch, channel, height, width)
        returns embedded vectors with size (batch, embed_dim) in the shared space
        """
        raise NotImplementedError()

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        """
        encode texts x with size (batch_size, sentence_size, embedding_size)
        returns embedded vectors with size (batch_size, sentence_size, embed_dim) in the shared space
        """
        raise NotImplementedError()

    def get_vocabulary(self) -> Tuple[List[str], np.ndarray]:
        """
        returns (words, embeddings)
        """
        raise NotImplementedError()