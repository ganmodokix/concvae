from typing import List, Optional, Tuple, Dict

import torch
from torch import nn

from tqdm import tqdm

from .vislang import get_pretrained_vlmodel
from .concvae import LatentEmbedding, ConceptualVAE

class ConceptualVAEScratch(ConceptualVAE):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.beta = float(hyperparameters["beta"])
        self.gamma = float(hyperparameters["gamma"])
        self.vlmodel = str(hyperparameters["vlmodel"])

        del self.enc_block
        del self.dec_block

        # X -> W
        vse_model = get_pretrained_vlmodel(name=self.vlmodel)
        self.enc_x_w = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, vse_model.embed_dim),
            nn.LeakyReLU(0.2, True),
        )

        # W -> Z
        self.enc_w_z = nn.Sequential(
            nn.Linear(vse_model.embed_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, True),
        )
        self.enc_mean = nn.Sequential(
            nn.Linear(256, self.z_dim),
        )
        self.enc_logvar = nn.Linear(256, self.z_dim)

        # Z -> W
        self.dec_z_w = LatentEmbedding(self.z_dim, vse_model.word_dim, vse_model)
        self.dec_z_w_skip = nn.Sequential(
            nn.Linear(self.z_dim, vse_model.word_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.dec_sentence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(vse_model.word_dim * (self.z_dim * 3 + 3), 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, vse_model.embed_dim),
            nn.LeakyReLU(0.2, True),
        )

        # W -> X
        self.dec_w_x = nn.Sequential(
            nn.Linear(vse_model.embed_dim, 64 * 8 * 8),
            nn.LeakyReLU(0.2, True),
            nn.Unflatten(dim=1, unflattened_size=[64, 8, 8]),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode_gauss_w(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # X -> W
        w = self.enc_x_w(x)

        # W -> Z
        h = self.enc_w_z(w)
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)

        return mean, logvar, w

    def decode_w(self, z: torch.Tensor, progress: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # Z -> W
        e2, p_pos, p_neg = self.dec_z_w(z, progress)
        w2 = self.dec_sentence(e2)

        # W -> X
        x2 = self.dec_w_x(w2)

        return x2, w2, p_pos, p_neg
    