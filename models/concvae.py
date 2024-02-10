from typing import List, Optional, Tuple, Dict
from vaetc.utils import debug_print
import itertools

from tqdm import tqdm
import numpy as np

import torch
from torch import device, nn
from torch.nn import functional as F
import torchvision
from torch.distributions import Categorical

from tqdm import tqdm

from vaetc.models.utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.blocks import ResBlock
from .vislang import get_pretrained_vlmodel
from .vislang.clip import clip

from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian, neglogpxz_von_mises_fisher
from vaetc.models.vae import VAE
from .explainable import ExplainableElementVAE

def normalize_vector(x):
    
    return x / np.maximum(1e-12, np.linalg.norm(x, axis=-1))[...,None]

def softplus(x, inf=0):
    return torch.log1p((x - inf).exp()) + inf

def softminus(x, sup=0):
    return -torch.log1p((-x + sup).exp()) + sup

class LatentEmbedding(nn.Module):

    def __init__(self, z_dim: int, word_dim: int, vse_model, prefix_string = "a photo of", vocab_path = "vocab.txt"):
        super().__init__()

        self.z_dim = int(z_dim)
        self.word_dim = int(word_dim)
        assert self.z_dim > 0
        assert self.word_dim > 0

        self.prefix = clip.tokenize(prefix_string)[0]
        pad = self.prefix[-1:]
        self.prefix = self.prefix[self.prefix != 0][1:-1]
        self.prefix = vse_model.embed(self.prefix)
        self.pad_emb = vse_model.embed(pad)[0]

        sw, sb = 0.01, 0.01
        self.bias = nn.Parameter(self.pad_emb.cuda()[None,:].tile(self.z_dim, 1), requires_grad=False)

        self.word_prefix = nn.Parameter(torch.randn(size=(self.z_dim, 2, self.word_dim)) * sb, requires_grad=True)
        self.word_suffix = nn.Parameter(torch.randn(size=(self.z_dim, 2, self.word_dim)) * sb, requires_grad=True)

        self.emb_dict = vse_model.get_emb_weights(vocab_path)
        self.dic_size = self.emb_dict.shape[0]
        self.w_pos = nn.Parameter(torch.randn(size=(self.z_dim, self.dic_size)) * sw, requires_grad=True)
        self.w_neg = nn.Parameter(torch.randn(size=(self.z_dim, self.dic_size)) * sw, requires_grad=True)
        self.z_scale = nn.Parameter(torch.randn(size=(self.z_dim, )) * sw, requires_grad=True)

    def forward(self, z: torch.Tensor, progress: Optional[float] = None) -> torch.Tensor:

        p_pos = F.softmax(self.w_pos, dim=1)
        c_pos = p_pos @ self.emb_dict

        p_neg = F.softmax(self.w_neg, dim=1)
        c_neg = p_neg @ self.emb_dict

        z_scale = softplus(self.z_scale, inf=1)
        z_act = torch.tanh(z * z_scale[None,:])

        z_pos  = torch.relu(z_act)
        z_neg  = torch.relu(-z_act)
        z_bias = 1. - z_pos - z_neg

        v_pos  = c_pos[None,:,:] * z_pos[:,:,None]
        v_neg  = c_neg[None,:,:] * z_neg[:,:,None]
        v_bias = self.bias[None,:,:] * z_bias[:,:,None]

        v = v_pos + v_neg + v_bias

        batch_size, z_dim, word_dim = v.shape

        sentence = torch.cat([
            self.word_prefix[None,:,:,:].tile(batch_size, 1, 1, 1),
            v[:,:,None,:],
            self.word_suffix[None,:,:,:].tile(batch_size, 1, 1, 1),
        ], dim=2)
        sentence = sentence.view(batch_size, -1, self.word_dim)
        sentence = torch.cat([self.prefix[None,:,:].tile(batch_size, 1, 1), sentence], dim=1)
        return sentence, p_pos, p_neg

class Discriminator(nn.Module):

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        assert self.in_channels > 0

        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, 16, 4, 2, 1),
                nn.SiLU(True),
                nn.BatchNorm2d(16),
                nn.Dropout2d(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 4, 2, 1),
                nn.SiLU(True),
                nn.BatchNorm2d(32),
                nn.Dropout2d(0.5, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.SiLU(True),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.5, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.SiLU(True),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.5, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.SiLU(True),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5, inplace=True),
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 2 * 2, 256),
                nn.Dropout(0.5, inplace=True),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.Linear(256, 2),
                nn.LogSoftmax(dim=1),
            ),
        ])

    def forward(self, x):
        h = x
        for layer in self.net:
            h = layer(h)
        return h

    def hidden_loss(self, x, x2):

        losses = []
        h, l = x, x2
        for layer in self.net[:-1]:
            h, l = layer(h), layer(l)
            se = (h - l) ** 2
            se = se.view(se.shape[0], -1)
            losses += [se.mean(dim=1)]

        losses = torch.stack(losses, dim=1)
        return losses.sum(dim=1)

class ConceptualVAE(ExplainableElementVAE):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.beta = float(hyperparameters["beta"])
        self.gamma = float(hyperparameters["gamma"])
        self.vlmodel = str(hyperparameters["vlmodel"])
        self.vocab_path = str(hyperparameters["vocab"]) if "vocab" in hyperparameters else "vocab.txt"
        self.lr_disc = float(hyperparameters["lr_disc"])

        self.ablate_adv = bool(hyperparameters.get("ablate_adv", False))
        self.ablate_ent = bool(hyperparameters.get("ablate_ent", False))

        del self.enc_block
        del self.dec_block

        # X -> W
        vse_model = get_pretrained_vlmodel(name=self.vlmodel)

        # W -> Z
        self.enc_w_z = nn.Sequential(
            nn.Linear(vse_model.embed_dim, 256),
            nn.SiLU(True),
            nn.Linear(256, 256),
            nn.SiLU(True),
        )
        self.enc_mean = nn.Sequential(
            nn.Linear(256, self.z_dim, bias=False),
            nn.BatchNorm1d(self.z_dim),
        )
        self.enc_logvar = nn.Sequential(
            nn.Linear(256, self.z_dim, bias=False),
            nn.BatchNorm1d(self.z_dim),
        )

        # Z -> W
        self.dec_z_w = LatentEmbedding(self.z_dim, vse_model.word_dim, vse_model, vocab_path=self.vocab_path)

        # W -> X
        self.dec_w_x = nn.Sequential(
            nn.Linear(vse_model.embed_dim, 256 * 4 * 4),
            nn.SiLU(True),
            nn.Unflatten(dim=1, unflattened_size=[256, 4, 4]),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.SiLU(True),
            nn.BatchNorm2d(128),
            ResBlock(128, batchnorm=True),
            ResBlock(128, batchnorm=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(True),
            nn.BatchNorm2d(64),
            ResBlock(64, batchnorm=True),
            ResBlock(64, batchnorm=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(True),
            nn.BatchNorm2d(32),
            ResBlock(32, batchnorm=True),
            ResBlock(32, batchnorm=True),
            
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.SiLU(True),
            nn.BatchNorm2d(32),
            ResBlock(32, batchnorm=True),
            ResBlock(32, batchnorm=True),

            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # disc
        self.disc_block = Discriminator()

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:

        main_parameters = itertools.chain(
            self.enc_w_z.parameters(),
            self.enc_mean.parameters(),
            self.enc_logvar.parameters(),
            self.dec_z_w.parameters(),
            self.dec_w_x.parameters(),
        )
        disc_parameters = self.disc_block.parameters()

        return {
            "main": torch.optim.Adam(main_parameters, lr=self.lr, betas=(0.9, 0.999)),
            "disc": torch.optim.Adam(disc_parameters, lr=self.lr_disc, betas=(0.5, 0.9)),
        }

    def encode_gauss_w(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        vse_model = get_pretrained_vlmodel(name=self.vlmodel)

        # X -> W
        with torch.no_grad():
            w = vse_model.encode_image(x).detach()

        # W -> Z
        h = self.enc_w_z(w)
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)

        return mean, logvar, w

    def encode_gauss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mean, logvar, _ = self.encode_gauss_w(x)

        return mean, logvar

    def decode_w(self, z: torch.Tensor, progress: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        vse_model = get_pretrained_vlmodel(name=self.vlmodel)

        # Z -> W
        e2, p_pos, p_neg = self.dec_z_w(z, progress)
        w2 = vse_model.encode_text(e2)

        # W -> X
        x2 = self.dec_w_x(w2)

        return x2, w2, p_pos, p_neg
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        x2, _, _, _ = self.decode_w(z)

        return x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar, w = self.encode_gauss_w(x)
        z = reparameterize(mean, logvar)
        x2, w2, p_pos, p_neg = self.decode_w(z)

        return w, z, mean, logvar, x2, w2, p_pos, p_neg

    def loss(self, x, w, z, mean, logvar, x2, w2, p_pos, p_neg, progress: Optional[float] = None):

        # losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))
        loss_vse = torch.mean(neglogpxz_von_mises_fisher(w, w2))

        h_pos = torch.sum(-torch.xlogy(p_pos, p_pos), dim=1) / np.log(p_pos.shape[1])
        h_neg = torch.sum(-torch.xlogy(p_neg, p_neg), dim=1) / np.log(p_neg.shape[1])
        loss_ent = torch.sum(h_pos + h_neg)

        if not self.ablate_adv:
            zp = torch.randn_like(z)
            xp = self.decode(zp)
            loss_disc = -torch.mean(self.disc_block(x)[:,1] \
                + self.disc_block(x2)[:,0] \
                + self.disc_block(xp)[:,0])
            loss_adv = -loss_disc
            loss_amt = self.disc_block.hidden_loss(x, x2).mean()
        else:
            loss_adv = torch.zeros_like(loss_ae)
            loss_amt = torch.zeros_like(loss_ae)

        # Total loss
        loss = loss_ae \
                + loss_reg * self.beta \
                + loss_vse * self.gamma \
                + loss_ent * (1 if self.ablate_ent else 0) \
                + (loss_adv + loss_amt) * (1 if self.ablate_adv else 0)

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "loss_vse": loss_vse,
            "loss_ent": loss_ent,
            "h_pos": h_pos.mean(),
            "h_neg": h_neg.mean(),
            # "anneal_ent": anneal_ent,
            "loss_adv": loss_adv,
            "loss_amt": loss_amt,
        })

    def step_batch(
        self,
        batch, optimizers = None, progress: Optional[float] = None,
        training: Optional[bool] = False
    ):

        x, t = batch

        x = x.to(self.device)
        
        # train main
        w, z, mean, logvar, x2, w2, p_pos, p_neg = self(x)
        loss, loss_dict = self.loss(x, w, z, mean, logvar, x2, w2, p_pos, p_neg, progress)

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()
        
        # train disc
        if not self.ablate_adv:
            zp = torch.randn_like(z)
            xp = self.decode(zp)
            loss_disc = -torch.mean(self.disc_block(x)[:,1] \
                + self.disc_block(x2.detach())[:,0] \
                + self.disc_block(xp.detach())[:,0])
        else:
            loss_disc = torch.zeros_like(loss)
        loss_dict["loss_disc"] = loss_disc.item()

        if training and not self.ablate_adv:
            self.zero_grad()
            loss_disc.backward()
            optimizers["disc"].step()

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):

        return self.step_batch(batch, optimizers, progress, training=True)

    def eval_batch(self, batch):

        return self.step_batch(batch)

    def encode_embeddings(self, embs: torch.Tensor) -> torch.Tensor:

        vse_model = get_pretrained_vlmodel(name=self.vlmodel)

        data_size = embs.shape[0]
        batch_size = 32

        results = []
        for i in tqdm(range((data_size + batch_size - 1) // batch_size)):
            ib = i * batch_size
            ie = ib + batch_size

            embs_batch = embs[ib:ie,None,:]

            encoded_batch = vse_model.encode_text(embs_batch).detach()
            results += [encoded_batch]

        return torch.cat(results, dim=0)

    def explain(self):

        with torch.no_grad():

            tau = 10

            p_pos = F.softmax(self.dec_z_w.w_pos, dim=1)
            c_pos = (p_pos @ self.dec_z_w.emb_dict).detach().cpu().numpy()

            p_neg = F.softmax(self.dec_z_w.w_neg, dim=1)
            c_neg = (p_neg @ self.dec_z_w.emb_dict).detach().cpu().numpy()
        
        bias = self.dec_z_w.bias.detach().cpu().numpy()

        c_pos_unit = normalize_vector(c_pos)
        c_neg_unit = normalize_vector(c_neg)
        bias_unit = normalize_vector(bias)

        vse_model = get_pretrained_vlmodel(name=self.vlmodel)
        words, embs = vse_model.get_freq_embs(self.vocab_path)

        embs_unit = normalize_vector(embs)

        explanations = {}

        for i in tqdm(range(self.z_dim)):

            bases = {
                "pos": c_pos_unit[i],
                "neg": c_neg_unit[i],
                "bias": bias_unit[i],
            }

            k = 3
            explanations_zi = {
                "top-k": {
                    "k": k,
                },
                "norm": {},
            }

            for name, vec in bases.items():

                # top-k words
                coss = np.sum(embs_unit * vec[None,:], axis=-1)
                rank = np.argsort(coss)[::-1]

                topk = []
                for j in range(k):
                    idx = rank[j]
                    topi = {"word": words[idx], "cos": float(coss[idx])}
                    topk.append(topi)

                explanations_zi["top-k"][name] = topk

                # norm
                explanations_zi["norm"][name] = float(np.linalg.norm(vec))

            explanations[f"z_{i:03d}"] = explanations_zi

        return explanations

    def verbalize(self, z: torch.Tensor) -> List[List[str]]:

        with torch.no_grad():

            v, p_pos, p_neg = self.dec_z_w(z.cuda())
            v = v.detach()
            p_pos = p_pos.detach()
            p_neg = p_neg.detach()
            b, l, n = v.shape
            v = self.encode_embeddings(v.view(b*l, n).cuda()).view(b, l, -1)
            v = v.detach().cpu().numpy()
        
        vse_model = get_pretrained_vlmodel(name=self.vlmodel)
        words, embs = vse_model.get_embeddings(self.vocab_path)
        embs = embs.detach().cpu().numpy()

        v_unit = normalize_vector(v)
        embs_unit = normalize_vector(embs)

        coss = np.sum(v_unit[:,:,None,:] * embs_unit[None,None,:,:], axis=3)

        res = []

        for i in range(z.shape[0]):
            
            sentence = []
            
            for j in range(v.shape[1]):

                rank = np.argsort(coss[i,j])[::-1]
                sentence += [words[rank[0]]]

            res += [sentence]
        
        return res

