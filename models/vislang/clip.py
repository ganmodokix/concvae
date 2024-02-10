from typing import Tuple, List

import sys
import os

import csv

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torchvision import transforms
from torchvision.transforms.transforms import Resize

from vaetc.utils import debug_print

from .abstract import VisionLanguageEncoder

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer

class CLIP(VisionLanguageEncoder):

    def __init__(self, name="ViT-B/32", pretrained=True):

        super().__init__()

        self.clip, self.preprocess = clip.load(name=name, device="cuda")
        self.start_token = self.clip.vocab_size - 2
        self.pad_token = 0
        self.eot_token = self.clip.vocab_size - 1
        self.word_emb = self.clip.token_embedding.weight.detach()
        self.emb_start = self.word_emb[self.start_token]
        self.emb_pad = self.word_emb[self.pad_token]
        self.emb_eot = self.word_emb[self.eot_token]

        self.template_sentence = "a photo of"
        self.template_sentence = clip.tokenize(self.template_sentence)[0].cpu().numpy()
        self.template_sentence = self.template_sentence[self.template_sentence != self.pad_token]
        self.template_sentence = self.template_sentence[[0, -1]]
        self.prefix_tokens = self.template_sentence[0]
        self.embs_prefix = self.word_emb[self.prefix_tokens].detach().cpu().numpy()
        self.suffix_tokens = self.template_sentence[1]
        self.embs_suffix = self.word_emb[self.suffix_tokens].detach().cpu().numpy()

        res = self.clip.visual.input_resolution
        self.image_transform = transforms.Compose([
            transforms.Resize([res, res]),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

    @property
    def embed_dim(self) -> int:
        
        return self.clip.text_projection.shape[1]

    @property
    def word_dim(self) -> int:
        
        return self.clip.token_embedding.weight.shape[1]

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:

        x = self.image_transform(x)

        x = x.type(self.clip.dtype)
        x = self.clip.encode_image(x)
        x = x.type(torch.float)

        return x

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:

        assert x.shape[1] + 2 <= self.clip.context_length

        batch_size = x.shape[0]
        text_length = x.shape[1]

        x = x.cuda()
        embs_start = torch.tensor(self.embs_prefix).cuda()[None,None,:].tile(batch_size, 1, 1)
        embs_eot = torch.tensor(self.embs_suffix).cuda()[None,None,:].tile(batch_size, 1, 1)
        content = torch.cat([embs_start, x, embs_eot], dim=1) # [batch_size, n_ctx, d_model]
        content_length = content.shape[1]

        pad_length = self.clip.context_length - content_length
        embs_pad = self.emb_pad[None,None,:].tile(batch_size, pad_length, 1)
        x = torch.cat([content, embs_pad], dim=1) # [batch_size, n_ctx, d_model]
        
        x = x.type(self.clip.dtype)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.clip.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        idx_eot = content_length - 1
        final_state = x[:,idx_eot,:]
        result = final_state @ self.clip.text_projection
        result = result.type(torch.float)

        return result

    def get_vocabulary(self) -> Tuple[List[str], np.ndarray]:

        tokenizer = clip._tokenizer
        words = [tokenizer.decode([i]).strip() for i in range(self.clip.vocab_size)]
        
        embeddings = self.word_emb.cpu().numpy()
        
        return words, embeddings

    def frequent_words(self, vocab_path) -> List[str]:

        debug_print(vocab_path)

        with open(os.path.join("vocab", vocab_path), "r", encoding="utf8") as fp:
            words = fp.read().strip().split(" ")
        
        return words

    def get_emb_weights(self, vocab_path) -> torch.Tensor:

        debug_print(vocab_path)

        words = self.frequent_words(vocab_path)
        sentences = clip.tokenize(words)
        tokens = sentences[:,1]
        return self.clip.token_embedding.weight[tokens].detach()

    def get_freq_embs(self, vocab_path) -> Tuple[List[str], torch.Tensor]:

        debug_print(vocab_path)

        return self.frequent_words(vocab_path), self.get_emb_weights(vocab_path).cpu().numpy()

    def get_embeddings(self, vocab_path) -> Tuple[List[str], torch.Tensor]:

        debug_print(vocab_path)

        words = self.frequent_words(vocab_path)
        
        with torch.no_grad():
            sentences = [f"{word}" for word in words]
            embeddings = clip.tokenize(sentences)
            embeddings = self.clip.encode_text(embeddings.cuda())
            embeddings = embeddings.detach().cpu()

        return words, embeddings

    def tokenize(self, x: str):
        return clip.tokenize(x)

    def embed(self, x):
        return self.clip.token_embedding.weight.detach()[x]