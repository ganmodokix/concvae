from typing import List

from vaetc.models.vae import VAE

class ExplainableElementVAE(VAE):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

    def explain(self) -> List[dict]:
        """
        returns a dict of explanatory information about each latent variables et al.
        """
        raise NotImplementedError()