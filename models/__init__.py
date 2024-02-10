from vaetc.models.byname import register_model

from .explainable import ExplainableElementVAE

from .concvae import ConceptualVAE
register_model("concvae", ConceptualVAE)
from .concvaescratch import ConceptualVAEScratch
register_model("concvaescratch", ConceptualVAEScratch)
