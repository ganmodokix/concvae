import os

import yaml

import torch
from torch.utils.data import DataLoader, Subset

from vaetc.checkpoint import Checkpoint
from models import ExplainableElementVAE, ConceptualVAE

def verbalize(checkpoint: Checkpoint):

    logger_path = checkpoint.options["logger_path"]

    n = 10
    
    loader_test = DataLoader(
        dataset=Subset(checkpoint.dataset.test_set, range(n)),
        batch_size=n,
        shuffle=False,
        num_workers=os.cpu_count() - 1,
        pin_memory=True)

    for x, t in loader_test:

        with torch.no_grad():
            z = checkpoint.model.encode(x.cuda())

        verbalizations = checkpoint.model.verbalize(z)

        with open(os.path.join(logger_path, "verbalizations.txt"), "w") as fp:
            for sentence in verbalizations:
                fp.write(" ".join(sentence) + ".\n")

def visualize(checkpoint: Checkpoint):

    if not isinstance(checkpoint.model, ExplainableElementVAE):
        raise ValueError("Model is not explainable")

    logger_path = checkpoint.options["logger_path"]

    explanations = checkpoint.model.explain()

    if isinstance(checkpoint.model, ConceptualVAE):

        verbalize(checkpoint)

    with open(os.path.join(logger_path, "explanations.yaml"), "w") as fp:
        yaml.safe_dump(explanations, fp)