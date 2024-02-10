import os
import numpy as np
from vaetc.checkpoint import Checkpoint
from vaetc.utils import debug_print

def visualize(checkpoint: Checkpoint):
    
    logger_path = checkpoint.options["logger_path"]

    explanations_path = os.path.join(logger_path, "explanations.yaml")
    zt_path = os.path.join(logger_path, "zt_test.npz")

    if not os.path.isfile(explanations_path):
        debug_print("skipped; explanations.yaml not found")
        return

    if not os.path.isfile(zt_path):
        debug_print("skipped; zt_test.npz not found")
        return


    zt = np.load(zt_path)
    z, t = zt["z"], zt["t"]

    