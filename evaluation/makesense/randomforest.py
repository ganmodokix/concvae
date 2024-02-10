import os

import yaml
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from vaetc.checkpoint import Checkpoint
from vaetc.utils import debug_print

CELEBA_ATTRIBUTES = [
    # (4, "Bald"),
    # (5, "Bangs"),
    # (20, "Male"),
    # (26, "Pale Skin"),
    # (31, "Smiling"),
    # (39, "Young"),
    *enumerate("5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split(" "))
]

def load_zt(zt_path):

    zt = np.load(zt_path)
    z, t = zt["z"], zt["t"]

    t_threshold = 0.5
    t = np.where(t >= t_threshold, 1, 0)

    return z, t

def load_explanations(explanations_path):

    with open(explanations_path, "r", encoding="utf8") as fp:
        explanations = yaml.safe_load(fp)

    results = {}
    for key in explanations:
        if key.startswith("z_"):
            index = int(key[2:])
            u = explanations[key]["top-k"]["neg"][0]["word"]
            v = explanations[key]["top-k"]["pos"][0]["word"]
            results[index] = (u, v)

    return results

def fit(z_train, t_train):

    # debug_print("Fitting a classifier...")
    clf = RandomForestClassifier(n_estimators=10, max_depth=4)
    # debug_print("The classifier fitted")

    clf.fit(z_train, t_train)

    return clf

def visualize(checkpoint: Checkpoint):
    
    logger_path = checkpoint.options["logger_path"]
    output_path = os.path.join(logger_path, "makesense")
    os.makedirs(output_path, exist_ok=True)

    explanations_path = os.path.join(logger_path, "explanations.yaml")
    zt_path = os.path.join(logger_path, "zt_test.npz")

    if not os.path.isfile(explanations_path):
        debug_print("skipped; explanations.yaml not found")
        return

    if not os.path.isfile(zt_path):
        debug_print("skipped; zt_test.npz not found")
        return

    if not checkpoint.options["dataset"] == "celeba":
        debug_print("skipped; dataset is not CelebA")
        return

    attribute_name = "Male"
    attribute_index = 20

    z, t = load_zt(zt_path)
    explanations = load_explanations(explanations_path)
    data_size, z_dim = z.shape

    debug_print(f"Estimating attributes ...")
    for attribute_index, attribute_name in tqdm(CELEBA_ATTRIBUTES):

        clf = fit(z, t[:,attribute_index])

        imp = clf.feature_importances_
        names = [""] * z_dim
        for i in range(z_dim):
            wneg, wpos = explanations[i]
            names[i] = f"{wneg}$\\leftrightarrow${wpos} ($z_{{{i+1}}}$)"

        df = pd.DataFrame({"name": names, "importance": imp})
        df.sort_values(by="importance", ascending=False, inplace=True)

        fig = plt.figure(figsize=(8.54, 4.8))
        sns.set()
        ax = fig.add_subplot(111)
        sns.barplot(x="importance", y="name", data=df, ax=ax)
        ax.set_xlabel("Feature importances")
        ax.set_ylabel("Latent variables")
        fig.subplots_adjust(left=0.25)

        file_name = f"t_{attribute_index:03d}_{attribute_name}"
        plt.savefig(os.path.join(output_path, f"{file_name}.svg"))
        plt.savefig(os.path.join(output_path, f"{file_name}.pdf"))
        plt.close()