"""Python script with util functions."""

import os
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

DATA_DIR = "../figures"
os.makedirs(DATA_DIR, exist_ok=True)

label_to_idx = {
    "gram-negative": 0,
    "gram-positive": 1,
    "acid-fast": 2,
    "fungi": 3,
    "inactive": 4,
}


# get shared elements for each combination of sets
def get_shared(sets):
    IDs = sets.keys()
    combs = sum(
        [list(map(list, combinations(IDs, i))) for i in range(1, len(IDs) + 1)], []
    )

    shared = {}
    for comb in combs:
        ID = " and ".join(comb)
        if len(comb) == 1:
            shared.update({ID: sets[comb[0]]})
        else:
            setlist = [sets[c] for c in comb]
            u = set.intersection(*setlist)
            shared.update({ID: u})
    return shared


# get unique elements for each combination of sets
def get_unique(shared):
    unique = {}
    for shar in shared:
        if shar == list(shared.keys())[-1]:
            s = shared[shar]
            unique.update({shar: s})
            continue
        count = shar.count(" and ")
        if count == 0:
            setlist = [
                shared[k] for k in shared.keys() if k != shar and " and " not in k
            ]
            s = shared[shar].difference(*setlist)
        else:
            setlist = [
                shared[k]
                for k in shared.keys()
                if k != shar and k.count(" and ") >= count
            ]
            s = shared[shar].difference(*setlist)
        unique.update({shar: s})
    return unique


# plot Venn
def draw_venn(sets={}, size=3.5, save=False):
    shared = get_shared(sets)
    unique = get_unique(shared)
    ce = "bgrc"  # colors
    lw = size * 0.12  # line width
    fs = size * 2  # font size
    nc = 2  # legend cols
    cs = 4  # columnspacing

    ax = plt.figure(figsize=(size, size), dpi=200).add_subplot(111)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # 4 sets
    if len(sets) == 4:
        # draw ellipses
        ew = 45  # width
        eh = 75  # height
        xe = [35, 48, 52, 65]  # x coordinats
        ye = [35, 45, 45, 35]  # y coordinats
        ae = [225, 225, 315, 315]  # angles

        for i, s in enumerate(sets):
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]),
                    width=ew,
                    height=eh,
                    fc=ce[i],
                    angle=ae[i],
                    alpha=0.3,
                )
            )
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]),
                    width=ew,
                    height=eh,
                    fc="None",
                    angle=ae[i],
                    ec="black",
                    lw=lw,
                )
            )

        # annotate
        xt = [
            12,
            32,
            68,
            88,
            14,
            34,
            66,
            86,
            26,
            28,
            50,
            50,
            72,
            74,
            37,
            60,
            40,
            63,
            50,
        ]  # x
        yt = [
            67,
            79,
            79,
            67,
            41,
            70,
            70,
            41,
            59,
            26,
            11,
            60,
            26,
            59,
            51,
            17,
            17,
            51,
            35,
        ]  # y

        for j, s in enumerate(sets):
            ax.text(
                xt[j],
                yt[j],
                "",
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

        for k in unique:
            j += 1
            ax.text(
                xt[j],
                yt[j],
                len(unique[k]),
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

    # 3 sets
    if len(sets) == 3:
        # draw circles
        ew = 60  # width
        eh = 60  # height
        lw = size * 0.12  # line width
        xe = [37, 63, 50]  # x coordinats
        ye = [55, 55, 32]  # y coordinats
        nc = 3  # legend columns
        cs = 1  # columns spacing

        for i, s in enumerate(sets):
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]), width=ew, height=eh, fc=ce[i], angle=0, alpha=0.3
                )
            )
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]),
                    width=ew,
                    height=eh,
                    fc="None",
                    angle=0,
                    ec="black",
                    lw=lw,
                )
            )

        # annotate
        xt = [12, 88, 28, 22, 78, 50, 50, 30, 70, 50]  # x
        yt = [80, 80, 3, 60, 60, 17, 70, 35, 35, 50]  # y

        for j, s in enumerate(sets):
            ax.text(
                xt[j],
                yt[j],
                "",
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

        for k in unique:
            j += 1
            ax.text(
                xt[j],
                yt[j],
                len(unique[k]),
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

    # 2 sets
    if len(sets) == 2:
        # draw circles
        ew = 70  # width
        eh = 70  # height
        lw = size * 0.12  # line width
        xe = [37, 63]  # x coordinats
        ye = [45, 45]  # y coordinats

        for i, s in enumerate(sets):
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]), width=ew, height=eh, fc=ce[i], angle=0, alpha=0.3
                )
            )
            ax.add_artist(
                Ellipse(
                    xy=(xe[i], ye[i]),
                    width=ew,
                    height=eh,
                    fc="None",
                    angle=0,
                    ec="black",
                    lw=lw,
                )
            )

        # annotate
        xt = [20, 80, 18, 82, 50]  # x
        yt = [80, 80, 45, 45, 45]  # y

        for j, s in enumerate(sets):
            ax.text(
                xt[j],
                yt[j],
                "",
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

        for k in unique:
            j += 1
            ax.text(
                xt[j],
                yt[j],
                len(unique[k]),
                ha="center",
                va="center",
                fontsize=fs,
                transform=ax.transData,
            )

    # legend
    handles = [
        mpatches.Patch(color=ce[i], label=l, alpha=0.3) for i, l in enumerate(sets)
    ]
    ax.legend(
        labels=sets,
        handles=handles,
        fontsize=fs * 1.1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        bbox_transform=ax.transAxes,
        loc=9,
        handlelength=1.5,
        ncol=nc,
        columnspacing=cs,
        handletextpad=0.5,
    )
    plt.tight_layout()
    if save:
        plt.savefig(f"{DATA_DIR}/figure_1.png", dpi=400)


def _process_euos_data():
    euos_df = pd.DataFrame()

    for file in os.listdir("../data/benchmark"):
        if "EOS" not in file:
            continue
        df = pd.read_csv(f"../data/benchmark/{file}")
        df = df[["eos", "activity", "smiles", "inchikey"]]
        df.rename(columns={"activity": file.split(".")[0]}, inplace=True)
        df.set_index("eos", inplace=True)
        if euos_df.empty:
            euos_df = df
        else:
            euos_df = euos_df.join(df, how="outer", lsuffix="{file}_")

    cols_to_keep = [i for i in euos_df.columns if "{file}_" not in i]
    euos_df = euos_df[cols_to_keep]

    # Adding in latest results
    gram_negative_actives = """EOS2202
    EOS24472
    EOS100321
    EOS100573
    EOS100586
    EOS100698
    EOS100925
    EOS100994
    EOS101209
    EOS101641
    EOS101978
    EOS102146
    EOS102156
    EOS102273"""
    gram_negative_actives = gram_negative_actives.split("\n")

    euos_df["gram_negative"] = euos_df.index.isin(gram_negative_actives)
    euos_df["gram_negative"] = euos_df["gram_negative"].apply(
        lambda x: "active" if x else "inactive"
    )

    # cleaning
    euos_df.rename(
        columns={
            "CandidaAlb_fungi_EOS300076_65": "Fungi_candida",
            "AspFumigatus_fungi_EOS300074_64": "Fungi_aspergillus",
            "CandidaAuris_fungi_EOS300072_63": "Fungi_candida_auris",
            "StaphA_EOS300078_66_GPlus": "Gram_positive_staph",
            "E_faecalis_EOS300080_67_GPlus": "Gram_positive_enterococcus",
        },
        inplace=True,
    )

    final_class = []

    for fungi_1, fungi_2, fungi_2, gp1, gp2, gn in tqdm(
        euos_df[
            [
                "Fungi_candida",
                "Fungi_aspergillus",
                "Fungi_candida_auris",
                "Gram_positive_staph",
                "Gram_positive_enterococcus",
                "gram_negative",
            ]
        ].values
    ):
        if "active" in [fungi_1, fungi_2, fungi_2]:
            fungi = True
        elif "inactive" in [fungi_1, fungi_2, fungi_2]:
            fungi = False

        if "active" in [gp1, gp2]:
            gp = True
        elif "inactive" in [gp1, gp2]:
            gp = False

        if "active" in [gn]:
            gn = True
        elif "inactive" in [gn]:
            gn = False

        if fungi and gp and gn:
            final_class.append("fungi, gram-positive, gram-negative")
        elif fungi and gp:
            final_class.append("fungi, gram-positive")
        elif fungi and gn:
            final_class.append("fungi, gram-negative")
        elif gp and gn:
            final_class.append("gram-positive, gram-negative")
        elif fungi:
            final_class.append("fungi")
        elif gp:
            final_class.append("gram-positive")
        elif gn:
            final_class.append("gram-negative")
        else:
            final_class.append("inactive")

    euos_df["final_class"] = final_class
    euos_df.to_csv("../data/benchmark/euos_data_cleaned.tsv", sep="\t", index=False)
