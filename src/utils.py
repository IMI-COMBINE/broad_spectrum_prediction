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
