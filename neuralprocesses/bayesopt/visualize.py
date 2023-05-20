import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns


path = "../../../results/res"

# file = "hartmann3d_single_10.npy"
# file = "hartmann3d_sumprod_10.npy"
# file = "matern300_20.npy"
file = "hartmann6d_sumprod_5.npy"
# file = "hartmann6d_single_5.npy"
data = np.load(f"{path}/{file}", allow_pickle=True).item()

color = sns.color_palette("Paired")

Names = {
    "gp": ("GP", color[0]),
    "cnp": ("CNP", color[1]),
    "gnp": ("GNP", color[1]),
    "rcnp": ("RCNP", color[3]),
    "rgnp": ("RGNP", color[3]),
    # "contextrcnp": ("ConRCNP", color[6]),
    # "contextrgnp": ("ConRGNP", color[7]),
    # "np": ("NP", color[]),
    "acnp": ("ACNP", color[7]),
    "agnp": ("AGNP", color[7]),
    # "anp": ("ANP", color[])
}


h3dmin = -3.86278
h6dmin = -3.32237


# keys = ["gp", "cnp", "gnp", "acnp", "agnp", "rcnp", "rgnp"]
keys = ["gp", "cnp", "gnp", "rcnp", "rgnp"]

sns.set(style="white", font_scale=2.5)
# for i, key in enumerate(data.keys()):
for i, key in enumerate(keys):
    print(key)
    sub = np.stack(data[key])

    # sub = np.abs(h3dmin - sub)
    std = sub.std(0)

    if key in ["np", "anp"]:
        continue
    elif key == "gp":
        plt.plot(sub.mean(0), label=Names[key][0], color="black", ls="dotted", lw=3)
        plt.fill_between(
            np.arange(sub.shape[1]),
            y1=sub.mean(0) - std,
            y2=sub.mean(0) + std,
            alpha=0.1,
            color="black",
            interpolate=False,
        )
    else:
        if "c" in key:
            plt.plot(sub.mean(0), label=Names[key][0], color=Names[key][1], ls="dashed")
            plt.fill_between(
                np.arange(sub.shape[1]),
                y1=sub.mean(0) - std,
                y2=sub.mean(0) + std,
                color=Names[key][1],
                alpha=0.1,
                interpolate=False,
            )
        else:
            plt.plot(sub.mean(0), label=Names[key][0], color=Names[key][1])
            plt.fill_between(
                np.arange(sub.shape[1]),
                y1=sub.mean(0) - std,
                y2=sub.mean(0) + std,
                color=Names[key][1],
                alpha=0.1,
                interpolate=False,
            )
# plt.axhline(h3dmin, color="black", ls="dashed", alpha=0.3)
# plt.legend(loc="upper right")
sns.despine()
plt.xlim(0, sub.shape[1] - 1)
plt.ylim(0)
plt.xlabel("number of queries")
plt.ylabel(r"error $|\min_t$ $f(\mathbf{x}_t) - f_\min|$ ")
plt.title("Hartmann 3D")
plt.tight_layout()
plt.show()
