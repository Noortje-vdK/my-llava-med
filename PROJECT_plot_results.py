
import matplotlib.pyplot as plt
import numpy as np

metrics = [
    "RadGraph ER",
    "RaTEScore",
    "CHAIR Hallucination"
]

base = np.array([0.0429, 0.4086, 0.6756])
base_std = np.array([0.0533, 0.0595, 0.0000])

vcd = np.array([0.0441, 0.4028, 0.5676])
vcd_std = np.array([0.0544, 0.0658, 0.0000])

x = np.arange(len(metrics))
width = 0.35


plt.rcParams.update({
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
})


fig, ax = plt.subplots(figsize=(10, 5.5))

bars_base = ax.bar(
    x - width / 2,
    base,
    width,
    yerr=base_std,
    capsize=6,
    label="Base LLaVA-Med",
    error_kw={"elinewidth": 1.8, "capthick": 1.8}
)

bars_vcd = ax.bar(
    x + width / 2,
    vcd,
    width,
    yerr=vcd_std,
    capsize=6,
    label="LLaVA-Med + VCD",
    error_kw={"elinewidth": 1.8, "capthick": 1.8}
)

# Mean value labels, placed above the error bars
for bars, stds in [
    (bars_base, base_std),
    (bars_vcd, vcd_std)
]:
    for bar, std in zip(bars, stds):
        height = bar.get_height()

        ax.annotate(
            f"{height:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height + std
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

# Formatting
ax.set_ylabel("Score", fontsize=15, fontweight="bold")

ax.set_title(
    "Performance of X-Ray Report Generation:\n"
    "LLaVA-Med vs. LLaVA-Med + VCD",
    fontsize=17,
    fontweight="bold",
    pad=14
)

ax.set_xticks(x)
ax.set_xticklabels(
    metrics,
    fontsize=13,
    fontweight="bold"
)

ax.set_ylim(0, 1)

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    labelsize=12,
    width=1.5,
    length=6,
    top=True,
    right=True
)

ax.legend(
    fontsize=13,
    frameon=True,
    prop={"weight": "bold", "size": 13}
)

ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

# Compress the vertical dimension of the plotting box
ax.set_box_aspect(0.35)

plt.tight_layout()
plt.show()