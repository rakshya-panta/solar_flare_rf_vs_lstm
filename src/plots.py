import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ALL_PARTITIONS, DATA_DIR, PARAMS, RESULTS_DIR, ROOT, TRAIN, VAL

PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

BLUE = "tab:blue"
ORANGE = "tab:orange"

plt.style.use("seaborn-v0_8-whitegrid")


def partition_label(name):
    role = "train" if name in TRAIN else "val" if name in VAL else "test"
    return f"P{name[-1]} ({role})"


def save(fig, file_name, top=1.0):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(PLOTS_DIR / file_name, dpi=200)
    plt.close(fig)
    print("saved", file_name)


def mark_partitions(ax):
    for name in ALL_PARTITIONS:
        times = info.loc[info.partition == name, "end_time"]
        if name != ALL_PARTITIONS[0]:
            ax.axvline(times.min(), color=ORANGE, linewidth=1)
        middle = times.min() + (times.max() - times.min()) / 2
        ax.text(middle, 1.02, partition_label(name), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=ORANGE, fontsize=9)


####### Load data 
# end_time: the timestamp of the last time step of a instance.
# last_time_step_values: 24 parameters of the last time step of every training instance.
rows, last_time_step_values, missing = [], [], {}
mins, maxs, samples = [], [], []
for name in ALL_PARTITIONS:
    data = np.load(DATA_DIR / f"{name}.npz")
    X = data["X"]
    rows.append(pd.DataFrame({
        "partition": name,
        "end_time": data["end_time"],
        "label": data["y"],
        "flare": data["flare"],
    }))
    missing[partition_label(name)] = np.isnan(X).mean(axis=(0, 1)) * 100
    if name in TRAIN:
        last_time_step_values.append(X[:, -1]) # last time step values of all training instances
    if name == "partition1": # to create plot 07
        flare_example = X[np.where(data["flare"] == "M")[0][0]]
        quiet_example = X[np.where((data["flare"] == "FQ") & ~np.isnan(X).any(axis=(1, 2)))[0][0]]
    # to create plot 08
    mins.append(np.nanmin(X, axis=(0, 1))) 
    maxs.append(np.nanmax(X, axis=(0, 1)))
    samples.append(X[::10].reshape(-1, len(PARAMS)))  # every 10th instance, for percentiles

info = pd.concat(rows, ignore_index=True) # one row per instance, with partition, end_time, label, flare
last = pd.DataFrame(np.concatenate(last_time_step_values), columns=PARAMS) # last time step values of all training instances
monthly = info.set_index("end_time").resample("MS") # group by month, for plots 01 and 03


######### Plots 
def plot_instances_over_time():
    counts = monthly.size()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(counts.index, counts.values, width=25, color=BLUE)
    mark_partitions(ax)
    ax.set_title("Instances per month (bar chart)", pad=24)
    ax.set_xlabel("Month (end of observation window)")
    ax.set_ylabel("Number of instances")
    save(fig, "01_instances_over_time.png")

def plot_instances_per_partition():
    counts = info.groupby("partition")["label"].agg(["size", "sum"]).reindex(ALL_PARTITIONS)
    names = [partition_label(p) for p in ALL_PARTITIONS]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    bars = left.bar(names, counts["size"], width=0.6, color=BLUE)
    left.bar_label(bars, labels=[f"{n:,}" for n in counts["size"]], padding=3, color="black")
    left.set_ylim(top=counts["size"].max() * 1.15)
    left.set_title("All instances")
    left.set_ylabel("Number of instances")

    bars = right.bar(names, counts["sum"], width=0.6, color=ORANGE)
    rates = counts["sum"] / counts["size"] * 100
    right.bar_label(bars, labels=[f"{r:.2f}%\n{n:,}" for r, n in zip(rates, counts["sum"])],
                    padding=3, color="black")
    right.set_ylim(top=counts["sum"].max() * 1.25)
    right.set_title("M/X-flare instances (% of the partition)")
    right.set_ylabel("Number of M/X instances")
    for ax in (left, right):
        ax.yaxis.set_major_formatter("{x:,.0f}")  # 100,000 instead of 100000

    fig.suptitle(f"Instances per partition (bar charts, total {len(info):,}, of which {info.label.sum():,} M/X)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    save(fig, "02_instances_per_partition.png", top=0.96)
    
def plot_mx_over_time():
    counts = monthly["label"].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(counts.index, counts.values, width=25, color=ORANGE)
    mark_partitions(ax)
    ax.set_title("M/X-flare instances per month (bar chart)", pad=24)
    ax.set_xlabel("Month (end of observation window)")
    ax.set_ylabel("Number of M/X instances")
    save(fig, "03_mx_instances_over_time.png")


def plot_class_counts():
    classes = ["X", "M", "C", "B", "FQ"]
    counts = info.groupby(["flare", "partition"]).size().unstack().reindex(classes)
    x = np.arange(len(classes))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, name in enumerate(ALL_PARTITIONS):
        ax.bar(x + (k - 2) * width, counts[name], width * 0.85,
               label=partition_label(name))
    ax.set_yscale("log")
    ax.set_ylim(top=counts.max().max() * 20)  # room for the legend
    ax.set_xticks(x, ["X", "M", "C", "B", "FQ (flare-quiet)"])
    ax.set_title("Instances per flare class and partition (grouped bar chart)")
    ax.set_xlabel("Largest flare in the next 24 hours")
    ax.set_ylabel("Number of instances (log scale)")
    ax.legend(title="Partition", ncol=5, loc="upper left")
    save(fig, "04_class_counts.png")


def plot_missing_by_parameter():
    table = pd.DataFrame(missing, index=PARAMS)
    fig, ax = plt.subplots(figsize=(6, 9))
    image = ax.imshow(table, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(PARAMS)), PARAMS)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Values missing (%)")
    ax.set_title("Missing values per parameter (heatmap)")
    save(fig, "05_missing_by_parameter.png")


def plot_correlation():
    corr = last[PARAMS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PARAMS)), PARAMS, rotation=90)
    ax.set_yticks(range(len(PARAMS)), PARAMS)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Spearman correlation", shrink=0.8)
    ax.set_title("Correlation between parameters (heatmap, last time step of training partitions)")
    save(fig, "06_correlation.png")


def plot_example_instance():
    hours = np.arange(60) * 12 / 60
    fig, axes = plt.subplots(6, 4, figsize=(16, 19), sharex=True)
    for j, (ax, param) in enumerate(zip(axes.flat, PARAMS)):
        ax.plot(hours, flare_example[:, j], color=ORANGE, linewidth=2, label="Followed by an M-class flare")
        ax.plot(hours, quiet_example[:, j], color=BLUE, linewidth=2, label="Followed by no flare (FQ)")

        # Log scale only when all values are positive and they span more than 100x
        values = np.concatenate([flare_example[:, j], quiet_example[:, j]])
        if (values > 0).all() and values.max() / values.min() > 100:
            ax.set_yscale("log")
            ax.set_title(f"{param} (log scale)", fontsize=11)
        else:
            ax.set_title(param, fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Hours into the observation window")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2)
    fig.suptitle("Two example instances, all 24 parameters (line charts, partition 1)", fontsize=14, fontweight="bold",
                 x=0.01, ha="left")
    save(fig, "07_example_instance.png", top=0.97)


def plot_parameter_ranges():
    values = np.concatenate(samples)
    ranges = pd.DataFrame({
        "min": np.min(mins, axis=0),
        "p1": np.nanpercentile(values, 1, axis=0), # 1% of the values are below this value, low end
        "median": np.nanpercentile(values, 50, axis=0),
        "p99": np.nanpercentile(values, 99, axis=0), # 99% of the values are below this value, high end
        "max": np.max(maxs, axis=0),
    }, index=PARAMS).sort_values("median")
    ranges.to_csv(RESULTS_DIR / "parameter_ranges.csv", index_label="parameter")

    fig, ax = plt.subplots(figsize=(11, 9))
    rows = np.arange(len(ranges))
    ax.hlines(rows, ranges["min"], ranges["max"], color="gray", linewidth=1.5, label="Min to max")
    ax.hlines(rows, ranges["p1"], ranges["p99"], color=BLUE, linewidth=7, label="Middle 98% (1st to 99th percentile)")
    ax.scatter(ranges["median"], rows, color="black", s=30, zorder=3, edgecolor="white", linewidth=1.5, label="Median")

    # Symmetric log scale
    ax.set_xscale("symlog", linthresh=0.01)
    ticks = [-1e25, -1e20, -1e15, -1e10, -1e5, -1, 0, 1, 1e5, 1e10, 1e15, 1e20, 1e25]
    labels = ["$-10^{25}$", "$-10^{20}$", "$-10^{15}$", "$-10^{10}$", "$-10^{5}$", "$-1$", "0",
              "1", "$10^{5}$", "$10^{10}$", "$10^{15}$", "$10^{20}$", "$10^{25}$"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(rows, ranges.index)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_title("Value range of each parameter (range plot, all partitions)")
    ax.set_xlabel("Value (symmetric log scale)")
    ax.legend(loc="upper left")
    save(fig, "08_parameter_ranges.png")



plot_instances_over_time()
plot_instances_per_partition()
plot_mx_over_time()
plot_class_counts()
plot_missing_by_parameter()
plot_correlation()
plot_example_instance()
plot_parameter_ranges()
