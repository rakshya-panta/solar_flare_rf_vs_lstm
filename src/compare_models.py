import json

import matplotlib.pyplot as plt
import pandas as pd

from config import DEFAULT_N_ESTIMATORS, RESULTS_DIR, ROOT

METRICS = ["TSS", "HSS", "CSS", "accuracy", "precision", "recall", "F1", "ROC-AUC"]

paths = {"baseline": RESULTS_DIR / "baseline.json"}

for folder in sorted(RESULTS_DIR.glob("n_estimators_*"), key=lambda folder: int(folder.name.split("_")[-1])):
    n = folder.name.split("_")[-1]
    for name in ["rf_last", "rf_summary"]:
        paths[f"{name} (n={n})"] = folder / f"{name}.json"

# for name in ["lstm_last", "lstm_shuffled", "lstm_ordered"]:
#     paths[name] = RESULTS_DIR / f"{name}.json"

# One row per model in each table
val_rows, test_rows = {}, {}
for name, path in paths.items():
    if not path.exists():
        continue
    results = json.loads(path.read_text())
    val_rows[name] = {"threshold": results["threshold"], **results["val"]}
    test_rows[name] = {"threshold": results["threshold"], **results["test"]} # ** unpacks the dictionary into key-value pairs without being nested in a dictionary

val_table = pd.DataFrame.from_dict(val_rows, orient="index").astype({"threshold": float})
test_table = pd.DataFrame.from_dict(test_rows, orient="index").astype({"threshold": float})
val_table.round(5).to_csv(RESULTS_DIR / "val_results.csv")  # round to 5 decimals
test_table.round(5).to_csv(RESULTS_DIR / "test_results.csv")
print(f"Results saved in {RESULTS_DIR / 'val_results.csv'} and {RESULTS_DIR / 'test_results.csv'}")


def plot_scores(table, title, ylabel, file_name):
    ax = table[METRICS].T.plot.bar(figsize=(12, 4.5), rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(ROOT / "plots" / file_name, dpi=200)
    print(f"Plot saved in {ROOT / 'plots' / file_name}")


# Compare random forests only, on validation: number of trees, and last measurement vs 12 hour summary
rf_rows = [name for name in val_table.index if name.startswith("rf_")]
plot_scores(val_table.loc[rf_rows], "Random Forests on the validation set",
            "Score on validation set (P4)", "rf_comparison.png")

# Final comparison on test: baseline vs the picked rf_summary
final_rows = [name for name in ["baseline", f"rf_summary (n={DEFAULT_N_ESTIMATORS})"] if name in test_table.index]
plot_scores(test_table.loc[final_rows], "Model comparison on the test set",
            "Score on test set (P5)", "model_comparison.png")


# Plot validation scores of rf models for different numbers of trees choices
tree_counts = sorted(int(folder.name.split("_")[-1]) for folder in RESULTS_DIR.glob("n_estimators_*"))
fig, axes = plt.subplots(3, 1, figsize=(7, 10.5))
for ax, metric in zip(axes, ["ROC-AUC", "CSS", "F1"]):
    for model, label, style in [("rf_last", "Last step (24 features)", "o-"),
                                ("rf_summary", "12-h summary (120 features)", "s--")]:
        values = [val_table.loc[f"{model} (n={n})", metric] for n in tree_counts]
        ax.plot(range(len(tree_counts)), values, style, label=label)
    ax.set_xticks(range(len(tree_counts)), tree_counts)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel(f"Validation {metric}")
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "plots" / "rf_n_estimators.png", dpi=200)
print(f"Plot saved in {ROOT / 'plots' / 'rf_n_estimators.png'}")

# Plot feature importance (%) of rf_summary for the default number of trees
importance = pd.read_csv(RESULTS_DIR / f"n_estimators_{DEFAULT_N_ESTIMATORS}" / "rf_importance.csv", index_col=0)["importance"] * 100
stat_names = {"max": "Maximum", "mean": "Mean", "min": "Minimum", "std": "Std. dev.", "slope": "Slope"}
by_stat = importance.groupby(importance.index.str.split("_").str[0]).sum().rename(index=stat_names)  # "max_TOTUSJH" -> "max"
by_param = importance.groupby(importance.index.str.split("_", n=1).str[1]).sum()  # "max_TOTUSJH" -> "TOTUSJH"

fig, axes = plt.subplots(2, 1, figsize=(7, 7))
for ax, values, xlabel in [(axes[0], by_stat.sort_values(), "Aggregate feature importance (%)"),
                           (axes[1], by_param.nlargest(5).sort_values(), "Aggregate parameter importance (%)")]:
    bars = ax.barh(values.index, values, height=0.6)
    ax.bar_label(bars, fmt="%.0f%%", padding=5)
    ax.set_xlim(0, values.max() * 1.2)
    ax.set_xlabel(xlabel)
fig.tight_layout()
fig.savefig(ROOT / "plots" / "rf_importance.png", dpi=200)
print(f"Plot saved in {ROOT / 'plots' / 'rf_importance.png'}")
