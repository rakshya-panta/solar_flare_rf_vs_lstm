import json

import pandas as pd

from config import RESULTS_DIR, ROOT

PICKED_N_ESTIMATORS = 300 # number of trees chosen after multiple experiments 
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
final_rows = [name for name in ["baseline", f"rf_summary (n={PICKED_N_ESTIMATORS})"] if name in test_table.index]
plot_scores(test_table.loc[final_rows], "Model comparison on the test set",
            "Score on test set (P5)", "model_comparison.png")
