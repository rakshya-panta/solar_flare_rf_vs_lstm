import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    # average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import RESULTS_DIR, ROOT

CONFUSION_DIR = ROOT / "plots" / "confusion_matrices"


def evaluate(y_true, prob, threshold=0.5):
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tss = tp / (tp + fn) - fp / (fp + tn)
    hss = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    css = np.sqrt(tss * hss) if tss > 0 and hss > 0 else 0.0
    return {
        "TSS": tss,
        "HSS": hss,
        "CSS": css,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        # "PR-AUC": average_precision_score(y_true, prob),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def best_threshold(y_true, prob):
    """Pick the decision threshold with the highest CSS using validation data only."""
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2) # thresholds from 0.01 to 0.99 in increments of 0.01
    scores = [evaluate(y_true, prob, t)["CSS"] for t in thresholds]
    return float(thresholds[np.argmax(scores)])


def plot_confusion_matrix(name, results, folder=""):
    """Validation and test confusion matrices of one model"""
    (CONFUSION_DIR / folder).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, split, title in zip(axes, ["val", "test"], ["Validation (P4)", "Test (P5)"]):
        r = results[split]
        matrix = np.array([[r["TP"], r["FN"]],
                           [r["FP"], r["TN"]]])
        display = ConfusionMatrixDisplay(matrix, display_labels=["M/X", "No M/X"])
        display.plot(ax=ax, cmap="PuRd", values_format=",", colorbar=False, text_kw={"fontsize": 20})  # big numbers
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Predicted label", fontsize=14)
        ax.set_ylabel("True label", fontsize=14)
        ax.tick_params(labelsize=13)
        ax.set_title(f"{title}", pad=12, fontsize=15)
    fig.suptitle(f"Confusion matrix: {name}", fontweight="bold", fontsize=16)
    fig.tight_layout()
    fig.savefig(CONFUSION_DIR / folder / f"confusion_matrix_{name}.png", dpi=200)
    plt.close(fig)


def report(name, y_val, prob_val, y_test, prob_test, tune_threshold=True, folder=""):
    """tune_threshold=False for models that do not use a threshold (i.e., the baseline)."""
    threshold = best_threshold(y_val, prob_val) if tune_threshold else None
    cutoff = threshold if tune_threshold else 0.5
    results = {
        "threshold": threshold,
        "val": evaluate(y_val, prob_val, cutoff),
        "test": evaluate(y_test, prob_test, cutoff),
    }
    (RESULTS_DIR / folder).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / folder / f"{name}.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved in {RESULTS_DIR / folder / f'{name}.json'}")
    plot_confusion_matrix(name, results, folder)
    print(f"Confusion matrix saved in {CONFUSION_DIR / folder / f'confusion_matrix_{name}.png'}")
    return results
