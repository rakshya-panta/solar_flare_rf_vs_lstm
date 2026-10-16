import argparse

import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import ROOT, SEED, TRAIN
from data import load
from features import stat_param_names, summary_features

# e.g. python src/decision_tree_example.py --criterion entropy
parser = argparse.ArgumentParser()
parser.add_argument("--criterion", default="gini", choices=["gini", "entropy"])
args = parser.parse_args()

# Training data, rf_summary
X, y = load(TRAIN)
x = SimpleImputer(strategy="median").fit_transform(summary_features(X))

# One depth only: the tree picks the single feature and cut-off that best separates M/X from no M/X
tree = DecisionTreeClassifier(max_depth=1, criterion=args.criterion, random_state=SEED, class_weight="balanced").fit(x, y)

# Information gain = impurity before the split - weighted impurity after it
t = tree.tree_
left, right = t.children_left[0], t.children_right[0]
n = t.n_node_samples
after = (n[left] * t.impurity[left] + n[right] * t.impurity[right]) / n[0]
gain = t.impurity[0] - after

fig, ax = plt.subplots(figsize=(9, 5))
plot_tree(tree, feature_names=stat_param_names(), class_names=["No M/X", "M/X"],
          filled=True, impurity=True, rounded=True, fontsize=11, ax=ax)
ax.set_title(f"Decision tree of depth 1 ({args.criterion}), information gain = {gain:.4f}")
fig.tight_layout()
fig.savefig(ROOT / "plots" / f"decision_tree_depth1_{args.criterion}.png", dpi=200)
print(f"Plot saved in {ROOT / 'plots' / f'decision_tree_depth1_{args.criterion}.png'}")
