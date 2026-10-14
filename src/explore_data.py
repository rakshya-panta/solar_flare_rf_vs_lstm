import numpy as np
import pandas as pd

from config import ALL_PARTITIONS, DATA_DIR, PARAMS, RESULTS_DIR

rows = []
missing = {}
for name in ALL_PARTITIONS:
    data = np.load(DATA_DIR / f"{name}.npz")
    X, y, flare = data["X"], data["y"], data["flare"]

    row = {"partition": name, "instances": len(y), "M/X rate": y.mean()}
    row.update(pd.Series(flare).value_counts().reindex(["X", "M", "C", "B", "FQ"], fill_value=0))
    rows.append(row)
    missing[name] = np.isnan(X).mean(axis=(0, 1)) * 100

classes = pd.DataFrame(rows)
missing = pd.DataFrame(missing, index=PARAMS).round(2)

print("Instances and flare classes per partition:")
print(classes.round(4).to_string(index=False))
print("\nMissing values (%) per parameter:")
print(missing.to_string())

classes.to_csv(RESULTS_DIR / "data_classes.csv", index=False)
missing.to_csv(RESULTS_DIR / "data_missing.csv", index_label="parameter")

#visualize one instance
from config import DATA_DIR, PARAMS

data = np.load(DATA_DIR / "partition1.npz")
X, y, flare = data["X"], data["y"], data["flare"]

i = 0  # pick any instance number
print("\n ################### Visualizing one instance ###################")
# print("\nlabel:", y[i], "class:", flare[i])
# print(pd.DataFrame(X[i], columns=PARAMS))
df = pd.DataFrame(X[i], columns=PARAMS)
print(f"instance {i}: label={y[i]}  class={flare[i]}")
print(df)


