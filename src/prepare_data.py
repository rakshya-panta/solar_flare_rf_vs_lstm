import tarfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ALL_PARTITIONS, DATA_DIR, PARAMS, RAW_DIR


def read_partition(name):
    X, y, flare, end_time = [], [], [], []
    with tarfile.open(RAW_DIR / f"{name}_instances.tar.gz", "r:gz") as tar:
        csv_files = (m for m in tar if m.name.endswith(".csv"))
        for member in tqdm(csv_files, desc=name):
            df = pd.read_csv(tar.extractfile(member), sep="\t", usecols=PARAMS)
            if len(df) != 60:
                continue

            # File names: M1.7@7623:Primary_ar3721_..._e2013-11-02T07:48:00.csv or "FQ_ar3503_..."
            # start with the flare class
            file_name = member.name.split("/")[-1]
            flare_class = "FQ" if file_name.startswith("FQ") else file_name[0]

            X.append(df[PARAMS].to_numpy(dtype=float))
            y.append(int(flare_class in ("M", "X")))
            flare.append(flare_class)
            end_time.append(file_name.split("_e")[-1].replace(".csv", ""))

    X = np.array(X)
    X[~np.isfinite(X)] = np.nan
    return X, np.array(y), np.array(flare), np.array(end_time, dtype="datetime64[s]")


for name in ALL_PARTITIONS:
    X, y, flare, end_time = read_partition(name)
    np.savez(DATA_DIR / f"{name}.npz", X=X, y=y, flare=flare, end_time=end_time)
    print(name, "instances:", X.shape, "M/X rate:", round(y.mean(), 4))
