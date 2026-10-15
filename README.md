# Solar Flare Prediction: Random Forest vs LSTM

COSC 50303 Data Mining project

Predicts whether an M- or X-class solar flare happens in the next 24 hours from 12 hours of
magnetic-field measurements (SWAN-SF dataset), comparing a Random Forest with an LSTM.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps

### Download the 5 SWAN-SF partitions
```bash
python src/download_data.py
```

### Preprocess the data to only keep 24 parameters and saves raw archives as .npz files
```bash
python src/prepare_data.py
```
### Explore the data and create plots to visualize them
```bash
python src/explore_data.py
python src/plots.py
```
### Train and evaluate models
```bash
python src/baseline.py # always predict no flare
python src/random_forest.py  # rf_last_timestamp and rf_summary
```

### Compare models
```bash
python src/compare_models.py
```

### (Optional) Random Forest experiments:
```bash
python src/random_forest.py --n_estimators 50              # different number of trees
python src/random_forest.py --n_estimators 300 --seed 43   # different seeds
```

## Project structure

```
dm_project/
├── src/
│   ├── config.py
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── data.py
│   ├── explore_data.py
│   ├── plots.py
│   ├── features.py
│   ├── metrics.py
│   ├── baseline.py
│   ├── random_forest.py
│   └── compare_models.py
├── data/
├── models/
├── results/
├── plots/
├── requirements.txt
```


## Key choices

- **Split:** partitions 1–3 train, 4 validation, 5 test (by time).
- **Main metric:** CSS, TSS, HSS, ROC-AUC, F1 and others are also reported.
- **Threshold:** chosen per model on validation to maximise CSS, then used unchanged on test.

