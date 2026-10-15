import numpy as np
from sklearn.dummy import DummyClassifier

from config import TEST, TRAIN, VAL
from data import load_labels
from metrics import report

y_train, y_val, y_test = load_labels(TRAIN), load_labels(VAL), load_labels(TEST)

# Always predicts the majority class: "no M/X flare".
# does not care about inputs, so X is just zeros with the right number of rows.
model = DummyClassifier(strategy="most_frequent")
model.fit(np.zeros((len(y_train), 1)), y_train)

prob_val = model.predict_proba(np.zeros((len(y_val), 1)))[:, 1]
prob_test = model.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
report("baseline", y_val, prob_val, y_test, prob_test, tune_threshold=False)
