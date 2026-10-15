import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from config import RESULTS_DIR, SEED
from data import load_splits
from features import last_step_features, summary_features, stat_param_names
from metrics import report

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits()


def run(name, make_features):
    model = make_pipeline(
        SimpleImputer(strategy="median"), # fill missing values with median of each parameter
        RandomForestClassifier(
            n_estimators=300, # builds 300 decision trees
            min_samples_leaf=5,
            # class weight = total training samples / (2 × class labels)
            #  M/X weight = 204,559 / (2 × 4,079) ≈ 25.1
            # no-M/X weight = 204,559 / (2 × 200,480) ≈ 0.51
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
    )
    model.fit(make_features(X_train), y_train)

    prob_val = model.predict_proba(make_features(X_val))[:, 1]
    prob_test = model.predict_proba(make_features(X_test))[:, 1]
    report(name, y_val, prob_val, y_test, prob_test)
    return model


# model using latest measurement only (24 features)
run("rf_last", last_step_features)

# model using 12-hour summary, mean, std, min, max and slope (120 features)
model = run("rf_summary", summary_features)

# The importance of a feature is the total purity gained from all its splits, across all 300 trees, scaled so all 120 add up to 1.
importance = pd.Series(model[-1].feature_importances_, index=stat_param_names(), name="importance")
importance.sort_values(ascending=False).to_csv(RESULTS_DIR / "rf_importance.csv")
