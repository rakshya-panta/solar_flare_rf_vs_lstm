import argparse
import joblib
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from config import MODELS_DIR, RESULTS_DIR, SEED, load_model_config
from data import calc_pos_weight, load_splits
from features import last_step_features, summary_features, stat_param_names
from metrics import best_threshold, evaluate, report, save_hyp_tuning_results

CONFIG = load_model_config("random_forest")
PARAMS = CONFIG["params"]

# Run: python src/random_forest.py for training
#      python src/random_forest.py --hyp_tuning --model rf_summary --trials 30 for W&B sweep
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=PARAMS["n_estimators"])
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--hyp_tuning", action="store_true")
parser.add_argument("--model", default="rf_summary", choices=["rf_last", "rf_summary"])
parser.add_argument("--trials", type=int, default=20)
args = parser.parse_args()
N_ESTIMATORS = args.n_estimators
RANDOM_SEED = args.seed
FOLDER = f"n_estimators_{N_ESTIMATORS}"
if RANDOM_SEED != SEED:  # try other seeds to check stability
    FOLDER = f"extra_seeds/n_estimators_{N_ESTIMATORS}/seed_{RANDOM_SEED}"

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits()


def make_model(params, seed):
    pos_weight = params["pos_weight"]
    if pos_weight == "full_weight":
        pos_weight = calc_pos_weight(y_train)
    return make_pipeline(
        SimpleImputer(strategy="median"), # fill missing values with median of each parameter
        RandomForestClassifier(
            n_estimators=params["n_estimators"],
            min_samples_leaf=params["min_samples_leaf"],
            max_depth=params["max_depth"],
            class_weight={0: 1.0, 1: pos_weight},
            n_jobs=-1,
            random_state=seed,
        ),
    )


def run(name, make_features):
    model = make_model({**PARAMS, "n_estimators": N_ESTIMATORS}, RANDOM_SEED)
    model.fit(make_features(X_train), y_train)

    # Save the selected model to use for inference
    if N_ESTIMATORS == PARAMS["n_estimators"] and RANDOM_SEED == SEED:
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"Model saved in {MODELS_DIR / f'{name}.joblib'}")

    prob_val = model.predict_proba(make_features(X_val))[:, 1]
    prob_test = model.predict_proba(make_features(X_test))[:, 1]
    report(name, y_val, prob_val, y_test, prob_test, folder=FOLDER)
    return model


def hyperparameter_tuning(name, make_features, trials):
    """Bayesian search (W&B sweep) over the search space in the config, scored by validation CSS."""
    train_x, val_x = make_features(X_train), make_features(X_val)
    results = []

    def run_trial():
        with wandb.init(group=f"{name}_tuning") as run:
            params = {**dict(run.config), "min_samples_leaf": int(run.config.min_samples_leaf)}
            model = make_model(params, SEED).fit(train_x, y_train)
            prob_val = model.predict_proba(val_x)[:, 1]
            threshold = best_threshold(y_val, prob_val)
            css = evaluate(y_val, prob_val, threshold)["CSS"]  # validation CSS at its best threshold
            run.log({"val_CSS": css, "threshold": threshold})
            results.append({**params, "threshold": threshold, "val_CSS": css})
            print(f"trial {len(results)}: val CSS={css:.4f}  {params}")

    CONFIG["sweep"]["parameters"]["pos_weight"]["max"] = calc_pos_weight(y_train)
    sweep_id = wandb.sweep(CONFIG["sweep"], project=CONFIG["wandb_project"])
    wandb.agent(sweep_id, function=run_trial, count=trials)
    save_hyp_tuning_results(name, results, "val_CSS")


feature_extractors = {"rf_last": last_step_features, "rf_summary": summary_features}

if args.hyp_tuning:
    hyperparameter_tuning(args.model, feature_extractors[args.model], args.trials)
else:
    # training with the default numbers or config
    # model using latest measurement only (24 features)
    run("rf_last", last_step_features)

    # model using 12-hour summary, mean, std, min, max and slope (120 features)
    rf_summary_model = run("rf_summary", summary_features)

    # The importance of a feature is the total purity gained from all its splits, across all trees, all 120 features add up to 1.
    # this is to see which statistic and which parameter is most important for the rf_summary model
    rf_summary_importance = pd.Series(rf_summary_model[-1].feature_importances_, index=stat_param_names(), name="importance")
    rf_summary_importance.sort_values(ascending=False).to_csv(RESULTS_DIR / FOLDER / "rf_summary_importance.csv")
