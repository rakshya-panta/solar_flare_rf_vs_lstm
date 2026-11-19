import argparse
import copy

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import MODELS_DIR, SEED, load_model_config
from data import calc_pos_weight, load_splits
from metrics import CONFUSION_DIR, best_threshold, evaluate, report, save_hyp_tuning_results

CONFIG = load_model_config("lstm")
PARAMS = CONFIG["params"]

# Run: python src/lstm.py               to train lstm_last, lstm_shuffled and lstm_ordered
#      python src/lstm.py --hyp_tuning        for W&B sweep over the search space in the config (on lstm_ordered)
parser = argparse.ArgumentParser()
parser.add_argument("--hyp_tuning", action="store_true")
parser.add_argument("--trials", type=int, default=20)
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class FlareLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, n_params=24):
        super().__init__()
        self.lstm = nn.LSTM(n_params, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.out(self.dropout(h[-1])).squeeze(1)


def standardize(X, mean, std):
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, copy=False)  # missing value replaced with training mean
    return X.astype(np.float32)


def load_standardized_splits():
    """Train, validation and test data, each parameter standardised with training-set statistics only."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits()
    mean = np.nanmean(X_train, axis=(0, 1))
    std = np.nanstd(X_train, axis=(0, 1)) + 1e-8
    X_train = standardize(X_train, mean, std)
    X_val = standardize(X_val, mean, std)
    X_test = standardize(X_test, mean, std)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def shuffle_time(X):
    """Randomly reorder the 60 time steps inside each instance."""
    rng = np.random.default_rng(SEED)
    X = X.copy()
    for instance in X:
        rng.shuffle(instance)
    return X


def predict(model, X):
    model.eval()
    probs = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=1024):
            probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(probs)


def fit(X_train, y_train, X_val, y_val, params, on_epoch=None):
    """Train with early stopping on validation CSS. Returns the model at the epoch with best CSS."""
    torch.manual_seed(SEED)
    pos_weight = params["pos_weight"]  # weight of an M/X instance in the loss
    if pos_weight == "full_weight":
        pos_weight = calc_pos_weight(y_train)
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)

    model = FlareLSTM(params["hidden_size"], params["num_layers"], params["dropout"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    best_css, best_state, epochs_without_gain = -1, None, 0
    for epoch in range(1, params["max_epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation CSS at its best threshold, so different M/X weights are compared fairly
        prob_val = predict(model, X_val)
        val_css = evaluate(y_val, prob_val, best_threshold(y_val, prob_val))["CSS"]
        if on_epoch:
            on_epoch(epoch, np.mean(losses), val_css)

        if val_css > best_css:
            best_css, best_state, epochs_without_gain = val_css, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= params["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_css


def train(name, X_train, y_train, X_val, y_val, X_test, y_test):
    run = wandb.init(project=CONFIG["wandb_project"], group="lstm_training", name=name, config={
        **PARAMS,
        "sequence_length": X_train.shape[1],
        "seed": SEED,
    })

    def log(epoch, train_loss, val_css):
        print(f"{name} epoch {epoch}: loss={train_loss:.4f}  val CSS={val_css:.3f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_CSS": val_css})

    model, _ = fit(X_train, y_train, X_val, y_val, PARAMS, on_epoch=log)
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    results = report(name, y_val, predict(model, X_val), y_test, predict(model, X_test))

    # Final scores of the best epoch, plus its confusion matrix
    run.summary["threshold"] = results["threshold"]
    for split in ["val", "test"]:
        for metric, value in results[split].items():
            run.summary[f"{split}_{metric}"] = value
    wandb.log({"confusion_matrix": wandb.Image(str(CONFUSION_DIR / f"confusion_matrix_{name}.png"))})
    run.finish()


def hyperparameter_tuning(X_train, y_train, X_val, y_val, trials):
    """Bayesian search (W&B sweep) over the search space in the config, scored by validation CSS."""
    results = []

    def run_trial():
        with wandb.init(group="lstm_tuning") as run:
            params = {**PARAMS, **dict(run.config)}  # settings not in the search space come from the config

            def log(epoch, train_loss, val_css):
                print(f"trial {len(results) + 1} epoch {epoch}: val CSS={val_css:.3f}")
                run.log({"epoch": epoch, "train_loss": train_loss, "val_CSS": val_css})

            _, best_css = fit(X_train, y_train, X_val, y_val, params, on_epoch=log)
            run.log({"best_val_CSS": best_css})
            results.append({**dict(run.config), "best_val_CSS": best_css})

    CONFIG["sweep"]["parameters"]["pos_weight"]["max"] = calc_pos_weight(y_train)
    sweep_id = wandb.sweep(CONFIG["sweep"], project=CONFIG["wandb_project"])
    wandb.agent(sweep_id, function=run_trial, count=trials)
    save_hyp_tuning_results("lstm", results, "best_val_CSS")


(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_standardized_splits()

if args.hyp_tuning:
    # Search on full ordered sequences
    hyperparameter_tuning(X_train, y_train, X_val, y_val, args.trials)
else:
    # Latest measurement only (a sequence of length 1)
    train("lstm_last", X_train[:, -1:], y_train, X_val[:, -1:], y_val, X_test[:, -1:], y_test)

    # 12-hour history, order removed
    train("lstm_shuffled", shuffle_time(X_train), y_train, shuffle_time(X_val), y_val, shuffle_time(X_test), y_test)

    # 12-hour history, in order
    train("lstm_ordered", X_train, y_train, X_val, y_val, X_test, y_test)
