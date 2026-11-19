from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"

for folder in (RAW_DIR, DATA_DIR, RESULTS_DIR, MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# The 24 magnetic-field parameters used as model input
PARAMS = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "TOTFZ", "MEANPOT", "EPSZ", "MEANSHR", "SHRGT45",
    "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH", "TOTFY",
    "MEANJZD", "MEANALP", "TOTFX", "EPSY", "EPSX", "R_VALUE",
]

# Temporal split: earlier partitions train, later ones validate and test
TRAIN = ["partition1", "partition2", "partition3"]
VAL = ["partition4"]
TEST = ["partition5"]
ALL_PARTITIONS = TRAIN + VAL + TEST

SEED = 42


def load_model_config(name):
    """Model settings and W&B sweep search space from configs/<name>.yaml"""
    return yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())

DEFAULT_N_ESTIMATORS = load_model_config("random_forest")["params"]["n_estimators"]

