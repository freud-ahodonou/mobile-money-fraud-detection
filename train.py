import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from features import add_features


DATA_DIR = Path(".")

ID_COL = "id"
TARGET_COL = "fraud_flag"

train = pd.read_csv(DATA_DIR / "train.csv")

# =====================
# Feature engineering
# =====================
train_fe, freq_maps = add_features(
    train,
    fit=True
)

X = train_fe.drop(
    columns=[ID_COL, TARGET_COL]
)

y = train_fe[TARGET_COL]

# =====================
# Split
# =====================
X_tr, X_val, y_tr, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================
# Modèle
# =====================
model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="PRAUC",
    class_weights=[1, 9],
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200
)

cat_features = ["operation"]

model.fit(
    X_tr,
    y_tr,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# =====================
# Sauvegarde
# =====================
joblib.dump(model, "fraud_model.pkl")
joblib.dump(freq_maps, "freq_maps.pkl")

print("Modèle sauvegardé.")