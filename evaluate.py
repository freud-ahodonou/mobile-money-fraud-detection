import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score
)

from features import add_features


DATA_DIR = Path(".")

ID_COL = "id"
TARGET_COL = "fraud_flag"

train = pd.read_csv(DATA_DIR / "train.csv")

train_fe, freq_maps = add_features(
    train,
    fit=True
)

X = train_fe.drop(
    columns=[ID_COL, TARGET_COL]
)

y = train_fe[TARGET_COL]

X_tr, X_val, y_tr, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = joblib.load("fraud_model.pkl")

val_pred = model.predict_proba(X_val)[:, 1]

score = average_precision_score(
    y_val,
    val_pred
)

print("PR-AUC =", score)

pred = (val_pred > 0.5).astype(int)

print()
print(confusion_matrix(
    y_val,
    pred
))

print()
print(classification_report(
    y_val,
    pred
))

print("\nRecherche du meilleur seuil")

for seuil in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:

    pred_temp = (
        val_pred > seuil
    ).astype(int)

    score_temp = f1_score(
        y_val,
        pred_temp
    )

    print(
        "Seuil =",
        seuil,
        "F1 =",
        round(score_temp, 4)
    )