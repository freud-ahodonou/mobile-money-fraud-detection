import pandas as pd
import joblib

from pathlib import Path

from features import add_features


DATA_DIR = Path(".")

ID_COL = "id"

test = pd.read_csv(
    DATA_DIR / "test.csv"
)

model = joblib.load(
    "fraud_model.pkl"
)

freq_maps = joblib.load(
    "freq_maps.pkl"
)

# =====================
# Feature engineering
# =====================
test_fe, _ = add_features(
    test,
    freq_maps=freq_maps
)

X_test = test_fe.drop(
    columns=[ID_COL]
)

# =====================
# Prédictions
# =====================
pred = model.predict_proba(
    X_test
)[:, 1]

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    "target": pred
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv généré avec succès.")