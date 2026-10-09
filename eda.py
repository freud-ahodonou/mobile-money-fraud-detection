# eda.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path(".")

train = pd.read_csv(DATA_DIR / "train.csv")

print("=== STATISTIQUES GLOBALES ===")
print(train.describe(include="all"))

print("\n=== NOMBRE DE TRANSACTIONS PAR OPÉRATION ===")
print(train["operation"].value_counts())

print("\n=== TAUX DE FRAUDE PAR OPÉRATION ===")
print(train.groupby("operation")["fraud_flag"].mean())

print("\n=== CORRÉLATION AVEC FRAUD_FLAG ===")
print(train.corr(numeric_only=True)["fraud_flag"].sort_values(ascending=False))

print("\n=== STATISTIQUES DES MONTANTS PAR OPÉRATION ===")
print(train.groupby("operation")["amount"].describe())