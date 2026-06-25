"""
train_model.py
==============
Run this ONCE to train all models and save them to ../models/
Usage: python train_model.py
"""

import os, pickle, json, zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
DATA_ZIP    = os.path.join(BASE_DIR, "..", "data", "covtype_csv.zip")   # put your zip here
DATA_CSV    = "covtype.csv"
SAMPLE_SIZE = 15000   # increase for better accuracy; full 581K is slow on KNN

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
]

# ── Load Dataset ───────────────────────────────────────────────────────────────
print(f"[1/5] Loading dataset from {DATA_ZIP} ...")
with zipfile.ZipFile(DATA_ZIP, "r") as z:
    with z.open(DATA_CSV) as f:
        df = pd.read_csv(f)

print(f"      Full dataset: {len(df):,} rows × {len(df.columns)} cols")
sample = df.sample(n=SAMPLE_SIZE, random_state=42)
print(f"      Using sample: {SAMPLE_SIZE:,} rows")

# ── Build Feature Matrix ───────────────────────────────────────────────────────
X_cont = sample[FEATURE_NAMES].values
X_wild = sample[[f"Wilderness_Area{i}" for i in range(1, 5)]].values
X_soil = sample[[f"Soil_Type{i}" for i in range(1, 41)]].values
X = np.hstack([X_cont, X_wild, X_soil])
y = sample["Cover_Type"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale Continuous Features (first 10 cols) ──────────────────────────────────
print("[2/5] Fitting scaler ...")
scaler = StandardScaler()
scaler.fit(X_train[:, :10])

X_train_s = X_train.copy(); X_train_s[:, :10] = scaler.transform(X_train[:, :10])
X_test_s  = X_test.copy();  X_test_s[:, :10]  = scaler.transform(X_test[:, :10])

# ── Train Models ───────────────────────────────────────────────────────────────
metrics = {}

def evaluate(name, model, y_pred):
    m = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred, average="weighted") * 100, 2),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, average="weighted") * 100, 2),
    }
    metrics[name] = m
    print(f"      Accuracy={m['accuracy']}%  F1={m['f1']}%  Precision={m['precision']}%  Recall={m['recall']}%")

print("[3/5] Training KNN ...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_s, y_train)
evaluate("knn", knn, knn.predict(X_test_s))

print("[3/5] Training Naïve Bayes ...")
nb = GaussianNB()
nb.fit(X_train_s, y_train)
evaluate("naive_bayes", nb, nb.predict(X_test_s))

print("[3/5] Training Random Forest ...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
evaluate("random_forest", rf, rf.predict(X_test_s))

print("[3/5] Training K-Means ...")
km = KMeans(n_clusters=7, random_state=42, n_init=10, max_iter=300)
km.fit(X_train_s)
# K-Means is unsupervised — no supervised metrics

# ── Save Everything ────────────────────────────────────────────────────────────
print("[4/5] Saving model files ...")
bundle = {
    "scaler":        scaler,
    "knn":           knn,
    "naive_bayes":   nb,
    "random_forest": rf,
    "kmeans":        km,
}
for name, obj in bundle.items():
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"      Saved → models/{name}.pkl")

metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"      Saved → models/metrics.json")

print("[5/5] Done! All models saved.\n")
print(json.dumps(metrics, indent=2))