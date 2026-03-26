import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

DATA_FILE = "master_full_features.csv"
OUTPUT_DIR = "."
TEST_RATIO = 0.20
RANDOM_STATE = 42

FEATURES = [
    "return_temp_C",
    "supply_humidity_pct",
    "humidity_diff",
    "temp_roc",
    "return_dp_Pa",
]

print("=" * 60)
print("HybridSense — Model Eğitim Pipeline (v3)")
print("=" * 60)

print("\n[1/6] Veri okunuyor:", DATA_FILE)
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

print("Toplam satır:", len(df))
print("Toplam sütun:", len(df.columns))
print("Zaman aralığı:", df["timestamp"].min(), "→", df["timestamp"].max())

print("\n[2/6] Etiketler hazırlanıyor...")

print("Dolu:", (df["occupancy_label"] == 1).sum())
print("Boş:", (df["occupancy_label"] == -1).sum())
print("Belirsiz:", (df["occupancy_label"] == 0).sum())

df_clean = df[df["occupancy_label"].isin([-1, 1])].copy()
df_clean["y"] = (df_clean["occupancy_label"] == 1).astype(int)

print("Temiz veri:", len(df_clean))
print("Dolu:", df_clean["y"].sum())
print("Boş:", (df_clean["y"] == 0).sum())

print("\n[3/6] Split...")

df_clean[FEATURES] = df_clean[FEATURES].ffill().fillna(0)
df_clean = df_clean.dropna(subset=FEATURES).reset_index(drop=True)

X = df_clean[FEATURES]
y = df_clean["y"]

split_idx = int(len(df_clean) * (1 - TEST_RATIO))

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("Train:", len(X_train))
print("Test:", len(X_test))

print("\n[4/6] Scaling...")

sc = StandardScaler()
sc.fit(X_train)

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

print("\n[5/6] Model training...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)

p_rf = rf.predict_proba(X_test)[:, 1]
pred_rf = rf.predict(X_test)

print("\nRandom Forest:")
print("AUC:", roc_auc_score(y_test, p_rf))
print("F1:", f1_score(y_test, pred_rf))
print("Acc:", accuracy_score(y_test, pred_rf))

cm = confusion_matrix(y_test, pred_rf)
print("Confusion Matrix:", cm)

fi = sorted(zip(FEATURES, rf.feature_importances_ * 100), key=lambda x: -x[1])

print("\nFeature Importance:")
for feat, imp in fi:
    print(feat, imp)

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE
)
lr.fit(X_train_sc, y_train)

p_lr = lr.predict_proba(X_test_sc)[:, 1]
pred_lr = lr.predict(X_test_sc)

print("\nLogistic Regression:")
print("AUC:", roc_auc_score(y_test, p_lr))
print("F1:", f1_score(y_test, pred_lr))
print("Acc:", accuracy_score(y_test, pred_lr))

cm = confusion_matrix(y_test, pred_lr)
print("Confusion Matrix:", cm)

coef = sorted(zip(FEATURES, lr.coef_[0]), key=lambda x: -abs(x[1]))

print("\nCoefficients:")
for feat, c in coef:
    direction = "dolu" if c > 0 else "boş"
    print(feat, c, direction)

print("\n[6/6] Saving...")

rf_path = os.path.join(OUTPUT_DIR, "model_rf.pkl")
lr_path = os.path.join(OUTPUT_DIR, "model_lr.pkl")
sc_path = os.path.join(OUTPUT_DIR, "scaler.pkl")

joblib.dump(rf, rf_path)
joblib.dump(lr, lr_path)
joblib.dump(sc, sc_path)

print("Saved:", rf_path, lr_path, sc_path)

print("\nDone")
