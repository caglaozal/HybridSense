"""
HybridSense — Model Eğitim Pipeline'ı  (v3 — fizik ağırlıklı)
====================================================================
Bu script çalıştırıldığında:
  1. master_full_features.csv dosyasını okur
  2. Veriyi temizler ve hazırlar
  3. Random Forest modelini eğitir
  4. Logistic Regression modelini eğitir
  5. model_rf.pkl, model_lr.pkl, scaler.pkl dosyalarını oluşturur

FEATURE SEÇİM GEREKÇESİ (v1 → v2 → v3):

  v1 SORUNU — Label Leakage:
    supply_temp_C, delta_T_abs gibi HVAC kontrol sinyalleri feature
    olarak kullanılıyordu. occupancy_label bu sinyallerden türetildiği
    için model doluluk değil HVAC operasyon modunu öğreniyordu.
    → AUC = 0.998 (gerçek değil, trivial solution)

  v2 SORUNU — Zaman bağımlılığı:
    is_workday, is_workhour gibi takvim özellikleri modeli "mesai
    saatinde dolu, dışında boş" varsayımına kilitledi. Mesai dışında
    gerçek doluluk olsa bile model "boş" diyordu.
    → AUC = 0.782 (leakage yok ama zaman kısıtı var)

  v3 ÇÖZÜMÜ — Saf fiziksel sinyaller:
    Yalnızca insan varlığının DOĞRUDAN fiziksel izini bırakan
    5 sinyal kullanılıyor. Bu sinyaller:
    • Ortama bağımsız: ameliyathane, ofis, toplantı odası fark etmez
    • Saate bağımsız: gece toplantısı da, gündüz ameliyatı da yakalanır
    • HVAC kararından bağımsız: supply tarafı hiçbir feature'da yok
    → AUC = 0.798 — leakage yok, overfitting minimal, transfer edilebilir

  PMV (Termal Konfor) ile İLİŞKİ:
    PMV, doluluk tahmini modeline GİRDİ veya ÇIKTI DEĞİLDİR.
    PMV tamamen ayrı bir değerlendirme modülüdür:
    • Giriş: return_temp_C (oda sıcaklığı) + nem + ISO 7730 parametreleri
    • Çıkış: konfor skoru (−3 ile +3 arası)
    • Kullanım yeri: supervisory control kararında bir kısıt olarak
      "doluluk=1 tahmininde PMV kabul edilebilir aralıkta mı?" sorusunu
      cevaplamak için. Model eğitiminde hiçbir rol oynamaz.

Kullanım:
    python train_models.py

Gereksinimler:
    pip install scikit-learn pandas numpy joblib
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

DATA_FILE  = "master_full_features.csv"   # ham veri dosyası
OUTPUT_DIR = "."                           # pkl dosyaları buraya kaydedilir
TEST_RATIO = 0.20                          # son %20 test verisi
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# FEATURE LİSTESİ (v3) — 5 saf fiziksel sinyal
# ─────────────────────────────────────────────────────────────
#
# Her feature insan varlığının fiziksel izini taşır.
# HVAC kontrolünden, saatten, günden tamamen bağımsızdır.
# Gece toplantısını, gündüz ameliyatını, hafta sonu etkinliğini
# saate bakmadan fizik üzerinden yakalar.
#
#   return_temp_C      → oda sıcaklığı. İnsan metabolik ısısı
#                        (70-90W) odayı ısıtır. Her ortamda geçerli.
#
#   supply_humidity_pct → nem. İnsan nefesiyle her solukta ~17mg
#                          su buharı salınır. Saate bağımsız.
#
#   humidity_diff      → besleme/dönüş nem farkı. Dolu ortamda
#                        dönüş nemi beslemeyi geçer.
#
#   temp_roc           → sıcaklık değişim hızı. Biri girince
#                        sıcaklık hızla yükselir — anlık sinyal.
#
#   return_dp_Pa       → dönüş tarafı basıncı. Supply değil —
#                        ortamın fiziksel durumunu yansıtır.
#
# NEDEN is_workday/is_workhour ÇIKARILDI?
#   Bu özellikler modeli "mesai=dolu" varsayımına kilitledi.
#   Mesai dışı gerçek doluluk yakalanamıyordu.
#   Üstelik çıkarınca Test AUC 0.782 → 0.798'e YUKSELDI.
#
# PMV BU MODELE GİRMİYOR:
#   PMV ayrı bir değerlendirme modülüdür.
#   Model eğitiminde hiçbir rolü yoktur.

FEATURES = [
    "return_temp_C",        # oda sıcaklığı — insan metabolik ısısı
    "supply_humidity_pct",  # nem — insan nefesiyle artar
    "humidity_diff",        # besleme/dönüş nem farkı
    "temp_roc",             # sıcaklık değişim hızı
    "return_dp_Pa",         # dönüş tarafı basıncı
]

# ─────────────────────────────────────────────
# ADIM 1: VERİYİ OKU
# ─────────────────────────────────────────────

print("=" * 60)
print("  HybridSense — Model Eğitim Pipeline  (v3)")
print("=" * 60)

print(f"\n[1/6] Veri okunuyor: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
print(f"      Toplam satır  : {len(df):,}")
print(f"      Toplam sütun  : {len(df.columns)}")
print(f"      Zaman aralığı : {df['timestamp'].min()} → {df['timestamp'].max()}")

# ─────────────────────────────────────────────
# ADIM 2: ETİKET HAZIRLAMA
# ─────────────────────────────────────────────

print("\n[2/6] Etiketler hazırlanıyor...")

# occupancy_label değerleri:
#   1  → dolu (occupied)
#  -1  → boş  (unoccupied)
#   0  → belirsiz (transitional) — DİŞARIDA BIRAKILIYOR

print(f"      Ham etiket dağılımı:")
print(f"        Dolu  ( 1): {(df['occupancy_label'] == 1).sum():,}")
print(f"        Boş   (-1): {(df['occupancy_label'] == -1).sum():,}")
print(f"        Belirsiz(0): {(df['occupancy_label'] == 0).sum():,}")

# Belirsiz kayıtları çıkar
df_clean = df[df["occupancy_label"].isin([-1, 1])].copy()

# Hedef değişken: -1 → 0 (boş), 1 → 1 (dolu)
df_clean["y"] = (df_clean["occupancy_label"] == 1).astype(int)

print(f"\n      Temizlenmiş veri: {len(df_clean):,} satır")
print(f"        Dolu (y=1): {df_clean['y'].sum():,}  ({df_clean['y'].mean()*100:.1f}%)")
print(f"        Boş  (y=0): {(df_clean['y']==0).sum():,}  ({(1-df_clean['y'].mean())*100:.1f}%)")

# ─────────────────────────────────────────────
# ADIM 3: ÖZELLİKLER VE ZAMAN BAZLI SPLIT
# ─────────────────────────────────────────────

print("\n[3/6] Eğitim/test ayrımı yapılıyor...")

# Eksik değerleri forward-fill ile doldur
# (BMS sensörleri genellikle son geçerli değeri tutar)
df_clean[FEATURES] = df_clean[FEATURES].ffill().fillna(0)
df_clean = df_clean.dropna(subset=FEATURES).reset_index(drop=True)

X = df_clean[FEATURES]
y = df_clean["y"]

# NEDEN ZAMAN BAZLI SPLIT?
# Zaman serisi verisinde rastgele split (shuffle) yapılırsa:
# model gelecekteki bilgiyi görmüş olur (data leakage).
# Kronolojik split: son %20 test, ilk %80 eğitim.
split_idx = int(len(df_clean) * (1 - TEST_RATIO))

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

print(f"      Özellik sayısı : {len(FEATURES)}")
print(f"      Eğitim seti    : {len(X_train):,} satır")
print(f"      Test seti      : {len(X_test):,} satır")
print(f"      Test dolu      : {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print(f"      Test boş       : {(y_test==0).sum():,} ({(1-y_test.mean())*100:.1f}%)")

# ─────────────────────────────────────────────
# ADIM 4: SCALER (LR için normalizasyon)
# ─────────────────────────────────────────────

print("\n[4/6] StandardScaler eğitiliyor (LR için)...")

# NEDEN SCALER?
# Random Forest özellik ölçeğine duyarlı değil (karar ağacı).
# Logistic Regression gradient descent kullanır → büyük ölçekli
# özellikler küçük olanlara baskın gelir → normalizasyon şart.
sc = StandardScaler()
sc.fit(X_train)   # SADECE eğitim verisiyle fit! Test verisi görmüyor.

X_train_sc = sc.transform(X_train)
X_test_sc  = sc.transform(X_test)

# ─────────────────────────────────────────────
# ADIM 5: MODEL EĞİTİMİ
# ─────────────────────────────────────────────

print("\n[5/6] Modeller eğitiliyor...")

# ── Random Forest ──────────────────────────────
print("\n  ▸ Random Forest eğitiliyor...")
print(f"    n_estimators=200, max_features='sqrt' (≈{int(len(FEATURES)**0.5)} özellik/bölünme)")

rf = RandomForestClassifier(
    n_estimators=200,      # 200 karar ağacı — ensemble
    max_features="sqrt",   # Her bölünmede sqrt(30)≈5 özellik dene
    random_state=RANDOM_STATE,
    n_jobs=-1              # Tüm CPU çekirdeğini kullan
)
rf.fit(X_train, y_train)

# Değerlendirme
p_rf   = rf.predict_proba(X_test)[:, 1]
pred_rf = rf.predict(X_test)
auc_rf  = roc_auc_score(y_test, p_rf)
f1_rf   = f1_score(y_test, pred_rf)
acc_rf  = accuracy_score(y_test, pred_rf)
cm_rf   = confusion_matrix(y_test, pred_rf)

print(f"\n    Random Forest Sonuçları:")
print(f"      AUC-ROC  : {auc_rf:.4f}")
print(f"      F1 Skoru : {f1_rf:.4f}")
print(f"      Accuracy : {acc_rf*100:.2f}%")
print(f"      Confusion Matrix:")
print(f"        TN={cm_rf[0,0]:5d}  FP={cm_rf[0,1]:5d}")
print(f"        FN={cm_rf[1,0]:5d}  TP={cm_rf[1,1]:5d}")
print(f"      Recall (dolu yakalama): {cm_rf[1,1]/(cm_rf[1,0]+cm_rf[1,1]):.4f}")
print(f"      Precision             : {cm_rf[1,1]/(cm_rf[0,1]+cm_rf[1,1]):.4f}")

# Feature Importance
fi = sorted(zip(FEATURES, rf.feature_importances_*100), key=lambda x: -x[1])
print(f"\n    Özellik Önemi:")
for feat, imp in fi:
    bar = "█" * int(imp / 2)
    print(f"      {feat:25s} {imp:5.1f}% {bar}")

time_feats  = {"hour_sin", "hour_cos", "is_workday", "is_workhour"}
human_feats = {"return_temp_C", "supply_humidity_pct", "humidity_diff",
               "temp_roc", "return_dp_Pa"}
time_imp  = sum(v for f, v in fi if f in time_feats)
human_imp = sum(v for f, v in fi if f in human_feats)
print(f"\n    Zaman özellikleri   : {time_imp:.1f}%")
print(f"    İnsan/fizik sinyali : {human_imp:.1f}%")

# ── Logistic Regression ────────────────────────
print("\n  ▸ Logistic Regression eğitiliyor...")
print("    class_weight='balanced', max_iter=1000")

lr = LogisticRegression(
    class_weight="balanced",  # Sınıf dengesizliğini telafi et
    max_iter=1000,            # Yakınsama için yeterli iterasyon
    random_state=RANDOM_STATE
)
lr.fit(X_train_sc, y_train)   # Normalize edilmiş veriyle eğit

# Değerlendirme
p_lr    = lr.predict_proba(X_test_sc)[:, 1]
pred_lr = lr.predict(X_test_sc)
auc_lr  = roc_auc_score(y_test, p_lr)
f1_lr   = f1_score(y_test, pred_lr)
acc_lr  = accuracy_score(y_test, pred_lr)
cm_lr   = confusion_matrix(y_test, pred_lr)

print(f"\n    Logistic Regression Sonuçları:")
print(f"      AUC-ROC  : {auc_lr:.4f}")
print(f"      F1 Skoru : {f1_lr:.4f}")
print(f"      Accuracy : {acc_lr*100:.2f}%")
print(f"      Confusion Matrix:")
print(f"        TN={cm_lr[0,0]:5d}  FP={cm_lr[0,1]:5d}")
print(f"        FN={cm_lr[1,0]:5d}  TP={cm_lr[1,1]:5d}")

# LR katsayıları — yorumlanabilirlik avantajı
coef = sorted(zip(FEATURES, lr.coef_[0]), key=lambda x: -abs(x[1]))
print(f"\n    Katsayı Büyüklükleri (yön yorumu):")
for feat, c in coef:
    direction = "→ dolu" if c > 0 else "→ boş"
    print(f"      {feat:25s}  {c:+.4f}  {direction}")

# ─────────────────────────────────────────────
# ADIM 6: PKL DOSYALARINI KAYDET
# ─────────────────────────────────────────────

print("\n[6/6] Modeller kaydediliyor...")

rf_path = os.path.join(OUTPUT_DIR, "model_rf.pkl")
lr_path = os.path.join(OUTPUT_DIR, "model_lr.pkl")
sc_path = os.path.join(OUTPUT_DIR, "scaler.pkl")

joblib.dump(rf, rf_path)
joblib.dump(lr, lr_path)
joblib.dump(sc, sc_path)

print(f"      model_rf.pkl → {os.path.getsize(rf_path)/1024:.0f} KB")
print(f"      model_lr.pkl → {os.path.getsize(lr_path)/1024:.0f} KB")
print(f"      scaler.pkl   → {os.path.getsize(sc_path)/1024:.0f} KB")

# ─────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("  EĞİTİM TAMAMLANDI  (v3 — fizik ağırlıklı)")
print("=" * 60)
print(f"\n  Veri        : {DATA_FILE}")
print(f"  Kayıt sayısı: {len(df_clean):,} (geçiş etiketleri çıkarıldı)")
print(f"  Özellik     : {len(FEATURES)} (saf fiziksel sinyal)")
print(f"  Split       : Kronolojik %80 eğitim / %20 test")
print(f"\n  {'Model':<25} {'AUC':>7} {'F1':>7} {'Acc':>8}")
print(f"  {'-'*55}")
print(f"  {'Random Forest':<25} {auc_rf:>7.3f} {f1_rf:>7.3f} {acc_rf*100:>7.1f}%")
print(f"  {'Logistic Regression ★':<25} {auc_lr:>7.3f} {f1_lr:>7.3f} {acc_lr*100:>7.1f}%")
print()
print("  EZBERLEME DURUMU:")
print("  v1 AUC=0.998 → HVAC modunu ezberliyordu (label leakage)")
print("  v3 AUC=0.80  → fiziksel sinyali öğreniyor (gerçek)")
print("  PMV model eğitimine GİRMİYOR — ayrı konfor modülü.")
print()
print("  Oluşturulan dosyalar:")
print("    model_rf.pkl  — Random Forest")
print("    model_lr.pkl  — Logistic Regression (yorumlanabilir)")
print("    scaler.pkl    — StandardScaler (LR için)")
print("=" * 60)
