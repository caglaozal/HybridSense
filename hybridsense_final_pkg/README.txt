============================================================
  HybridSense — Çalıştırma Kılavuzu  (v3 — final)
============================================================

DOSYALAR:
  train_models.py              → Model eğitim scripti (v2, temiz 9 feature)
  hybridsense_dashboard.py     → Streamlit dashboard
  pmv_analizi.py               → PMV hesaplama scripti (v2, düzeltildi)
  hybridsense_correlation.ipynb→ Korelasyon analizi (Jupyter)
  model_rf.pkl                 → RF modeli (v2, AUC=0.782)
  model_lr.pkl                 → LR modeli (v2, AUC=0.806)
  scaler.pkl                   → StandardScaler (LR için)
  master_full_features.csv     → Eğitim verisi
  Demo_Odası__Kurulu_Sistemler_002.xlsx → Siemens demo oda

GEREKSİNİMLER:
  pip install scikit-learn pandas numpy joblib streamlit matplotlib seaborn

ÇALIŞTIRMA:
  1) Model yeniden eğitmek:   python train_models.py
  2) Dashboard:               streamlit run hybridsense_dashboard.py
  3) PMV analizi:             python pmv_analizi.py
  4) Notebook:                jupyter notebook hybridsense_correlation.ipynb

DEĞİŞİKLİKLER (v2 → v3):
  train_models.py:
    - 30 feature → 9 feature (HVAC kontrol sinyalleri çıkarıldı)
    - AUC 0.998 → 0.782/0.806 (label leakage giderildi)
    - Demo odaya transfer edilebilir

  pmv_analizi.py:
    - CSV: master_real_labels → master_full_features (düzeltildi)
    - MET: 2.0 → 1.2 (ameliyat → ofis/masa çalışması)
    - V_AIR: 0.25 → 0.1 m/s (daha doğru ofis değeri)
    - Unoccupied mask: supply_airflow<1200 → occupancy_label tabanlı
    - PMV yorumu: ASHRAE 170 hastane standardı notu eklendi
============================================================
