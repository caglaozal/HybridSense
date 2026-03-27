import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

try:
    from pythermalcomfort.models import pmv_ppd as _ptc_pmv_ppd
    USE_PTC = True
    print("[PMV] pythermalcomfort kütüphanesi yüklendi [OK]")
except ImportError:
    USE_PTC = False
    print("[PMV] pythermalcomfort bulunamadı → ISO 7730 fallback kullanılıyor")

CSV_PATH = "master_full_features.csv"
OUTPUT_DIR = "pmv_output"

MET = 2.0
CLO = 0.65
V_AIR = 0.25

SETBACKS = [0, 1, 2, 3]
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _pmv_ppd_fallback(tdb, tr, vr, rh, met, clo):
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (tdb + 235))
    icl = 0.155 * clo
    m = met * 58.15
    fcl = (1 + 1.29 * icl) if icl <= 0.078 else (1.05 + 0.645 * icl)
    hcf = 12.1 * np.sqrt(vr)
    taa = tdb + 273
    tra = tr + 273
    xn = (taa + (35.5 - tdb) / (3.5 * icl + 0.1)) / 100
    xf = xn
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * m + p2 * (tra / 100) ** 4

    for _ in range(150):
        xf = (xn + xf) / 2
        hc = max(hcf, 2.38 * abs(100 * xf - taa) ** 0.25)
        xn_ = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
        if abs(xn_ - xn) < 1e-6:
            xn = xn_
            break
        xn = xn_

    tcl = 100 * xn - 273
    hl1 = 3.05e-3 * (5733 - 6.99 * m - pa)
    hl2 = max(0, 0.42 * (m - 58.15))
    hl3 = 1.7e-5 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - tdb)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4)
    hl6 = fcl * hc * (tcl - tdb)
    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (m - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100 - 95 * np.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)

    return round(float(pmv), 4), round(float(ppd), 2)

def calc_pmv_ppd(tdb, tr, vr, rh, met, clo):
    if USE_PTC:
        result = _ptc_pmv_ppd(tdb=tdb, tr=tr, v=vr, rh=rh, met=met, clo=clo, standard='ISO')
        if isinstance(result, dict):
            return round(float(result['pmv']), 4), round(float(result['ppd']), 2)
        else:
            return round(float(result.pmv), 4), round(float(result.ppd), 2)
    else:
        return _pmv_ppd_fallback(tdb, tr, vr, rh, met, clo)

def hesapla_pmv_serisi(tdb_arr, rh_arr, vr=V_AIR, met=MET, clo=CLO):
    pmv_list, ppd_list = [], []
    for t, r in zip(tdb_arr, rh_arr):
        if pd.isna(t) or pd.isna(r):
            pmv_list.append(np.nan)
            ppd_list.append(np.nan)
        else:
            p, d = calc_pmv_ppd(tdb=t, tr=t, vr=vr, rh=r, met=met, clo=clo)
            pmv_list.append(p)
            ppd_list.append(d)
    return np.array(pmv_list), np.array(ppd_list)

print("=" * 60)
print("HybridSense — PMV Analizi")
print("=" * 60)

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], index_col="timestamp")

hour = df.index.hour
dow = df.index.dayofweek

if "occupancy_label" in df.columns:
    unocc = df["occupancy_label"] == -1
else:
    unocc = (hour >= 22) | (hour < 6) | (dow >= 5)

pmv_b, ppd_b = hesapla_pmv_serisi(
    df["return_temp_C"].values,
    df["supply_humidity_pct"].values
)

df["pmv_baseline"] = pmv_b
df["ppd_baseline"] = ppd_b

senaryo_sonuclari = {}

for sb in SETBACKS:
    if sb == 0:
        pmv_s = pmv_b.copy()
        ppd_s = ppd_b.copy()
    else:
        temp_s = df["return_temp_C"].copy()
        temp_s[unocc] -= sb
        pmv_s, ppd_s = hesapla_pmv_serisi(temp_s.values, df["supply_humidity_pct"].values)

    pmv_ser = pd.Series(pmv_s, index=df.index)

    n_total = pmv_ser.notna().sum()
    n_konfor = ((pmv_ser >= -0.5) & (pmv_ser <= 0.5)).sum()

    occ_mask = (hour >= 8) & (hour < 18) & (dow < 5)
    n_occ_risk = ((pmv_ser < -0.5) & occ_mask).sum()

    lbl = "Baseline" if sb == 0 else f"Setback {sb}°C"

    senaryo_sonuclari[lbl] = {
        "pmv_mean": float(np.nanmean(pmv_s)),
        "konfor_pct": n_konfor / n_total * 100,
        "occupied_risk": int(n_occ_risk),
    }

print("\nSONUÇ:")
for k, v in senaryo_sonuclari.items():
    print(k, v)
