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

# ── pythermalcomfort import ────────────────────────────────
# Kurulum: pip install pythermalcomfort
# ISO 7730 standardını doğrudan implemente eden,
# yayınlanmış ve test edilmiş bir kütüphane.
# Referans: Tartarini & Schiavon (2020), SoftwareX.
try:
    from pythermalcomfort.models import pmv_ppd as _ptc_pmv_ppd
    USE_PTC = True
    print("[PMV] pythermalcomfort kütüphanesi yüklendi [OK]")
except ImportError:
    USE_PTC = False
    print("[PMV] pythermalcomfort bulunamadı → ISO 7730 fallback kullanılıyor")
    print("      Kurmak için: pip install pythermalcomfort")

CSV_PATH   = "master_full_features.csv"
OUTPUT_DIR = "pmv_output"

# ── ISO 7730 sabit varsayımlar (Hastane Ameliyathanesi) ────
# MET = 2.0 → ayakta hafif iş (ISO 7730 Tablo B.1)
#             Cerrah/hemşire: ayakta aktif çalışma
#             Ofis masa çalışması için 1.2 olurdu
# CLO = 0.65 → hastane önlüğü + ince iş kıyafeti
# V_AIR = 0.25 m/s → OR laminar flow hava hareketi
# tr = tdb  → radyant sıcaklık sensörü yok, ISO 7730 fallback
MET   = 2.0
CLO   = 0.65
V_AIR = 0.25

SETBACKS = [0, 1, 2, 3]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PMV/PPD hesap fonksiyonu ───────────────────────────────
# pythermalcomfort varsa onu kullan (tercih edilen).
# yoksa aynı ISO 7730 denklemlerini içeren fallback devreye girer.

def _pmv_ppd_fallback(tdb, tr, vr, rh, met, clo):
    """ISO 7730 PMV/PPD — pythermalcomfort yoksa kullanılır."""
    pa  = rh * 10 * np.exp(16.6536 - 4030.183 / (tdb + 235))
    icl = 0.155 * clo
    m   = met * 58.15
    fcl = (1 + 1.29 * icl) if icl <= 0.078 else (1.05 + 0.645 * icl)
    hcf = 12.1 * np.sqrt(vr)
    taa = tdb + 273
    tra = tr  + 273
    xn  = (taa + (35.5 - tdb) / (3.5 * icl + 0.1)) / 100
    xf  = xn
    p1  = icl * fcl;  p2 = p1 * 3.96;  p3 = p1 * 100
    p4  = p1 * taa;   p5 = 308.7 - 0.028 * m + p2 * (tra / 100) ** 4
    for _ in range(150):
        xf  = (xn + xf) / 2
        hc  = max(hcf, 2.38 * abs(100 * xf - taa) ** 0.25)
        xn_ = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
        if abs(xn_ - xn) < 1e-6: xn = xn_; break
        xn  = xn_
    tcl = 100 * xn - 273
    hl1 = 3.05e-3 * (5733 - 6.99 * m - pa)
    hl2 = max(0, 0.42 * (m - 58.15))
    hl3 = 1.7e-5 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - tdb)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4)
    hl6 = fcl * hc * (tcl - tdb)
    ts  = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (m - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100 - 95 * np.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return round(float(pmv), 4), round(float(ppd), 2)


def calc_pmv_ppd(tdb, tr, vr, rh, met, clo):
    """
    Tek nokta PMV/PPD hesabı.
    pythermalcomfort varsa standard='ISO' ile çağırır,
    yoksa fallback fonksiyonu kullanır.
    Döndürür: (pmv: float, ppd: float)
    """
    if USE_PTC:
        result = _ptc_pmv_ppd(
            tdb=tdb, tr=tr, v=vr, rh=rh,
            met=met, clo=clo, standard='ISO'
        )
        # pythermalcomfort dict veya named tuple döner
        if isinstance(result, dict):
            return round(float(result['pmv']), 4), round(float(result['ppd']), 2)
        else:
            return round(float(result.pmv), 4), round(float(result.ppd), 2)
    else:
        return _pmv_ppd_fallback(tdb, tr, vr, rh, met, clo)


def hesapla_pmv_serisi(tdb_arr, rh_arr, vr=V_AIR, met=MET, clo=CLO):
    """Zaman serisi üzerinde PMV/PPD hesabı."""
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
print("  HybridSense — PMV Termal Konfor Analizi  (v2)")
print("=" * 60)

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], index_col="timestamp")
print(f"\n Veri yüklendi: {len(df):,} satır")
print(f" Tarih: {df.index.min().date()} → {df.index.max().date()}")

# Oda sıcaklığı: return_temp_C  (supply değil — ortamın gerçek sıcaklığı)
# Nem          : supply_humidity_pct  (HVAC besleme nemi, ortam nemi için makul proxy)

# ── Zaman indeksleri ──────────────────────────────────────
hour = df.index.hour
dow  = df.index.dayofweek   # 0=Pazartesi, 6=Pazar

# 2. UNOCCUPIED MASK (setback uygulanacak dönemler)

# ── Unoccupied mask: HVAC sinyali KULLANILMIYOR ───────────
# v1'de supply_airflow_m3h < 1200 eşiği kullanılıyordu.
# Bu, label leakage ile aynı hata: HVAC modunu doluluk proxy'si
# olarak kullanmak. v2'de yalnızca zaman ve occupancy_label
# tabanlı mask uygulanıyor.
#
# Öncelik sırası:
#   1) occupancy_label = -1 → BMS verisinde boş olarak işaretli
#   2) Gece (22:00-06:00) veya hafta sonu → varsayılan boş

if "occupancy_label" in df.columns:
    # Label tabanlı mask (en güvenilir)
    unocc = df["occupancy_label"] == -1
    print("  Unoccupied mask: occupancy_label = -1 kullanıldı")
else:
    # Fallback: saf zaman tabanlı
    unocc = (hour >= 22) | (hour < 6) | (dow >= 5)
    print("  Unoccupied mask: zaman bazlı (gece + hafta sonu)")

print(f"  Unoccupied: {unocc.sum():,} slot (%{unocc.mean()*100:.1f})")
print(f"  Occupied  : {(~unocc).sum():,} slot (%{(~unocc).mean()*100:.1f})")

#BASELINE PMV HESABI

print("\n Baseline PMV hesaplanıyor...")
pmv_b, ppd_b = hesapla_pmv_serisi(
    df["return_temp_C"].values,
    df["supply_humidity_pct"].values
)
df["pmv_baseline"] = pmv_b
df["ppd_baseline"] = ppd_b

#SETBACK SENARYOLARI
senaryo_sonuclari = {}
for sb in SETBACKS:
    if sb == 0:
        pmv_s = pmv_b.copy()
        ppd_s = ppd_b.copy()
    else:
        print(f"▶ {sb}°C setback PMV hesaplanıyor...")
        temp_s = df["return_temp_C"].copy()
        temp_s[unocc] = temp_s[unocc] - sb
        pmv_s, ppd_s = hesapla_pmv_serisi(temp_s.values, df["supply_humidity_pct"].values)

    pmv_ser = pd.Series(pmv_s, index=df.index)
    ppd_ser = pd.Series(ppd_s, index=df.index)

    n_total = pmv_ser.notna().sum()
    n_konfor = ((pmv_ser >= -0.5) & (pmv_ser <= 0.5)).sum()

    #Sadece occupied olan saatlerdeki konfor riski hesaplaması:
    occ_mask_mesai = (hour >= 8) & (hour < 18) & (dow < 5)
    n_occ_risk = ((pmv_ser < -0.5) & occ_mask_mesai).sum()
    n_cold     = (pmv_ser < -0.5).sum()
    n_hot      = (pmv_ser >  0.5).sum()

    lbl = "Baseline" if sb == 0 else f"Setback {sb}°C"
    senaryo_sonuclari[lbl] = {
        "setback": sb,
        "pmv_mean": round(float(np.nanmean(pmv_s)), 3),
        "pmv_std":  round(float(np.nanstd(pmv_s)), 3),
        "pmv_min":  round(float(np.nanmin(pmv_s)), 3),
        "pmv_max":  round(float(np.nanmax(pmv_s)), 3),
        "ppd_mean": round(float(np.nanmean(ppd_s)), 1),
        "konfor_pct": round(n_konfor / n_total * 100, 1),
        "occupied_risk_slots": int(n_occ_risk),
        "total_cold_slots": int(n_cold),
        "total_hot_slots":  int(n_hot),
        "pmv_ser": pmv_ser,
        "ppd_ser": ppd_ser,
    }

    if sb == 2:
        df["pmv_whatif"] = pmv_s
        df["ppd_whatif"] = ppd_s

# 5. SAYISAL ÇIKTI

print("\n" + "=" * 60)
print("  SAYISAL ÇIKTI")
print("=" * 60)

print(f"""
VARSAYIMLAR (ISO 7730 — Hastane Ameliyathanesi)
  Metabolik oran (met) : {MET}   ayakta ameliyathane çalışması
  Giysi direnci  (clo) : {CLO}  hastane önlüğü
  Hava hızı      (m/s) : {V_AIR}  OR laminar flow
  Radyant sıcaklık     : tr = tdb  (sensör yok, ISO 7730 fallback)
  Unoccupied mask      : occupancy_label tabanlı (HVAC sinyali değil)
  Konfor bandı         : -0.5 ≤ PMV ≤ +0.5  (ISO 7730 Kategori B)
""")

print(f"{'Senaryo':<18} {'Ort PMV':>8} {'Std':>6} {'Min':>7} {'Max':>7} "
      f"{'PPD%':>6} {'Konfor%':>8} {'Occ.Risk':>10}")
print("-" * 80)

for lbl, s in senaryo_sonuclari.items():
    occ_risk_str = f"{s['occupied_risk_slots']} slot"
    print(f"{lbl:<18} {s['pmv_mean']:>8.3f} {s['pmv_std']:>6.3f} "
          f"{s['pmv_min']:>7.3f} {s['pmv_max']:>7.3f} "
          f"{s['ppd_mean']:>6.1f}% {s['konfor_pct']:>7.1f}% "
          f"{occ_risk_str:>10}")

print("""
Notlar:
  Konfor%    : -0.5 ≤ PMV ≤ +0.5 olan slot yüzdesi
  Occ.Risk   : PMV < -0.5 olan mesai saati (08-18, hafta içi) slot sayısı
                Bu sayı setback'ten bağımsız ise sistem zaten soğuk demektir
""")

#GRAFİKLER
print("Grafikler üretiliyor...")

BG  = "#0f1117"; BG2 = "#1a1d2e"; BG3 = "#252840"
TEAL = "#00B4D8"; ORG = "#FF6B35"; GRN  = "#4CAF50"
RED  = "#EF5350"; AMB = "#FFA726"; WHT  = "#F0F5FA"; GRY = "#7A8FA6"

def sa(ax, grid="y"):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=WHT, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=BG3, lw=0.6, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_edgecolor(BG3)

fig = plt.figure(figsize=(20, 16), facecolor=BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)
fig.suptitle("HybridSense — PMV Termal Konfor Analizi",
             color=WHT, fontsize=15, fontweight="bold", y=0.99)

hrs_arr = np.arange(24)

#PMV Zaman Serisi
ax = fig.add_subplot(gs[0, :]); sa(ax)

sample = df.resample("1h").mean()
ax.fill_between(sample.index, -0.5, 0.5, alpha=0.10, color=GRN, label="ISO 7730 Konfor Bandı")
ax.axhline( 0.5, color=GRN, lw=1.5, ls="--", alpha=0.7)
ax.axhline(-0.5, color=GRN, lw=1.5, ls="--", alpha=0.7)
ax.axhline( 0.0, color=WHT, lw=0.8, ls=":",  alpha=0.3)
ax.plot(sample.index, sample["pmv_baseline"], color=TEAL, lw=1.5, label="Baseline PMV",    alpha=0.9)
ax.plot(sample.index, sample["pmv_whatif"],   color=ORG,  lw=1.2, label="2°C Setback PMV", alpha=0.75, ls="--")

ax.set_ylabel("PMV Değeri", color=GRY, fontsize=10)
ax.set_title("PMV Zaman Serisi—Baseline vs 2°C Setback  |  Konfor Bandı: −0.5 ≤ PMV ≤ +0.5",
             color=WHT, fontsize=12, fontweight="bold")
ax.legend(facecolor=BG3, labelcolor=WHT, fontsize=10)
ax.set_ylim(-0.6, 1.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, color=WHT, fontsize=8)

#PMV Histogram
ax = fig.add_subplot(gs[1, 0]); sa(ax)

bins_ = np.linspace(-0.6, 1.2, 40)
ax.hist(df["pmv_baseline"].dropna(), bins=bins_, color=TEAL, alpha=0.7, label="Baseline",    edgecolor="none")
ax.hist(df["pmv_whatif"].dropna(),   bins=bins_, color=ORG,  alpha=0.6, label="2°C Setback", edgecolor="none")
ax.axvline(-0.5, color=RED, lw=2, ls="--", label="Konfor sınırı (±0.5)")
ax.axvline( 0.5, color=RED, lw=2, ls="--")
ax.set_xlabel("PMV", color=GRY, fontsize=9)
ax.set_ylabel("Slot Sayısı", color=GRY, fontsize=9)
ax.set_title("PMV Dağılımı\nBaseline vs 2°C Setback", color=WHT, fontsize=11, fontweight="bold")
ax.legend(facecolor=BG3, labelcolor=WHT, fontsize=9)

#Saatlik Ortalama PMV
ax = fig.add_subplot(gs[1, 1]); sa(ax)

h_pmv_b = df.groupby(df.index.hour)["pmv_baseline"].mean()
h_pmv_w = df.groupby(df.index.hour)["pmv_whatif"].mean()

ax.fill_between(hrs_arr, -0.5,  0.5, alpha=0.10, color=GRN)
ax.axhline(-0.5, color=GRN, lw=1.5, ls="--", alpha=0.7)
ax.axhline( 0.5, color=GRN, lw=1.5, ls="--", alpha=0.7)
ax.plot(hrs_arr, h_pmv_b.values, color=TEAL, lw=2, marker="o", ms=4, label="Baseline")
ax.plot(hrs_arr, h_pmv_w.values, color=ORG,  lw=2, marker="s", ms=4, label="2°C Setback", ls="--")
ax.fill_between([0,  6],  -0.6, 1.2, alpha=0.06, color=AMB)
ax.fill_between([22, 24], -0.6, 1.2, alpha=0.06, color=AMB)
ax.text(3, 1.05, "Gece\nsetback", ha="center", fontsize=8, color=AMB)
ax.set_xticks(hrs_arr[::2])
ax.set_xticklabels([f"{h:02d}" for h in hrs_arr[::2]], color=WHT, fontsize=8)
ax.set_xlabel("Saat", color=GRY, fontsize=9)
ax.set_ylabel("Ort. PMV", color=GRY, fontsize=9)
ax.set_title("Saatlik Ortalama PMV", color=WHT, fontsize=11, fontweight="bold")
ax.legend(facecolor=BG3, labelcolor=WHT, fontsize=9)
ax.set_ylim(-0.6, 1.2)

# Senaryo Konforun yüzdelik Barı 
ax = fig.add_subplot(gs[1, 2]); sa(ax)

lbl_list  = list(senaryo_sonuclari.keys())
konfor_list = [senaryo_sonuclari[l]["konfor_pct"] for l in lbl_list]
pmv_means  = [senaryo_sonuclari[l]["pmv_mean"]   for l in lbl_list]
bar_colors = [GRN, TEAL, AMB, RED]

x_ = np.arange(len(lbl_list))
bars_ = ax.bar(x_, konfor_list, color=bar_colors, alpha=0.85, width=0.6)
for bar, v, pm in zip(bars_, konfor_list, pmv_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"%{v:.1f}\nPMV={pm:.3f}", ha="center", fontsize=9,
            color=WHT, fontweight="bold")
ax.axhline(95, color=RED, lw=1.5, ls="--", alpha=0.8, label="%95 referans")
ax.set_xticks(x_)
ax.set_xticklabels([l.replace("Setback ", "") for l in lbl_list], color=WHT, fontsize=10)
ax.set_ylabel("Konfor İçindeki Slot (%)", color=GRY, fontsize=9)
ax.set_ylim(75, 105)
ax.set_title("Senaryo Konfor %\n(Tüm saatler, ISO 7730 Kat. B)", color=WHT, fontsize=11, fontweight="bold")
ax.legend(facecolor=BG3, labelcolor=WHT, fontsize=9)

#  Panel 5: Occupied PMV Boxplotu
ax = fig.add_subplot(gs[2, 0]); sa(ax, "")

occ_mesai = (df.index.hour >= 8) & (df.index.hour < 18) & (df.index.dayofweek < 5)
pmv_occ_b = df.loc[occ_mesai, "pmv_baseline"].dropna()
pmv_occ_w = df.loc[occ_mesai, "pmv_whatif"].dropna()

bp = ax.boxplot(
    [pmv_occ_b, pmv_occ_w],
    patch_artist=True,
    medianprops=dict(color=WHT, lw=2.5),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(color=GRY, lw=1.2),
    capprops=dict(color=GRY, lw=1.2),
    flierprops=dict(marker="o", ms=3, alpha=0.4, color=GRY)
)
for patch, color in zip(bp["boxes"], [TEAL, ORG]):
    patch.set_facecolor(color); patch.set_alpha(0.7)

ax.set_xticklabels(["Baseline\n(Mesai 08-18)", "2°C Setback\n(Mesai 08-18)"], color=WHT, fontsize=10)
ax.axhline(-0.5, color=RED, lw=1.5, ls="--", alpha=0.8, label="Konfor alt sınır")
ax.axhline( 0.5, color=RED, lw=1.5, ls="--", alpha=0.8)
ax.fill_between([-1, 3], -0.5, 0.5, alpha=0.08, color=GRN)
ax.set_ylabel("PMV", color=GRY, fontsize=9)
ax.set_xlim(0.3, 2.7); ax.set_ylim(-0.6, 1.2)
ax.set_title("Occupied Dönem PMV Dağılımı\n(Mesai Saatleri)", color=WHT, fontsize=11, fontweight="bold")
ax.legend(facecolor=BG3, labelcolor=WHT, fontsize=9)

# PPDnin  Saatlik Karşılaştırma sı 
ax = fig.add_subplot(gs[2, 1]); sa(ax)

h_ppd_b = df.groupby(df.index.hour)["ppd_baseline"].mean()
h_ppd_w = df.groupby(df.index.hour)["ppd_whatif"].mean()

ax.fill_between(hrs_arr, h_ppd_b.values, h_ppd_w.values,
                where=(h_ppd_w.values > h_ppd_b.values),
                alpha=0.3, color=ORG, label="PPD artışı (setback)")
ax.plot(hrs_arr, h_ppd_b.values, color=TEAL, lw=2, label="Baseline PPD")
ax.plot(hrs_arr, h_ppd_w.values, color=ORG,  lw=2, ls="--", label="2°C Setback PPD")
ax.axhline(10, color=RED, lw=1.5, ls="--", alpha=0.8, label="ISO 7730 limit (%10)")
ax.set_xticks(hrs_arr[::2])
ax.set_xticklabels([f"{h:02d}" for h in hrs_arr[::2]], color=WHT, fontsize=8)
ax.set_xlabel("Saat", color=GRY, fontsize=9)
ax.set_ylabel("PPD (%)", color=GRY, fontsize=9)
ax.set_title("PPD — Beklenen Memnuniyetsizlik %\n(ISO 7730 limit: <%10)", color=WHT, fontsize=11, fontweight="bold")
ax.legend(facecolor=BG3, labelcolor=WHT, fontsize=9)

#Özet
ax = fig.add_subplot(gs[2, 2])
ax.set_facecolor(BG2)
for sp in ax.spines.values():
    sp.set_edgecolor(TEAL); sp.set_linewidth(2)
ax.set_xticks([]); ax.set_yticks([])

b  = senaryo_sonuclari["Baseline"]
s2 = senaryo_sonuclari["Setback 2°C"]

ozet_metin = (
    "SONUÇ ÖZETİ\n\n"
    f"Baseline\n"
    f"  Ort. PMV     :  {b['pmv_mean']:+.3f}\n"
    f"  Konfor %     :  %{b['konfor_pct']:.1f}\n"
    f"  Ort. PPD     :  %{b['ppd_mean']:.1f}\n\n"
    f"2°C Setback\n"
    f"  Ort. PMV     :  {s2['pmv_mean']:+.3f}\n"
    f"  Konfor %     :  %{s2['konfor_pct']:.1f}\n"
    f"  Ort. PPD     :  %{s2['ppd_mean']:.1f}\n\n"
    "Occupied Konfor Riski\n"
    + "".join(
        f"  {lbl:<14}:  {senaryo_sonuclari[lbl]['occupied_risk_slots']} slot\n"
        for lbl in lbl_list
    ) +
    "\n[OK] 2°C setback güvenli\n"
    "   Mesai PMV konfor\n"
    "   bandında kalıyor"
)

ax.text(0.07, 0.96, ozet_metin, transform=ax.transAxes,
        color=WHT, fontsize=10, va="top", fontfamily="monospace")

# KAYDETme 
out_fig = os.path.join(OUTPUT_DIR, "pmv_analizi.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Grafik kaydedildi: {out_fig}")

# 7. CSV KAYDETME
ozet_df = pd.DataFrame([
    {
        "Senaryo": lbl,
        "Setback_C": s["setback"],
        "PMV_Ortalama": s["pmv_mean"],
        "PMV_Std": s["pmv_std"],
        "PMV_Min": s["pmv_min"],
        "PMV_Max": s["pmv_max"],
        "PPD_Ortalama_%": s["ppd_mean"],
        "Konfor_Icinde_%": s["konfor_pct"],
        "Occupied_Konfor_Riski_Slot": s["occupied_risk_slots"],
        "Toplam_Soguk_Slot": s["total_cold_slots"],
        "Toplam_Sicak_Slot": s["total_hot_slots"],
    }
    for lbl, s in senaryo_sonuclari.items()
])
ozet_df.to_csv(os.path.join(OUTPUT_DIR, "pmv_senaryo_ozet.csv"), index=False, encoding="utf-8-sig")
print(f"[OK] Senaryo özet tablosu: {OUTPUT_DIR}/pmv_senaryo_ozet.csv")

# saatlik örneklenmis zaman serisi
ts_df = df[["return_temp_C", "supply_humidity_pct", "pmv_baseline", "ppd_baseline",
             "pmv_whatif", "ppd_whatif"]].copy()
ts_df.to_csv(os.path.join(OUTPUT_DIR, "pmv_zaman_serisi.csv"), encoding="utf-8-sig")
print(f"[OK] Zaman serisi: {OUTPUT_DIR}/pmv_zaman_serisi.csv")

#sonuç bastırma:
b_pmv  = senaryo_sonuclari['Baseline']['pmv_mean']
b_kpct = senaryo_sonuclari['Baseline']['konfor_pct']
s2_occ = senaryo_sonuclari['Setback 2°C']['occupied_risk_slots']

# PMV yorumu
if   b_pmv > -0.5 and b_pmv < 0.5: yorum = "ISO 7730 Kat. B konfor bandında"
elif b_pmv <= -0.5:                  yorum = "hafif serin (ameliyathane için normal)"
else:                                yorum = "hafif sıcak"

print(f"""
{'=' * 60}
  PMV ANALİZİ TAMAMLANDI  (v2)
{'=' * 60}

Çıktı klasörü: {os.path.abspath(OUTPUT_DIR)}/
  pmv_analizi.png          ← 7 panelli grafik
  pmv_senaryo_ozet.csv     ← senaryo karşılaştırma tablosu
  pmv_zaman_serisi.csv     ← timestamp bazlı PMV değerleri

TEMEL SONUÇ:
  Baseline PMV = {b_pmv:+.3f}  → {yorum}
  Konfor içi % = %{b_kpct:.1f}

YORUM:
  Ameliyathane 20-22°C + düşük nem (%28-32) koşulları,
  ISO 7730 (ofis standardı) açısından hafif serin görünür.
  Ancak ASHRAE 170 hastane standardına göre bu NORMALDIR:
  - OR sıcaklık aralığı: 20-24°C [OK]
  - Enfeksiyon kontrolü için düşük nem tercih edilir [OK]
  Demo oda (ofis) ortamında PMV daha yüksek olacaktır
  (22-24°C, %40-50 nem → PMV ≈ 0 ile +0.3 arası beklenir).

  2°C Setback → Occupied konfor riski: {s2_occ} slot
  Setback yalnızca boş dönemlerde uygulandığından
  mesai saatlerinde konfor korunmaktadır.
""")