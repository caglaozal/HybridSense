# HybridSense
**A Hybrid AI Framework for Energy Optimization and Occupant Comfort**

---

## 1. Overview

HybridSense is an AI-assisted supervisory HVAC control framework that operates transparently on top of an existing Building Management System (BMS). It predicts short-term room occupancy using supervised machine learning and uses those predictions to issue adaptive, energy-aware HVAC setpoints, without modifying any low-level HVAC controller or physical infrastructure.

The system was trained on real sensor data from Siemens Desigo CC at Bilkent Hospital (87-day dataset, Nov 2025–Jan 2026) and is being validated in a Siemens Smart Infrastructure demo room environment, communicating via BACnet/IP and Modbus TCP.

---

## 2. Motivation

Traditional HVAC systems rely on static schedules and rule-based logic. They cannot adapt to time-varying occupancy, leading to:

- Unnecessary energy consumption during unoccupied periods (20–30% of HVAC budget in a typical commercial building)
- Comfort degradation during high-occupancy periods not anticipated by the static schedule
- Zero real-time adaptability in shared or irregular-use spaces

Academic literature has proposed model predictive control and reinforcement learning approaches, but most require complete overhaul of low-level control logic — not feasible in operational industrial BMS environments.

**HybridSense targets the supervisory layer: it reads from the BMS, predicts, decides, and writes setpoints back without touching any field-level device.**

---

## 3. System Architecture

**Four-layer control hierarchy:**

| Layer | Component | HybridSense Role |
|-------|-----------|-----------------|
| 4 | AI Supervisor — HybridSense | Active: occupancy prediction, PMV check, setpoint decision, 15-min loop |
| 3 | BMS — Siemens Desigo CC | Receives setpoint commands · Validates against hardcoded safety limits (20–26°C) |
| 2 | DDC Controllers (PXC/DXR) | Not modified — PID loops and BACnet/IP network completely unchanged |
| 1 | Field & Actuators | Not modified — VAV boxes, valves, fan drives, T/H/P sensors |

**Closed-loop cycle (15-minute interval):**

1. **READ** — BAC0 reads `return_temp_C`, `supply_humidity_pct`, `return_dp_Pa` from BACnet Analog Input objects
2. **COMPUTE** — Five physical features derived from raw signals
3. **PREDICT** — Random Forest classifier outputs Occupied (1) or Empty (0)
4. **PMV CHECK** — ISO 7730 comfort index calculated; if setback would violate PMV ≤ −0.5, action is blocked
5. **WRITE** — Setpoint written to Desigo CC via BACnet Write Property; decision logged to SQLite

---

## 4. Key Components

| Component | Description | Technology |
|-----------|-------------|------------|
| Occupancy Predictor | Binary classification from 5 physical BMS sensor features | scikit-learn (Random Forest, Logistic Regression) |
| Supervisory Controller | Translates occupancy state + PMV check to HVAC setpoint command | Python rule engine |
| PMV Calculator | Real-time ISO 7730 thermal comfort index — independent of model training | pythermalcomfort (ISO 7730) |
| BMS Connector | Protocol-level read/write interface to Siemens Desigo CC | BAC0 (BACnet/IP), pymodbus |
| Data Preprocessor | Cleaning, normalization, feature engineering pipeline | pandas, numpy |
| Logging & Monitoring | Audit trail for every decision cycle | SQLite, CSV |

---

## 5. Data Pipeline

**Training data:** Siemens Desigo CC, Bilkent Hospital — 8,182 labelled records (87 days, 15-min interval, Nov 2025–Jan 2026). 171 ambiguous transition records excluded.

### 5.1 Feature Selection — v3 (Final)

Three design iterations were required to arrive at a valid, transferable feature set:

**v1 — Label Leakage (AUC=0.998 — invalid)**  
Supply-side signals (`supply_temp_C`, `delta_T_abs`, `supply_airflow_m3h`) were used both to generate occupancy labels AND as model inputs. AUC=0.998 is a red flag, not a success.

**v2 — Time Feature Dependency (AUC=0.782)**  
Time features (`is_workday`, `is_workhour`) locked the model into a "working hours = occupied" assumption. Removing them caused AUC to rise to 0.798.

**v3 — Pure Physical Signals (AUC=0.798–0.832, deployed) ✓**
```python
FEATURES = [
    'return_temp_C',        # Return air temp — human metabolic heat raises zone T
    'supply_humidity_pct',  # Supply humidity — occupants exhale ~17mg water/breath
    'humidity_diff',        # Supply/return humidity difference — captures transitions
    'temp_roc',             # Temperature rate of change — occupancy onset signal
    'return_dp_Pa',         # Return duct pressure — occupancy-driven ventilation demand
]
```

All 5 features are return-side or derived signals — they carry the physical trace of human presence without encoding HVAC control decisions. Independent of time-of-day, day-of-week, or building type.

**Excluded features:**
```python
EXCLUDED = [
    'supply_temp_C',       # HVAC setpoint proxy — label leakage
    'supply_airflow_m3h',  # Directly encodes occupancy-driven control state
    'delta_T_abs',         # Derived from supply_temp → leakage chain
    'supply_dp_*',         # Supply-side pressure — HVAC mode indicator
    'af_mean_*', 'af_lag_*',  # Rolling/lagged airflow → same leakage chain
    'is_workday', 'is_workhour', 'hour_sin', 'hour_cos',  # Time dependency
]
```

### 5.2 Train/Test Split

Chronological 80/20 split — **not** random shuffle. Random shuffle leaks future context into training. Chronological split mirrors actual deployment.

- Train: 6,545 records (Nov 1, 2025 → Jan 8, 2026)  
- Test: 1,637 records (Jan 8–27, 2026)

---

## 6. Occupancy Prediction Model

### 6.1 Model Selection

| Model | Notes |
|-------|-------|
| Random Forest | Handles non-linearity; feature importance available; no scaling required. **Selected for deployment — lowest False Negative count.** |
| Logistic Regression | Interpretable baseline with explicit coefficients; `class_weight='balanced'`; scaling required. |

### 6.2 Error Cost Asymmetry

| Error Type | Meaning | Impact | Risk |
|-----------|---------|--------|------|
| **False Negative (FN)** | Model says "empty" — room is actually occupied | HVAC reduces output → comfort violation | **HIGH** |
| **False Positive (FP)** | Model says "occupied" — room is actually empty | HVAC runs unnecessarily → energy waste only | LOW |

Random Forest is selected because it produces fewer False Negatives at deployment threshold — protecting occupant comfort is the primary constraint. PMV safety layer provides a second line of defence.

### 6.3 Performance Results

| Metric | Random Forest | Logistic Regression | Notes |
|--------|--------------|--------------------|----|
| AUC-ROC | **0.832** | 0.874 | From `roc_analysis.py` (n_estimators=300) |
| F1-Score | 0.786 ✓ | 0.660 | At default threshold (0.5); deployed pkl |
| Precision | 0.843 | 0.913 | |
| Recall | 0.736 | 0.517 | RF catches more occupied periods |
| Accuracy | 72.5% | 63.5% | |
| Specificity | 0.701 | 0.893 | |
| False Negatives | **296** | 542 | RF: 46% fewer comfort violations |
| False Positives | 154 | 55 | |

---

## 7. Supervisory Control Logic
```
Occupancy = 1 (Occupied)
  → Setpoint stays at 22°C

Occupancy = 0 (Unoccupied) AND PMV ≥ −0.5
  → Setpoint raised to 24°C (2°C setback applied)

Occupancy = 0 (Unoccupied) BUT PMV < −0.5
  → Setback BLOCKED by PMV constraint
  → Setpoint stays at 22°C

All commands hardcoded within 20–26°C safety range.
Controller firmware and PID parameters NEVER modified.
```

---

## 8. Thermal Comfort (PMV)

PMV is computed per ISO 7730:2005 using `pythermalcomfort`. Comfort zone: −0.5 ≤ PMV ≤ +0.5 (ISO 7730 Category B).

**PMV is architecturally independent from the occupancy model.** It shares two input signals but is processed separately and has no influence on model training.

| Parameter | Source |
|-----------|--------|
| Air temperature | BMS sensor — `return_temp_C` |
| Relative humidity | BMS sensor — `supply_humidity_pct` |
| Radiant temperature | Fixed: T_r = T_air |
| Air velocity | Fixed: 0.1 m/s |
| Metabolic rate | Fixed: 1.2 met (seated office work) |
| Clothing insulation | Fixed: 1.0 clo (winter) |

**PMV scenario results:**

| Scenario | Mean PMV | PPD | In-comfort | Occ. Risk | Energy |
|----------|----------|-----|------------|-----------|--------|
| Baseline (static BMS) | +0.348 | 8.0% | 87.5% | 0 slots | — |
| 1°C Setback | +0.294 | 7.3% | 90.8% | 0 slots | −6.6% |
| **2°C Setback ★ OPTIMAL** | **+0.241** | **7.1%** | **90.8%** | **0 slots** | **−16.7%** |
| 3°C Setback | +0.189 | 7.3% | 90.7% | 11 slots ⚠ | −25.1% |

---

## 9. BMS Integration
```
[HybridSense Python Process]
  |
  +-- BACnet/IP (primary)       via BAC0
  |     Read : Analog Input objects (sensor values)
  |     Write: Analog Output objects (setpoints)
  |
  +-- Modbus TCP (fallback)     via pymodbus
        Read/Write: holding registers

Connection: Laptop --[Ethernet]--> Siemens PXC00.ED
No cloud. No intermediate hardware. No controller modifications.
```

---

## 10. WP4 Demo Room (April 2026)

**Hardware confirmed:** Desigo CC · PXC00.ED · PXC7.E400L · PXC4.E16 · DXR2 room controllers · RDG thermostats · TXM1 I/O modules

**Test scenarios:**

| ID | return_temp_C | supply_hum % | Expected Decision |
|----|--------------|-------------|------------------|
| S1 | 23.5°C | 52% | No setback (occupied) |
| S2 | 20.8°C | 48% | 2°C setback applied |
| S3 ★ | 18.5°C | 45% | PMV constraint → setback CANCELLED |
| S4 | 26.0°C | 68% | No setback (occupied) |
| S5 | 22.0°C | 50% | Threshold-dependent, logged |

S3 is the critical safety test: the system must cancel setback when the PMV boundary is approached, regardless of the occupancy prediction.

---

## 11. Work Package Progress

| WP | Period | Deliverable | Status |
|----|--------|-------------|--------|
| WP1 | Nov–Dec 2025 | Data collection, ETL pipeline, BACnet point mapping | ✓ Complete |
| WP2 | Dec 2025–Jan 2026 | System architecture, algorithm selection, PMV methodology | ✓ Complete |
| WP3 | Feb–Mar 2026 | Python pipeline, PMV module, Streamlit dashboard, model evaluation | ✓ Complete |
| WP4 | April 2026 | Siemens demo room integration, closed-loop BACnet/IP validation, final report | In Progress |

---

## 12. Repository Structure
```
HybridSense/
├── hybridsense_final_pkg/
│   ├── train_models.py             Model training pipeline (v3, 5 physical features)
│   ├── roc_analysis.py             ROC curve + AUC analysis (n_estimators=300)
│   ├── pmv_analizi.py              PMV scenario analysis (ISO 7730)
│   ├── hybridsense_dashboard.py    Streamlit monitoring dashboard
│   ├── hybridsense_correlation.ipynb  Feature correlation analysis
│   ├── model_rf.pkl                Trained Random Forest (AUC=0.798 at threshold=0.5)
│   ├── model_lr.pkl                Trained Logistic Regression
│   ├── scaler.pkl                  StandardScaler for LR input
│   ├── master_full_features.csv    Training dataset (8,182 records)
│   ├── roc.png                     ROC curve
│   └── roc_metrics.csv             AUC and optimal threshold values
└── HybridSense_ProgressPresentation.pdf
```

**Setup:**
```bash
pip install scikit-learn pandas numpy joblib streamlit matplotlib seaborn pythermalcomfort

python train_models.py                                          # Re-train models
python roc_analysis.py --data master_full_features.csv         # ROC analysis
streamlit run hybridsense_dashboard.py                         # Dashboard
python pmv_analizi.py                                          # PMV scenarios
```
