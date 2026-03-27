  HybridSense
A Hybrid AI Framework for Energy Optimization and Occupant Comfort
1. Overview
HybridSense is an AI-assisted supervisory HVAC control framework that operates transparently on top of an existing Building Management System (BMS). It predicts short-term room occupancy using supervised machine learning and uses those predictions to issue adaptive, energy-aware HVAC setpoints, without modifying any low-level HVAC controller or physical infrastructure.
The system was trained on anonymized sensor data from a hospital facility and validated in a Siemens Desigo CC demo room environment, communicating via BACnet/IP and Modbus TCP.
 
  Motivation

Traditional HVAC systems rely on static schedules and rule-based logic. They cannot adapt to time-varying occupancy, leading to:
•	Unnecessary energy consumption during unoccupied periods
•	Comfort degradation during high-occupancy periods that were not anticipated
•	Zero real-time adaptability in shared or irregular-use spaces

Academic literature has proposed model predictive control and reinforcement learning approaches, but most of these require complete overhaul of low-level control logic, which is not feasible in operational industrial BMS environments.
HybridSense targets the supervisory layer: it reads from the BMS, predicts, decides, and writes setpoints back without touching any field-level device.
 
  System Architecture
 
The closed-loop sequence:
•	Data acquisition from BMS sensors
•	Occupancy prediction via ML classifier
•	Supervisory setpoint generation
•	HVAC actuation through Desigo CC
•	PMV comfort evaluation on live measurements
•	Feedback-driven re-prediction at each cycle
 
  Key Components

Component	Description	Technology
Occupancy Predictor	Binary classification from BMS sensor features	scikit-learn / XGBoost / LightGBM
Supervisory Controller	Translates occupancy state to HVAC setpoint commands	Python rule engine
PMV Calculator	Real-time thermal comfort index computation	ISO 7730 / ASHRAE 55
BMS Connector	Protocol-level read/write interface to Desigo CC	BAC0 (BACnet/IP), pymodbus
Data Preprocessor	Cleaning, normalization, feature engineering pipeline	pandas / numpy
Logging & Monitoring	Trend logging for energy proxy and comfort assessment	CSV / time-series store

 Data Pipeline

Training data was sourced from a hospital facility BMS. The raw signal set captures HVAC supply/return behavior and environmental conditions across 8,350 timesteps.
5.1 Feature Philosophy
A deliberate design decision was made to exclude domain-specific HVAC control signals (supply temperature, supply airflow, delta-T variants) from the final model. These signals directly encode the control state of a bimodal hospital OR system and would fail to generalize to the continuously-modulated office HVAC in the Siemens demo room.

Retained Features - Physically Transferable Across Domains
TRANSFERABLE_FEATURES = [
    # Return-side thermal signals (passive, not control-driven)
    'return_temp_C',
    'return_dp_Pa',
 
    # Humidity (environment response, not setpoint-driven)
    'supply_humidity_pct',
    'humidity_diff',
 
    # Rate-of-change (dynamics, not absolute level)
    'temp_roc',
 
    # Temporal context
    'hour_sin', 'hour_cos',   # Continuous encoding of time-of-day
    'is_workday',
    'is_workhour',
]
 
Excluded Features - Leakage / Domain-Specific
EXCLUDED_FEATURES = [
    'supply_temp_C',            # HVAC setpoint proxy - bimodal in OR, continuous in office
    'supply_airflow_m3h',        # Directly encodes occupancy-driven control state
    'delta_T_abs', 'delta_T',    # Derived from supply temp --> label leakage
    'supply_dp_*',              # Supply-side pressure - HVAC mode indicator
    'af_mean_*', 'af_lag_*',    # Rolling/lagged airflow --> same leakage chain
]

The original feature set yielded AUC ≈ 0.998, a near-perfect score that is a red flag, not a success. The occupancy_label was constructed from HVAC signal thresholds (supply airflow ~1,200 m³/h standby vs ~1,950 m³/h active), meaning features and labels were derived from the same source (structural label leakage). The revised transferable feature set yields AUC ≈ 0.65–0.78, which is honest, defensible, and generalizes to the Siemens demo room.
6. Occupancy Prediction Model
6.1 Problem Definition
Binary classification task:
•	1 → Room is occupied (HVAC should operate in comfort mode)
•	0 → Room is unoccupied (HVAC can reduce to standby/setback mode)
6.2 Model Selection

  Model	Notes

Logistic Regression	Interpretable baseline
Random Forest	Handles non-linearity, feature importance available
Gradient Boosting (XGBoost / LightGBM)	Best AUC on transferable feature set
Decision Tree	Fully interpretable, jury-presentable

Model complexity is deliberately constrained. The project prioritizes robustness and interpretability over accuracy maximization, consistent with deployment in an industrial BMS where black-box decisions are not acceptable.
Evaluation Metrics

•	AUC-ROC: primary metric (threshold-independent)
•	F1-Score: balance between precision and recall on imbalanced occupancy windows
•	Confusion matrix at operating threshold (false negatives are tolerable - false positives cause unnecessary HVAC activation)
 
  Supervisory Control Logic

The supervisory controller translates the binary occupancy prediction into actionable BMS setpoints:

Occupancy = 1 (Occupied)
  --> Temperature setpoint : COMFORT range (e.g., 21-23 C)
  --> Airflow              : NOMINAL operating level
  --> Humidity control     : ACTIVE
 
Occupancy = 0 (Unoccupied)
  --> Temperature setpoint : SETBACK range (e.g., 18 C / 26 C)
  --> Airflow              : MINIMUM ventilation
  --> Humidity control     : PASSIVE
 
PMV check on both paths:
  --> If PMV exceeds [-0.5, +0.5] during occupied period
      --> override toward comfort

All setpoint commands remain within predefined safety margins. The controller never modifies low-level PID parameters or controller firmware only supervisory-layer variables exposed through the BMS interface.

 Thermal Comfort Evaluation (PMV)

PMV (Predicted Mean Vote) is computed per ISO 7730 and ASHRAE Standard 55:

PMV = f(T_air, T_radiant, v_air, RH, M, I_cl)
 
Comfort zone:  -0.5 <= PMV <= +0.5

Parameter	Source
Air temperature (T_air)	BMS sensor (real-time)
Relative humidity (RH)	BMS sensor (real-time)
Mean radiant temperature	Fixed assumption ≈ T_air (documented)
Air velocity	Fixed assumption: 0.1 m/s (sedentary indoor)
Metabolic rate	Fixed: 1.2 met (seated office work)
Clothing insulation	Fixed: 1.0 clo (winter) / 0.5 clo (summer)

PMV is computed at the same temporal resolution as occupancy prediction and energy proxy logging. It is used as a comparative indicator between baseline static control and HybridSense predictive control.
 
BMS Integration: BACnet/IP + Modbus TCP

  [HybridSense Python Process]
          |
          +-- BACnet/IP (primary)
          |     +-- BAC0 library
          |           +-- Read : sensor present values
          |           +-- Write: supervisory setpoint objects
          |
          +-- Modbus TCP (secondary / fallback)
                +-- pymodbus
                      +-- Read : holding registers (sensor data)
                      +-- Write: holding registers (setpoints)
 
  Connection: Laptop --[Ethernet]--> Siemens Controller Card
  No intermediate hardware.

The system operates entirely within the local BMS network. No cloud connectivity. No modifications to internal controller configurations.

  Demo Room Deployment (WP4)

The Siemens demo room contains:
•	Siemens Desigo CC workstation with BMS access
•	PXC/DXR HVAC controllers (read/write via supervisory interface only)
•	Temperature, humidity, and CO₂ sensors connected to BMS
•	No dedicated sub-metering → energy performance assessed via operational proxy indicators

  Operational Proxies for Energy Comparison
•	HVAC operating duration (baseline vs. predictive mode)
•	Supervisory setpoint adjustment frequency and magnitude
•	Actuator command levels (fan duty, valve position integrals)
•	System runtime characteristics from BMS trend logs

Work Package Progress

  Work Package	Period	Status
WP1 – Requirement Analysis	Nov–Dec 2025	Complete
WP2 – System & Algorithm Design	Dec 2025 – Jan 2026	Complete
WP3 – Implementation & Integration	Feb 2026	Complete
WP4 – Test & Verification	Mar–Apr 2026	In Progress

