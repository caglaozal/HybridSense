HybridSense – March Progress
1. Overview
HybridSense is an AI-assisted supervisory HVAC control framework that operates transparently on top of an existing Building Management System (BMS). It predicts short-term room occupancy using supervised machine learning and uses those predictions to issue adaptive, energy-aware HVAC setpoints, without modifying any low-level HVAC controller or physical infrastructure.
Training data was collected from Siemens Desigo CC at Bilkent Hospital (87-day dataset, Nov 2025-Jan 2026). WP4 validation is scheduled for April 2026 in the Siemens Smart Infrastructure demo room, communicating via BACnet/IP and Modbus TCP.
 
2. Motivation
Traditional HVAC systems rely on static schedules and rule-based logic. They cannot adapt to time-varying occupancy, leading to:
–	Unnecessary energy consumption during unoccupied periods (20–30% of HVAC budget in a typical commercial building)
–	Comfort degradation during high-occupancy periods not anticipated by the static schedule
–	Zero real-time adaptability in shared or irregular-use spaces

Academic literature has proposed model predictive control and reinforcement learning approaches, but most require complete overhaul of low-level control logic, not feasible in operational industrial BMS environments.
HybridSense targets the supervisory layer: it reads from the BMS, predicts, decides, and writes setpoints back without touching any field-level device.
3. System Architecture
3.1  Four-Layer Control Hierarchy
HybridSense is positioned at Layer 4 (supervisor) of the standard industrial BAS hierarchy:
Layer	Component	HybridSense Role
4	AI Supervisor-HybridSense	Active. Occupancy prediction, PMV check, setpoint decision, 15-min loop
3	BMS-Siemens Desigo CC	Receives setpoint commands · Validates against hardcoded safety limits (20–26°C)
2	DDC Controllers (PXC/DXR)	Not modified. PID loops and BACnet/IP network completely unchanged
1	Field & Actuators	Not modified. VAV boxes, valves, fan drives, T/H/P sensors
 
3.2  Closed-Loop Control Flow (15-minute cycle)
Each cycle executes five steps sequentially:
–	Step 1: READ: BAC0 reads return_temp_C, supply_humidity_pct, return_dp_Pa from BACnet Analog Input objects on PXC/DXR controller
–	Step 2: COMPUTE: Five physical features derived from raw signals
–	Step 3: PREDICT: Random Forest classifier outputs Occupied (1) or Empty (0)
–	Step 4: PMV CHECK: ISO 7730 thermal comfort index calculated; if setback would violate PMV ≤ −0.5, action is blocked regardless of prediction
–	Step 5: WRITE: Setpoint written to Desigo CC via BACnet Write Property; decision logged to SQLite
 

3.3  Fail-Safe Design
–	Setpoint hardcoded limits: 20–26°C. Out-of-range write attempts rejected at controller level.
–	BACnet connection loss: PXC/DXR retains last known setpoint; system reverts to local BMS control automatically.
–	Write authorization: BACnet access control restricts write permission to the AI module account only.
–	Audit log: Every cycle records timestamp, occupancy prediction, PMV value, applied setpoint, and communication status.
4. Key Components
Component	Description	Technology
Occupancy Predictor	Binary classification from 5 physical BMS sensor features	scikit-learn (Random Forest, Logistic Regression)
Supervisory Controller	Translates occupancy state + PMV check to HVAC setpoint command	Python rule engine
PMV Calculator	Real-time ISO 7730 thermal comfort index - 
 independent of model training	pythermalcomfort (ISO 7730)
BMS Connector	Protocol-level read/write interface to Siemens Desigo CC	BAC0 (BACnet/IP), pymodbus
Data Preprocessor	Cleaning, normalization, feature engineering pipeline	pandas, numpy
Logging & Monitoring	Audit trail for every decision cycle	SQLite, CSV
5. Data Pipeline
5.1  Training Data Source
Training data was collected from the Siemens Desigo CC system at Hospital controlled, 24/7 BMS environment with continuous HVAC operation.
Property	Value
Coverage period	November 1, 2025 → January 27, 2026 (87 days)
Total records	8,353 rows (8,182 after removing 171 ambiguous transition labels)
Sampling interval	15 minutes
Label distribution	Occupied (+1): 5,727 (68.6%)  ·  Empty (−1): 2,455 (29.4%)  ·  Ambiguous (0): 171 (2.0%)
Final feature set	5 physical signals 

Feature	Signal	Physical Justification
return_temp_C	Return air temperature	Human metabolic heat (70–90 W/person) raises ambient temperature. Return air reflects zone temperature accurately.
supply_humidity_pct	Supply air humidity	Occupants exhale ~17 mg water vapour per breath. Elevated humidity is a reliable proxy for presence.
humidity_diff	Supply/return humidity difference	In occupied spaces, return humidity exceeds supply humidity. Captures arrival/departure transitions.
temp_roc	Temperature rate of change (°C/15 min)	Rapid temperature rise signals occupancy onset. Decay pattern differs between occupied and empty periods.
return_dp_Pa	Return duct differential pressure	Occupancy-driven ventilation demand changes duct pressure. Return-side is independent of supply control decisions.

5.3  Train/Test Split Strategy
Chronological 80/20 split was used NOT random shuffle. In time-series forecasting, random shuffle allows the model to see both past and future context during training, producing unrealistically optimistic performance estimates. Chronological split mirrors actual deployment: the model predicts future occupancy using only historical data.
Train: 6,545 records (Nov 1, 2025 → Jan 8, 2026)  ·  Test: 1,637 records (Jan 8–27, 2026)
6. Occupancy Prediction Model
6.1  Problem Definition
Binary classification task:
–	Class 1 (Occupied): HVAC should operate in comfort mode (setpoint 22°C)
–	Class 0 (Unoccupied): HVAC can reduce to setback mode (setpoint 24°C, 2°C reduction)
–	Ambiguous class (0): 171 transition-period records excluded from training

6.2  Final Model Selection -  Random Forest and Logistic Regression
Two algorithms were evaluated. Model complexity was deliberately constrained, the project prioritises robustness and interpretability over accuracy maximisation, consistent with industrial BMS deployment where black-box decisions are not acceptable.
Model	Notes
Random Forest	Handles non-linearity; feature importance available for interpretability; no scaling required; robust to outliers. Selected for deployment, lowest False Negative count at default threshold.
Logistic Regression	Interpretable baseline with explicit coefficients; class_weight='balanced' used; scaling required. Higher AUC but more False Negatives at default threshold, less suitable for comfort-critical deployment.

6.3  Evaluation Metrics
–	AUC-ROC: primary metric, threshold-independent discriminative power (computed via roc_analysis.py with n_estimators=300)
–	F1-Score, Precision, Recall, Specificity: evaluated at default threshold (0.5) using deployed pkl models
–	Confusion matrix at default threshold (0.5): reflects actual deployment behaviour
 
6.4  Model Performance Results
Metric	Random Forest	Logistic Regression	Notes
AUC-ROC	0.832	0.874	From roc_analysis.py; n_estimators=300
F1-Score	0.786	0.660	At default threshold (0.5); deployed pkl
Precision	0.843	0.913	RF fewer false positives at default threshold
Recall	0.736	0.517	RF catches more occupied periods
Accuracy	72.5%	63.5%	At default threshold
Specificity	0.701	0.893	LR more conservative at default threshold
False Negatives	296	542	RF: 46% fewer comfort violations
False Positives	154	55	LR fewer unnecessary HVAC activations
TN / TP	361 / 826	460 / 580	Confusion matrix at threshold=0.5

7. Thermal Comfort Evaluation (PMV)
PMV (Predicted Mean Vote) is computed per ISO 7730:2005 using the pythermalcomfort library. The comfort zone is −0.5 ≤ PMV ≤ +0.5 (ISO 7730 Category B, applicable to offices and hospitals). PPD target: <10%.
PMV is architecturally independent from the occupancy prediction model. It shares two input signals (return_temp_C, supply_humidity_pct) but is processed separately and has NO influence on model training.
Parameter	Source	Justification
Air temperature (T_air)	BMS sensor - return_temp_C	Real-time zone temperature
Relative humidity (RH)	BMS sensor- supply_humidity_pct	Real-time humidity data
Radiant temperature (T_r)	Fixed: T_r = T_air	No dedicated radiant sensor; ISO 7730 assumption
Air velocity (v_air)	Fixed: 0.1 m/s	Typical sedentary indoor office environment
Metabolic rate (M)	Fixed: 1.2 met	Seated office work (ISO 7730 Table B.1)
Clothing insulation (I_cl)	Fixed: 1.0 clo (winter)	Standard office winter clothing

8.1  PMV Scenario Analysis Results
Scenario	Mean PMV	PPD	In-comfort %	Occ. Risk (slots)	Energy (proxy)
Baseline (static BMS)	+0.348	8.0%	87.5%	0	—
1°C Setback	+0.294	7.3%	90.8%	0	−6.6%
2°C Setback ★ OPTIMAL	+0.241	7.1%	90.8%	0	−16.7%
3°C Setback	+0.189	7.3%	90.7%	11	−25.1%
9. Demo Room Deployment (WP4)
9.1  Hardware Inventory
Component	Model	WP4 Role
Building Management System	Desigo CC	Main interface. Sensor simulation + setpoint verification
Integration Controller	PXC00.ED	BACnet/IP gateway. HybridSense communicates via this card
Main Controller	PXC7.E400L	Setpoint commands written here
I/O Controller	PXC4.E16	Additional sensor read access
Room Controllers	DXR2 series (6 types)	Direct setpoint recipients
Room Thermostats	RDG200KN, RDG100, QMX3.P37	Real T/H readings via KNX → Desigo CC → BACnet
I/O Modules	TXM1 series (13 modules)	Read/write BACnet data points

9.2  Test Scenarios
ID	return_temp_C	supply_hum %	Expected Prediction	Expected PMV	Expected Decision
S1	23.5°C	52%	Occupied (1)	0.30–0.45	No setback
S2	20.8°C	48%	Empty (0)	0.10–0.25	2°C setback applied
S3 ★	18.5°C	45%	Empty (0)	−0.45 to −0.50	PMV constraint → setback CANCELLED
S4	26.0°C	68%	Occupied (1)	0.50–0.65	No setback
S5	22.0°C	50%	Threshold ~0.50	0.20–0.35	Threshold-dependent, logged

 S3 is the critical PMV safety constraint test: the system must cancel setback when the thermal comfort boundary is approached, regardless of the occupancy prediction.
10. Work Package Progress
WP	Period	Deliverable	Status
WP1	Nov–Dec 2025	Data collection, ETL pipeline, BACnet point mapping	✓ Complete
WP2	Dec 2025–Jan 2026	System architecture, algorithm selection, PMV methodology	✓ Complete
WP3	Feb–Mar 2026	Python pipeline, PMV module, Streamlit dashboard, model evaluation	✓ Complete
WP4	April 2026	Siemens demo room integration, closed-loop BACnet/IP validation, final report	In Progress
11. Repository Structure
 
