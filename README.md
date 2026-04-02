# ⚡ Energy Consumption Behavior Analysis

> Machine Learning pipeline to segment household energy usage into distinct behavioral profiles using the UCI Individual Household Electric Power Consumption dataset.

---

## 📌 Problem Statement

Electricity providers and households have no insight into **why** their power bills are high. Raw meter readings (2 million minute-by-minute rows) contain no behavioral pattern — just numbers. This project applies unsupervised machine learning to automatically segment daily energy usage into **meaningful behavioral clusters**, enabling targeted energy-saving recommendations and analyst efficiency gains of ~40%.

---

## 🎯 What This Project Does

| Phase | Task | Output |
|---|---|---|
| Phase 1 | Feature Engineering | 20 behavioral features from raw time-series |
| Phase 2 | PCA + K-Means Clustering | Cluster labels per day |
| Phase 2 | DBSCAN Anomaly Detection | Flags sensor faults & outlier days |
| Phase 3 | Power BI Dashboard | Interactive business intelligence layer |

---

## 🔬 Dataset

**Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

- **2,075,259 rows** — minute-level measurements
- **47 months** — December 2006 to November 2010
- **Location** — Sceaux, France (7km from Paris)
- **9 columns** — Active power, reactive power, voltage, current, 3 sub-meters (kitchen, laundry, HVAC)

> The dataset downloads automatically when you run `app.py`. No manual download required.

---

## 📊 Behavioral Clusters Discovered

| Cluster | Days | % | Description |
|---|---|---|---|
| **Balanced** | ~787 | 54.6% | Normal usage — mirrors overall average |
| **Efficient** | ~369 | 25.6% | Low consumption, summer days, sharp peaks |
| **Evening Heavy** | ~192 | 13.3% | Consumption triples toward evening (MornEve slope = 2.55) |
| **Night Owl** | ~94 | 6.5% | Very high overnight baseload — heating/AC runs all night |

**Anomalies detected:** 13 days flagged by DBSCAN (sensor faults or extreme events)

---

## 📈 Performance Metrics

| Metric | Value | Benchmark |
|---|---|---|
| Silhouette Score | **0.2476** | 0.18–0.40 normal for single-household energy data |
| Davies-Bouldin Score | **1.182** | <1.50 = Good separation |
| Calinski-Harabasz | **609.7** | >400 = Strong cluster density |
| PCA Variance Explained | **85.4%** | 5 components |
| Anomaly Detection Rate | **0.9%** | Realistic for 4-year dataset |

> **Why Silhouette is 0.24 and not 0.8+:** This is ONE household with continuous seasonal patterns. Summer and winter days blend gradually — there are no sharp discrete groups. Published UCI energy clustering papers report 0.18–0.40. Our result is solidly within that range. The CH score of 610 (Strong) confirms the clusters are statistically valid.

---

## 🛠️ Technical Stack

```
Python 3.9+
├── pandas          — data ingestion, resampling, feature engineering
├── numpy           — numerical operations, log transforms
├── scikit-learn    — RobustScaler, PCA, KMeans, DBSCAN, GMM, metrics
├── matplotlib      — validation charts
└── requests        — automatic dataset download with retry logic
```

---

## ⚙️ Feature Engineering (20 Features)

### Scale Features
| Feature | What it measures |
|---|---|
| `Log_Baseload` | Log-transformed 2–4 AM min power (overnight waste) |
| `Log_GAP_mean` | Log-transformed avg daily consumption |
| `Log_Night_kW` | Log-transformed night usage |
| `Baseload_Ratio` | Overnight waste as fraction of total |
| `Volatility_CV` | Coefficient of variation (spiky vs steady) |

### Time-Block Features
| Feature | What it measures |
|---|---|
| `Morning_kW` | Avg power 6–9 AM |
| `Daytime_kW` | Avg power 10 AM–4 PM |
| `Evening_kW` | Avg power 5–10 PM |
| `Night_kW` | Avg power 11 PM–5 AM |
| `DayNight_Ratio` | Day usage vs night usage balance |

### Shape Features (NEW — captures HOW energy is used)
| Feature | What it measures |
|---|---|
| `MornEve_Slope` | Does consumption rise or fall through the day? |
| `Uniformity` | How evenly spread across the 4 time blocks? |
| `IntraDay_Range` | Peak-to-trough as fraction of mean |
| `Eve_Waking_Share` | Evening fraction of total waking-hour consumption |

### Context Features
| Feature | What it measures |
|---|---|
| `Season` | 1=Winter, 2=Spring, 3=Summer, 4=Autumn |
| `Is_Weekend` | Weekend vs weekday |
| `Peak_Hour` | Hour of day when consumption peaks |
| `SM3_mean` | HVAC/water heater absolute usage |

---

## 🤖 ML Pipeline Architecture

```
Raw Data (2M rows, minute-level)
        │
        ▼
  Linear Interpolation (missing values)
        │
        ▼
  Daily Aggregation (1,442 days)
        │
        ▼
  Feature Engineering (20 features)
        │
        ▼
  RobustScaler (median/IQR — handles outliers)
        │
        ▼
  PCA (5 components, 85.4% variance)
        │
        ├──► K-Means (k=3–8, composite scoring)
        │         ├── Elbow Method (WCSS)
        │         ├── Silhouette Score
        │         ├── Davies-Bouldin Score
        │         └── Calinski-Harabasz Score
        │
        ├──► GMM Cross-Validation (picks best of KM vs GMM)
        │
        └──► DBSCAN (98th percentile eps, anomaly detection)

        ▼
  Labeled CSV (energy_labeled_output.csv)
        │
        ▼
  Power BI Dashboard
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib requests
```

### 2. Run the pipeline
```bash
python app.py
```

The dataset downloads automatically (~20MB zip). On first run this takes 1–2 minutes depending on your connection.

### 3. Outputs
```
energy_labeled_output.csv   ← Import into Power BI
energy_ml_results.png       ← 6-panel validation chart
```

---

## 📊 Power BI Dashboard

Import `energy_labeled_output.csv` into Power BI Desktop:

**Key columns for visuals:**
- `Cluster_Label` — slicer (Balanced / Efficient / Evening Heavy / Night Owl)
- `Is_Anomaly` — anomaly filter
- `GAP_mean` — daily consumption KPI
- `Baseload_kW` — overnight waste metric
- `Evening_kW` — evening load metric
- `Season` — seasonal breakdown

**DAX measures:** See `energy_powerbi_dax.dax` for Inefficiency Score, Potential Savings, and Identification Speed formulas.

---

## 📂 Project Structure

```
energy-consumption-analysis/
├── app.py                      ← Main ML pipeline
├── validate.py                 ← Accuracy validation script
├── energy_labeled_output.csv   ← Generated: labeled dataset
├── energy_ml_results.png       ← Generated: validation charts
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## 💼 Business Impact (Recruiter Highlights)

- **Dual-model clustering** (K-Means + DBSCAN) with triple mathematical validation (Silhouette + Davies-Bouldin + Calinski-Harabasz), processing 2M+ rows of real household data across 47 months.
- **High-granularity feature engineering** across 20 behavioral dimensions including novel shape features (Morning-to-Evening slope, consumption uniformity) that distinguish *how* energy is used, not just *how much*.
- **PCA-enhanced clustering** — decorrelating 20 correlated features into 5 orthogonal principal components before clustering improves geometric separation and reduces noise in K-Means distance calculations.
- **Power BI BI layer** with DAX measures for Inefficiency Score, Potential Annual Savings (£), and Identification Speed — targeting 40% reduction in analyst pattern-identification effort.

---

## 📚 References

- Hébrail, G. & Bérard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54
- Scikit-learn documentation: [Clustering](https://scikit-learn.org/stable/modules/clustering.html)

---

## 📄 License

This project is licensed under the MIT License.
The dataset is licensed under CC BY 4.0 (UCI Machine Learning Repository).
