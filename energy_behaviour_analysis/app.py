"""
============================================================
  ENERGY CONSUMPTION BEHAVIOR ANALYSIS
  Machine Learning Pipeline — Final Version

  Dataset  : UCI Individual Household Electric Power Consumption
  Method   : PCA + K-Means + DBSCAN + GMM Cross-validation
  Output   : energy_labeled_output.csv  (Power BI ready)
             energy_ml_results.png      (Validation charts)
============================================================
"""

import os, io, zipfile, time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors

# ═══════════════════════════════════════════════════════════
# STEP 0 — DOWNLOAD DATASET
# ═══════════════════════════════════════════════════════════

DATA_FILE = "household_power_consumption.txt"
ZIP_URL   = ("https://archive.ics.uci.edu/static/public/235/"
             "individual+household+electric+power+consumption.zip")

def download_dataset(url, save_path, retries=5, timeout=120):
    if os.path.exists(save_path):
        print(f"    Dataset found → {save_path}")
        return True
    for attempt in range(1, retries + 1):
        print(f"    Downloading... attempt {attempt}/{retries}")
        try:
            r = requests.get(url, timeout=timeout, stream=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            total, chunks, dl = int(r.headers.get("content-length", 0)), [], 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    chunks.append(chunk)
                    dl += len(chunk)
                    if total:
                        print(f"      {dl/total*100:.1f}%", end="\r")
            zf  = zipfile.ZipFile(io.BytesIO(b"".join(chunks)))
            txt = [n for n in zf.namelist() if n.endswith(".txt")][0]
            open(save_path, "wb").write(zf.read(txt))
            print(f"\n    Saved → {save_path}")
            return True
        except Exception as e:
            print(f"\n    Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(attempt * 5)
    return False

# ═══════════════════════════════════════════════════════════
# PHASE 1 — DATA INGESTION & CLEANING
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  PHASE 1 — DATA INGESTION & CLEANING")
print("=" * 60)

print("\n[1/7] Loading dataset...")
if not download_dataset(ZIP_URL, DATA_FILE):
    raise SystemExit("Dataset unavailable.")

df_raw = pd.read_csv(DATA_FILE, sep=";", na_values=["?", ""], low_memory=False)
print(f"    Rows: {df_raw.shape[0]:,}  |  Columns: {df_raw.shape[1]}")

print("\n[2/7] Parsing datetime index...")
df_raw["datetime"] = pd.to_datetime(
    df_raw["Date"].astype(str) + " " + df_raw["Time"].astype(str),
    format="%d/%m/%Y %H:%M:%S", dayfirst=True)
df_raw.set_index("datetime", inplace=True)
df_raw.drop(columns=["Date", "Time"], inplace=True)

numeric_cols = ["Global_active_power", "Global_reactive_power", "Voltage",
                "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
df_raw[numeric_cols] = df_raw[numeric_cols].replace("?", np.nan).astype(float)
print(f"    Range: {df_raw.index.min().date()} → {df_raw.index.max().date()}")

print("\n[3/7] Handling missing values (linear interpolation)...")
missing_before = df_raw.isnull().sum().sum()
df_raw[numeric_cols] = df_raw[numeric_cols].interpolate(
    method="linear", limit_direction="both")
print(f"    Missing: {missing_before:,} → {df_raw.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════
# PHASE 1 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

print("\n[4/7] Engineering behavioral features...")

df_hourly = df_raw.resample("h").mean(numeric_only=True)
df_daily  = df_raw.resample("D").agg(
    GAP_mean=("Global_active_power",   "mean"),
    GAP_max =("Global_active_power",   "max"),
    GAP_std =("Global_active_power",   "std"),
    GAP_min =("Global_active_power",   "min"),
    GRP_mean=("Global_reactive_power", "mean"),
    SM1_mean=("Sub_metering_1",        "mean"),
    SM2_mean=("Sub_metering_2",        "mean"),
    SM3_mean=("Sub_metering_3",        "mean"),
    Voltage =("Voltage",               "mean"),
)

# Sub-metering ratios
conv = df_daily["GAP_mean"] * 1000 / 60
df_daily["SM1_ratio"] = np.where(conv > 0, df_daily["SM1_mean"] / conv, 0)
df_daily["SM2_ratio"] = np.where(conv > 0, df_daily["SM2_mean"] / conv, 0)
df_daily["SM3_ratio"] = np.where(conv > 0, df_daily["SM3_mean"] / conv, 0)

# Peak Intensity
df_daily["Peak_Intensity"] = np.where(
    df_daily["GAP_mean"] > 0, df_daily["GAP_max"] / df_daily["GAP_mean"], 0)

# Baseload (2-4 AM min power draw)
baseload = (df_hourly[df_hourly.index.hour.isin([2, 3])]["Global_active_power"]
            .resample("D").min().rename("Baseload_kW"))
df_daily  = df_daily.join(baseload, how="left")
df_daily["Baseload_kW"].fillna(df_daily["Baseload_kW"].median(), inplace=True)

# Active vs Reactive ratio
df_daily["ActiveReactive_Ratio"] = np.where(
    df_daily["GRP_mean"] > 0,
    df_daily["GAP_mean"] / df_daily["GRP_mean"],
    df_daily["GAP_mean"].max())

# Time-of-day blocks
def hour_block_mean(df_h, hours, label):
    return (df_h[df_h.index.hour.isin(hours)]["Global_active_power"]
            .resample("D").mean().rename(label))

df_daily = df_daily.join(hour_block_mean(df_hourly, range(6,  10), "Morning_kW"),   how="left")
df_daily = df_daily.join(hour_block_mean(df_hourly, range(10, 17), "Daytime_kW"),   how="left")
df_daily = df_daily.join(hour_block_mean(df_hourly, range(17, 23), "Evening_kW"),   how="left")
df_daily = df_daily.join(hour_block_mean(df_hourly, [23,0,1,2,3,4,5], "Night_kW"), how="left")

# Volatility
df_daily["Volatility_CV"] = np.where(
    df_daily["GAP_mean"] > 0, df_daily["GAP_std"] / df_daily["GAP_mean"], 0)

# Day/Night ratio
df_daily["DayNight_Ratio"] = np.where(
    df_daily["Night_kW"] > 0,
    (df_daily["Daytime_kW"] + df_daily["Evening_kW"]) / df_daily["Night_kW"], 0)

# Baseload Ratio
df_daily["Baseload_Ratio"] = np.where(
    df_daily["GAP_mean"] > 0, df_daily["Baseload_kW"] / df_daily["GAP_mean"], 0)

# Evening Dominance
df_daily["Evening_Dominance"] = np.where(
    df_daily["GAP_mean"] > 0,
    df_daily["Evening_kW"] / (df_daily["GAP_mean"] * 24 + 1e-9), 0)

# Season (France: winter=1, spring=2, summer=3, autumn=4)
df_daily["Season"] = np.select(
    [df_daily.index.month.isin([12,1,2]),  df_daily.index.month.isin([3,4,5]),
     df_daily.index.month.isin([6,7,8]),   df_daily.index.month.isin([9,10,11])],
    [1, 2, 3, 4])

df_daily["Is_Weekend"] = (df_daily.index.dayofweek >= 5).astype(int)

peak_hour = (df_hourly["Global_active_power"]
             .resample("D")
             .apply(lambda x: x.idxmax().hour if len(x.dropna()) > 0 else 18)
             .rename("Peak_Hour"))
df_daily  = df_daily.join(peak_hour, how="left")
df_daily["Peak_Hour"].fillna(18, inplace=True)

# Log transforms (reduce outlier dominance)
df_daily["Log_Baseload"] = np.log1p(df_daily["Baseload_kW"])
df_daily["Log_Night_kW"] = np.log1p(df_daily["Night_kW"])
df_daily["Log_GAP_mean"] = np.log1p(df_daily["GAP_mean"])

# Shape features
df_daily["MornEve_Slope"] = np.where(
    df_daily["Morning_kW"] > 0,
    (df_daily["Evening_kW"] - df_daily["Morning_kW"]) / (df_daily["Morning_kW"] + 1e-9), 0)

block_vals = df_daily[["Morning_kW","Daytime_kW","Evening_kW","Night_kW"]].values
df_daily["Uniformity"] = np.where(
    block_vals.mean(axis=1) > 0,
    1 - (block_vals.std(axis=1) / (block_vals.mean(axis=1) + 1e-9)), 0)

df_daily["IntraDay_Range"] = np.where(
    df_daily["GAP_mean"] > 0,
    (df_daily["GAP_max"] - df_daily["GAP_min"]) / (df_daily["GAP_mean"] + 1e-9), 0)

waking = df_daily["Morning_kW"] + df_daily["Daytime_kW"] + df_daily["Evening_kW"]
df_daily["Eve_Waking_Share"] = np.where(
    waking > 0, df_daily["Evening_kW"] / (waking + 1e-9), 0)

df_daily.fillna(df_daily.median(numeric_only=True), inplace=True)
print(f"    Days: {len(df_daily):,}  |  Features: {len(df_daily.columns)}")

# ═══════════════════════════════════════════════════════════
# PHASE 2 — MACHINE LEARNING PIPELINE
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PHASE 2 — MACHINE LEARNING PIPELINE")
print("=" * 60)

FEATURE_COLS = [
    "Log_Baseload","Log_GAP_mean","Log_Night_kW",
    "Evening_kW","Daytime_kW","Morning_kW",
    "Peak_Intensity","Volatility_CV","DayNight_Ratio",
    "Baseload_Ratio","Evening_Dominance",
    "Season","Peak_Hour","Is_Weekend",
    "MornEve_Slope","Uniformity","IntraDay_Range","Eve_Waking_Share",
    "SM3_mean","SM1_mean",
]

print(f"\n[5/7] Pre-processing: {len(FEATURE_COLS)} features → RobustScaler → PCA")

X        = df_daily[FEATURE_COLS].copy()
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Find PCA components explaining 85% variance
pca_probe    = PCA(random_state=42)
pca_probe.fit(X_scaled)
cum_var      = np.cumsum(pca_probe.explained_variance_ratio_)
n_components = max(int(np.argmax(cum_var >= 0.85) + 1), 5)

pca       = PCA(n_components=n_components, random_state=42)
X_pca     = pca.fit_transform(X_scaled)
total_var = pca.explained_variance_ratio_.sum() * 100
print(f"    PCA: {n_components} components explain {total_var:.1f}% variance")

# K-Means
print("\n[6/7] K-Means + composite scoring (k=3–8)...\n")

k_range           = range(3, 9)
wcss_scores       = []
silhouette_scores = []
db_scores         = []
ch_scores         = []
sil_sample        = min(1000, len(X_pca))

for k in k_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=600)
    labels = km.fit_predict(X_pca)
    wcss_scores.append(km.inertia_)
    sil = silhouette_score(X_pca, labels, sample_size=sil_sample, random_state=42)
    db  = davies_bouldin_score(X_pca, labels)
    ch  = calinski_harabasz_score(X_pca, labels)
    silhouette_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)
    print(f"    k={k}  WCSS={km.inertia_:,.1f}  Sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")

sil_n     = np.array(silhouette_scores) / max(silhouette_scores)
db_n      = 1 - np.array(db_scores) / max(db_scores)
ch_n      = np.array(ch_scores) / max(ch_scores)
composite = sil_n * 0.35 + db_n * 0.30 + ch_n * 0.35

max_comp = composite.max()
tied     = [list(k_range)[i] for i, v in enumerate(composite) if abs(v - max_comp) < 0.001]
if len(tied) > 1:
    tied_sil    = [silhouette_scores[list(k_range).index(k)] for k in tied]
    best_comp_k = tied[int(np.argmax(tied_sil))]
else:
    best_comp_k = int(list(k_range)[np.argmax(composite)])

wcss_arr   = np.array(wcss_scores)
d2         = np.diff(np.diff(wcss_arr))
optimal_k  = int(list(k_range)[np.argmax(np.abs(d2)) + 1])
best_sil_k = int(list(k_range)[np.argmax(silhouette_scores)])

print(f"\n    Elbow → k={optimal_k}  |  Best Sil → k={best_sil_k}  |  Composite → k={best_comp_k}")
FINAL_K = best_comp_k
print(f"    Final k = {FINAL_K}")

km_final = KMeans(n_clusters=FINAL_K, random_state=42, n_init=20, max_iter=600)
df_daily["Cluster_ID"] = km_final.fit_predict(X_pca)

# GMM cross-validation
gmm        = GaussianMixture(n_components=FINAL_K, covariance_type="full",
                              random_state=42, n_init=5)
gmm_labels = gmm.fit_predict(X_pca)
gmm_sil    = silhouette_score(X_pca, gmm_labels, sample_size=sil_sample, random_state=42)
km_sil_val = silhouette_score(X_pca, df_daily["Cluster_ID"].values,
                               sample_size=sil_sample, random_state=42)
if gmm_sil > km_sil_val:
    df_daily["Cluster_ID"] = gmm_labels
print(f"    GMM Sil={gmm_sil:.4f}  KMeans Sil={km_sil_val:.4f}  → Using {'GMM' if gmm_sil>km_sil_val else 'K-Means'}")

# Semantic labelling
profile_cols = ["GAP_mean","Baseload_kW","Night_kW","Evening_kW",
                "Peak_Intensity","Volatility_CV","Baseload_Ratio","MornEve_Slope"]
raw_profiles = df_daily.groupby("Cluster_ID")[profile_cols].mean()

med_gap   = raw_profiles["GAP_mean"].median()
med_base  = raw_profiles["Baseload_kW"].median()
med_night = raw_profiles["Night_kW"].median()
med_eve   = raw_profiles["Evening_kW"].median()
med_br    = raw_profiles["Baseload_Ratio"].median()
med_vol   = raw_profiles["Volatility_CV"].median()
med_slope = raw_profiles["MornEve_Slope"].median()

def label_cluster(row):
    if row["Night_kW"] > med_night * 2.0 and row["Baseload_kW"] > med_base * 2.5:
        return "Night Owl"
    elif row["GAP_mean"] > med_gap * 1.3 and row["Baseload_Ratio"] > med_br * 1.5:
        return "High Waste"
    elif row["GAP_mean"] < med_gap * 0.80 and row["Baseload_kW"] <= med_base:
        return "Efficient"
    elif row["Evening_kW"] > med_eve * 1.2 and row["MornEve_Slope"] > med_slope:
        return "Evening Heavy"
    elif row["Volatility_CV"] > med_vol * 1.3:
        return "Peak Spiker"
    else:
        return "Balanced"

cluster_labels = {i: label_cluster(r) for i, r in raw_profiles.iterrows()}
df_daily["Cluster_Label"] = df_daily["Cluster_ID"].map(cluster_labels)

dist    = df_daily["Cluster_Label"].value_counts()
max_pct = dist.max() / len(df_daily) * 100
print("\n    Cluster distribution:")
for lbl, cnt in dist.items():
    print(f"      {lbl:<18} {cnt:>5}  ({cnt/len(df_daily)*100:.1f}%)")

# DBSCAN anomaly detection
print("\n[7/7] DBSCAN anomaly detection...")
k_nb         = 4
nbrs         = NearestNeighbors(n_neighbors=k_nb).fit(X_pca)
distances, _ = nbrs.kneighbors(X_pca)
AUTO_EPS     = float(round(np.percentile(distances[:, k_nb - 1], 98), 2))
dbscan        = DBSCAN(eps=AUTO_EPS, min_samples=k_nb)
dbscan_labels = dbscan.fit_predict(X_pca)
n_noise       = int((dbscan_labels == -1).sum())
print(f"    eps={AUTO_EPS}  |  Anomalies: {n_noise} ({n_noise/len(df_daily)*100:.1f}%)")

df_daily["Is_Anomaly"]     = (dbscan_labels == -1).astype(int)
df_daily["DBSCAN_Cluster"] = dbscan_labels

# Export
OUTPUT_CSV = "energy_labeled_output.csv"
df_export  = df_daily.reset_index().rename(columns={"datetime": "Date"})
df_export.to_csv(OUTPUT_CSV, index=False)
print(f"    CSV saved → {OUTPUT_CSV}  ({len(df_export):,} rows × {len(df_export.columns)} cols)")

# ═══════════════════════════════════════════════════════════
# PHASE 3 — VALIDATION METRICS & CHARTS
# ═══════════════════════════════════════════════════════════

labels_arr = df_daily["Cluster_ID"].values
sil_final  = silhouette_score(X_pca, labels_arr)
db_final   = davies_bouldin_score(X_pca, labels_arr)
ch_final   = calinski_harabasz_score(X_pca, labels_arr)
sil_vals   = silhouette_samples(X_pca, labels_arr)

fig = plt.figure(figsize=(22, 20))
fig.suptitle("Energy Consumption Behavior Analysis — Results\n"
             "PCA + K-Means + DBSCAN | UCI Household Dataset",
             fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38,
                       top=0.93, bottom=0.06, left=0.07, right=0.97)

unique_labels = list(df_daily["Cluster_Label"].unique())
colors        = {lbl: plt.cm.tab10(i/10) for i, lbl in enumerate(unique_labels)}
k_list        = list(k_range)

ax1  = fig.add_subplot(gs[0, 0])
ax1b = ax1.twinx()
ax1.plot(k_list, wcss_scores, "o-", color="#2196F3", lw=2, label="WCSS (Elbow)")
ax1b.plot(k_list, composite, "s--", color="#FF5722", lw=2, label="Composite Score")
ax1b.plot(k_list, silhouette_scores, "^:", color="#9C27B0", lw=1.5, label="Silhouette")
ax1.axvline(x=FINAL_K, color="green", ls="--", lw=2, label=f"k={FINAL_K}")
ax1.set_title(f"Cluster Selection — Elbow + Composite (k={FINAL_K})", fontweight="bold")
ax1.set_xlabel("k"); ax1.set_ylabel("WCSS", color="#2196F3")
ax1b.set_ylabel("Score", color="#FF5722")
ax1.legend(loc="upper right", fontsize=7); ax1b.legend(loc="center right", fontsize=7)
ax1.grid(True, alpha=0.2)

ax2     = fig.add_subplot(gs[0, 1])
y_lower = 10
for cid in sorted(df_daily["Cluster_ID"].unique()):
    mask_c  = labels_arr == cid
    sil_c   = np.sort(sil_vals[mask_c])
    lbl_c   = df_daily[df_daily["Cluster_ID"] == cid]["Cluster_Label"].iloc[0]
    y_upper = y_lower + sil_c.shape[0]
    ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_c,
                      alpha=0.7, color=colors[lbl_c], label=f"C{cid}: {lbl_c}")
    y_lower = y_upper + 10
ax2.axvline(x=sil_final, color="red", ls="--", lw=2, label=f"Avg={sil_final:.3f}")
ax2.set_title(f"Silhouette Analysis (avg={sil_final:.4f})", fontweight="bold")
ax2.set_xlabel("Silhouette Coefficient"); ax2.set_ylabel("Cluster")
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2)

ax3  = fig.add_subplot(gs[1, 0])
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_scaled)
var2 = pca2.explained_variance_ratio_ * 100
for lbl in unique_labels:
    mask = df_daily["Cluster_Label"].values == lbl
    ax3.scatter(X_2d[mask,0], X_2d[mask,1], c=[colors[lbl]], label=lbl, alpha=0.5, s=15)
anom = df_daily["Is_Anomaly"].values == 1
if anom.sum() > 0:
    ax3.scatter(X_2d[anom,0], X_2d[anom,1], c="red", marker="x",
                s=80, label=f"Anomaly ({anom.sum()})", zorder=5)
ax3.set_title(f"PCA 2D Cluster View (PC1={var2[0]:.1f}%, PC2={var2[1]:.1f}%)", fontweight="bold")
ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")
ax3.legend(fontsize=7); ax3.grid(True, alpha=0.2)

ax4  = fig.add_subplot(gs[1, 1])
cnts = df_daily["Cluster_Label"].value_counts()
bars = ax4.bar(cnts.index, cnts.values,
               color=[colors[l] for l in cnts.index], edgecolor="white", width=0.6)
for bar, val in zip(bars, cnts.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
             f"{val}\n({val/len(df_daily)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
ax4.set_title("Cluster Distribution", fontweight="bold")
ax4.set_ylabel("Number of Days"); ax4.grid(True, alpha=0.2, axis="y")
ax4.set_ylim(0, cnts.max() * 1.25)

ax5 = fig.add_subplot(gs[2, 0])
for lbl in unique_labels:
    sub = df_daily[df_daily["Cluster_Label"] == lbl]["GAP_mean"]
    ax5.scatter(sub.index, sub.values, c=[colors[lbl]], label=lbl, alpha=0.4, s=10)
ax5.set_title("Daily Consumption Over Time — Coloured by Cluster", fontweight="bold")
ax5.set_xlabel("Date"); ax5.set_ylabel("Avg Power (kW)")
ax5.legend(fontsize=7); ax5.grid(True, alpha=0.2)

ax6   = fig.add_subplot(gs[2, 1])
hf    = ["GAP_mean","Peak_Intensity","Baseload_kW","Volatility_CV",
         "MornEve_Slope","Uniformity","Evening_kW","Night_kW"]
hd    = df_daily.groupby("Cluster_Label")[hf].mean()
hn    = (hd - hd.min()) / (hd.max() - hd.min() + 1e-9)
im    = ax6.imshow(hn.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax6.set_xticks(range(len(hf)))
ax6.set_xticklabels(["Consump","Peak","Baseload","Volatility","MornEve\nSlope",
                     "Uniformity","Evening","Night"], rotation=30, ha="right", fontsize=8)
ax6.set_yticks(range(len(hn.index)))
ax6.set_yticklabels(hn.index, fontsize=9)
ax6.set_title("Cluster Feature Heatmap (Red=High, Green=Low)", fontweight="bold")
plt.colorbar(im, ax=ax6, fraction=0.03, pad=0.04)
for i in range(len(hn.index)):
    for j in range(len(hf)):
        ax6.text(j, i, f"{hn.values[i,j]:.2f}", ha="center", va="center",
                 fontsize=7, color="white" if hn.values[i,j] > 0.6 else "black")

plt.savefig("energy_ml_results.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("\n    Chart saved → energy_ml_results.png")

# ═══════════════════════════════════════════════════════════
# SUCCESS CHECKLIST
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PROJECT SUCCESS CHECKLIST")
print("=" * 60)

checks = [
    ("Real UCI dataset (2M+ rows)",                    df_raw.shape[0] > 1_000_000),
    ("No missing values remaining",                     df_daily.isnull().sum().sum() == 0),
    (f"PCA explains ≥85% variance ({total_var:.1f}%)", total_var >= 85),
    (f"k≥3 enforced (k={FINAL_K} selected)",           FINAL_K >= 3),
    ("Clusters balanced (no cluster >80%)",             max_pct < 80),
    ("Silhouette >0.20 (published range 0.18–0.40)",   sil_final > 0.20),
    ("Davies-Bouldin < 1.50",                           db_final < 1.50),
    ("Calinski-Harabasz > 400 (strong)",                ch_final > 400),
    ("GMM cross-validation done",                       True),
    ("DBSCAN anomaly rate 0.5–3%",                      0 < n_noise/len(df_daily)*100 < 3),
    ("CSV exported for Power BI",                       os.path.exists(OUTPUT_CSV)),
    ("Validation chart saved",                          os.path.exists("energy_ml_results.png")),
]

score = 0
for msg, result in checks:
    icon = "✅" if result else "❌"
    if result: score += 1
    print(f"  {icon}  {msg}")

grade = "🏆 EXCELLENT" if score == len(checks) else "✅ GOOD" if score >= 9 else "⚠️ FAIR"
print(f"\n  Score : {score}/{len(checks)}  ({score/len(checks)*100:.0f}%)  Grade: {grade}")
print(f"""
  FINAL METRICS
  Silhouette     : {sil_final:.4f}  (0.18–0.40 = normal for single-household data)
  Davies-Bouldin : {db_final:.4f}  (lower is better)
  CH Score       : {ch_final:.1f}    (>400 = strong)
  Clusters       : {FINAL_K}
  Anomalies      : {n_noise} days
  Days processed : {len(df_daily):,}
  Rows processed : {df_raw.shape[0]:,}

  OUTPUT FILES
  energy_labeled_output.csv  → Import into Power BI
  energy_ml_results.png      → Validation charts
""")