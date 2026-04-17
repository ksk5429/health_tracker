"""
Tier 1 Derived Intelligence: compute decision-grade metrics from raw health data.

Metrics:
  1. Recovery Score (0-100): weighted HRV + Sleep Score + inverse Resting HR
  2. Training Load & Acute:Chronic Workload Ratio (ACR)
  3. Rolling Z-scores with anomaly flags
  4. Blood biomarker trajectory projections
  5. Body recomposition rates from DEXA

Run from repo root: python scripts/derived_metrics.py
Outputs to data/processed/derived_*.csv
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


# ═══════════════════════════════════════════════════════════════════════════
# 1. RECOVERY SCORE
#
# Methodology:
#   - Each component normalized to a personal 30-day rolling z-score
#   - Z-scores mapped to 0-100 via sigmoid (mean=50, ±2σ spans 10-90)
#   - Weighted: HRV 40%, Sleep Score 35%, Resting HR 25% (inverted)
#
# Rationale:
#   HRV is the strongest single predictor of autonomic readiness (Plews 2013).
#   Sleep score captures both duration and architecture.
#   Resting HR captures cardiovascular recovery state (lower = better).
#   Personal baselines (not population norms) because inter-individual
#   variation dwarfs intra-individual signal (Buchheit 2014).
# ═══════════════════════════════════════════════════════════════════════════
def compute_recovery_score(df):
    """Compute daily Recovery Score (0-100)."""
    WINDOW = 30
    MIN_PERIODS = 14
    WEIGHTS = {"hrv": 0.40, "sleep": 0.35, "rhr": 0.25}

    def personal_zscore(series):
        """Z-score relative to rolling 30-day personal baseline."""
        roll_mean = series.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
        roll_std = series.rolling(WINDOW, min_periods=MIN_PERIODS).std()
        # Clamp std to avoid division by zero
        roll_std = roll_std.clip(lower=1e-6)
        return (series - roll_mean) / roll_std

    def zscore_to_score(z):
        """Map z-score to 0-100 via sigmoid. z=0 → 50, z=±2 → ~12/88."""
        return 100 / (1 + np.exp(-1.5 * z))

    out = df[["date"]].copy()

    # HRV component (higher = better)
    hrv_z = personal_zscore(df["hrv_last_night_avg"])
    out["recovery_hrv"] = zscore_to_score(hrv_z)

    # Sleep score component (already 0-100, but normalize to personal baseline)
    sleep_z = personal_zscore(df["sleep_score"])
    out["recovery_sleep"] = zscore_to_score(sleep_z)

    # Resting HR component (INVERTED: lower = better)
    rhr_z = personal_zscore(df["activity_resting_hr"])
    out["recovery_rhr"] = zscore_to_score(-rhr_z)  # negate: low HR → high score

    # Weighted composite
    out["recovery_score"] = (
        WEIGHTS["hrv"] * out["recovery_hrv"] +
        WEIGHTS["sleep"] * out["recovery_sleep"] +
        WEIGHTS["rhr"] * out["recovery_rhr"]
    )

    # Classification
    out["recovery_class"] = pd.cut(
        out["recovery_score"],
        bins=[0, 33, 66, 100],
        labels=["Low", "Moderate", "High"],
        include_lowest=True,
    )

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. TRAINING LOAD & ACUTE:CHRONIC WORKLOAD RATIO
#
# Methodology:
#   - Session TRIMP (Training Impulse) = duration × HR_fraction
#     where HR_fraction = (avg_hr - resting_hr) / (max_hr - resting_hr)
#   - Daily load = sum of session TRIMPs
#   - Acute load = 7-day rolling sum (represents fatigue)
#   - Chronic load = 28-day EWMA (represents fitness)
#   - ACR = acute / chronic
#
# Reference zones (Gabbett 2016):
#   ACR < 0.8  → undertrained (detraining risk)
#   0.8 - 1.3  → sweet spot (optimal adaptation)
#   1.3 - 1.5  → caution (elevated injury risk)
#   > 1.5      → danger zone (high injury risk)
# ═══════════════════════════════════════════════════════════════════════════
def compute_training_load(garmin_daily, activities):
    """Compute daily training load and acute:chronic workload ratio."""
    # Estimate resting HR from garmin daily (personal baseline)
    resting_hr_baseline = garmin_daily["activity_resting_hr"].median()
    # Estimate max HR (simple: 220 - age; but we have data)
    max_hr_observed = activities["max_hr"].quantile(0.95)
    if pd.isna(max_hr_observed) or max_hr_observed < 150:
        max_hr_observed = 190  # fallback

    hr_reserve = max_hr_observed - resting_hr_baseline
    if hr_reserve <= 0:
        hr_reserve = 100  # safety

    # Session TRIMP
    activities = activities.copy()
    activities["hr_fraction"] = (
        (activities["avg_hr"].fillna(resting_hr_baseline) - resting_hr_baseline) / hr_reserve
    ).clip(0, 1)
    activities["trimp"] = activities["duration_min"].fillna(0) * activities["hr_fraction"]

    # Aggregate to daily
    daily_trimp = activities.groupby("date")["trimp"].sum().reset_index()
    daily_trimp.columns = ["date", "daily_trimp"]

    # Merge with full date range from garmin
    date_range = pd.DataFrame({"date": garmin_daily["date"]})
    out = date_range.merge(daily_trimp, on="date", how="left")
    out["daily_trimp"] = out["daily_trimp"].fillna(0)

    # Acute load (7-day rolling sum)
    out["acute_load"] = out["daily_trimp"].rolling(7, min_periods=1).sum()

    # Chronic load (28-day EWMA, decay matches ~28-day half-life)
    out["chronic_load"] = out["daily_trimp"].ewm(span=28, min_periods=14).mean() * 7

    # ACR
    out["acr"] = out["acute_load"] / out["chronic_load"].clip(lower=1)

    # Zone classification
    out["acr_zone"] = pd.cut(
        out["acr"],
        bins=[0, 0.8, 1.3, 1.5, float("inf")],
        labels=["Undertrained", "Sweet Spot", "Caution", "Danger"],
        include_lowest=True,
    )

    # Monotony (Banister): how uniform is the training? High monotony + high load → overtraining
    roll_mean = out["daily_trimp"].rolling(7, min_periods=4).mean()
    roll_std = out["daily_trimp"].rolling(7, min_periods=4).std()
    out["monotony"] = roll_mean / roll_std.clip(lower=0.01)
    out["strain"] = out["acute_load"] * out["monotony"]

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. ROLLING Z-SCORES & ANOMALY DETECTION
#
# For each daily metric, compute z-score relative to 30-day rolling window.
# Flag days where |z| > 2 (outside 95th percentile of personal variation).
# This answers: "Was this day unusually good or bad for ME?"
# ═══════════════════════════════════════════════════════════════════════════
def compute_zscores(df):
    """Compute rolling z-scores for key metrics, flag anomalies."""
    WINDOW = 30
    MIN_PERIODS = 14

    metrics = [
        "activity_steps", "activity_active_calories", "activity_resting_hr",
        "sleep_total_hours", "sleep_score", "sleep_deep_hours",
        "hrv_last_night_avg", "body_battery_charged", "stress_overall",
    ]

    out = df[["date"]].copy()
    anomaly_flags = []

    for col in metrics:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        roll_mean = s.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
        roll_std = s.rolling(WINDOW, min_periods=MIN_PERIODS).std().clip(lower=1e-6)
        z = (s - roll_mean) / roll_std
        out[f"z_{col}"] = z

        # Flag anomalies
        anomalies = df.loc[z.abs() > 2, ["date"]].copy()
        anomalies["metric"] = col
        anomalies["z_score"] = z[z.abs() > 2].values
        anomalies["value"] = s[z.abs() > 2].values
        anomalies["direction"] = np.where(anomalies["z_score"] > 0, "high", "low")
        anomaly_flags.append(anomalies)

    # Daily composite anomaly score: mean absolute z-score across all metrics
    z_cols = [c for c in out.columns if c.startswith("z_")]
    out["anomaly_score"] = out[z_cols].abs().mean(axis=1)

    anomaly_df = pd.concat(anomaly_flags, ignore_index=True) if anomaly_flags else pd.DataFrame()

    return out, anomaly_df


# ═══════════════════════════════════════════════════════════════════════════
# 4. BLOOD BIOMARKER TRAJECTORY PROJECTIONS
#
# Fit OLS to each numeric biomarker's time series (4 data points).
# Project forward 12 months with prediction intervals.
# Flag biomarkers trending toward clinical thresholds.
#
# Clinical thresholds (sources: AHA, KDIGO, ADA):
#   LDL > 130 mg/dL (borderline high)
#   HDL < 40 mg/dL (low, risk factor)
#   Creatinine > 1.3 mg/dL (possible CKD stage 2)
#   HbA1c > 5.7% (pre-diabetic)
#   eGFR < 60 (CKD stage 3)
#   AST/ALT > 40 IU/L (liver concern)
#   Triglycerides > 150 mg/dL (high)
# ═══════════════════════════════════════════════════════════════════════════
CLINICAL_THRESHOLDS = {
    "LDL CHOLESTEROL": {"upper": 130, "label": "Borderline high"},
    "TOTAL CHOLESTEROL": {"upper": 200, "label": "Desirable limit"},
    "HDL CHOLESTEROL": {"lower": 40, "label": "Low (risk factor)"},
    "CREATININE": {"upper": 1.3, "label": "Possible CKD"},
    "HbA1c": {"upper": 5.7, "label": "Pre-diabetic"},
    "GOT (AST)": {"upper": 40, "label": "Liver concern"},
    "GPT (ALT)": {"upper": 40, "label": "Liver concern"},
    "GAMMA-GT": {"upper": 50, "label": "Elevated"},
    "FBS": {"upper": 100, "label": "Pre-diabetic"},
}


def compute_blood_trajectories(blood_df):
    """Compute linear trajectories and flag threshold crossings."""
    df = blood_df.copy()
    df = df[df["is_numeric"] == True].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Convert dates to numeric (days from first measurement)
    ref_date = df["date"].min()
    df["days"] = (df["date"] - ref_date).dt.days

    # Projection horizon: 365 days from last measurement
    last_date = df["date"].max()
    proj_days = np.arange(0, (last_date - ref_date).days + 365, 30)
    proj_dates = ref_date + pd.to_timedelta(proj_days, unit="D")

    results = []
    markers = df.groupby("marker").filter(lambda g: g["value"].nunique() >= 2)

    for marker_name, group in markers.groupby("marker"):
        if len(group) < 2:
            continue

        x = group["days"].values.astype(float)
        y = group["value"].values.astype(float)

        # OLS fit
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

        # Prediction with confidence interval
        n = len(x)
        x_mean = x.mean()
        ss_x = ((x - x_mean) ** 2).sum()
        residual_std = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / max(n - 2, 1))

        for d, dt in zip(proj_days, proj_dates):
            y_pred = slope * d + intercept
            # Prediction interval (t-distribution)
            if n > 2 and ss_x > 0:
                t_val = sp_stats.t.ppf(0.975, n - 2)
                se = residual_std * np.sqrt(1 + 1/n + (d - x_mean)**2 / ss_x)
                ci_lower = y_pred - t_val * se
                ci_upper = y_pred + t_val * se
            else:
                ci_lower = ci_upper = y_pred

            results.append({
                "marker": marker_name,
                "date": dt,
                "predicted": y_pred,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "is_observed": d in x,
                "slope_per_year": slope * 365,
                "r_squared": r_value ** 2,
                "p_value": p_value,
            })

    proj_df = pd.DataFrame(results)

    # Flag threshold crossings
    alerts = []
    for marker_name, group in proj_df.groupby("marker"):
        # Check each known threshold
        for thresh_key, thresh_info in CLINICAL_THRESHOLDS.items():
            if thresh_key.lower() not in marker_name.lower():
                continue

            future = group[group["date"] > last_date]
            if future.empty:
                continue

            if "upper" in thresh_info:
                crossings = future[future["predicted"] > thresh_info["upper"]]
                if not crossings.empty:
                    cross_date = crossings["date"].iloc[0]
                    alerts.append({
                        "marker": marker_name,
                        "threshold": thresh_info["upper"],
                        "threshold_label": thresh_info["label"],
                        "direction": "above",
                        "projected_crossing_date": cross_date,
                        "current_value": group[group["is_observed"]]["predicted"].iloc[-1] if any(group["is_observed"]) else None,
                        "slope_per_year": group["slope_per_year"].iloc[0],
                    })

            if "lower" in thresh_info:
                crossings = future[future["predicted"] < thresh_info["lower"]]
                if not crossings.empty:
                    cross_date = crossings["date"].iloc[0]
                    alerts.append({
                        "marker": marker_name,
                        "threshold": thresh_info["lower"],
                        "threshold_label": thresh_info["label"],
                        "direction": "below",
                        "projected_crossing_date": cross_date,
                        "current_value": group[group["is_observed"]]["predicted"].iloc[-1] if any(group["is_observed"]) else None,
                        "slope_per_year": group["slope_per_year"].iloc[0],
                    })

    alerts_df = pd.DataFrame(alerts) if alerts else pd.DataFrame()

    return proj_df, alerts_df


# ═══════════════════════════════════════════════════════════════════════════
# 5. BODY RECOMPOSITION ANALYSIS (DEXA)
#
# Compute period-over-period changes in:
#   - Fat mass, lean mass, total mass
#   - Monthly rates of change
#   - Fat-Free Mass Index trajectory
#   - Partitioning ratio: what fraction of mass change was lean?
#
# The partitioning ratio (P-ratio) is critical:
#   P > 0.5 during weight loss = losing more fat than muscle (good)
#   P > 0.5 during weight gain = gaining more muscle than fat (good)
# ═══════════════════════════════════════════════════════════════════════════
def compute_recomposition(dexa_df):
    """Compute body recomposition metrics from DEXA scans."""
    df = dexa_df.copy()
    df["fat_kg"] = df["total_fat_g"] / 1000
    df["lean_kg"] = df["total_lean_bmc_g"] / 1000
    df["total_kg"] = df["total_mass_g"] / 1000

    records = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        days = (curr["date"] - prev["date"]).days
        months = days / 30.44

        d_fat = curr["fat_kg"] - prev["fat_kg"]
        d_lean = curr["lean_kg"] - prev["lean_kg"]
        d_total = curr["total_kg"] - prev["total_kg"]

        # Partitioning ratio: fraction of mass change that was lean
        if abs(d_total) > 0.01:
            p_ratio = d_lean / d_total
        else:
            p_ratio = 0.5

        records.append({
            "period": f"{prev['date'].strftime('%b %y')} → {curr['date'].strftime('%b %y')}",
            "date_start": prev["date"],
            "date_end": curr["date"],
            "days": days,
            "delta_fat_kg": d_fat,
            "delta_lean_kg": d_lean,
            "delta_total_kg": d_total,
            "fat_rate_kg_month": d_fat / months if months > 0 else 0,
            "lean_rate_kg_month": d_lean / months if months > 0 else 0,
            "p_ratio": p_ratio,
            "fat_pct_start": prev["total_pct_fat"],
            "fat_pct_end": curr["total_pct_fat"],
            "fmi_start": prev["fmi"],
            "fmi_end": curr["fmi"],
            "lmi_start": prev["lmi"],
            "lmi_end": curr["lmi"],
            "vat_start": prev["vat_area_cm2"],
            "vat_end": curr["vat_area_cm2"],
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# 6. LAG-CORRELATION MATRIX
#
# For each pair of metrics (X, Y), compute Pearson correlation between
# X on day N and Y on day N+lag, for lags 0..3 days.
# This reveals temporal causality: "does yesterday's X predict today's Y?"
# ═══════════════════════════════════════════════════════════════════════════
def compute_lag_correlations(df):
    """Compute lag-correlation matrix for key metrics."""
    metrics = [
        ("sleep_total_hours", "Sleep Hours"),
        ("sleep_score", "Sleep Score"),
        ("hrv_last_night_avg", "HRV"),
        ("activity_steps", "Steps"),
        ("activity_active_calories", "Active Cal"),
        ("activity_resting_hr", "Resting HR"),
        ("body_battery_charged", "Battery"),
        ("stress_overall", "Stress"),
    ]

    available = [(col, label) for col, label in metrics if col in df.columns]
    max_lag = 3
    results = []

    for lag in range(max_lag + 1):
        for col_x, label_x in available:
            for col_y, label_y in available:
                x = df[col_x].astype(float)
                y = df[col_y].shift(-lag).astype(float)
                mask = x.notna() & y.notna()
                if mask.sum() < 20:
                    continue
                r, p = sp_stats.pearsonr(x[mask], y[mask])
                results.append({
                    "x_metric": label_x,
                    "y_metric": label_y,
                    "x_col": col_x,
                    "y_col": col_y,
                    "lag_days": lag,
                    "correlation": r,
                    "p_value": p,
                    "significant": p < 0.05,
                    "n": int(mask.sum()),
                })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=== Tier 1: Derived Intelligence ===")

    garmin = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])
    activities = pd.read_csv(PROCESSED / "activities.csv", parse_dates=["date"])
    dexa = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])
    blood = pd.read_csv(PROCESSED / "blood_tests.csv", parse_dates=["date"])

    # 1. Recovery Score
    recovery = compute_recovery_score(garmin)
    recovery.to_csv(PROCESSED / "derived_recovery.csv", index=False)
    valid = recovery["recovery_score"].dropna()
    print(f"  Recovery Score: {len(valid)} days, mean={valid.mean():.1f}, "
          f"range=[{valid.min():.0f}, {valid.max():.0f}]")

    # 2. Training Load
    training = compute_training_load(garmin, activities)
    training.to_csv(PROCESSED / "derived_training_load.csv", index=False)
    valid_acr = training["acr"].dropna()
    print(f"  Training Load: {len(valid_acr)} days, mean ACR={valid_acr.mean():.2f}, "
          f"sweet spot: {(training['acr_zone'] == 'Sweet Spot').sum()} days")

    # 3. Z-scores & Anomalies
    zscores, anomalies = compute_zscores(garmin)
    zscores.to_csv(PROCESSED / "derived_zscores.csv", index=False)
    anomalies.to_csv(PROCESSED / "derived_anomalies.csv", index=False)
    print(f"  Z-scores: {len(zscores)} days, {len(anomalies)} anomaly flags "
          f"({anomalies['direction'].value_counts().to_dict() if len(anomalies) > 0 else 'none'})")

    # 4. Blood Trajectories
    projections, alerts = compute_blood_trajectories(blood)
    projections.to_csv(PROCESSED / "derived_blood_projections.csv", index=False)
    alerts.to_csv(PROCESSED / "derived_blood_alerts.csv", index=False)
    n_markers = projections["marker"].nunique() if len(projections) > 0 else 0
    print(f"  Blood Trajectories: {n_markers} markers projected, {len(alerts)} threshold alerts")
    if len(alerts) > 0:
        for _, a in alerts.iterrows():
            print(f"    ! {a['marker']}: trending {a['direction']} {a['threshold']} "
                  f"({a['threshold_label']}) by {a['projected_crossing_date'].strftime('%b %Y')}")

    # 5. Body Recomposition
    recomp = compute_recomposition(dexa)
    recomp.to_csv(PROCESSED / "derived_recomposition.csv", index=False)
    print(f"  Recomposition: {len(recomp)} periods")
    for _, r in recomp.iterrows():
        print(f"    {r['period']}: fat {r['delta_fat_kg']:+.1f}kg, "
              f"lean {r['delta_lean_kg']:+.1f}kg, P-ratio={r['p_ratio']:.2f}")

    # 6. Lag Correlations
    lag_corr = compute_lag_correlations(garmin)
    lag_corr.to_csv(PROCESSED / "derived_lag_correlations.csv", index=False)
    sig = lag_corr[(lag_corr["significant"]) & (lag_corr["lag_days"] > 0) &
                   (lag_corr["x_col"] != lag_corr["y_col"])]
    print(f"  Lag Correlations: {len(lag_corr)} pairs, {len(sig)} significant cross-lag relationships")
    top = sig.reindex(sig["correlation"].abs().sort_values(ascending=False).index).head(5)
    for _, row in top.iterrows():
        print(f"    {row['x_metric']} → {row['y_metric']} (lag={row['lag_days']}d): "
              f"r={row['correlation']:.3f}, p={row['p_value']:.4f}")

    print("=== Done. Derived metrics in data/processed/derived_*.csv ===")


if __name__ == "__main__":
    main()
