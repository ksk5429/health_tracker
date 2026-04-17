"""
Generate all dashboard charts from processed data.
Run from repo root: python scripts/generate_charts.py
Outputs SVG/PNG to dashboard/charts/
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CHARTS = ROOT / "dashboard" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#c9d1d9",
    "text_dim": "#8b949e",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "orange": "#d29922",
    "purple": "#bc8cff",
    "pink": "#f778ba",
    "cyan": "#39d2c0",
    "gradient": ["#58a6ff", "#bc8cff", "#f778ba", "#39d2c0", "#3fb950", "#d29922"],
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["border"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text_dim"],
    "ytick.color": COLORS["text_dim"],
    "grid.color": COLORS["border"],
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 10,
})


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(CHARTS / f"{name}.svg", format="svg", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    fig.savefig(CHARTS / f"{name}.png", format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {name}")


def add_trend_line(ax, x, y, color, alpha=0.3):
    """Add a smooth rolling average trend line."""
    mask = ~np.isnan(y)
    if mask.sum() < 14:
        return
    s = pd.Series(y[mask], index=x[mask])
    trend = s.rolling(14, min_periods=7, center=True).mean()
    ax.plot(trend.index, trend.values, color=color, linewidth=2.5, alpha=0.9, zorder=5)


def format_date_axis(ax, df_dates):
    """Smart date axis formatting."""
    span = (df_dates.max() - df_dates.min()).days
    if span > 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    elif span > 60:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))


# ═══════════════════════════════════════════════════════════════════════════
# 1. HERO: Body Composition Timeline (DEXA)
# ═══════════════════════════════════════════════════════════════════════════
def chart_body_composition():
    df = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("DEXA Body Composition Timeline", fontsize=16, fontweight="bold",
                 color=COLORS["accent"], y=0.98)

    metrics = [
        ("weight_kg", "Weight (kg)", COLORS["accent"], (70, 85)),
        ("total_pct_fat", "Body Fat %", COLORS["orange"], (15, 35)),
        ("fmi", "Fat Mass Index", COLORS["red"], (3, 9)),
        ("lmi", "Lean Mass Index", COLORS["green"], (15, 20)),
        ("vat_area_cm2", "Visceral Fat Area (cm²)", COLORS["pink"], (0, 150)),
        ("ag_ratio", "Android/Gynoid Ratio", COLORS["purple"], (0.5, 1.8)),
    ]

    for ax, (col, title, color, ylim) in zip(axes.flatten(), metrics):
        vals = df[col].dropna()
        dates = df.loc[vals.index, "date"]
        ax.plot(dates, vals, "o-", color=color, markersize=10, linewidth=2.5, zorder=5)
        for x, y in zip(dates, vals):
            fmt = f"{y:.1f}" if isinstance(y, float) else str(y)
            ax.annotate(fmt, (x, y), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, fontweight="bold", color=color)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.2)
        format_date_axis(ax, dates)
        # Fill between first and last to show direction
        if len(vals) >= 2:
            delta = vals.iloc[-1] - vals.iloc[0]
            direction_color = COLORS["green"] if (
                (col in ["total_pct_fat", "fmi", "vat_area_cm2", "ag_ratio"] and delta < 0) or
                (col in ["lmi", "weight_kg"] and delta > 0 and col == "lmi")
            ) else COLORS["red"] if delta != 0 else COLORS["text_dim"]
            sign = "↓" if delta < 0 else "↑"
            ax.text(0.98, 0.05, f"{sign}{abs(delta):.1f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=12, fontweight="bold",
                    color=direction_color, alpha=0.8)

    save(fig, "01_body_composition")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Garmin Daily Activity Overview
# ═══════════════════════════════════════════════════════════════════════════
def chart_activity_overview():
    df = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle("Daily Activity Overview (Garmin)", fontsize=16, fontweight="bold",
                 color=COLORS["accent"], y=0.98)

    panels = [
        (gs[0, 0], "activity_steps", "Daily Steps", COLORS["accent"], None),
        (gs[0, 1], "activity_distance_km", "Distance (km)", COLORS["cyan"], None),
        (gs[1, 0], "activity_active_calories", "Active Calories", COLORS["orange"], None),
        (gs[1, 1], "activity_resting_hr", "Resting HR (bpm)", COLORS["red"], (45, 70)),
        (gs[2, 0], "activity_floors_climbed", "Floors Climbed", COLORS["green"], None),
        (gs[2, 1], None, "Intensity Minutes", COLORS["purple"], None),
    ]

    for spec, col, title, color, ylim in panels:
        ax = fig.add_subplot(spec)
        if col is None:
            # Combined intensity minutes
            mod = df["activity_moderate_intensity_min"].fillna(0)
            vig = df["activity_vigorous_intensity_min"].fillna(0)
            ax.bar(df["date"], mod, color=COLORS["orange"], alpha=0.6, label="Moderate", width=1)
            ax.bar(df["date"], vig, bottom=mod, color=COLORS["red"], alpha=0.6, label="Vigorous", width=1)
            total = mod + vig
            add_trend_line(ax, df["date"].values.astype("datetime64[ns]"),
                           total.values.astype(float), COLORS["purple"])
            ax.legend(loc="upper right", fontsize=8)
        else:
            vals = df[col].values.astype(float)
            ax.bar(df["date"], vals, color=color, alpha=0.4, width=1)
            add_trend_line(ax, df["date"].values.astype("datetime64[ns]"), vals, color)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.15)
        if ylim:
            ax.set_ylim(ylim)
        format_date_axis(ax, df["date"])

    save(fig, "02_activity_overview", tight=False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sleep Architecture
# ═══════════════════════════════════════════════════════════════════════════
def chart_sleep():
    df = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])
    df = df.dropna(subset=["sleep_total_hours"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    fig.suptitle("Sleep Architecture", fontsize=16, fontweight="bold",
                 color=COLORS["accent"], y=0.98)

    # Stacked sleep stages
    ax = axes[0, 0]
    ax.stackplot(df["date"],
                 df["sleep_deep_hours"].fillna(0),
                 df["sleep_light_hours"].fillna(0),
                 df["sleep_rem_hours"].fillna(0),
                 labels=["Deep", "Light", "REM"],
                 colors=[COLORS["purple"], COLORS["accent"], COLORS["cyan"]],
                 alpha=0.7)
    ax.set_title("Sleep Stages (hours)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.15)
    format_date_axis(ax, df["date"])

    # Sleep score
    ax = axes[0, 1]
    vals = df["sleep_score"].values.astype(float)
    ax.scatter(df["date"], vals, c=vals, cmap="RdYlGn", s=15, alpha=0.6, vmin=50, vmax=100)
    add_trend_line(ax, df["date"].values.astype("datetime64[ns]"), vals, COLORS["green"])
    ax.axhline(y=80, color=COLORS["green"], linestyle="--", alpha=0.4, label="Good (80)")
    ax.set_title("Sleep Score", fontsize=11, fontweight="bold")
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8)
    format_date_axis(ax, df["date"])

    # HRV
    ax = axes[1, 0]
    hrv = df.dropna(subset=["hrv_last_night_avg"])
    if len(hrv) > 0:
        vals = hrv["hrv_last_night_avg"].values.astype(float)
        ax.bar(hrv["date"], vals, color=COLORS["cyan"], alpha=0.4, width=1)
        add_trend_line(ax, hrv["date"].values.astype("datetime64[ns]"), vals, COLORS["cyan"])
    ax.set_title("HRV (last night avg, ms)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)
    format_date_axis(ax, df["date"])

    # Body Battery
    ax = axes[1, 1]
    ax.fill_between(df["date"], df["body_battery_charged"].fillna(0),
                     alpha=0.4, color=COLORS["green"], label="Charged")
    ax.fill_between(df["date"], -df["body_battery_drained"].fillna(0),
                     alpha=0.4, color=COLORS["red"], label="Drained")
    ax.axhline(y=0, color=COLORS["text_dim"], linewidth=0.5)
    ax.set_title("Body Battery (Charged ↑ / Drained ↓)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.15)
    format_date_axis(ax, df["date"])

    save(fig, "03_sleep_architecture")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Blood Test Biomarkers
# ═══════════════════════════════════════════════════════════════════════════
def chart_blood_tests():
    df = pd.read_csv(PROCESSED / "blood_tests.csv", parse_dates=["date"])
    df_num = df[df["is_numeric"] == True].copy()
    df_num["value"] = pd.to_numeric(df_num["value"], errors="coerce")

    # Key biomarkers to track
    panels = [
        ("TOTAL CHOLESTEROL(mg/dL)", "Total Cholesterol", COLORS["accent"], (100, 250),
         [(200, "Desirable", COLORS["green"]), (240, "Borderline", COLORS["orange"])]),
        ("LDL CHOLESTEROL(mg/dL)", "LDL Cholesterol", COLORS["red"], (50, 180),
         [(100, "Optimal", COLORS["green"]), (130, "Near-optimal", COLORS["orange"])]),
        ("HDL CHOLESTEROL(mg/dL)", "HDL Cholesterol", COLORS["green"], (30, 90),
         [(40, "Low risk", COLORS["orange"]), (60, "Protective", COLORS["green"])]),
        ("CREATININE (mg/dL)", "Creatinine", COLORS["orange"], (0.5, 1.8),
         [(1.2, "Upper normal", COLORS["orange"])]),
        ("HbA1c", "HbA1c (%)", COLORS["purple"], (4, 7),
         [(5.7, "Pre-diabetic", COLORS["orange"]), (6.5, "Diabetic", COLORS["red"])]),
        ("GOT (AST) (IU/L)", "AST (Liver)", COLORS["cyan"], (0, 60),
         [(40, "Upper normal", COLORS["orange"])]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Blood Test Biomarkers", fontsize=16, fontweight="bold",
                 color=COLORS["accent"], y=0.98)

    for ax, (marker, title, color, ylim, ref_lines) in zip(axes.flatten(), panels):
        sub = df_num[df_num["marker"].str.contains(marker.split("(")[0].strip(), case=False, na=False)]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center",
                    color=COLORS["text_dim"])
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        ax.plot(sub["date"], sub["value"], "o-", color=color, markersize=10, linewidth=2.5, zorder=5)
        for x, y in zip(sub["date"], sub["value"]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, fontweight="bold", color=color)
        for ref_val, ref_label, ref_color in ref_lines:
            ax.axhline(y=ref_val, color=ref_color, linestyle="--", alpha=0.4, linewidth=1)
            ax.text(ax.get_xlim()[1], ref_val, f" {ref_label}", va="center", fontsize=7,
                    color=ref_color, alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.15)
        # Sparse data (<=6 points): show exact dates, rotated
        ax.set_xticks(sub["date"])
        ax.set_xticklabels([d.strftime("%b\n%Y") for d in sub["date"]], fontsize=7)

    save(fig, "04_blood_biomarkers")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Workout Analysis
# ═══════════════════════════════════════════════════════════════════════════
def chart_workouts():
    df = pd.read_csv(PROCESSED / "activities.csv", parse_dates=["date"])

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Workout Analysis", fontsize=16, fontweight="bold",
                 color=COLORS["accent"], y=0.98)

    # Workout frequency by type (donut)
    ax = fig.add_subplot(gs[0, 0])
    type_counts = df["type"].value_counts().head(8)
    wedges, texts, autotexts = ax.pie(
        type_counts.values, labels=type_counts.index, autopct="%1.0f%%",
        colors=COLORS["gradient"] + [COLORS["text_dim"]] * 5,
        pctdistance=0.75, startangle=90
    )
    for t in texts + autotexts:
        t.set_fontsize(8)
        t.set_color(COLORS["text"])
    centre_circle = plt.Circle((0, 0), 0.5, fc=COLORS["card"])
    ax.add_artist(centre_circle)
    ax.text(0, 0, f"{len(df)}\nTotal", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLORS["text"])
    ax.set_title("Workout Types", fontsize=11, fontweight="bold")

    # Weekly workout count
    ax = fig.add_subplot(gs[0, 1])
    weekly = df.set_index("date").resample("W").size()
    ax.bar(weekly.index, weekly.values, color=COLORS["accent"], alpha=0.6, width=5)
    ax.axhline(y=weekly.mean(), color=COLORS["green"], linestyle="--", alpha=0.5,
               label=f"Avg: {weekly.mean():.1f}/wk")
    ax.set_title("Weekly Workout Count", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    format_date_axis(ax, pd.Series(weekly.index))

    # Duration distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df["duration_min"].dropna(), bins=30, color=COLORS["purple"], alpha=0.6, edgecolor=COLORS["border"])
    ax.axvline(x=df["duration_min"].median(), color=COLORS["orange"], linestyle="--",
               label=f"Median: {df['duration_min'].median():.0f}min")
    ax.set_title("Duration Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Minutes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # Calories per workout over time
    ax = fig.add_subplot(gs[1, 0])
    vals = df["calories"].values.astype(float)
    ax.scatter(df["date"], vals, c=COLORS["orange"], alpha=0.3, s=15)
    add_trend_line(ax, df["date"].values.astype("datetime64[ns]"), vals, COLORS["orange"])
    ax.set_title("Calories per Workout", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)
    format_date_axis(ax, df["date"])

    # HR zones (avg_hr distribution)
    ax = fig.add_subplot(gs[1, 1])
    hr_data = df["avg_hr"].dropna()
    if len(hr_data) > 0:
        zones = [(0, 100, "Recovery", COLORS["green"]),
                 (100, 120, "Fat Burn", COLORS["cyan"]),
                 (120, 140, "Cardio", COLORS["orange"]),
                 (140, 200, "Peak", COLORS["red"])]
        zone_counts = []
        zone_labels = []
        zone_colors = []
        for lo, hi, label, color in zones:
            count = ((hr_data >= lo) & (hr_data < hi)).sum()
            if count > 0:
                zone_counts.append(count)
                zone_labels.append(f"{label}\n({lo}-{hi})")
                zone_colors.append(color)
        ax.bar(zone_labels, zone_counts, color=zone_colors, alpha=0.7)
    ax.set_title("HR Zone Distribution", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)

    # VO2max trend
    ax = fig.add_subplot(gs[1, 2])
    vo2 = df.dropna(subset=["vo2max"])
    if len(vo2) > 0:
        ax.plot(vo2["date"], vo2["vo2max"], "o-", color=COLORS["green"], markersize=4, alpha=0.5)
        add_trend_line(ax, vo2["date"].values.astype("datetime64[ns]"),
                       vo2["vo2max"].values.astype(float), COLORS["green"])
    ax.set_title("VO2max Trend", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)
    format_date_axis(ax, df["date"])

    save(fig, "05_workout_analysis", tight=False)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Summary KPI Cards
# ═══════════════════════════════════════════════════════════════════════════
def chart_kpi_cards():
    garmin = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])
    dexa = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])
    activities = pd.read_csv(PROCESSED / "activities.csv", parse_dates=["date"])
    blood = pd.read_csv(PROCESSED / "blood_tests.csv", parse_dates=["date"])

    last_30 = garmin[garmin["date"] >= garmin["date"].max() - timedelta(days=30)]

    kpis = [
        ("Weight", f"{dexa['weight_kg'].iloc[-1]:.1f} kg",
         f"from {dexa['weight_kg'].iloc[0]:.1f}", COLORS["accent"],
         dexa['weight_kg'].iloc[-1] < dexa['weight_kg'].iloc[0]),
        ("Body Fat", f"{dexa['total_pct_fat'].iloc[-1]:.1f}%",
         f"from {dexa['total_pct_fat'].iloc[0]:.1f}%", COLORS["orange"],
         dexa['total_pct_fat'].iloc[-1] < dexa['total_pct_fat'].iloc[0]),
        ("Avg Steps", f"{int(last_30['activity_steps'].mean()):,}",
         "last 30 days", COLORS["green"], last_30['activity_steps'].mean() >= 8000),
        ("Avg Sleep", f"{last_30['sleep_total_hours'].mean():.1f}h",
         "last 30 days", COLORS["purple"], last_30['sleep_total_hours'].mean() >= 7),
        ("Resting HR", f"{last_30['activity_resting_hr'].mean():.0f} bpm",
         "last 30 days", COLORS["red"],
         last_30['activity_resting_hr'].mean() <= 60),
        ("Workouts", f"{len(activities)}",
         f"in {(activities['date'].max() - activities['date'].min()).days} days",
         COLORS["cyan"], True),
        ("HbA1c", f"{blood[blood['marker'].str.contains('HbA1c', na=False)]['value'].iloc[-1]}",
         "latest", COLORS["pink"],
         float(blood[blood['marker'].str.contains('HbA1c', na=False)]['value'].iloc[-1]) < 5.7),
        ("Visceral Fat", f"{dexa['vat_area_cm2'].iloc[-1]:.0f} cm²",
         f"from {dexa['vat_area_cm2'].iloc[0]:.0f}", COLORS["purple"],
         dexa['vat_area_cm2'].iloc[-1] < dexa['vat_area_cm2'].iloc[0]),
    ]

    fig, axes = plt.subplots(1, 8, figsize=(20, 2.8))
    fig.patch.set_facecolor(COLORS["bg"])

    for ax, (title, value, subtitle, color, good) in zip(axes, kpis):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        # Card background
        rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.05",
                               facecolor=COLORS["card"], edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        # Status dot
        dot_color = COLORS["green"] if good else COLORS["orange"]
        ax.plot(0.85, 0.85, "o", color=dot_color, markersize=8, transform=ax.transAxes)
        # Text
        ax.text(0.5, 0.75, title, ha="center", va="center", fontsize=8,
                color=COLORS["text_dim"], fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.45, value, ha="center", va="center", fontsize=16,
                color=color, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.18, subtitle, ha="center", va="center", fontsize=7,
                color=COLORS["text_dim"], transform=ax.transAxes)

    save(fig, "00_kpi_cards")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Multi-Variable Heatmap Calendar
# ═══════════════════════════════════════════════════════════════════════════
def chart_activity_heatmap():
    df = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon
    df["week_idx"] = (df["date"] - df["date"].min()).dt.days // 7

    heatmaps = [
        ("activity_steps", "Steps", "YlGn"),
        ("sleep_total_hours", "Sleep (hours)", "PuBu"),
        ("sleep_score", "Sleep Score", "RdYlGn"),
        ("activity_active_calories", "Active Calories", "YlOrRd"),
        ("hrv_last_night_avg", "HRV (ms)", "BuGn"),
        ("activity_resting_hr", "Resting HR (bpm)", "RdYlGn_r"),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(18, 16), sharex=True)
    fig.suptitle("Health Calendar Heatmaps", fontsize=16, fontweight="bold",
                 color=COLORS["accent"], y=0.99)

    # Month labels from first row only
    months = df.groupby(df["date"].dt.to_period("M")).first()

    for ax, (col, title, cmap) in zip(axes, heatmaps):
        pivot = df.pivot_table(index="dow", columns="week_idx", values=col, aggfunc="mean")
        im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                           cmap=cmap, shading="nearest", edgecolors=COLORS["bg"], linewidth=1)
        ax.set_yticks(range(7))
        ax.set_yticklabels(["M", "T", "W", "T", "F", "S", "S"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_ylabel(title, fontsize=8, fontweight="bold", color=COLORS["text"], rotation=0,
                      ha="right", va="center", labelpad=5)

        cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.015, pad=0.01)
        cb.ax.tick_params(labelsize=7, colors=COLORS["text_dim"])

    # Month labels on top axis
    for _, row in months.iterrows():
        axes[0].text(row["week_idx"], -1.0, row["date"].strftime("%b '%y"), fontsize=8,
                     color=COLORS["text_dim"], ha="left")

    fig.subplots_adjust(hspace=0.08)
    save(fig, "06_health_heatmaps", tight=False)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Correlation: Sleep vs Recovery
# ═══════════════════════════════════════════════════════════════════════════
def chart_correlations():
    df = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cross-Domain Correlations", fontsize=14, fontweight="bold",
                 color=COLORS["accent"], y=0.98)

    # Sleep vs Steps next day
    ax = axes[0]
    sub = df.dropna(subset=["sleep_total_hours", "activity_steps"])
    if len(sub) > 10:
        ax.scatter(sub["sleep_total_hours"], sub["activity_steps"],
                   c=COLORS["accent"], alpha=0.3, s=20)
        z = np.polyfit(sub["sleep_total_hours"], sub["activity_steps"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(sub["sleep_total_hours"].min(), sub["sleep_total_hours"].max(), 50)
        ax.plot(x_range, p(x_range), color=COLORS["orange"], linewidth=2, linestyle="--")
        r = sub["sleep_total_hours"].corr(sub["activity_steps"])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color=COLORS["accent"], va="top")
    ax.set_xlabel("Sleep (hours)")
    ax.set_ylabel("Steps")
    ax.set_title("Sleep Duration vs Activity", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)

    # HRV vs Sleep Score
    ax = axes[1]
    sub = df.dropna(subset=["hrv_last_night_avg", "sleep_score"])
    if len(sub) > 10:
        ax.scatter(sub["hrv_last_night_avg"], sub["sleep_score"],
                   c=COLORS["cyan"], alpha=0.3, s=20)
        r = sub["hrv_last_night_avg"].corr(sub["sleep_score"])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color=COLORS["cyan"], va="top")
    ax.set_xlabel("HRV (ms)")
    ax.set_ylabel("Sleep Score")
    ax.set_title("HRV vs Sleep Quality", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)

    # Body Battery vs Stress
    ax = axes[2]
    sub = df.dropna(subset=["body_battery_charged", "stress_overall"])
    if len(sub) > 10:
        ax.scatter(sub["stress_overall"], sub["body_battery_charged"],
                   c=COLORS["pink"], alpha=0.3, s=20)
        r = sub["stress_overall"].corr(sub["body_battery_charged"])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
                fontweight="bold", color=COLORS["pink"], va="top")
    ax.set_xlabel("Stress Score")
    ax.set_ylabel("Body Battery Charged")
    ax.set_title("Stress vs Recovery", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)

    save(fig, "07_correlations")


# ═══════════════════════════════════════════════════════════════════════════
# 9. DEXA Regional Body Composition Radar
# ═══════════════════════════════════════════════════════════════════════════
def chart_dexa_radar():
    df = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])

    categories = ["Body Fat %", "FMI", "LMI", "ALMI", "A/G Ratio", "VAT Area"]
    cat_cols = ["total_pct_fat", "fmi", "lmi", "almi", "ag_ratio", "vat_area_cm2"]
    # Normalize each to 0-1 range for radar
    ranges = [(15, 35), (3, 9), (15, 20), (6, 10), (0.5, 1.8), (0, 150)]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card"])

    for i, (_, row) in enumerate(df.iterrows()):
        values = []
        for col, (lo, hi) in zip(cat_cols, ranges):
            v = row[col]
            if pd.isna(v):
                values.append(0)
            else:
                values.append((v - lo) / (hi - lo))
        values += values[:1]
        color = COLORS["gradient"][i % len(COLORS["gradient"])]
        label = row["date"].strftime("%Y-%m-%d")
        ax.plot(angles, values, "o-", linewidth=2, markersize=6, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, color=COLORS["text"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%", "50%", "75%"], fontsize=7, color=COLORS["text_dim"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Body Composition Radar (DEXA)", fontsize=14, fontweight="bold",
                 color=COLORS["accent"], pad=20)
    ax.grid(color=COLORS["border"], alpha=0.3)

    save(fig, "08_dexa_radar")


def main():
    print("=== Generating Dashboard Charts ===")
    chart_kpi_cards()
    chart_body_composition()
    chart_activity_overview()
    chart_sleep()
    chart_blood_tests()
    chart_workouts()
    chart_activity_heatmap()
    chart_correlations()
    chart_dexa_radar()
    print(f"=== Done. {len(list(CHARTS.glob('*.png')))} PNGs + SVGs in dashboard/charts/ ===")


if __name__ == "__main__":
    main()
