"""
Generate interactive Plotly HTML dashboard for GitHub Pages.
Run from repo root: python scripts/generate_interactive.py
Outputs a single self-contained index.html with all charts + tab navigation.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "dashboard" / "interactive"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "bg": "#0d1117", "card": "#161b22", "border": "#30363d",
    "text": "#c9d1d9", "dim": "#8b949e",
    "accent": "#58a6ff", "green": "#3fb950", "red": "#f85149",
    "orange": "#d29922", "purple": "#bc8cff", "pink": "#f778ba", "cyan": "#39d2c0",
}
SEQ = [COLORS["accent"], COLORS["green"], COLORS["red"],
       COLORS["orange"], COLORS["purple"], COLORS["pink"], COLORS["cyan"]]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"], family="Inter, -apple-system, sans-serif", size=12),
    hovermode="x unified",
    xaxis=dict(gridcolor=COLORS["border"], gridwidth=0.5),
    yaxis=dict(gridcolor=COLORS["border"], gridwidth=0.5),
    margin=dict(l=50, r=30, t=50, b=40),
)


def fig_to_div(fig, div_id):
    """Convert a Plotly figure to an HTML div string (no full page wrapper)."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displayModeBar": True, "responsive": True,
                               "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                               "displaylogo": False})


def build_garmin_fig():
    df = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=("Daily Steps", "Distance (km)",
                        "Sleep Stages (hours)", "Sleep Score & HRV",
                        "Resting Heart Rate (bpm)", "Body Battery",
                        "Active Calories", "Intensity Minutes"),
        vertical_spacing=0.055, horizontal_spacing=0.06,
    )

    # Steps
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_steps"], name="Steps",
                         marker_color=SEQ[0], opacity=0.4), row=1, col=1)
    roll = df["activity_steps"].rolling(14, min_periods=7).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=roll, name="14d trend",
                             line=dict(color=SEQ[0], width=3)), row=1, col=1)

    # Distance
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_distance_km"], name="Distance",
                         marker_color=SEQ[6], opacity=0.4, showlegend=False), row=1, col=2)
    roll = df["activity_distance_km"].rolling(14, min_periods=7).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=roll, name="14d",
                             line=dict(color=SEQ[6], width=3), showlegend=False), row=1, col=2)

    # Sleep stages stacked
    for col_name, color, label in [
        ("sleep_deep_hours", SEQ[4], "Deep"), ("sleep_light_hours", SEQ[0], "Light"),
        ("sleep_rem_hours", SEQ[6], "REM"),
    ]:
        fig.add_trace(go.Bar(x=df["date"], y=df[col_name], name=label,
                             marker_color=color, opacity=0.7), row=2, col=1)

    # Sleep score + HRV
    fig.add_trace(go.Scatter(x=df["date"], y=df["sleep_score"].rolling(14).mean(),
                             name="Sleep Score (14d)", line=dict(color=SEQ[1], width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=df["date"], y=df["sleep_score"], name="Sleep Score (raw)",
                             line=dict(color=SEQ[1], width=1), opacity=0.3), row=2, col=2)
    hrv = df.dropna(subset=["hrv_last_night_avg"])
    fig.add_trace(go.Bar(x=hrv["date"], y=hrv["hrv_last_night_avg"], name="HRV (ms)",
                         marker_color=SEQ[6], opacity=0.25), row=2, col=2)

    # Resting HR
    fig.add_trace(go.Scatter(x=df["date"], y=df["activity_resting_hr"], name="Resting HR",
                             mode="markers", marker=dict(color=SEQ[2], size=3, opacity=0.3),
                             showlegend=False), row=3, col=1)
    roll = df["activity_resting_hr"].rolling(14, min_periods=7).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=roll, name="HR 14d",
                             line=dict(color=SEQ[2], width=3)), row=3, col=1)

    # Body Battery
    fig.add_trace(go.Scatter(x=df["date"], y=df["body_battery_charged"], name="Charged",
                             fill="tozeroy", fillcolor="rgba(63,185,80,0.15)",
                             line=dict(color=SEQ[1], width=1.5)), row=3, col=2)
    fig.add_trace(go.Scatter(x=df["date"], y=-df["body_battery_drained"].fillna(0), name="Drained",
                             fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
                             line=dict(color=SEQ[2], width=1.5)), row=3, col=2)

    # Calories
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_active_calories"], name="Calories",
                         marker_color=SEQ[3], opacity=0.4, showlegend=False), row=4, col=1)
    roll = df["activity_active_calories"].rolling(14, min_periods=7).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=roll, name="Cal 14d",
                             line=dict(color=SEQ[3], width=3), showlegend=False), row=4, col=1)

    # Intensity minutes
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_moderate_intensity_min"],
                         name="Moderate", marker_color=SEQ[3], opacity=0.6), row=4, col=2)
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_vigorous_intensity_min"],
                         name="Vigorous", marker_color=SEQ[2], opacity=0.6), row=4, col=2)

    fig.update_layout(
        height=1500, barmode="stack",
        title=dict(text="Daily Activity & Sleep", font=dict(size=18, color=SEQ[0])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        **LAYOUT_DEFAULTS,
    )
    for i in range(1, 9):
        fig.update_xaxes(gridcolor=COLORS["border"], row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS["border"], row=(i-1)//2+1, col=(i-1)%2+1)

    return fig


def build_dexa_fig():
    df = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Weight (kg)", "Body Fat %", "Fat Mass Index",
                        "Lean Mass Index", "Visceral Fat Area (cm\u00b2)", "Android/Gynoid Ratio"),
        vertical_spacing=0.15, horizontal_spacing=0.06,
    )

    metrics = [
        ("weight_kg", 1, 1), ("total_pct_fat", 1, 2), ("fmi", 1, 3),
        ("lmi", 2, 1), ("vat_area_cm2", 2, 2), ("ag_ratio", 2, 3),
    ]

    for i, (col, r, c) in enumerate(metrics):
        vals = df[col].fillna(0)
        fig.add_trace(go.Scatter(
            x=df["date"], y=vals, mode="lines+markers+text",
            text=[f"{v:.1f}" for v in vals], textposition="top center",
            textfont=dict(size=11, color=SEQ[i]),
            marker=dict(size=14, color=SEQ[i], line=dict(width=2, color=COLORS["bg"])),
            line=dict(color=SEQ[i], width=3),
            name=col.replace("_", " ").title(), showlegend=False,
        ), row=r, col=c)
        # Delta annotation
        if len(vals) >= 2:
            delta = vals.iloc[-1] - vals.iloc[0]
            sign = "+" if delta > 0 else ""
            fig.add_annotation(
                x=df["date"].iloc[-1], y=vals.iloc[-1],
                text=f"<b>{sign}{delta:.1f}</b>", showarrow=False,
                xshift=45, font=dict(size=12,
                    color=COLORS["green"] if (col in ["lmi", "almi"] and delta > 0) or
                          (col not in ["lmi", "almi"] and delta < 0) else COLORS["red"]),
                row=r, col=c,
            )

    fig.update_layout(
        height=650,
        title=dict(text="DEXA Body Composition Timeline", font=dict(size=18, color=SEQ[0])),
        **LAYOUT_DEFAULTS,
    )
    return fig


def build_dexa_radar_fig():
    df = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])

    categories = ["Body Fat %", "FMI", "LMI", "ALMI", "A/G Ratio", "VAT Area"]
    cat_cols = ["total_pct_fat", "fmi", "lmi", "almi", "ag_ratio", "vat_area_cm2"]
    ranges = [(15, 35), (3, 9), (15, 20), (6, 10), (0.5, 1.8), (0, 150)]

    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        values = []
        for col, (lo, hi) in zip(cat_cols, ranges):
            v = row[col]
            values.append((v - lo) / (hi - lo) if pd.notna(v) else 0)
        values.append(values[0])  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", fillcolor=f"rgba({','.join(str(int(c, 16)) for c in [SEQ[i][1:3], SEQ[i][3:5], SEQ[i][5:7]])},0.1)",
            line=dict(color=SEQ[i], width=2),
            marker=dict(size=8, color=SEQ[i]),
            name=row["date"].strftime("%Y-%m-%d"),
        ))

    fig.update_layout(
        height=550,
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"],
                            tickvals=[0.25, 0.5, 0.75], ticktext=["25%", "50%", "75%"],
                            tickfont=dict(size=9, color=COLORS["dim"])),
            angularaxis=dict(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["text"])),
        ),
        title=dict(text="Body Composition Radar", font=dict(size=18, color=SEQ[0])),
        legend=dict(orientation="h", y=-0.1),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ["xaxis", "yaxis", "hovermode"]},
    )
    return fig


def build_blood_fig():
    df = pd.read_csv(PROCESSED / "blood_tests.csv", parse_dates=["date"])
    df_num = df[df["is_numeric"] == True].copy()
    df_num["value"] = pd.to_numeric(df_num["value"], errors="coerce")

    panels = [
        ("TOTAL CHOLESTEROL", "Total Cholesterol", [(200, "Desirable"), (240, "Borderline")]),
        ("LDL CHOLESTEROL", "LDL Cholesterol", [(100, "Optimal"), (130, "Near-optimal")]),
        ("HDL CHOLESTEROL", "HDL Cholesterol", [(40, "Low risk"), (60, "Protective")]),
        ("CREATININE", "Creatinine", [(1.2, "Upper normal")]),
        ("HbA1c", "HbA1c (%)", [(5.7, "Pre-diabetic"), (6.5, "Diabetic")]),
        ("GOT (AST)", "AST (Liver)", [(40, "Upper normal")]),
        ("GPT (ALT)", "ALT (Liver)", [(40, "Upper normal")]),
        ("GAMMA-GT", "GGT", [(50, "Upper normal")]),
        ("FBS", "Fasting Glucose", [(100, "Pre-diabetic"), (126, "Diabetic")]),
        ("HEMOGLOBIN", "Hemoglobin", [(13.5, "Low normal")]),
    ]

    fig = make_subplots(rows=2, cols=5,
                        subplot_titles=[p[1] for p in panels],
                        vertical_spacing=0.18, horizontal_spacing=0.05)

    for i, (marker_prefix, title, ref_lines) in enumerate(panels):
        r, c = i // 5 + 1, i % 5 + 1
        sub = df_num[df_num["marker"].str.contains(marker_prefix, case=False, na=False)]
        if len(sub) == 0:
            continue
        color = SEQ[i % len(SEQ)]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["value"], mode="lines+markers+text",
            text=[f"{v:.1f}" for v in sub["value"]], textposition="top center",
            textfont=dict(size=9, color=color),
            marker=dict(size=10, color=color, line=dict(width=1, color=COLORS["bg"])),
            line=dict(color=color, width=2.5),
            name=title, showlegend=False,
            hovertemplate=f"<b>{title}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
        ), row=r, col=c)
        for ref_val, ref_label in ref_lines:
            fig.add_hline(y=ref_val, line=dict(color=COLORS["orange"], width=1, dash="dash"),
                          annotation=dict(text=ref_label, font=dict(size=8, color=COLORS["dim"])),
                          row=r, col=c)

    fig.update_layout(
        height=550,
        title=dict(text="Blood Test Biomarkers", font=dict(size=18, color=SEQ[0])),
        **LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(tickfont=dict(size=8))
    return fig


def build_workout_fig():
    df = pd.read_csv(PROCESSED / "activities.csv", parse_dates=["date"])

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Workout Types", "Weekly Frequency", "Duration vs Calories (color=HR)", "VO2max Trend"),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # Pie
    tc = df["type"].value_counts().head(8)
    fig.add_trace(go.Pie(labels=tc.index, values=tc.values, hole=0.5,
                         marker=dict(colors=SEQ), textinfo="percent+label",
                         textfont=dict(size=9)), row=1, col=1)

    # Weekly
    weekly = df.set_index("date").resample("W").size()
    fig.add_trace(go.Bar(x=weekly.index, y=weekly.values, name="Workouts/week",
                         marker_color=SEQ[0], opacity=0.6), row=1, col=2)
    avg = weekly.mean()
    fig.add_shape(type="line", x0=weekly.index.min(), x1=weekly.index.max(),
                  y0=avg, y1=avg, line=dict(color=SEQ[1], dash="dash", width=1.5),
                  xref="x2", yref="y2")
    fig.add_annotation(x=weekly.index[-1], y=avg, text=f"Avg: {avg:.1f}",
                       font=dict(color=SEQ[1], size=10), showarrow=False, xshift=40,
                       xref="x2", yref="y2")

    # Scatter
    fig.add_trace(go.Scatter(
        x=df["duration_min"], y=df["calories"], mode="markers",
        marker=dict(size=6, color=df["avg_hr"], colorscale="Turbo", opacity=0.7,
                    showscale=True, colorbar=dict(title="HR", len=0.3, y=0.2, thickness=12)),
        text=df["type"], name="Workouts", showlegend=False,
        hovertemplate="<b>%{text}</b><br>%{x:.0f}min, %{y:.0f}cal, HR=%{marker.color:.0f}<extra></extra>",
    ), row=2, col=1)

    # VO2max
    vo2 = df.dropna(subset=["vo2max"])
    if len(vo2) > 0:
        fig.add_trace(go.Scatter(x=vo2["date"], y=vo2["vo2max"], mode="markers",
                                 marker=dict(size=4, color=SEQ[1], opacity=0.3),
                                 name="VO2max", showlegend=False), row=2, col=2)
        roll = vo2.set_index("date")["vo2max"].rolling("30D").mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name="VO2max 30d",
                                 line=dict(color=SEQ[1], width=3)), row=2, col=2)

    fig.update_layout(
        height=700,
        title=dict(text="Workout Analysis", font=dict(size=18, color=SEQ[0])),
        **LAYOUT_DEFAULTS,
    )
    return fig


def build_correlations_fig():
    df = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Sleep vs Activity", "HRV vs Sleep Quality", "Stress vs Recovery"),
                        horizontal_spacing=0.07)

    pairs = [
        ("sleep_total_hours", "activity_steps", "Sleep (h)", "Steps", SEQ[0]),
        ("hrv_last_night_avg", "sleep_score", "HRV (ms)", "Sleep Score", SEQ[6]),
        ("stress_overall", "body_battery_charged", "Stress", "Battery Charged", SEQ[5]),
    ]

    for i, (x_col, y_col, x_label, y_label, color) in enumerate(pairs):
        sub = df.dropna(subset=[x_col, y_col])
        if len(sub) < 10:
            continue
        r = sub[x_col].corr(sub[y_col])
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col], mode="markers",
            marker=dict(size=5, color=color, opacity=0.25),
            name=f"r={r:.3f}", showlegend=True,
            hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.0f}}<extra></extra>",
        ), row=1, col=i+1)
        # Trend line
        z = np.polyfit(sub[x_col], sub[y_col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
        fig.add_trace(go.Scatter(x=x_range, y=p(x_range), mode="lines",
                                 line=dict(color=color, width=2, dash="dash"),
                                 showlegend=False), row=1, col=i+1)
        fig.update_xaxes(title_text=x_label, row=1, col=i+1)
        fig.update_yaxes(title_text=y_label, row=1, col=i+1)

    fig.update_layout(
        height=400,
        title=dict(text="Cross-Domain Correlations", font=dict(size=18, color=SEQ[0])),
        **LAYOUT_DEFAULTS,
    )
    return fig


def build_single_page():
    """Build a single-page interactive dashboard with tab navigation."""
    garmin = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])
    dexa = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])
    activities = pd.read_csv(PROCESSED / "activities.csv", parse_dates=["date"])
    blood = pd.read_csv(PROCESSED / "blood_tests.csv", parse_dates=["date"])
    last_30 = garmin[garmin["date"] >= garmin["date"].max() - timedelta(days=30)]

    # Build all figures
    figs = {
        "garmin": build_garmin_fig(),
        "dexa": build_dexa_fig(),
        "radar": build_dexa_radar_fig(),
        "blood": build_blood_fig(),
        "workout": build_workout_fig(),
        "correlations": build_correlations_fig(),
    }

    # Convert to divs
    divs = {k: fig_to_div(v, f"chart-{k}") for k, v in figs.items()}

    # KPI data
    def delta_html(latest, first, name, unit="", invert=False):
        d = latest - first
        good = (d < 0) if not invert else (d > 0)
        arrow = "&#x2193;" if d < 0 else "&#x2191;"
        cls = "good" if good else "bad"
        return f"""<div class="kpi">
            <div class="kpi-label">{name}</div>
            <div class="kpi-value">{latest:.1f}{unit}</div>
            <div class="kpi-sub {cls}">{arrow} {abs(d):.1f} from {first:.1f}</div>
        </div>"""

    kpi_html = "".join([
        delta_html(dexa["weight_kg"].iloc[-1], dexa["weight_kg"].iloc[0], "Weight", " kg"),
        delta_html(dexa["total_pct_fat"].iloc[-1], dexa["total_pct_fat"].iloc[0], "Body Fat", "%"),
        f"""<div class="kpi"><div class="kpi-label">Avg Steps (30d)</div>
            <div class="kpi-value">{int(last_30['activity_steps'].mean()):,}</div>
            <div class="kpi-sub neutral">daily average</div></div>""",
        f"""<div class="kpi"><div class="kpi-label">Avg Sleep</div>
            <div class="kpi-value">{last_30['sleep_total_hours'].mean():.1f}h</div>
            <div class="kpi-sub {'good' if last_30['sleep_total_hours'].mean() >= 7 else 'bad'}">
            {'above' if last_30['sleep_total_hours'].mean() >= 7 else 'below'} 7h target</div></div>""",
        f"""<div class="kpi"><div class="kpi-label">Resting HR</div>
            <div class="kpi-value">{last_30['activity_resting_hr'].mean():.0f} bpm</div>
            <div class="kpi-sub neutral">30-day avg</div></div>""",
        f"""<div class="kpi"><div class="kpi-label">Workouts</div>
            <div class="kpi-value">{len(activities)}</div>
            <div class="kpi-sub neutral">{(activities['date'].max() - activities['date'].min()).days}d tracked</div></div>""",
        delta_html(dexa["vat_area_cm2"].iloc[-1], dexa["vat_area_cm2"].iloc[0], "Visceral Fat", " cm\u00b2"),
        delta_html(dexa["lmi"].iloc[-1], dexa["lmi"].iloc[0], "Lean Mass Idx", "", invert=True),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Health Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg: {COLORS["bg"]}; --card: {COLORS["card"]}; --border: {COLORS["border"]};
    --text: {COLORS["text"]}; --dim: {COLORS["dim"]}; --accent: {COLORS["accent"]};
    --green: {COLORS["green"]}; --red: {COLORS["red"]}; --orange: {COLORS["orange"]};
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:Inter,-apple-system,sans-serif; }}
.container {{ max-width:1400px; margin:0 auto; padding:1.5rem; }}
header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; }}
h1 {{ font-size:1.6rem; color:var(--accent); }}
.date-range {{ color:var(--dim); font-size:0.85rem; }}

/* KPI Grid */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:0.8rem; margin-bottom:1.5rem; }}
.kpi {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem; text-align:center; transition:border-color 0.2s, transform 0.2s; }}
.kpi:hover {{ border-color:var(--accent); transform:translateY(-2px); }}
.kpi-label {{ font-size:0.7rem; color:var(--dim); text-transform:uppercase; letter-spacing:0.06em; }}
.kpi-value {{ font-size:1.6rem; font-weight:700; color:var(--accent); margin:0.2rem 0; }}
.kpi-sub {{ font-size:0.75rem; }}
.kpi-sub.good {{ color:var(--green); }}
.kpi-sub.bad {{ color:var(--red); }}
.kpi-sub.neutral {{ color:var(--dim); }}

/* Tabs */
.tabs {{ display:flex; gap:0.4rem; margin-bottom:1rem; flex-wrap:wrap; }}
.tab {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:0.5rem 1rem;
        color:var(--dim); cursor:pointer; font-size:0.85rem; transition:all 0.2s; user-select:none; }}
.tab:hover {{ border-color:var(--accent); color:var(--text); }}
.tab.active {{ background:var(--accent); color:var(--bg); border-color:var(--accent); font-weight:600; }}

/* Panels */
.panel {{ display:none; }}
.panel.active {{ display:block; }}
.chart-card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:0.5rem; margin-bottom:1rem; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
@media (max-width:900px) {{ .two-col {{ grid-template-columns:1fr; }} }}

footer {{ text-align:center; color:var(--dim); font-size:0.75rem; margin-top:2rem; padding:1rem 0; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>&#x1F3E5; Health Dashboard</h1>
        <span class="date-range">{garmin['date'].min().strftime('%b %Y')} &ndash; {garmin['date'].max().strftime('%b %Y')} | Updated {pd.Timestamp.now().strftime('%Y-%m-%d')}</span>
    </header>

    <div class="kpi-grid">{kpi_html}</div>

    <div class="tabs">
        <div class="tab active" data-tab="activity">&#x1F3C3; Activity & Sleep</div>
        <div class="tab" data-tab="dexa">&#x1F9B4; DEXA</div>
        <div class="tab" data-tab="blood">&#x1FA78; Blood Tests</div>
        <div class="tab" data-tab="workout">&#x1F4AA; Workouts</div>
        <div class="tab" data-tab="correlations">&#x1F517; Correlations</div>
    </div>

    <div class="panel active" id="panel-activity">
        <div class="chart-card">{divs["garmin"]}</div>
    </div>

    <div class="panel" id="panel-dexa">
        <div class="two-col">
            <div class="chart-card">{divs["dexa"]}</div>
            <div class="chart-card">{divs["radar"]}</div>
        </div>
    </div>

    <div class="panel" id="panel-blood">
        <div class="chart-card">{divs["blood"]}</div>
    </div>

    <div class="panel" id="panel-workout">
        <div class="chart-card">{divs["workout"]}</div>
    </div>

    <div class="panel" id="panel-correlations">
        <div class="chart-card">{divs["correlations"]}</div>
    </div>

    <footer>
        Data: {garmin['date'].nunique()} Garmin days &middot; {len(activities)} workouts &middot;
        {dexa.shape[0]} DEXA scans &middot; {blood['date'].nunique()} blood panels &middot;
        Auto-generated by <a href="https://github.com/ksk5429/health_tracker" style="color:var(--accent)">health_tracker</a>
    </footer>
</div>

<script>
document.querySelectorAll('.tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
        // Trigger Plotly resize for hidden charts
        window.dispatchEvent(new Event('resize'));
    }});
}});
</script>
</body>
</html>"""

    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("  index.html (single-page dashboard)")


def main():
    print("=== Generating Interactive Dashboard ===")
    build_single_page()
    print(f"=== Done. dashboard/interactive/index.html ===")


if __name__ == "__main__":
    main()
