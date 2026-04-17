"""
Generate interactive Plotly HTML dashboard for GitHub Pages.
Run from repo root: python scripts/generate_interactive.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "dashboard" / "interactive"
OUT.mkdir(parents=True, exist_ok=True)

THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="Inter, -apple-system, sans-serif"),
)
COLORS = ["#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff", "#f778ba", "#39d2c0"]


def build_garmin_dashboard():
    """Build the main Garmin daily metrics interactive dashboard."""
    df = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=("Daily Steps", "Distance (km)",
                        "Sleep Duration & Stages", "Sleep Score & HRV",
                        "Resting Heart Rate", "Body Battery",
                        "Active Calories", "Intensity Minutes"),
        vertical_spacing=0.06, horizontal_spacing=0.08,
    )

    # Steps
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_steps"], name="Steps",
                         marker_color=COLORS[0], opacity=0.5), row=1, col=1)
    roll = df["activity_steps"].rolling(14, min_periods=7).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=roll, name="14d avg",
                             line=dict(color=COLORS[0], width=3)), row=1, col=1)

    # Distance
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_distance_km"], name="Distance",
                         marker_color=COLORS[6], opacity=0.5, showlegend=False), row=1, col=2)

    # Sleep stages stacked
    for col_name, color, label in [
        ("sleep_deep_hours", COLORS[4], "Deep"),
        ("sleep_light_hours", COLORS[0], "Light"),
        ("sleep_rem_hours", COLORS[6], "REM"),
    ]:
        fig.add_trace(go.Bar(x=df["date"], y=df[col_name], name=label,
                             marker_color=color, opacity=0.7), row=2, col=1)
    fig.update_layout(barmode="stack")

    # Sleep score + HRV dual axis
    fig.add_trace(go.Scatter(x=df["date"], y=df["sleep_score"], name="Sleep Score",
                             line=dict(color=COLORS[1], width=1), opacity=0.5), row=2, col=2)
    fig.add_trace(go.Scatter(x=df["date"], y=df["sleep_score"].rolling(14).mean(),
                             name="Score 14d", line=dict(color=COLORS[1], width=3)), row=2, col=2)
    hrv = df.dropna(subset=["hrv_last_night_avg"])
    fig.add_trace(go.Bar(x=hrv["date"], y=hrv["hrv_last_night_avg"], name="HRV",
                         marker_color=COLORS[6], opacity=0.3), row=2, col=2)

    # Resting HR
    fig.add_trace(go.Scatter(x=df["date"], y=df["activity_resting_hr"], name="Resting HR",
                             mode="markers", marker=dict(color=COLORS[2], size=3, opacity=0.4),
                             showlegend=False), row=3, col=1)
    roll = df["activity_resting_hr"].rolling(14, min_periods=7).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=roll, name="HR 14d avg",
                             line=dict(color=COLORS[2], width=3)), row=3, col=1)

    # Body Battery
    fig.add_trace(go.Scatter(x=df["date"], y=df["body_battery_charged"], name="Charged",
                             fill="tozeroy", fillcolor="rgba(63,185,80,0.2)",
                             line=dict(color=COLORS[1], width=1)), row=3, col=2)
    fig.add_trace(go.Scatter(x=df["date"], y=-df["body_battery_drained"].fillna(0), name="Drained",
                             fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
                             line=dict(color=COLORS[2], width=1)), row=3, col=2)

    # Calories
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_active_calories"], name="Calories",
                         marker_color=COLORS[3], opacity=0.4, showlegend=False), row=4, col=1)

    # Intensity minutes
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_moderate_intensity_min"],
                         name="Moderate", marker_color=COLORS[3], opacity=0.6), row=4, col=2)
    fig.add_trace(go.Bar(x=df["date"], y=df["activity_vigorous_intensity_min"],
                         name="Vigorous", marker_color=COLORS[2], opacity=0.6), row=4, col=2)

    fig.update_layout(
        height=1600, width=1200,
        title=dict(text="<b>Garmin Daily Health Dashboard</b>", font=dict(size=20, color=COLORS[0])),
        **THEME,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#30363d", gridwidth=0.5)
    fig.update_yaxes(gridcolor="#30363d", gridwidth=0.5)

    fig.write_html(OUT / "garmin_daily.html", include_plotlyjs="cdn")
    print("  garmin_daily.html")


def build_dexa_dashboard():
    """Interactive DEXA body composition timeline."""
    df = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Weight (kg)", "Body Fat %", "Fat Mass Index",
                        "Lean Mass Index", "Visceral Fat (cm²)", "Android/Gynoid Ratio"),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    metrics = [
        ("weight_kg", 1, 1), ("total_pct_fat", 1, 2), ("fmi", 1, 3),
        ("lmi", 2, 1), ("vat_area_cm2", 2, 2), ("ag_ratio", 2, 3),
    ]

    for i, (col, r, c) in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines+markers+text",
            text=[f"{v:.1f}" for v in df[col].fillna(0)],
            textposition="top center", textfont=dict(size=11, color=COLORS[i]),
            marker=dict(size=12, color=COLORS[i]),
            line=dict(color=COLORS[i], width=3),
            name=col, showlegend=False,
        ), row=r, col=c)

    fig.update_layout(
        height=700, width=1200,
        title=dict(text="<b>DEXA Body Composition Timeline</b>", font=dict(size=20, color=COLORS[0])),
        **THEME, hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#30363d")
    fig.update_yaxes(gridcolor="#30363d")

    fig.write_html(OUT / "dexa.html", include_plotlyjs="cdn")
    print("  dexa.html")


def build_blood_dashboard():
    """Interactive blood test biomarker explorer."""
    df = pd.read_csv(PROCESSED / "blood_tests.csv", parse_dates=["date"])
    df_num = df[df["is_numeric"] == True].copy()
    df_num["value"] = pd.to_numeric(df_num["value"], errors="coerce")

    key_markers = [
        "TOTAL CHOLESTEROL", "LDL CHOLESTEROL", "HDL CHOLESTEROL",
        "CREATININE", "HbA1c", "GOT (AST)", "GPT (ALT)", "GAMMA-GT",
        "FBS", "HEMOGLOBIN",
    ]

    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[m.split("(")[0].strip() for m in key_markers],
        vertical_spacing=0.15, horizontal_spacing=0.06,
    )

    for i, marker_prefix in enumerate(key_markers):
        r = i // 5 + 1
        c = i % 5 + 1
        sub = df_num[df_num["marker"].str.contains(marker_prefix, case=False, na=False)]
        if len(sub) > 0:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["value"], mode="lines+markers+text",
                text=[f"{v:.1f}" for v in sub["value"]],
                textposition="top center", textfont=dict(size=9),
                marker=dict(size=8, color=COLORS[i % len(COLORS)]),
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                name=marker_prefix, showlegend=False,
            ), row=r, col=c)

    fig.update_layout(
        height=600, width=1400,
        title=dict(text="<b>Blood Test Biomarker Explorer</b>", font=dict(size=20, color=COLORS[0])),
        **THEME, hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#30363d", tickfont=dict(size=8))
    fig.update_yaxes(gridcolor="#30363d")

    fig.write_html(OUT / "blood_tests.html", include_plotlyjs="cdn")
    print("  blood_tests.html")


def build_workout_dashboard():
    """Interactive workout analysis."""
    df = pd.read_csv(PROCESSED / "activities.csv", parse_dates=["date"])

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Workout Types", "Weekly Frequency",
                        "Duration vs Calories", "VO2max Trend"),
        vertical_spacing=0.12, horizontal_spacing=0.1,
    )

    # Pie chart
    type_counts = df["type"].value_counts().head(8)
    fig.add_trace(go.Pie(
        labels=type_counts.index, values=type_counts.values,
        hole=0.5, marker=dict(colors=COLORS),
        textinfo="percent+label", textfont=dict(size=10),
    ), row=1, col=1)

    # Weekly frequency
    weekly = df.set_index("date").resample("W").size()
    fig.add_trace(go.Bar(
        x=weekly.index, y=weekly.values, name="Workouts/week",
        marker_color=COLORS[0], opacity=0.6,
    ), row=1, col=2)

    # Duration vs Calories scatter
    fig.add_trace(go.Scatter(
        x=df["duration_min"], y=df["calories"], mode="markers",
        marker=dict(size=5, color=df["avg_hr"], colorscale="Turbo",
                    showscale=True, colorbar=dict(title="Avg HR", len=0.3, y=0.2)),
        text=df["type"], name="Workouts", showlegend=False,
    ), row=2, col=1)

    # VO2max trend
    vo2 = df.dropna(subset=["vo2max"])
    if len(vo2) > 0:
        fig.add_trace(go.Scatter(
            x=vo2["date"], y=vo2["vo2max"], mode="markers",
            marker=dict(size=5, color=COLORS[1], opacity=0.4),
            name="VO2max", showlegend=False,
        ), row=2, col=2)
        roll = vo2.set_index("date")["vo2max"].rolling("30D").mean()
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll.values, name="30d avg",
            line=dict(color=COLORS[1], width=3),
        ), row=2, col=2)

    fig.update_layout(
        height=800, width=1200,
        title=dict(text="<b>Workout Analysis</b>", font=dict(size=20, color=COLORS[0])),
        **THEME, hovermode="closest",
    )
    fig.update_xaxes(gridcolor="#30363d")
    fig.update_yaxes(gridcolor="#30363d")

    fig.write_html(OUT / "workouts.html", include_plotlyjs="cdn")
    print("  workouts.html")


def build_index_page():
    """Build the main index.html for GitHub Pages."""
    garmin = pd.read_csv(PROCESSED / "garmin_daily.csv", parse_dates=["date"])
    dexa = pd.read_csv(PROCESSED / "dexa.csv", parse_dates=["date"])
    activities = pd.read_csv(PROCESSED / "activities.csv", parse_dates=["date"])

    last_30 = garmin[garmin["date"] >= garmin["date"].max() - timedelta(days=30)]

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Dashboard — KSK</title>
    <style>
        :root {{
            --bg: #0d1117; --card: #161b22; --border: #30363d;
            --text: #c9d1d9; --dim: #8b949e;
            --accent: #58a6ff; --green: #3fb950; --red: #f85149;
            --orange: #d29922; --purple: #bc8cff; --pink: #f778ba; --cyan: #39d2c0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: var(--bg); color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6; padding: 2rem; max-width: 1400px; margin: 0 auto;
        }}
        h1 {{ color: var(--accent); font-size: 2rem; margin-bottom: 0.5rem; }}
        h2 {{ color: var(--text); font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
        .subtitle {{ color: var(--dim); margin-bottom: 2rem; }}
        .kpi-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem; margin-bottom: 2rem;
        }}
        .kpi {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 12px; padding: 1.2rem; text-align: center;
            transition: border-color 0.2s;
        }}
        .kpi:hover {{ border-color: var(--accent); }}
        .kpi-label {{ font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; }}
        .kpi-value {{ font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; }}
        .kpi-delta {{ font-size: 0.8rem; }}
        .kpi-delta.good {{ color: var(--green); }}
        .kpi-delta.bad {{ color: var(--red); }}
        .kpi-delta.neutral {{ color: var(--dim); }}
        .nav {{
            display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 1.5rem 0;
        }}
        .nav a {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; padding: 0.6rem 1.2rem;
            color: var(--accent); text-decoration: none; font-size: 0.9rem;
            transition: all 0.2s;
        }}
        .nav a:hover {{ background: var(--accent); color: var(--bg); }}
        .chart-frame {{
            width: 100%; border: 1px solid var(--border);
            border-radius: 12px; margin: 1rem 0;
            background: var(--card);
        }}
        .updated {{ color: var(--dim); font-size: 0.75rem; margin-top: 3rem; text-align: center; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
        @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <h1>🏥 Health Dashboard</h1>
    <p class="subtitle">Automated visualization of personal health metrics from Garmin, DEXA, blood tests, and cognitive assessments.</p>

    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-label">Weight</div>
            <div class="kpi-value" style="color: var(--accent)">{dexa['weight_kg'].iloc[-1]:.1f} kg</div>
            <div class="kpi-delta {'good' if dexa['weight_kg'].iloc[-1] < dexa['weight_kg'].iloc[0] else 'bad'}">
                {"↓" if dexa['weight_kg'].iloc[-1] < dexa['weight_kg'].iloc[0] else "↑"}{abs(dexa['weight_kg'].iloc[-1] - dexa['weight_kg'].iloc[0]):.1f} from {dexa['weight_kg'].iloc[0]:.1f}
            </div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Body Fat</div>
            <div class="kpi-value" style="color: var(--orange)">{dexa['total_pct_fat'].iloc[-1]:.1f}%</div>
            <div class="kpi-delta {'good' if dexa['total_pct_fat'].iloc[-1] < dexa['total_pct_fat'].iloc[0] else 'bad'}">
                {"↓" if dexa['total_pct_fat'].iloc[-1] < dexa['total_pct_fat'].iloc[0] else "↑"}{abs(dexa['total_pct_fat'].iloc[-1] - dexa['total_pct_fat'].iloc[0]):.1f}% from {dexa['total_pct_fat'].iloc[0]:.1f}%
            </div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Avg Steps (30d)</div>
            <div class="kpi-value" style="color: var(--green)">{int(last_30['activity_steps'].mean()):,}</div>
            <div class="kpi-delta neutral">last 30 days</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Avg Sleep</div>
            <div class="kpi-value" style="color: var(--purple)">{last_30['sleep_total_hours'].mean():.1f}h</div>
            <div class="kpi-delta {'good' if last_30['sleep_total_hours'].mean() >= 7 else 'bad'}">
                {"✓ above 7h target" if last_30['sleep_total_hours'].mean() >= 7 else "below 7h target"}
            </div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Resting HR</div>
            <div class="kpi-value" style="color: var(--red)">{last_30['activity_resting_hr'].mean():.0f} bpm</div>
            <div class="kpi-delta neutral">30-day average</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Total Workouts</div>
            <div class="kpi-value" style="color: var(--cyan)">{len(activities)}</div>
            <div class="kpi-delta neutral">{(activities['date'].max() - activities['date'].min()).days} days tracked</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Visceral Fat</div>
            <div class="kpi-value" style="color: var(--pink)">{dexa['vat_area_cm2'].iloc[-1]:.0f} cm²</div>
            <div class="kpi-delta {'good' if dexa['vat_area_cm2'].iloc[-1] < dexa['vat_area_cm2'].iloc[0] else 'bad'}">
                {"↓" if dexa['vat_area_cm2'].iloc[-1] < dexa['vat_area_cm2'].iloc[0] else "↑"}{abs(dexa['vat_area_cm2'].iloc[-1] - dexa['vat_area_cm2'].iloc[0]):.0f} from {dexa['vat_area_cm2'].iloc[0]:.0f}
            </div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Lean Mass Index</div>
            <div class="kpi-value" style="color: var(--green)">{dexa['lmi'].iloc[-1]:.1f}</div>
            <div class="kpi-delta neutral">kg/m²</div>
        </div>
    </div>

    <h2>📈 Interactive Charts</h2>
    <div class="nav">
        <a href="garmin_daily.html">🏃 Garmin Daily</a>
        <a href="dexa.html">🦴 DEXA Scans</a>
        <a href="blood_tests.html">🩸 Blood Tests</a>
        <a href="workouts.html">💪 Workouts</a>
    </div>

    <h2>🏃 Daily Activity & Sleep</h2>
    <iframe class="chart-frame" src="garmin_daily.html" height="1650" frameborder="0"></iframe>

    <h2>🦴 DEXA Body Composition</h2>
    <iframe class="chart-frame" src="dexa.html" height="750" frameborder="0"></iframe>

    <div class="grid-2">
        <div>
            <h2>🩸 Blood Biomarkers</h2>
            <iframe class="chart-frame" src="blood_tests.html" height="650" frameborder="0"></iframe>
        </div>
        <div>
            <h2>💪 Workouts</h2>
            <iframe class="chart-frame" src="workouts.html" height="650" frameborder="0"></iframe>
        </div>
    </div>

    <p class="updated">Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Data: {garmin['date'].min().strftime('%b %Y')} – {garmin['date'].max().strftime('%b %Y')}</p>
</body>
</html>"""

    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("  index.html")


def main():
    print("=== Generating Interactive Dashboard ===")
    build_garmin_dashboard()
    build_dexa_dashboard()
    build_blood_dashboard()
    build_workout_dashboard()
    build_index_page()
    print(f"=== Done. Files in dashboard/interactive/ ===")


if __name__ == "__main__":
    main()
