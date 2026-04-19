from __future__ import annotations

from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from forecasting import (
    ForecastBundle,
    forecast_hours,
    history_window,
    load_saved_forecasting_suite,
    parse_future_index,
    recommend_capacity_defaults,
    summarize_forecast,
)

st.set_page_config(
    page_title="TransitAI Companion",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f4f6f8;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero-band {
            padding: 0.2rem 0 1.2rem 0;
            border-bottom: 1px solid #d8dee6;
            margin-bottom: 1rem;
        }
        .hero-band h1 {
            font-size: 2.3rem;
            line-height: 1.05;
            margin: 0 0 0.4rem 0;
            color: #0f1720;
        }
        .hero-band p {
            margin: 0;
            max-width: 64rem;
            font-size: 1.02rem;
            color: #475467;
        }
        .eyebrow {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #0c7c59;
            margin-bottom: 0.5rem;
        }
        .section-label {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.55rem;
        }
        [data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #d8dee6;
            padding: 0.85rem 1rem;
            border-radius: 8px;
        }
        .minor-note {
            color: #667085;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading saved forecasting artifacts...")
def load_bundle() -> ForecastBundle:
    return load_saved_forecasting_suite()


def parse_custom_points(raw_text: str) -> list[pd.Timestamp]:
    tokens = [token.strip() for token in raw_text.replace(",", "\n").splitlines()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return []

    parsed = pd.to_datetime(tokens, errors="coerce")
    valid = [value.floor("h") for value in parsed if pd.notna(value)]
    if not valid:
        raise ValueError("No valid timestamps were found in the custom points list.")
    return valid


def accuracy_table(bundle: ForecastBundle) -> pd.DataFrame:
    rows = []
    for key, metrics in bundle.metrics.items():
        rows.append(
            {
                "Signal": metrics.get("label", key.title()),
                "MAE": metrics.get("mae"),
                "MSE": metrics.get("mse"),
                "R2": metrics.get("r2"),
                "Source": metrics.get("source", "Saved artifact"),
            }
        )

    return pd.DataFrame(rows)


def format_accuracy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in ["MAE", "MSE", "R2"]:
        formatted[column] = formatted[column].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:,.3f}"
        )
    return formatted


def render_demand_chart(
    history: pd.DataFrame | None, forecast: pd.DataFrame, history_label: str
) -> None:
    forecast_band = (
        alt.Chart(forecast)
        .mark_area(color="#0c7c59", opacity=0.18)
        .encode(
            x="timestamp:T",
            y=alt.Y("trips_lower:Q", title="Trips"),
            y2="trips_upper:Q",
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("trips:Q", title="Forecast trips", format=",.1f"),
                alt.Tooltip("trips_lower:Q", title="Lower band", format=",.1f"),
                alt.Tooltip("trips_upper:Q", title="Upper band", format=",.1f"),
            ],
        )
    )

    forecast_line = (
        alt.Chart(forecast)
        .mark_line(color="#0c7c59", strokeWidth=3)
        .encode(x="timestamp:T", y="trips:Q")
    )

    layers = []
    if history is not None and not history.empty:
        history_base = (
            alt.Chart(history)
            .mark_line(color="#7c8798", strokeWidth=2, strokeDash=[4, 4])
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("trips:Q", title="Trips"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("trips:Q", title=history_label, format=",.1f"),
                ],
            )
        )
        layers.append(history_base)

    layers.extend([forecast_band, forecast_line])
    chart = alt.layer(*layers).properties(height=360)
    st.altair_chart(chart, use_container_width=True)


def render_capacity_chart(forecast: pd.DataFrame, available_capacity: int, metric: str, title: str) -> None:
    series = (
        forecast[["timestamp", metric]]
        .rename(columns={metric: "value"})
        .assign(label=title)
    )
    rule_frame = pd.DataFrame(
        {
            "value": [available_capacity],
            "label": [f"Available {title.lower()}"],
        }
    )

    line = (
        alt.Chart(series)
        .mark_line(color="#111827", strokeWidth=3)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", title=title),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("value:Q", title=title, format=",.1f"),
            ],
        )
    )

    capacity_rule = (
        alt.Chart(rule_frame)
        .mark_rule(color="#c2410c", strokeDash=[6, 5], strokeWidth=2)
        .encode(y="value:Q")
    )

    st.altair_chart(
        alt.layer(line, capacity_rule).properties(height=260, title=title),
        use_container_width=True,
    )


def render_performance_chart(forecast: pd.DataFrame, otp_available: bool) -> None:
    metric_columns = ["timestamp", "status"]
    if otp_available and "otp" in forecast.columns and forecast["otp"].notna().any():
        metric_columns.append("otp")

    performance = forecast[metric_columns].melt(
        "timestamp", var_name="metric", value_name="value"
    )
    performance["value"] = performance["value"] * 100
    labels = {"otp": "On-time performance", "status": "Completion rate"}
    performance["metric"] = performance["metric"].map(labels)
    color_domain = performance["metric"].dropna().unique().tolist()
    color_range = ["#0c7c59", "#b42318"][: len(color_domain)]

    chart = (
        alt.Chart(performance)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", title="Rate (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "metric:N",
                title=None,
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("metric:N", title="Signal"),
                alt.Tooltip("value:Q", title="Rate", format=".1f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_pressure_heatmap(forecast: pd.DataFrame) -> None:
    if len(forecast) > 31 * 24:
        return

    heatmap = forecast.copy()
    heatmap["service_day"] = heatmap["timestamp"].dt.strftime("%b %d")
    heatmap["hour"] = heatmap["timestamp"].dt.hour
    day_order = (
        forecast["timestamp"].dt.strftime("%b %d").drop_duplicates().tolist()
    )

    heat = (
        alt.Chart(heatmap)
        .mark_rect()
        .encode(
            x=alt.X("service_day:N", sort=day_order, title=None),
            y=alt.Y("hour:O", title="Hour"),
            color=alt.Color(
                "driver_shortfall:Q",
                title="Driver shortfall",
                scale=alt.Scale(range=["#eef2f6", "#f59e0b", "#b42318"]),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("driver_shortfall:Q", title="Driver shortfall", format=",.0f"),
                alt.Tooltip("trips:Q", title="Trips", format=",.1f"),
                alt.Tooltip("drivers_plan:Q", title="Planned drivers", format=",.0f"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(heat, use_container_width=True)


def forecast_table(forecast: pd.DataFrame) -> pd.DataFrame:
    table = forecast.copy()
    table["time"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    table["trips"] = table["trips"].round(1)
    for column in ["drivers", "vehicles", "routes", "drivers_plan", "vehicles_plan"]:
        table[column] = table[column].round(0).astype(int)

    for column in ["otp", "status"]:
        if column in table.columns:
            table[column] = (table[column] * 100).round(1)
    table["confidence_pct"] = table["confidence_pct"].round(0)

    columns = [
        "time",
        "trips",
        "drivers_plan",
        "vehicles_plan",
        "routes",
        "status",
        "confidence_pct",
        "driver_shortfall",
        "vehicle_shortfall",
        "risk_level",
    ]
    if "otp" in table.columns and table["otp"].notna().any():
        columns.insert(5, "otp")
    return table[columns].rename(
        columns={
            "time": "Time",
            "trips": "Trips",
            "drivers_plan": "Planned drivers",
            "vehicles_plan": "Planned vehicles",
            "routes": "Routes",
            "otp": "OTP %",
            "status": "Completion %",
            "confidence_pct": "Confidence %",
            "driver_shortfall": "Driver shortfall",
            "vehicle_shortfall": "Vehicle shortfall",
            "risk_level": "Risk",
        }
    )


apply_app_style()

try:
    bundle = load_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    """
    <div class="hero-band">
        <div class="eyebrow">TransitAI Companion</div>
        <h1>Forecast demand, staffing pressure, and service reliability before the work starts.</h1>
        <p>
            Plan hourly or daily operating windows, project the drivers and vehicles you will need,
            and see where confidence narrows or service risk begins to build.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

app_now = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).tz_localize(None).floor("h")
default_start = app_now + pd.Timedelta(hours=1)
default_end = default_start + pd.Timedelta(days=14) - pd.Timedelta(hours=1)
default_index = pd.date_range(default_start, default_end, freq="h")
default_drivers, default_vehicles = recommend_capacity_defaults(
    bundle, default_index, staffing_buffer_pct=10
)

caption_parts = [
    f"Deployment mode uses saved model artifacts. The default forecast starts at {default_start:%B %d, %Y %I:%M %p}."
]
if bundle.history_loaded and bundle.history_start is not None and bundle.history_end is not None:
    caption_parts.append(
        "Saved demand snapshot available from "
        f"{bundle.history_start:%B %d, %Y %I:%M %p} through {bundle.history_end:%B %d, %Y %I:%M %p}."
    )
else:
    caption_parts.append(
        "Observed trip history is not loaded, so recent context charts use model-generated baseline demand."
    )
if bundle.metrics:
    caption_parts.append(
        "Accuracy in the review tab comes from the saved notebook evaluation bundled with the artifacts."
    )
st.caption(" ".join(caption_parts))

for note in bundle.notes:
    st.info(note)

with st.sidebar:
    st.markdown("### Forecast setup")
    with st.form("forecast_form"):
        forecast_mode = st.radio(
            "Forecast input",
            options=["Range", "Custom points"],
            horizontal=True,
        )

        if forecast_mode == "Range":
            start_date = st.date_input(
                "Start date",
                value=default_start.date(),
                min_value=default_start.date(),
            )
            start_hour = st.selectbox(
                "Start hour",
                options=list(range(24)),
                index=default_start.hour,
                format_func=lambda value: f"{value:02d}:00",
            )
            end_date = st.date_input(
                "End date",
                value=default_end.date(),
                min_value=start_date,
            )
            end_hour = st.selectbox(
                "End hour",
                options=list(range(24)),
                index=default_end.hour,
                format_func=lambda value: f"{value:02d}:00",
            )
            custom_points_text = ""
        else:
            st.caption("One timestamp per line. Forecasts are rounded to the nearest hour.")
            custom_points_text = st.text_area(
                "Future timestamps",
                height=170,
                value="\n".join(
                    [
                        default_start.strftime("%Y-%m-%d %H:%M"),
                        (default_start + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
                        (default_start + pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M"),
                    ]
                ),
            )
            start_date = default_start.date()
            start_hour = default_start.hour
            end_date = default_end.date()
            end_hour = default_end.hour

        display_granularity = st.radio(
            "Display",
            options=["Hourly", "Daily"],
            horizontal=True,
        )
        available_drivers = st.number_input(
            "Available drivers",
            min_value=0,
            value=default_drivers,
            step=1,
        )
        available_vehicles = st.number_input(
            "Available vehicles",
            min_value=0,
            value=default_vehicles,
            step=1,
        )
        staffing_buffer_pct = st.slider("Reserve buffer", 0, 30, 10, 1)
        if bundle.otp_available:
            otp_target_pct = st.slider("On-time target", 70, 100, 92, 1)
        else:
            otp_target_pct = None
        completion_target_pct = st.slider("Completion target", 80, 100, 96, 1)

        submitted = st.form_submit_button("Run forecast", use_container_width=True)

if submitted or "forecast_hourly" not in st.session_state:
    try:
        if forecast_mode == "Range":
            range_start = pd.Timestamp(start_date) + pd.Timedelta(hours=int(start_hour))
            range_end = pd.Timestamp(end_date) + pd.Timedelta(hours=int(end_hour))
            if range_end < range_start:
                raise ValueError("The end timestamp must be later than the start timestamp.")
            future_index = parse_future_index(
                default_start, range_start=range_start, range_end=range_end
            )
        else:
            future_index = parse_future_index(
                default_start, custom_points=parse_custom_points(custom_points_text)
            )

        forecast_hourly = forecast_hours(
            bundle=bundle,
            requested_index=future_index,
            available_drivers=int(available_drivers),
            available_vehicles=int(available_vehicles),
            staffing_buffer_pct=float(staffing_buffer_pct),
            status_target=float(completion_target_pct / 100),
            otp_target=None if otp_target_pct is None else float(otp_target_pct / 100),
        )

        st.session_state["forecast_hourly"] = forecast_hourly
        st.session_state["forecast_display"] = summarize_forecast(
            forecast_hourly, display_granularity
        )
        st.session_state["display_granularity"] = display_granularity
        st.session_state["available_drivers"] = int(available_drivers)
        st.session_state["available_vehicles"] = int(available_vehicles)
        st.session_state["staffing_buffer_pct"] = float(staffing_buffer_pct)
        st.session_state["otp_target_pct"] = (
            None if otp_target_pct is None else int(otp_target_pct)
        )
        st.session_state["completion_target_pct"] = int(completion_target_pct)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

forecast_hourly = st.session_state["forecast_hourly"]
display_forecast = st.session_state["forecast_display"]
display_granularity = st.session_state["display_granularity"]

history_periods = 24 * 14 if display_granularity == "Hourly" else 24 * 45
history = history_window(
    bundle,
    history_periods,
    display_granularity,
    anchor_timestamp=default_start - pd.Timedelta(hours=1),
)
history_label = "Observed trips" if bundle.history_loaded else "Baseline trips"

projected_trips = display_forecast["trips"].sum()
peak_row = forecast_hourly.loc[forecast_hourly["trips"].idxmax()]
risk_windows = (
    int(forecast_hourly["at_risk"].sum())
    if display_granularity == "Hourly"
    else int((display_forecast["at_risk"] > 0).sum())
)
average_confidence = float(forecast_hourly["confidence_pct"].mean())
peak_driver_plan = int(np.ceil(forecast_hourly["drivers_plan"].max()))
peak_vehicle_plan = int(np.ceil(forecast_hourly["vehicles_plan"].max()))

metric_columns = st.columns(4)
metric_columns[0].metric("Projected trips", f"{projected_trips:,.0f}")
metric_columns[1].metric(
    "Peak staffing plan", f"{peak_driver_plan:,} drivers / {peak_vehicle_plan:,} vehicles"
)
metric_columns[2].metric(
    f"{'Days' if display_granularity == 'Daily' else 'Hours'} at risk", f"{risk_windows:,}"
)
metric_columns[3].metric("Average confidence", f"{average_confidence:.0f}%")

st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
overview_left, overview_right = st.columns([1.8, 1])

with overview_left:
    render_demand_chart(history, display_forecast, history_label)

with overview_right:
    st.markdown(
        f"""
        <div class="minor-note">
            Peak demand lands on <strong>{peak_row["timestamp"]:%B %d, %Y at %I:%M %p}</strong>
            with <strong>{peak_row["trips"]:.0f} projected trips</strong>.
            The current staffing plan peaks at <strong>{peak_driver_plan} drivers</strong> and
            <strong>{peak_vehicle_plan} vehicles</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    render_pressure_heatmap(forecast_hourly)
    if risk_windows == 0:
        st.success("No capacity or service-risk windows were flagged in this forecast.")
    else:
        first_risk = forecast_hourly.loc[forecast_hourly["at_risk"] == 1, "timestamp"].min()
        st.warning(
            f"Risk starts at {first_risk:%B %d, %Y %I:%M %p}. Review the shortfall table below."
        )

staffing_tab, performance_tab, accuracy_tab = st.tabs(
    ["Staffing plan", "Performance outlook", "Accuracy and confidence"]
)

with staffing_tab:
    st.markdown('<div class="section-label">Capacity plan</div>', unsafe_allow_html=True)
    staffing_left, staffing_right = st.columns(2)
    with staffing_left:
        render_capacity_chart(
            display_forecast,
            st.session_state["available_drivers"],
            "drivers_plan",
            "Planned drivers",
        )
    with staffing_right:
        render_capacity_chart(
            display_forecast,
            st.session_state["available_vehicles"],
            "vehicles_plan",
            "Planned vehicles",
        )

    risk_table = (
        forecast_hourly.sort_values(
            ["driver_shortfall", "vehicle_shortfall", "trips"],
            ascending=[False, False, False],
        )
        .head(12)
        .copy()
    )
    risk_table["timestamp"] = risk_table["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    risk_table = risk_table[
        [
            "timestamp",
            "trips",
            "drivers_plan",
            "vehicles_plan",
            "driver_shortfall",
            "vehicle_shortfall",
            "confidence_pct",
            "risk_level",
        ]
    ].rename(
        columns={
            "timestamp": "Time",
            "trips": "Trips",
            "drivers_plan": "Planned drivers",
            "vehicles_plan": "Planned vehicles",
            "driver_shortfall": "Driver shortfall",
            "vehicle_shortfall": "Vehicle shortfall",
            "confidence_pct": "Confidence %",
            "risk_level": "Risk",
        }
    )

    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with performance_tab:
    st.markdown('<div class="section-label">Service health</div>', unsafe_allow_html=True)
    render_performance_chart(display_forecast, bundle.otp_available)
    perf_metrics = st.columns(3)
    if bundle.otp_available and display_forecast["otp"].notna().any():
        perf_metrics[0].metric("Expected OTP", f"{display_forecast['otp'].mean() * 100:.1f}%")
        perf_metrics[1].metric(
            "Expected completion", f"{display_forecast['status'].mean() * 100:.1f}%"
        )
        perf_metrics[2].metric("Projected routes", f"{display_forecast['routes'].max():,.0f}")
    else:
        perf_metrics[0].metric(
            "Expected completion", f"{display_forecast['status'].mean() * 100:.1f}%"
        )
        perf_metrics[1].metric("Projected routes", f"{display_forecast['routes'].max():,.0f}")
        perf_metrics[2].metric("Average confidence", f"{average_confidence:.0f}%")
        st.caption(
            "OTP is not part of the saved deployment artifacts. This build reports demand, "
            "staffing, routes, completion, and forecast confidence directly from the models."
        )

with accuracy_tab:
    st.markdown(
        '<div class="section-label">Model reliability</div>',
        unsafe_allow_html=True,
    )
    accuracy = accuracy_table(bundle)
    if not accuracy.empty:
        st.dataframe(
            format_accuracy_frame(accuracy),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "Saved evaluation metrics are not available in the current artifact bundle."
        )

    importance = bundle.feature_importance.head(10)
    importance_chart = (
        alt.Chart(importance)
        .mark_bar(color="#111827", cornerRadiusEnd=4)
        .encode(
            x=alt.X("importance:Q", title="Trip model importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", format=".3f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(importance_chart, use_container_width=True)
    with st.expander("How confidence is calculated"):
        st.write(
            "Confidence reflects model agreement on the selected forecast hour. Demand "
            "confidence comes from how tightly the random-forest trees agree, while staffing "
            "and completion confidence come from repeated dropout-enabled passes through the "
            "saved Keras model. Narrower intervals push the score higher."
        )

st.markdown('<div class="section-label">Forecast table</div>', unsafe_allow_html=True)
display_table = forecast_table(forecast_hourly if display_granularity == "Hourly" else display_forecast)
st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Confidence %": st.column_config.ProgressColumn(
            "Confidence %",
            min_value=0,
            max_value=100,
            format="%d%%",
        ),
    },
)

csv_data = forecast_hourly.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download hourly forecast",
    data=csv_data,
    file_name="transit_forecast.csv",
    mime="text/csv",
)
