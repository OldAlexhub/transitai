from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import RandomForestRegressor

RF_MODEL_FILE = Path(__file__).resolve().parent / "rf_model.pkl"
OPS_MODEL_FILE = Path(__file__).resolve().parent / "best_model.npz"
LEGACY_OPS_MODEL_FILE = Path(__file__).resolve().parent / "best_model.h5"
TRIPS_ARTIFACT_FILE = Path(__file__).resolve().parent / "trips.pkl"
NOTEBOOK_FILE = Path(__file__).resolve().parent / "code.ipynb"
MC_DROPOUT_SAMPLES = 30

TRIP_FEATURE_COLUMNS = [
    "Month",
    "Day",
    "Year",
    "SchHour",
    "season",
    "is_weekend",
    "is_holiday",
]
OPS_FEATURE_COLUMNS = ["Month", "Day", "Year", "SchHour", "trips", "otp_pass"]
COUNT_TARGETS = ("drivers", "vehicles", "routes")


@dataclass(frozen=True)
class DenseLayer:
    name: str
    kernel: np.ndarray
    bias: np.ndarray
    activation: str


@dataclass(frozen=True)
class DropoutLayer:
    name: str
    rate: float


class NumpyOpsModel:
    def __init__(self, layers: list[DenseLayer | DropoutLayer], seed: int | None = None):
        self.layers = layers
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_npz(cls, path: Path) -> "NumpyOpsModel":
        with np.load(path, allow_pickle=False) as artifact:
            model_config = artifact["model_config"].item()
            weights = {name: artifact[name] for name in artifact.files if name != "model_config"}
        return cls._from_serialized_artifact(model_config, weights)

    @classmethod
    def from_legacy_h5(cls, path: Path) -> "NumpyOpsModel":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional legacy conversion path
            raise ImportError(
                "Legacy HDF5 ops-model loading requires h5py. Prefer the bundled "
                "best_model.npz artifact for deployment."
            ) from exc

        with h5py.File(path, "r") as artifact:
            model_config = artifact.attrs["model_config"]
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")

            config = json.loads(model_config)
            weights: dict[str, np.ndarray] = {}
            for layer in config["config"]["layers"]:
                if layer["class_name"] != "Dense":
                    continue
                name = layer["config"]["name"]
                layer_group = artifact["model_weights"][name]["sequential_1"][name]
                weights[f"{name}__kernel"] = layer_group["kernel"][()]
                weights[f"{name}__bias"] = layer_group["bias"][()]

        return cls._from_serialized_artifact(model_config, weights)

    @classmethod
    def _from_serialized_artifact(
        cls, model_config: str, weights: dict[str, np.ndarray]
    ) -> "NumpyOpsModel":
        config = json.loads(model_config)
        layers: list[DenseLayer | DropoutLayer] = []

        for layer in config["config"]["layers"]:
            class_name = layer["class_name"]
            layer_config = layer["config"]
            name = layer_config["name"]

            if class_name == "Dense":
                layers.append(
                    DenseLayer(
                        name=name,
                        kernel=np.asarray(weights[f"{name}__kernel"], dtype=np.float32),
                        bias=np.asarray(weights[f"{name}__bias"], dtype=np.float32),
                        activation=layer_config.get("activation", "linear"),
                    )
                )
            elif class_name == "Dropout":
                layers.append(
                    DropoutLayer(name=name, rate=float(layer_config.get("rate", 0.0)))
                )

        if not layers:
            raise ValueError("The operations model artifact did not contain any supported layers.")

        return cls(layers)

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        return self._forward(inputs, training=False)

    def sample_predict(self, inputs: np.ndarray, sample_count: int) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float32)
        return np.stack(
            [self._forward(inputs, training=True) for _ in range(sample_count)],
            axis=0,
        )

    def _forward(self, inputs: np.ndarray, training: bool) -> np.ndarray:
        activations = np.asarray(inputs, dtype=np.float32)
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                activations = activations @ layer.kernel + layer.bias
                activations = _apply_activation(activations, layer.activation)
                continue

            if training:
                activations = self._apply_dropout(activations, layer.rate)

        return activations

    def _apply_dropout(self, values: np.ndarray, rate: float) -> np.ndarray:
        if rate <= 0.0:
            return values

        keep_prob = 1.0 - rate
        if keep_prob <= 0.0:
            return np.zeros_like(values)

        mask = self._rng.binomial(1, keep_prob, size=values.shape).astype(values.dtype)
        return (values * mask) / keep_prob


@dataclass
class ForecastBundle:
    trip_model: RandomForestRegressor
    ops_model: Any
    metrics: dict[str, dict[str, Any]]
    feature_importance: pd.DataFrame
    trip_feature_columns: list[str]
    ops_feature_columns: list[str]
    history: pd.DataFrame | None = None
    history_start: pd.Timestamp | None = None
    history_end: pd.Timestamp | None = None
    history_loaded: bool = False
    otp_available: bool = False
    notes: list[str] = field(default_factory=list)


def load_saved_forecasting_suite(
    rf_model_path: str | Path = RF_MODEL_FILE,
    ops_model_path: str | Path = OPS_MODEL_FILE,
    trips_artifact_path: str | Path = TRIPS_ARTIFACT_FILE,
    notebook_path: str | Path = NOTEBOOK_FILE,
) -> ForecastBundle:
    rf_path = Path(rf_model_path)
    ops_path = _resolve_ops_model_path(Path(ops_model_path))

    if not rf_path.exists():
        raise FileNotFoundError(f"Saved demand model not found: {rf_path}")

    trip_model = joblib.load(rf_path)

    notes: list[str] = []
    ops_model = _load_ops_model(ops_path, notes)
    history = _load_trip_history_artifact(Path(trips_artifact_path), notes)
    metrics = _load_notebook_metrics(Path(notebook_path), notes)

    trip_feature_columns = list(
        getattr(trip_model, "feature_names_in_", TRIP_FEATURE_COLUMNS)
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": trip_feature_columns,
                "importance": trip_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    history_start = None
    history_end = None
    history_loaded = history is not None and not history.empty
    if history_loaded:
        history_start = pd.to_datetime(history["timestamp"]).min()
        history_end = pd.to_datetime(history["timestamp"]).max()

    if not history_loaded:
        notes.append(
            "Recent demand history could not be restored from trips.pkl, so the overview "
            "uses model-generated baseline demand instead of observed history."
        )

    return ForecastBundle(
        trip_model=trip_model,
        ops_model=ops_model,
        metrics=metrics,
        feature_importance=feature_importance,
        trip_feature_columns=trip_feature_columns,
        ops_feature_columns=OPS_FEATURE_COLUMNS,
        history=history,
        history_start=history_start,
        history_end=history_end,
        history_loaded=history_loaded,
        otp_available=False,
        notes=notes,
    )


def parse_future_index(
    minimum_timestamp: pd.Timestamp,
    range_start: pd.Timestamp | None = None,
    range_end: pd.Timestamp | None = None,
    custom_points: list[pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    if custom_points:
        future_index = pd.DatetimeIndex(pd.to_datetime(custom_points)).floor("h")
        future_index = future_index.sort_values().unique()
    else:
        if range_start is None or range_end is None:
            raise ValueError("A start and end timestamp are required for range forecasts.")
        future_index = pd.date_range(
            pd.Timestamp(range_start).floor("h"),
            pd.Timestamp(range_end).floor("h"),
            freq="h",
        )

    future_index = future_index[future_index >= minimum_timestamp]
    if future_index.empty:
        raise ValueError(
            f"Forecast timestamps must be at or after {minimum_timestamp:%Y-%m-%d %H:%M}."
        )
    return future_index


def forecast_hours(
    bundle: ForecastBundle,
    requested_index: pd.DatetimeIndex,
    available_drivers: float,
    available_vehicles: float,
    staffing_buffer_pct: float,
    status_target: float,
    otp_target: float | None = None,
) -> pd.DataFrame:
    requested_index = pd.DatetimeIndex(requested_index).sort_values().unique()
    feature_frame = build_feature_frame(requested_index)

    trip_inputs = feature_frame[bundle.trip_feature_columns].to_numpy(dtype=float)
    trip_tree_predictions = np.vstack(
        [tree.predict(trip_inputs) for tree in bundle.trip_model.estimators_]
    )
    trip_mean = trip_tree_predictions.mean(axis=0)
    trip_lower = np.quantile(trip_tree_predictions, 0.1, axis=0)
    trip_upper = np.quantile(trip_tree_predictions, 0.9, axis=0)

    ops_inputs = np.column_stack(
        [
            feature_frame["Month"].to_numpy(dtype=float),
            feature_frame["Day"].to_numpy(dtype=float),
            feature_frame["Year"].to_numpy(dtype=float),
            feature_frame["SchHour"].to_numpy(dtype=float),
            trip_mean.astype(float),
            np.ones(len(feature_frame), dtype=float),
        ]
    ).astype(np.float32)

    ops_samples = bundle.ops_model.sample_predict(ops_inputs, MC_DROPOUT_SAMPLES)
    ops_mean = ops_samples.mean(axis=0)
    ops_lower = np.quantile(ops_samples, 0.1, axis=0)
    ops_upper = np.quantile(ops_samples, 0.9, axis=0)

    drivers = _clip_count(ops_mean[:, 0])
    vehicles = _clip_count(ops_mean[:, 1])
    routes = _clip_count(ops_mean[:, 2])
    status = np.clip(ops_mean[:, 3], 0.0, 1.0)

    drivers_lower = _clip_count(ops_lower[:, 0])
    drivers_upper = np.maximum(drivers, _clip_count(ops_upper[:, 0]))
    vehicles_lower = _clip_count(ops_lower[:, 1])
    vehicles_upper = np.maximum(vehicles, _clip_count(ops_upper[:, 1]))
    routes_lower = _clip_count(ops_lower[:, 2])
    routes_upper = np.maximum(routes, _clip_count(ops_upper[:, 2]))
    status_lower = np.clip(ops_lower[:, 3], 0.0, 1.0)
    status_upper = np.clip(np.maximum(status, ops_upper[:, 3]), 0.0, 1.0)

    forecast = pd.DataFrame(
        {
            "timestamp": requested_index,
            "trips": np.clip(trip_mean, 0.0, None),
            "trips_lower": np.clip(trip_lower, 0.0, None),
            "trips_upper": np.maximum(np.clip(trip_mean, 0.0, None), trip_upper),
            "drivers": drivers,
            "drivers_lower": drivers_lower,
            "drivers_upper": drivers_upper,
            "vehicles": vehicles,
            "vehicles_lower": vehicles_lower,
            "vehicles_upper": vehicles_upper,
            "routes": routes,
            "routes_lower": routes_lower,
            "routes_upper": routes_upper,
            "status": status,
            "status_lower": status_lower,
            "status_upper": status_upper,
            "otp": np.nan,
            "otp_lower": np.nan,
            "otp_upper": np.nan,
        }
    )

    forecast["drivers_plan"] = np.ceil(
        forecast["drivers"] * (1 + staffing_buffer_pct / 100)
    )
    forecast["vehicles_plan"] = np.ceil(
        forecast["vehicles"] * (1 + staffing_buffer_pct / 100)
    )
    forecast["driver_shortfall"] = np.maximum(
        0.0, forecast["drivers_plan"] - float(available_drivers)
    )
    forecast["vehicle_shortfall"] = np.maximum(
        0.0, forecast["vehicles_plan"] - float(available_vehicles)
    )
    forecast["confidence_pct"] = _confidence_score(forecast, bundle)

    risk_score = (
        forecast["driver_shortfall"] * 4
        + forecast["vehicle_shortfall"] * 3
        + np.maximum(0.0, status_target - forecast["status"]) * 100
    )
    if otp_target is not None and bundle.otp_available:
        risk_score = risk_score + np.maximum(0.0, otp_target - forecast["otp"]) * 100

    forecast["risk_level"] = np.select(
        [risk_score == 0, risk_score < 10],
        ["Stable", "Watch"],
        default="Intervene",
    )
    forecast["at_risk"] = (forecast["risk_level"] != "Stable").astype(int)
    return forecast


def summarize_forecast(forecast: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "Hourly":
        return forecast.copy()

    summary = (
        forecast.assign(service_date=forecast["timestamp"].dt.floor("D"))
        .groupby("service_date", as_index=False)
        .agg(
            timestamp=("service_date", "first"),
            trips=("trips", "sum"),
            trips_lower=("trips_lower", "sum"),
            trips_upper=("trips_upper", "sum"),
            drivers=("drivers", "max"),
            vehicles=("vehicles", "max"),
            routes=("routes", "max"),
            drivers_plan=("drivers_plan", "max"),
            vehicles_plan=("vehicles_plan", "max"),
            driver_shortfall=("driver_shortfall", "max"),
            vehicle_shortfall=("vehicle_shortfall", "max"),
            status=("status", "mean"),
            confidence_pct=("confidence_pct", "mean"),
            at_risk=("at_risk", "sum"),
        )
    )

    summary["otp"] = np.nan
    summary["risk_level"] = np.where(
        summary["at_risk"] == 0,
        "Stable",
        np.where(summary["at_risk"] <= 4, "Watch", "Intervene"),
    )
    return summary


def history_window(
    bundle: ForecastBundle,
    periods: int,
    granularity: str,
    anchor_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    anchor = pd.Timestamp(anchor_timestamp).floor("h")
    if bundle.history_loaded and bundle.history is not None:
        history = bundle.history[bundle.history["timestamp"] <= anchor].copy().tail(periods)
    else:
        history_index = pd.date_range(
            anchor - pd.Timedelta(hours=periods - 1),
            anchor,
            freq="h",
        )
        history = predict_trip_baseline(bundle, history_index)

    history = history[["timestamp", "trips"]].copy()
    if granularity == "Hourly":
        return history

    return (
        history.assign(service_date=history["timestamp"].dt.floor("D"))
        .groupby("service_date", as_index=False)
        .agg(timestamp=("service_date", "first"), trips=("trips", "sum"))
    )


def recommend_capacity_defaults(
    bundle: ForecastBundle,
    requested_index: pd.DatetimeIndex,
    staffing_buffer_pct: float,
) -> tuple[int, int]:
    feature_frame = build_feature_frame(pd.DatetimeIndex(requested_index))
    trip_inputs = feature_frame[bundle.trip_feature_columns]
    trips = np.clip(bundle.trip_model.predict(trip_inputs), 0.0, None)

    ops_inputs = np.column_stack(
        [
            feature_frame["Month"].to_numpy(dtype=float),
            feature_frame["Day"].to_numpy(dtype=float),
            feature_frame["Year"].to_numpy(dtype=float),
            feature_frame["SchHour"].to_numpy(dtype=float),
            trips.astype(float),
            np.ones(len(feature_frame), dtype=float),
        ]
    ).astype(np.float32)
    ops_point = bundle.ops_model.predict(ops_inputs, verbose=0)

    driver_plan = np.ceil(np.clip(ops_point[:, 0], 0.0, None) * (1 + staffing_buffer_pct / 100))
    vehicle_plan = np.ceil(np.clip(ops_point[:, 1], 0.0, None) * (1 + staffing_buffer_pct / 100))

    return (
        int(max(1, np.ceil(np.quantile(driver_plan, 0.95)))),
        int(max(1, np.ceil(np.quantile(vehicle_plan, 0.95)))),
    )


def predict_trip_baseline(
    bundle: ForecastBundle, requested_index: pd.DatetimeIndex
) -> pd.DataFrame:
    feature_frame = build_feature_frame(pd.DatetimeIndex(requested_index))
    trip_inputs = feature_frame[bundle.trip_feature_columns]
    trips = np.clip(bundle.trip_model.predict(trip_inputs), 0.0, None)
    return pd.DataFrame({"timestamp": requested_index, "trips": trips})


def build_feature_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(index).sort_values()
    holiday_lookup = build_holiday_lookup(timestamps.min(), timestamps.max())
    frame = pd.DataFrame({"timestamp": timestamps})
    frame["Month"] = frame["timestamp"].dt.month
    frame["Day"] = frame["timestamp"].dt.day
    frame["Year"] = frame["timestamp"].dt.year
    frame["SchHour"] = frame["timestamp"].dt.hour
    frame["season"] = frame["Month"].map(lambda value: int((value % 12 + 3) // 3))
    frame["is_weekend"] = (frame["timestamp"].dt.dayofweek >= 5).astype(int)
    frame["is_holiday"] = frame["timestamp"].dt.date.map(
        lambda value: int(value in holiday_lookup)
    )
    return frame


def build_holiday_lookup(start: pd.Timestamp, end: pd.Timestamp) -> set:
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=start - pd.Timedelta(days=365), end=end + pd.Timedelta(days=365))
    return set(pd.to_datetime(holidays).date)


def _clip_count(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 0.0, None)


def _apply_activation(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "linear":
        return values
    if activation == "relu":
        return np.maximum(values, 0.0)
    raise ValueError(f"Unsupported activation in operations model artifact: {activation}")


def _confidence_score(forecast: pd.DataFrame, bundle: ForecastBundle) -> np.ndarray:
    trip_width = forecast["trips_upper"] - forecast["trips_lower"]
    driver_width = forecast["drivers_upper"] - forecast["drivers_lower"]
    vehicle_width = forecast["vehicles_upper"] - forecast["vehicles_lower"]
    status_width = forecast["status_upper"] - forecast["status_lower"]

    spread_penalty = (
        0.4 * (trip_width / np.maximum(forecast["trips"], 1.0))
        + 0.25 * (driver_width / np.maximum(forecast["drivers"], 1.0))
        + 0.2 * (vehicle_width / np.maximum(forecast["vehicles"], 1.0))
        + 0.15 * status_width
    )

    trip_r2 = float(bundle.metrics.get("trips", {}).get("r2", 0.85))
    model_strength = np.clip(0.65 + max(trip_r2, 0.0) * 0.25, 0.55, 0.92)
    confidence = model_strength * (1 / (1 + spread_penalty))
    return np.clip(confidence * 100, 35.0, 98.0)


def _load_trip_history_artifact(path: Path, notes: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        notes.append("No saved trip history artifact was found.")
        return None

    try:
        with path.open("rb") as handle:
            trip_frame = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - deployment artifact compatibility
        notes.append(
            "trips.pkl could not be loaded in this environment. Save the trips artifact "
            "as Parquet or CSV if you want observed demand history in the deployed app."
        )
        return None

    if not isinstance(trip_frame, pd.DataFrame):
        notes.append("trips.pkl did not contain a pandas DataFrame.")
        return None

    required = {"Month", "Day", "Year", "SchHour", "trips"}
    if not required.issubset(trip_frame.columns):
        notes.append("trips.pkl is missing required demand-history columns.")
        return None

    history = trip_frame.copy()
    history["timestamp"] = pd.to_datetime(
        history[["Year", "Month", "Day"]]
    ) + pd.to_timedelta(history["SchHour"], unit="h")
    history["trips"] = pd.to_numeric(history["trips"], errors="coerce").fillna(0.0)
    history = history[["timestamp", "trips"]].sort_values("timestamp").reset_index(drop=True)
    return history


def _resolve_ops_model_path(path: Path) -> Path:
    if path.exists():
        return path
    if path == OPS_MODEL_FILE and LEGACY_OPS_MODEL_FILE.exists():
        return LEGACY_OPS_MODEL_FILE
    raise FileNotFoundError(f"Saved operations model not found: {path}")


def _load_ops_model(path: Path, notes: list[str]) -> NumpyOpsModel:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return NumpyOpsModel.from_npz(path)

    if suffix in {".h5", ".keras"}:
        notes.append(
            "Loaded the legacy HDF5 operations artifact. Keeping best_model.npz in the "
            "deployment bundle avoids TensorFlow-era runtime dependencies."
        )
        return NumpyOpsModel.from_legacy_h5(path)

    raise ValueError(f"Unsupported operations model artifact: {path.name}")


def _load_notebook_metrics(path: Path, notes: list[str]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    if not path.exists():
        notes.append("Notebook metrics were not found, so saved-model accuracy is unavailable.")
        return metrics

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        notes.append(f"Notebook metrics could not be read: {exc}")
        return metrics

    extracted: list[dict[str, float]] = []
    pattern = re.compile(
        r"Mean Squared Error:\s*([-\d.]+).*?R\^2 Score:\s*([-\d.]+).*?Mean Absolute Error:\s*([-\d.]+)",
        re.S,
    )
    for cell in notebook.get("cells", []):
        output_text = "".join(
            "".join(output.get("text", [])) for output in cell.get("outputs", [])
        )
        match = pattern.search(output_text)
        if match:
            extracted.append(
                {
                    "mse": float(match.group(1)),
                    "r2": float(match.group(2)),
                    "mae": float(match.group(3)),
                }
            )

    if extracted:
        metrics["trips"] = {
            "label": "Demand / trips",
            **extracted[0],
            "source": "Saved notebook evaluation",
        }
    if len(extracted) > 1:
        metrics["operations"] = {
            "label": "Operations model",
            **extracted[1],
            "source": "Saved notebook evaluation",
        }
        notes.append(
            "Operations accuracy comes from the original notebook's combined multi-output "
            "evaluation. Per-target holdout metrics were not saved with the artifact bundle."
        )
    return metrics
