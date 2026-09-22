#!/usr/bin/env python3
"""Fit and diagnose bivariate exponential Hawkes information-diffusion models."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kstest

STREAMS = ("news", "price")
TIME_SCALE = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}


@dataclass
class FitSpec:
    model: str
    free_pairs: tuple[tuple[int, int], ...]

    @property
    def n_params(self) -> int:
        return 2 + 2 * len(self.free_pairs)


SPECS = {
    "market": FitSpec("market", ((1, 0), (1, 1))),
    "full": FitSpec("full", ((0, 0), (0, 1), (1, 0), (1, 1))),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def read_timestamp_column(path: Path, timestamp_col: str, time_unit: str) -> tuple[np.ndarray, str]:
    frame = pd.read_csv(path)
    if timestamp_col not in frame.columns:
        raise ValueError(f"{path}: missing timestamp column {timestamp_col!r}")
    raw = frame[timestamp_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float) * TIME_SCALE[time_unit], f"numeric_{time_unit}"
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        samples = raw[bad].astype(str).head(3).tolist()
        raise ValueError(f"{path}: unparseable timestamps, examples: {samples}")
    return parsed.astype("int64").to_numpy(dtype=float) / 1e9, "datetime_utc"


def parse_bound(value: str, timestamp_kind: str, time_unit: str) -> float:
    if timestamp_kind.startswith("numeric_"):
        return float(value) * TIME_SCALE[time_unit]
    stamp = pd.to_datetime(value, utc=True, errors="raise")
    return float(stamp.value / 1e9)


def load_events(args: argparse.Namespace) -> tuple[list[np.ndarray], dict[str, Any]]:
    audit: dict[str, Any] = {}
    if args.events_csv:
        path = Path(args.events_csv)
        frame = pd.read_csv(path)
        for col in (args.timestamp_col, args.event_type_col):
            if col not in frame.columns:
                raise ValueError(f"{path}: missing column {col!r}")
        numeric = pd.to_numeric(frame[args.timestamp_col], errors="coerce")
        if numeric.notna().all():
            seconds = numeric.to_numpy(dtype=float) * TIME_SCALE[args.time_unit]
            source_kind = f"numeric_{args.time_unit}"
        else:
            parsed = pd.to_datetime(frame[args.timestamp_col], utc=True, errors="coerce")
            if parsed.isna().any():
                raise ValueError(f"{path}: one or more timestamps are unparseable")
            seconds = parsed.astype("int64").to_numpy(dtype=float) / 1e9
            source_kind = "datetime_utc"
        labels = frame[args.event_type_col].astype(str)
        news = seconds[labels.eq(str(args.news_label)).to_numpy()]
        price = seconds[labels.eq(str(args.price_label)).to_numpy()]
        known = labels.isin([str(args.news_label), str(args.price_label)])
        audit["ignored_unknown_event_types"] = int((~known).sum())
        audit["source_files"] = [str(path)]
    else:
        if not args.news_csv or not args.price_csv:
            raise ValueError("provide --events-csv or both --news-csv and --price-csv")
        news, news_kind = read_timestamp_column(Path(args.news_csv), args.timestamp_col, args.time_unit)
        price, price_kind = read_timestamp_column(Path(args.price_csv), args.timestamp_col, args.time_unit)
        if news_kind != price_kind:
            raise ValueError("news and price timestamps must both be datetimes or both be numeric")
        source_kind = news_kind
        audit["source_files"] = [str(Path(args.news_csv)), str(Path(args.price_csv))]

    if len(news) == 0 or len(price) == 0:
        raise ValueError("both news and price streams must contain at least one event")
    if not np.isfinite(news).all() or not np.isfinite(price).all():
        raise ValueError("timestamps must be finite")

    raw = [np.sort(news.astype(float)), np.sort(price.astype(float))]
    start_abs = min(x[0] for x in raw) if args.window_start is None else parse_bound(args.window_start, source_kind, args.time_unit)
    end_abs = max(x[-1] for x in raw) if args.window_end is None else parse_bound(args.window_end, source_kind, args.time_unit)
    if end_abs <= start_abs:
        raise ValueError("observation window end must be after start")
    clipped = [x[(x >= start_abs) & (x <= end_abs)] - start_abs for x in raw]
    if any(len(x) == 0 for x in clipped):
        raise ValueError("the selected observation window leaves an empty stream")

    duration = end_abs - start_abs
    if args.window_end is None:
        all_gaps = [np.diff(x) for x in clipped if len(x) > 1]
        positive_gaps = np.concatenate(all_gaps) if all_gaps else np.asarray([])
        positive_gaps = positive_gaps[positive_gaps > 0]
        tail = float(np.median(positive_gaps)) if len(positive_gaps) else 1.0
        duration += max(tail, 1e-6)

    audit.update(
        {
            "timestamp_kind": source_kind,
            "internal_time_unit": "seconds",
            "observation_duration_seconds": float(duration),
            "event_counts": {"news": int(len(clipped[0])), "price": int(len(clipped[1]))},
            "duplicate_timestamps_within_stream": {
                "news": int(len(clipped[0]) - len(np.unique(clipped[0]))),
                "price": int(len(clipped[1]) - len(np.unique(clipped[1]))),
            },
            "window_start_input_scale": float(start_abs),
            "window_end_input_scale": float(start_abs + duration),
        }
    )
    return clipped, audit


def unpack(theta: np.ndarray, spec: FitSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.exp(np.clip(theta, -30.0, 30.0))
    mu = values[:2]
    alpha = np.zeros((2, 2), dtype=float)
    beta = np.zeros((2, 2), dtype=float)
    cursor = 2
    for i, j in spec.free_pairs:
        alpha[i, j] = values[cursor]
        beta[i, j] = values[cursor + 1]
        cursor += 2
    return mu, alpha, beta


def excitation_sum(target_times: np.ndarray, source_times: np.ndarray, beta: float) -> np.ndarray:
    out = np.zeros(len(target_times), dtype=float)
    state = 0.0
    cursor = 0
    last_t = 0.0
    for n, t in enumerate(target_times):
        state *= math.exp(-beta * max(t - last_t, 0.0))
        while cursor < len(source_times) and source_times[cursor] < t:
            state += math.exp(-beta * (t - source_times[cursor]))
            cursor += 1
        out[n] = state
        last_t = t
    return out


def branching_matrix(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    out = np.zeros_like(alpha)
    mask = beta > 0
    out[mask] = alpha[mask] / beta[mask]
    return out


def log_likelihood(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, events: list[np.ndarray], horizon: float, spec: FitSpec) -> float:
    total = 0.0
    for i in range(2):
        intensity = np.full(len(events[i]), mu[i], dtype=float)
        integral = mu[i] * horizon
        for ii, j in spec.free_pairs:
            if ii != i:
                continue
            intensity += alpha[i, j] * excitation_sum(events[i], events[j], beta[i, j])
            integral += (alpha[i, j] / beta[i, j]) * np.sum(1.0 - np.exp(-beta[i, j] * (horizon - events[j])))
        if np.any(intensity <= 0) or not np.isfinite(intensity).all():
            return -np.inf
        total += float(np.log(intensity).sum() - integral)
    return total


def objective(theta: np.ndarray, spec: FitSpec, events: list[np.ndarray], horizon: float) -> float:
    mu, alpha, beta = unpack(theta, spec)
    rho = float(max(abs(np.linalg.eigvals(branching_matrix(alpha, beta)))))
    if rho >= 0.999:
        return 1e9 + 1e8 * (rho - 0.999) ** 2
    ll = log_likelihood(mu, alpha, beta, events, horizon, spec)
    return -ll if np.isfinite(ll) else 1e12


def initial_theta(spec: FitSpec, events: list[np.ndarray], horizon: float, scale: float) -> np.ndarray:
    rates = np.array([len(x) / horizon for x in events], dtype=float)
    values: list[float] = [max(rates[0] * 0.8, 1e-9), max(rates[1] * 0.5, 1e-9)]
    default_beta = 1.0 / max(scale, 1e-6)
    for i, _j in spec.free_pairs:
        ratio = 0.12 if i == 0 else 0.2
        values.extend([ratio * default_beta, default_beta])
    return np.log(np.asarray(values))


def fit_model(events: list[np.ndarray], horizon: float, spec: FitSpec, restarts: int, seed: int) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    gaps = np.concatenate([np.diff(x) for x in events if len(x) > 1])
    positive = gaps[gaps > 0]
    scale = float(np.median(positive)) if len(positive) else max(horizon / sum(map(len, events)), 1.0)
    base = initial_theta(spec, events, horizon, scale)
    rng = np.random.default_rng(seed)
    results = []
    for attempt in range(max(1, restarts)):
        start = base if attempt == 0 else base + rng.normal(0, 0.8, size=len(base))
        results.append(minimize(objective, start, args=(spec, events, horizon), method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-11}))
    valid = [result for result in results if np.isfinite(result.fun)]
    if not valid:
        raise RuntimeError("all optimizer restarts failed")
    result = min(valid, key=lambda item: item.fun)
    mu, alpha, beta = unpack(result.x, spec)
    return result, mu, alpha, beta


def cumulative_component(t: np.ndarray | float, source: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    target = np.atleast_1d(t).astype(float)
    out = np.zeros_like(target)
    if alpha <= 0 or beta <= 0 or len(source) == 0:
        return out
    for idx, value in enumerate(target):
        prior = source[source < value]
        out[idx] = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (value - prior)))
    return out


def compensator_at(t: np.ndarray, i: int, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, events: list[np.ndarray], spec: FitSpec) -> np.ndarray:
    out = mu[i] * np.asarray(t, dtype=float)
    for ii, j in spec.free_pairs:
        if ii == i:
            out = out + cumulative_component(t, events[j], alpha[i, j], beta[i, j])
    return out


def residual_diagnostics(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, events: list[np.ndarray], spec: FitSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for i, name in enumerate(STREAMS):
        cumulative = compensator_at(events[i], i, mu, alpha, beta, events, spec)
        gaps = np.diff(np.concatenate([[0.0], cumulative]))
        first_event_index = 0
        if len(gaps) and events[i][0] == 0.0:
            # When an inferred window begins exactly on an event, the first
            # compensator gap is structurally zero rather than an Exp(1) draw.
            gaps = gaps[1:]
            first_event_index = 1
        for idx, gap in enumerate(gaps):
            rows.append({"stream": name, "event_index": idx + first_event_index, "rescaled_gap": float(gap)})
        if len(gaps) >= 2:
            stat, pvalue = kstest(gaps, "expon")
            summary[name] = {
                "n": int(len(gaps)),
                "mean": float(np.mean(gaps)),
                "variance": float(np.var(gaps, ddof=1)),
                "ks_statistic": float(stat),
                "ks_pvalue": float(pvalue),
            }
        else:
            summary[name] = {"n": int(len(gaps)), "warning": "too few gaps for a KS test"}
    return pd.DataFrame(rows), summary


def price_contributions(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, events: list[np.ndarray], spec: FitSpec) -> pd.DataFrame:
    times = events[1]
    frame = pd.DataFrame({"time_seconds": times, "baseline": np.full(len(times), mu[1])})
    total = frame["baseline"].to_numpy(copy=True)
    for label, j in (("news_excitation", 0), ("price_echo", 1)):
        values = alpha[1, j] * excitation_sum(times, events[j], beta[1, j]) if (1, j) in spec.free_pairs else np.zeros(len(times))
        frame[label] = values
        total += values
    frame["total_intensity"] = total
    return frame


def make_summary(args: argparse.Namespace, audit: dict[str, Any], result: Any, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, events: list[np.ndarray], horizon: float, spec: FitSpec, residuals: dict[str, Any]) -> dict[str, Any]:
    kernel = branching_matrix(alpha, beta)
    rho = float(max(abs(np.linalg.eigvals(kernel))))
    ll = log_likelihood(mu, alpha, beta, events, horizon, spec)
    n = sum(map(len, events))
    integrals: dict[str, float] = {"baseline": float(mu[1] * horizon)}
    for label, j in (("news_excitation", 0), ("price_echo", 1)):
        if (1, j) in spec.free_pairs:
            integrals[label] = float((alpha[1, j] / beta[1, j]) * np.sum(1.0 - np.exp(-beta[1, j] * (horizon - events[j]))))
        else:
            integrals[label] = 0.0
    denominator = sum(integrals.values())
    shares = {key: (value / denominator if denominator > 0 else None) for key, value in integrals.items()}

    pairs: dict[str, Any] = {}
    for i, j in spec.free_pairs:
        key = f"{STREAMS[j]}_to_{STREAMS[i]}"
        pairs[key] = {
            "jump_alpha_per_second": float(alpha[i, j]),
            "decay_beta_per_second": float(beta[i, j]),
            "half_life_seconds": float(math.log(2) / beta[i, j]),
            "expected_direct_events": float(kernel[i, j]),
        }
        if i == 1 and j == 1:
            pairs[key]["branching_ratio"] = float(kernel[i, j])

    warnings: list[str] = []
    if not result.success:
        warnings.append(f"optimizer did not declare convergence: {result.message}")
    if rho > 0.95:
        warnings.append("spectral radius is near the stationarity boundary")
    if min(map(len, events)) < 30:
        warnings.append("at least one stream has fewer than 30 events; estimates may be unstable")
    if any(value > 0 for value in audit["duplicate_timestamps_within_stream"].values()):
        warnings.append("duplicate timestamps exist within a stream; timestamp resolution may be too coarse")

    return {
        "model": spec.model,
        "audit": audit,
        "optimizer": {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit), "restarts": int(args.restarts)},
        "fit": {
            "log_likelihood": float(ll),
            "free_parameters": spec.n_params,
            "aic": float(2 * spec.n_params - 2 * ll),
            "bic": float(spec.n_params * math.log(max(n, 1)) - 2 * ll),
        },
        "baseline_rates_per_second": {"news": float(mu[0]), "price": float(mu[1])},
        "kernels": pairs,
        "branching_matrix_target_rows_source_columns": kernel,
        "spectral_radius": rho,
        "price_compensator_integrals": integrals,
        "price_compensator_shares": shares,
        "residual_diagnostics": residuals,
        "warnings": warnings,
    }


def make_plot(path: Path, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, events: list[np.ndarray], horizon: float, spec: FitSpec, residual_frame: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hawkes-matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for label, j, color in (("News to price", 0, "#d81b60"), ("Price to price", 1, "#1565c0")):
        if (1, j) in spec.free_pairs:
            half = math.log(2) / beta[1, j]
            grid = np.linspace(0, max(5 * half, 1e-9), 300)
            ax.plot(grid, alpha[1, j] * np.exp(-beta[1, j] * grid), label=f"{label} (half-life={half:.3g}s)", color=color)
    ax.set(title="Price-intensity impulse responses", xlabel="Seconds after event", ylabel="Intensity contribution / second")
    ax.legend()

    ax = axes[0, 1]
    grid = np.linspace(0, horizon, 350)
    base = mu[1] * grid
    news_cum = cumulative_component(grid, events[0], alpha[1, 0], beta[1, 0]) if (1, 0) in spec.free_pairs else np.zeros_like(grid)
    echo_cum = cumulative_component(grid, events[1], alpha[1, 1], beta[1, 1]) if (1, 1) in spec.free_pairs else np.zeros_like(grid)
    ax.stackplot(grid, base, news_cum, echo_cum, labels=["Baseline", "News-driven", "Price echo"], colors=["#9e9e9e", "#ec407a", "#42a5f5"], alpha=0.85)
    ax.set(title="Cumulative fitted price compensator", xlabel="Elapsed seconds", ylabel="Expected cumulative events")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    theory_x = np.linspace(0, 5, 300)
    ax.plot(theory_x, 1 - np.exp(-theory_x), color="black", linestyle="--", label="Exp(1)")
    for name, color in (("news", "#d81b60"), ("price", "#1565c0")):
        sample = np.sort(residual_frame.loc[residual_frame["stream"] == name, "rescaled_gap"].to_numpy())
        if len(sample):
            ax.step(sample, np.arange(1, len(sample) + 1) / len(sample), where="post", label=name.title(), color=color)
    ax.set(xlim=(0, 5), ylim=(0, 1), title="Time-rescaled gap CDF", xlabel="Rescaled gap", ylabel="Cumulative probability")
    ax.legend()

    ax = axes[1, 1]
    matrix = branching_matrix(alpha, beta)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.01, float(matrix.max())))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["News source", "Price source"])
    ax.set_yticks([0, 1], ["News target", "Price target"])
    ax.set_title("Branching matrix (alpha/beta)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"Information diffusion Hawkes fit — {spec.model} model", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def command_fit(args: argparse.Namespace) -> int:
    events, audit = load_events(args)
    horizon = float(audit["observation_duration_seconds"])
    spec = SPECS[args.model]
    result, mu, alpha, beta = fit_model(events, horizon, spec, args.restarts, args.seed)
    residual_frame, residual_summary = residual_diagnostics(mu, alpha, beta, events, spec)
    summary = make_summary(args, audit, result, mu, alpha, beta, events, horizon, spec, residual_summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_jsonable)
    residual_frame.to_csv(output / "rescaled_gaps.csv", index=False)
    price_contributions(mu, alpha, beta, events, spec).to_csv(output / "event_contributions.csv", index=False)
    make_plot(output / "fit_diagnostics.png", mu, alpha, beta, events, horizon, spec, residual_frame)
    print(json.dumps({"output_dir": str(output.resolve()), "optimizer_success": bool(result.success), "spectral_radius": summary["spectral_radius"], "warnings": summary["warnings"]}, indent=2))
    return 0 if result.success else 2


def command_price_events(args: argparse.Namespace) -> int:
    path = Path(args.input_csv)
    frame = pd.read_csv(path)
    for col in (args.timestamp_col, args.price_col):
        if col not in frame.columns:
            raise ValueError(f"{path}: missing column {col!r}")
    price = pd.to_numeric(frame[args.price_col], errors="coerce")
    if price.isna().any() or (price <= 0).any():
        raise ValueError("price column must contain only positive numeric values")
    log_return_bps = np.log(price).diff() * 10000.0
    mask = log_return_bps.abs() >= args.threshold_bps
    selected = pd.DataFrame(
        {
            "timestamp": frame.loc[mask, args.timestamp_col],
            "return_bps": log_return_bps[mask],
            "direction": np.where(log_return_bps[mask] >= 0, "up", "down"),
        }
    )
    if args.cooldown_seconds > 0 and len(selected):
        raw_times, _kind = read_timestamp_column(path, args.timestamp_col, args.time_unit)
        chosen_positions = np.flatnonzero(mask.to_numpy())
        keep_positions: list[int] = []
        last = -np.inf
        for pos in chosen_positions:
            if raw_times[pos] - last >= args.cooldown_seconds:
                keep_positions.append(pos)
                last = raw_times[pos]
        selected = selected.loc[keep_positions]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(json.dumps({"output_csv": str(output.resolve()), "events": int(len(selected)), "threshold_bps": args.threshold_bps, "cooldown_seconds": args.cooldown_seconds}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit a bivariate exponential Hawkes model")
    inputs = fit.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--events-csv", help="combined CSV containing timestamps and event types")
    inputs.add_argument("--news-csv", help="CSV containing news-event timestamps")
    fit.add_argument("--price-csv", help="CSV containing price-event timestamps; required with --news-csv")
    fit.add_argument("--timestamp-col", default="timestamp")
    fit.add_argument("--event-type-col", default="event_type")
    fit.add_argument("--news-label", default="news")
    fit.add_argument("--price-label", default="price")
    fit.add_argument("--time-unit", choices=tuple(TIME_SCALE), default="seconds", help="unit for numeric timestamps")
    fit.add_argument("--window-start", help="optional inclusive observation start, in input format")
    fit.add_argument("--window-end", help="optional inclusive observation end, in input format")
    fit.add_argument("--model", choices=tuple(SPECS), default="market")
    fit.add_argument("--restarts", type=int, default=6)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--output-dir", default="hawkes_output")
    fit.set_defaults(func=command_fit)

    prep = sub.add_parser("price-events", help="convert a price series to thresholded move events")
    prep.add_argument("--input-csv", required=True)
    prep.add_argument("--timestamp-col", default="timestamp")
    prep.add_argument("--price-col", default="price")
    prep.add_argument("--threshold-bps", type=float, default=5.0)
    prep.add_argument("--cooldown-seconds", type=float, default=0.0)
    prep.add_argument("--time-unit", choices=tuple(TIME_SCALE), default="seconds")
    prep.add_argument("--output-csv", required=True)
    prep.set_defaults(func=command_price_events)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
