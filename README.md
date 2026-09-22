# Information Diffusion Hawkes

A model for measuring how timestamped news events propagate into market price-move events with an exponential Hawkes process.

It helps answer five practical questions:

- Did price activity increase after the news?
- How large was the estimated immediate response?
- How quickly did that response fade?
- How much later activity looks like market self-excitation or “echo”?
- Does the fitted model pass basic stability and residual checks?

The model identifies timing patterns that are **consistent with** information diffusion. It does not, by itself, prove that news caused a price move.

## What you provide

You need two timestamped event streams:

1. **News events** — announcement, headline, filing, or release times.
2. **Price events** — times when the price made a move large enough to count as an event.

You can supply one combined CSV:

```csv
timestamp,event_type
2026-09-21T08:30:00Z,news
2026-09-21T08:30:03Z,price
2026-09-21T08:30:08Z,price
```

Or separate news and price-event CSV files with a `timestamp` column.

If you only have a regularly sampled price series, the included script can turn threshold-crossing log returns into price events.

## Run the script directly

Fit separate news and price-event files:

```bash
python3 scripts/hawkes_diffusion.py fit \
  --news-csv news.csv \
  --price-csv price_events.csv \
  --timestamp-col timestamp \
  --output-dir hawkes_output
```

Fit a combined event file:

```bash
python3 scripts/hawkes_diffusion.py fit \
  --events-csv events.csv \
  --timestamp-col timestamp \
  --event-type-col event_type \
  --news-label news \
  --price-label price \
  --output-dir hawkes_output
```

Convert a price series into price-move events using a 5-basis-point threshold:

```bash
python3 scripts/hawkes_diffusion.py price-events \
  --input-csv prices.csv \
  --timestamp-col timestamp \
  --price-col price \
  --threshold-bps 5 \
  --cooldown-seconds 1 \
  --output-csv price_events.csv
```

## Outputs

The fitter writes:

- `summary.json` — parameter estimates, half-lives, branching ratios, model fit, and warnings.
- `event_contributions.csv` — baseline, news-driven, and market-echo intensity at each price event.
- `rescaled_gaps.csv` — transformed event gaps used for residual checking.
- `fit_diagnostics.png` — impulse responses, decomposition, residual CDF, and branching matrix.

The most useful interpretation is usually:

> The event had a weak/moderate/strong estimated immediate impact. Half of that effect faded after X seconds or minutes. Later activity contained little/some/substantial market echo. The diagnostics indicate that the result is reliable/cautious/inconclusive.

## Model structure

The default market model treats news as externally timed and estimates:

- news → price excitation;
- price → price self-excitation; and
- baseline event rates.

An optional full bivariate model estimates all four directed excitation kernels, but it needs substantially more data. See [`references/model-and-output.md`](references/model-and-output.md) for the equations and diagnostic definitions.

## Important limitations

- Hawkes excitation is not automatic proof of causality.
- Results depend on the price-event threshold, observation window, timestamp precision, and trading-session filters.
- Sparse data, duplicate timestamps, intraday seasonality, overnight gaps, or changing market regimes can make a simple stationary model unreliable.
- The bundled fitter reports point estimates, not confidence intervals.
