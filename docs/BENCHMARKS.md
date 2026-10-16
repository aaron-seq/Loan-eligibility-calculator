# Benchmarks

Measured with [`scripts/benchmark.py`](../scripts/benchmark.py). **Re-run it
rather than trusting these figures** — latency depends heavily on the CPU, the
scikit-learn build and the worker count:

```bash
python scripts/benchmark.py            # in-process latency, ~30 s
python scripts/benchmark.py --http     # plus end-to-end HTTP under gunicorn
```

## Environment of the figures below

| | |
|---|---|
| CPUs | 4 |
| Python | 3.11.17 |
| scikit-learn / numpy | 1.9.1 / 2.4.6 |
| Model | calibrated logistic regression (5-fold Platt) |

## In-process latency

One application, 300 timed runs after 20 warm-up runs.

| Operation | p50 | p95 | p99 |
|---|---|---|---|
| Score only | 10.5 ms | 16.3 ms | 18.6 ms |
| Full assessment (score + explanation + guidance) | **33.2 ms** | 45.4 ms | 51.9 ms |
| Assessment without guidance | 21.7 ms | 30.3 ms | 33.8 ms |

### Batching counterfactuals: 4.7×

A full assessment originally made **12** sequential model calls — one for the
estimate, one per explanation counterfactual, one per guidance candidate — and
took **159.7 ms** at p50 (217.1 ms p99).

Profiling showed the cost was almost entirely per call, not per row: the
calibrated pipeline averages five fold models, each validating its input on
every call, so one row took 12.9 ms and sixteen took 14.4 ms. Scoring each
group of variants in one call brought it to **3 calls and 33.2 ms**.

Equivalence was proven, not assumed: a 300-assessment golden snapshot is
byte-identical before and after at the API level. Raw probabilities are equal
to within one float64 ulp (2.2e-16) — numpy sums in a different order for a
different matrix shape — and nothing differs at the four decimal places the
API exposes.

## HTTP throughput

`POST /api/assess` through real gunicorn and real sockets, 400 requests at a
concurrency of 16, after warming every worker. The requests are anonymous,
so nothing is written to the database: this measures the scoring path.

| Workers | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 1 | 20.9 req/s | 758.6 ms | 901.4 ms | 957.5 ms | 0 |
| 2 | 41.2 req/s | 380.3 ms | 583.8 ms | 653.4 ms | 0 |
| 4 | **86.3 req/s** | 171.8 ms | 296.3 ms | 367.8 ms | 0 |

### Reading these honestly

- **Throughput scales linearly with worker processes** — 1 : 2 : 4.1. Scoring
  is CPU-bound and holds the GIL, so threads add nothing and processes add
  everything. Set `WEB_CONCURRENCY` to the number of CPUs.
- **The single-worker latency is queueing, not slowness.** Sixteen clients
  share one CPU; the service time per request is about 48 ms (1000 / 20.9),
  of which ~33 ms is the assessment and the rest HTTP, JSON and validation.
  Latency under load is a capacity question, answered by adding workers.
- **4 workers on 4 CPUs is the ceiling here.** More workers than CPUs only
  adds contention.

## What these numbers do not cover

- **Signed-in requests**, which also write the assessment to SQLite. SQLite
  has a single writer; that, not scoring, would be the first bottleneck for
  write-heavy traffic — see the Postgres migration triggers in
  [`DEPLOYMENT.md`](DEPLOYMENT.md).
- **Cold start.** The first request per worker loads the model; the benchmark
  warms every worker first, deliberately.
- **Page rendering.** Measured through the JSON API only.
- **Other hardware.** These are one machine's figures. Re-run the script.
