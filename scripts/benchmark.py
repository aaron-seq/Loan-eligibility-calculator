"""Reproducible performance benchmark.

    python scripts/benchmark.py              # in-process latency
    python scripts/benchmark.py --http       # plus end-to-end HTTP under gunicorn

Numbers in docs/BENCHMARKS.md come from this script. Re-run it rather than
trusting a figure in a document: latency depends heavily on the CPU, the
scikit-learn build and the worker count, so the report states all three.

Standard library only for the load generator, so it measures this app rather
than a client library.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAYLOAD = {
    "applicant_income": 40_000,
    "coapplicant_income": 0,
    "loan_amount": 2_500_000,
    "loan_term_months": 240,
    "credit_history": 1,
    "dependents": 1,
    "property_area": "urban",
    "cibil_score": 710,
}


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    pick = lambda q: ordered[min(len(ordered) - 1, int(len(ordered) * q))]  # noqa: E731
    return {
        "p50": round(statistics.median(ordered), 1),
        "p95": round(pick(0.95), 1),
        "p99": round(pick(0.99), 1),
    }


def time_calls(fn: Callable[[], object], runs: int, warmup: int = 20) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return percentiles(samples)


def in_process(runs: int) -> dict[str, object]:
    from loanapp.assessment import Assessor
    from loanapp.config import load_config
    from loanapp.features import LoanApplication
    from loanapp.ml.registry import load_bundle

    config = load_config("testing")
    assessor = Assessor(load_bundle(config.model_dir), config)
    app = LoanApplication(**{**PAYLOAD, "education": "graduate", "self_employed": False})

    calls = 0
    original = assessor.bundle.pipeline.predict_proba

    def counting(frame: object) -> object:
        nonlocal calls
        calls += 1
        return original(frame)

    assessor.bundle.pipeline.predict_proba = counting
    assessor.assess(app)
    assessor.bundle.pipeline.predict_proba = original

    return {
        "model_calls_per_assessment": calls,
        "score_one": time_calls(lambda: assessor.score(app), runs),
        "assess_full": time_calls(lambda: assessor.assess(app), runs),
        "assess_no_guidance": time_calls(
            lambda: assessor.assess(app, with_guidance=False), runs
        ),
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post(url: str, body: bytes) -> tuple[float, int]:
    # S310: the URL is built from a localhost base this script chose, never
    # from input, so the file:// concern the rule targets cannot arise.
    request = urllib.request.Request(  # noqa: S310
        url, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    return (time.perf_counter() - start) * 1000, status


def over_http(workers: int, requests: int, concurrency: int) -> dict[str, object]:
    """Real gunicorn, real sockets. Rate limiting is raised for the run only."""
    port = _free_port()
    env = {
        **os.environ,
        "FLASK_ENV": "production",
        "SECRET_KEY": "benchmark-secret-key-of-at-least-32-characters",
        "SESSION_COOKIE_SECURE": "true",
        "DATABASE_PATH": str(ROOT / ".benchmark.db"),
        "RATE_LIMIT_PREDICT": f"{requests * 10}/3600",
    }
    server = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "gunicorn",
            "--workers",
            str(workers),
            "--threads",
            "4",
            "--bind",
            f"127.0.0.1:{port}",
            "--log-level",
            "warning",
            "app:app",
        ],
        cwd=ROOT,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(base + "/api/health", timeout=2)  # noqa: S310
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("gunicorn did not become healthy")

        body = json.dumps(PAYLOAD).encode()
        url = base + "/api/assess"
        for _ in range(workers * 10):  # warm every worker's model
            _post(url, body)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: _post(url, body), range(requests)))
        elapsed = time.perf_counter() - started
    finally:
        server.terminate()
        server.wait(timeout=30)
        for suffix in ("", "-wal", "-shm"):
            Path(str(ROOT / ".benchmark.db") + suffix).unlink(missing_ok=True)

    latencies = [ms for ms, status in results if status == 200]
    errors = sum(1 for _, status in results if status != 200)
    return {
        "workers": workers,
        "concurrency": concurrency,
        "requests": requests,
        "errors": errors,
        "throughput_rps": round(len(latencies) / elapsed, 1),
        "latency_ms": percentiles(latencies) if latencies else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    import numpy
    import sklearn

    report: dict[str, object] = {
        "environment": {
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "numpy": numpy.__version__,
        },
        "in_process": in_process(args.runs),
    }
    if args.http:
        report["http"] = [
            over_http(w, args.requests, args.concurrency) for w in args.workers
        ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
