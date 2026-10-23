"""Check a running deployment's health and the README assessment example."""

from __future__ import annotations

import argparse
import http.client
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent


def request_json(
    base: str, path: str, body: bytes | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    target = urlsplit(base)
    if target.scheme not in {"http", "https"} or not target.hostname:
        raise ValueError("--base-url must be an HTTP or HTTPS URL")
    connection_type = (
        http.client.HTTPSConnection
        if target.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(target.hostname, target.port, timeout=5)
    try:
        connection.request(
            "POST" if body else "GET",
            target.path.rstrip("/") + path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status >= 300:
            raise RuntimeError(f"{path}: HTTP {response.status}")
        return json.load(response), {
            key.lower(): value for key, value in response.getheaders()
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    deadline = time.monotonic() + 30
    while True:
        try:
            health, _ = request_json(base, "/api/health")
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    if health["status"] != "ok" or health["model_status"] != "ready":
        raise RuntimeError(f"Model is not ready: {health}")

    payload, headers = request_json(
        base,
        "/api/assess",
        (ROOT / "examples" / "application.json").read_bytes(),
    )
    result = payload["data"]
    if headers.get("x-frame-options") != "DENY":
        raise RuntimeError("Response security headers are missing")
    if result["out_of_range"] or not 0 <= result["probability"] <= 1:
        raise RuntimeError("The documented sample did not produce a supported estimate")
    if result["saved"] or not result["disclaimer"]:
        raise RuntimeError("Anonymous assessment contract was not satisfied")
    print(
        json.dumps(
            {
                "model_status": health["model_status"],
                "probability": result["probability"],
                "band": result["band"],
                "monthly_instalment": result["affordability"]["monthly_instalment"],
                "saved": result["saved"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
