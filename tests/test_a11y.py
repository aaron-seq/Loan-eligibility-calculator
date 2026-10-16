"""Automated accessibility audit: axe-core in a real browser.

Runs axe-core (via axe-playwright-python, which bundles it, so no Node is
needed) against every page in both user flows, served by a live app and
rendered by real Chromium with the real stylesheet.

Two lessons are built into this file:

* **It must run against the real styles.** An earlier audit ran while the
  stylesheet could not load. Every page rendered as unstyled HTML, so the
  colour-contrast check measured default black on white and passed. Against
  the real styles it found a 1.41:1 failure. ``test_styles_are_loaded`` makes
  that situation a hard failure instead of a silent pass.
* **Zero violations here is not "accessible".** Automated checkers catch
  roughly a third of WCAG issues. Form-error wiring, the skip link and spoken
  rule severity are asserted separately in test_pages.py, because axe cannot
  see any of them. Manual screen-reader testing is tracked in #22.

Skipped, not failed, when no browser is available. To run locally:

    playwright install chromium
    pytest -m a11y

or point at an existing binary with PLAYWRIGHT_CHROMIUM_EXECUTABLE.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.a11y, pytest.mark.requires_model]

#: Set in the dedicated CI job. There, "no browser" or "no model" means the
#: job is broken, and skipping would let a job of nothing-but-skips pass green.
REQUIRED = os.environ.get("A11Y_REQUIRED") == "1"


def _unavailable(reason: str) -> None:
    if REQUIRED:
        pytest.fail(f"A11Y_REQUIRED is set but the audit cannot run: {reason}")
    pytest.skip(reason)


try:
    from axe_playwright_python import sync_playwright as axe_module
    from playwright import sync_api
except ImportError as exc:  # pragma: no cover - depends on the environment
    if REQUIRED:
        raise
    pytest.skip(f"accessibility tooling not installed: {exc}", allow_module_level=True)

PAGES = (
    "home",
    "apply",
    "apply-with-errors",
    "result",
    "how-it-works",
    "login",
    "signup",
    "not-found",
    "dashboard",
)


def _launch(playwright: Any) -> Any:
    """Find a browser, or skip with instructions rather than fail."""
    executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    try:
        if executable:
            return playwright.chromium.launch(executable_path=executable)
        return playwright.chromium.launch()
    except Exception as exc:
        _unavailable(
            "No Chromium available for the accessibility audit "
            f"({str(exc).splitlines()[0][:120]}). Run `playwright install "
            "chromium`, or set PLAYWRIGHT_CHROMIUM_EXECUTABLE."
        )
        raise  # unreachable: _unavailable always skips or fails


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The real app on an ephemeral port, in a background thread."""
    from werkzeug.serving import make_server

    from loanapp import create_app
    from loanapp.config import Config
    from loanapp.ml.registry import bundle_path

    model_dir = str(Path(__file__).resolve().parent.parent / "models")
    if not os.path.exists(bundle_path(model_dir)):
        _unavailable("No trained model. Run: python -m loanapp.ml.train")

    tmp = tmp_path_factory.mktemp("a11y")
    app = create_app(
        Config(
            env="testing",
            secret_key="a11y-audit-secret-key-at-least-32-characters",
            database_path=str(tmp / "a11y.db"),
            model_dir=model_dir,
            cookie_secure=False,
        )
    )
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


@pytest.fixture(scope="module")
def audit(live_server: str) -> dict[str, Any]:
    """Walk both flows once and collect axe results per page."""
    axe = axe_module.Axe()
    results: dict[str, Any] = {}

    with sync_api.sync_playwright() as playwright:
        browser = _launch(playwright)
        page = browser.new_page()

        def check(name: str) -> None:
            results[name] = axe.run(page).response["violations"]

        page.goto(live_server + "/")
        results["_cta_background"] = page.eval_on_selector(
            "a.bg-indigo-600", "e => getComputedStyle(e).backgroundColor"
        )
        check("home")

        for path, name in (
            ("/apply", "apply"),
            ("/how-it-works", "how-it-works"),
            ("/login", "login"),
            ("/signup", "signup"),
            ("/no-such-page", "not-found"),
        ):
            page.goto(live_server + path)
            check(name)

        page.goto(live_server + "/apply")
        page.fill("#applicant_income", "60000")
        page.fill("#loan_amount", "1800000")
        page.click("button[type=submit]")
        page.wait_for_load_state()
        check("result")

        page.goto(live_server + "/apply")
        page.fill("#applicant_income", "60000")
        page.fill("#loan_amount", "-5")
        page.evaluate("document.querySelector('#loan_amount').removeAttribute('min')")
        page.click("button[type=submit]")
        page.wait_for_load_state()
        check("apply-with-errors")

        page.goto(live_server + "/signup")
        page.fill("#user_id", "a11y_user")
        page.fill("#full_name", "Access Ible")
        page.fill("#email", "a11y@example.com")
        page.fill("#password", "correct-horse-battery")
        page.fill("#confirm_password", "correct-horse-battery")
        page.click("form button")
        page.wait_for_load_state()
        page.goto(live_server + "/apply")
        page.fill("#applicant_income", "60000")
        page.fill("#loan_amount", "1800000")
        page.click("button[type=submit]")
        page.wait_for_load_state()
        page.goto(live_server + "/dashboard")
        check("dashboard")

        browser.close()
    return results


def test_styles_are_loaded(audit: dict[str, Any]) -> None:
    """Guards the audit itself. If the stylesheet does not load, every
    contrast check below passes against unstyled HTML and means nothing."""
    assert audit["_cta_background"] not in ("rgba(0, 0, 0, 0)", "transparent"), (
        "Stylesheet did not load; the contrast results would be meaningless."
    )


@pytest.mark.parametrize("page_name", PAGES)
def test_page_has_no_axe_violations(audit: dict[str, Any], page_name: str) -> None:
    violations = audit[page_name]
    detail = "\n".join(
        f"  [{v['impact']}] {v['id']}: {v['help']}\n    {v['nodes'][0]['html'][:160]}"
        for v in violations
    )
    assert not violations, f"{page_name}: {len(violations)} violation(s)\n{detail}"
