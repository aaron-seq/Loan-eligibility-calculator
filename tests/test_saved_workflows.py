"""Saved results retain inputs and the original deterministic checks."""

import json
import re
from html import unescape
from unittest.mock import patch

import pytest
from test_api import VALID

from loanapp.db import connect


def _save(client, **changes):
    response = client.post(
        "/api/assess",
        json={**VALID, **changes},
        headers={"X-CSRF-Token": client.csrf},
    )
    assert response.status_code == 200
    return response.get_json()["data"]


@pytest.mark.parametrize("dependents", [0, 3, 4, 10])
def test_reopening_and_resubmitting_preserves_dependents_and_label(
    registered_client, dependents
):
    label = 'Home loan & "longer term"'
    result = _save(registered_client, dependents=dependents, label=label)
    original = registered_client.get(
        f"/api/applications/{result['application_id']}"
    ).get_json()["data"]
    body = registered_client.get(f"/apply?from={result['application_id']}").get_data(
        as_text=True
    )
    select = re.search(r'<select[^>]*name="dependents"[^>]*>(.*?)</select>', body, re.S)
    selected = re.search(r'<option value="([^"]+)"[^>]*selected', select.group(1))
    assert selected is not None, "Saved dependents need a matching selected option"
    label_field = re.search(r'<input[^>]*name="label"[^>]*value="([^"]*)"', body)
    form_label = unescape(label_field.group(1))
    assert int(selected.group(1)) == dependents
    assert form_label == label

    form = {
        **original["payload"],
        "dependents": selected.group(1),
        "label": form_label,
        "self_employed": "1" if original["payload"]["self_employed"] else "0",
        "csrf_token": registered_client.csrf,
    }
    assert registered_client.post("/apply", data=form).status_code == 200
    latest = registered_client.get("/api/applications").get_json()["data"][
        "applications"
    ][0]
    assert latest["payload"] == original["payload"]
    assert latest["label"] == label


@pytest.mark.parametrize(
    "unsupported,credit_score,severity",
    [(False, 620, "fail"), (True, 620, "fail"), (False, 780, "pass")],
)
def test_saved_result_shows_original_costs_and_rules_without_rescoring(
    app, registered_client, unsupported, credit_score, severity
):
    result = _save(
        registered_client,
        applicant_income=0 if unsupported else 6000,
        coapplicant_income=0,
        loan_amount=180000,
        cibil_score=credit_score,
    )
    assert result["out_of_range"] is unsupported
    warning = next(rule for rule in result["rules"] if rule["code"] == "cibil_score")
    assert warning["severity"] == severity
    with patch.object(app.extensions["assessor"], "assess", side_effect=AssertionError):
        response = registered_client.get(f"/applications/{result['application_id']}")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert warning["title"] in body
    assert warning["detail"] in unescape(body)
    assert "Monthly instalment" in body
    assert f"{result['affordability']['monthly_instalment']:,.0f}" in body
    assert result["model"]["version"] in body


@pytest.mark.parametrize(
    "missing", [("affordability",), ("rules",), ("affordability", "rules", "factors")]
)
def test_saved_legacy_result_tolerates_missing_optional_sections(
    app, registered_client, missing
):
    result = _save(registered_client, cibil_score=620)
    repository = app.extensions["application_repository"]
    record = repository.get(result["application_id"], user_id="testuser")
    for field in missing:
        record["assessment"].pop(field)
    with connect(repository.database_path) as conn:
        conn.execute(
            "UPDATE applications SET assessment=? WHERE id=?",
            (json.dumps(record["assessment"]), result["application_id"]),
        )
    response = registered_client.get(f"/applications/{result['application_id']}")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Estimate at the time" in body
    assert ("Monthly instalment" in body) is ("affordability" not in missing)
    assert ("Credit score below common cut-offs" in body) is ("rules" not in missing)
