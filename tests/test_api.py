"""Flask API tests (no LLM calls)."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["DISABLE_WEAVE"] = "true"


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Denial Defense" in resp.data


def test_list_cases(client):
    resp = client.get("/cases")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert len(data) >= 1


@patch("web.app.run_harness")
def test_run_case_mock(mock_harness, client):
    mock_harness.return_value = {
        "final_verdict": "Test appeal",
        "round_1_outputs": {"medical_necessity": {"claim": "x"}},
        "round_2_outputs": {},
        "critiques": [],
        "_metrics": {"elapsed_seconds": 1.0},
        "_evidence_checklist": ["Letter of medical necessity"],
    }
    resp = client.post("/run/case_01_oscar_cromolyn")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"
    assert "Test appeal" in data["final_verdict"]


def test_run_custom_missing_fields(client):
    resp = client.post("/run/custom", json={"denial_letter": ""})
    assert resp.status_code == 400


def test_export_txt(client):
    resp = client.post("/export", json={"appeal_text": "Hello appeal", "format": "txt"})
    assert resp.status_code == 200
