"""Golden regression structure tests for demo cases."""

import json
from pathlib import Path


DEMO_DIR = Path(__file__).parent.parent / "data" / "demo"


def test_demo_cases_have_golden_expectations():
    for case_file in DEMO_DIR.glob("case_*.json"):
        case = json.loads(case_file.read_text())
        assert "expected_round_1" in case
        assert "denial_letter" in case
        assert "denial_text" in case["denial_letter"]


def test_demo_case_structure():
    case = json.loads((DEMO_DIR / "case_01_oscar_cromolyn.json").read_text())
    assert case["expected_round_1"]["medical_necessity_output"]["claim"]
    assert case["expected_insurer_defense_round_1"]["weakest_claim_attacked"]
