"""Tests for denial playbook matching."""

from agents.playbook import Playbook


def test_playbook_loads():
    pb = Playbook.load()
    assert len(pb.codes) >= 10


def test_match_medical_necessity():
    pb = Playbook.load()
    text = "Service is not medically necessary per policy criteria."
    matches = pb.match_denial(text)
    assert len(matches) >= 1
    assert matches[0].code == "50"


def test_applicable_protections_mhpaea():
    pb = Playbook.load()
    text = "Residential substance use disorder treatment denied."
    protections = pb.get_applicable_protections(text)
    names = [p["name"] for p in protections]
    assert "mental_health_parity_addiction_equity_act" in names


def test_build_context_helpers():
    pb = Playbook.load()
    matches = pb.match_denial("not medically necessary")
    entry = matches[0]
    assert "CARC" in pb.build_insurer_defense_context(entry)
    assert "Counter-arguments" in pb.build_patient_counter_context(entry)
