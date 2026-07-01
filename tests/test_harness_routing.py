"""Tests for harness routing helpers."""

import importlib.util
import sys
from pathlib import Path

# Import harness module functions without triggering weave at decoration if disabled
import os

os.environ["DISABLE_WEAVE"] = "true"
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-import")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.harness import _map_weakest_to_agents, should_continue_rounds


def test_map_weakest_policy():
    assert _map_weakest_to_agents("policy_citation") == ["policy_citation"]


def test_map_weakest_medical():
    assert _map_weakest_to_agents("Medical Necessity") == ["medical_necessity"]


def test_skip_round_2_when_not_needed():
    state = {"round_num": 1, "skip_round_2": True}
    assert should_continue_rounds(state) == "finish"


def test_continue_to_round_2():
    state = {"round_num": 1, "skip_round_2": False}
    assert should_continue_rounds(state) == "round_2"
