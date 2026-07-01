"""
Multi-agent insurance appeal harness with adversarial critic.

Architecture: Supervisor-worker with adversarial critic and round-based revision.
Three patient agents work in parallel. The critic simulates an insurer medical reviewer.

Flow:
1. Round 1 — patient agents (parallel) → critic
2. Round 2 — targeted revision of weakest agent(s) → critic (optional if revision_needed=false)
3. Supervisor synthesizes final appeal from preserved round 1 + round 2 outputs
"""

from __future__ import annotations

import json
import sys
import io
from pathlib import Path
from typing import Annotated, Any, TypedDict
from operator import add

from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "buffer") and sys.stdout.buffer is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer") and sys.stderr.buffer is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except (ValueError, AttributeError):
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END

from agents.config import get_settings
from agents.imr_retrieval import format_precedent_context, retrieve_similar_cases
from agents.llm import chat_json_with_repair, chat_text, get_session_tokens, reset_session_tokens
from agents.logging_config import log
from agents.playbook import Playbook
from agents.progress import report
from agents.prompts import (
    MEDICAL_NECESSITY_SYSTEM,
    POLICY_CITATION_SYSTEM,
    PRECEDENT_SYSTEM,
    INSURER_DEFENSE_SYSTEM,
    SUPERVISOR_SYNTHESIS_SYSTEM,
)
from agents.schemas import AgentClaimOutput, CritiqueOutput, PrecedentOutput, validate_agent_output

_playbook: Playbook | None = None
_harness = None
_weave_ready = False


def _init_weave() -> None:
    global _weave_ready
    if _weave_ready:
        return
    import weave

    weave.init(get_settings().weave_project)
    _weave_ready = True


def _get_playbook() -> Playbook:
    global _playbook
    if _playbook is None:
        _playbook = Playbook.load()
    return _playbook


def _apply_weave_op(fn):
    """Apply weave.op when tracing is enabled; otherwise return fn unchanged."""
    if __import__("os").getenv("DISABLE_WEAVE", "").lower() == "true":
        return fn
    try:
        _init_weave()
        import weave

        return weave.op()(fn)
    except Exception as exc:
        log.warning("Weave tracing disabled: %s", exc)
        return fn


AGENT_KEYS = ("medical_necessity", "policy_citation", "precedent")


class RoundOutputs(TypedDict, total=False):
    medical_necessity: dict
    policy_citation: dict
    precedent: dict


class HarnessState(TypedDict):
    denial_letter: str
    patient_context: str
    denial_summary: str
    playbook_context: str
    precedent_context: str
    evidence_checklist: list[str]
    medical_necessity: dict
    policy_citation: dict
    precedent: dict
    round_1_outputs: RoundOutputs
    round_2_outputs: RoundOutputs
    critiques: Annotated[list, add]
    round_num: int
    agents_to_revise: list[str]
    skip_round_2: bool
    final_verdict: str


def _summarize_denial(denial_letter: str) -> str:
    """Cache a short summary once per run for agent prompts."""
    if len(denial_letter) <= 1200:
        return denial_letter
    return denial_letter[:1200] + "\n...[truncated]"


def _build_playbook_context(denial_letter: str) -> tuple[str, list[str]]:
    pb = _get_playbook()
    matches = pb.match_denial(denial_letter)
    if not matches:
        return "", []
    primary = matches[0]
    patient_ctx = pb.build_patient_counter_context(primary)
    protections = pb.get_applicable_protections(denial_letter)
    prot_text = ""
    if protections:
        prot_text = "\n\nApplicable federal protections:\n" + json.dumps(protections, indent=2)
    checklist = list(primary.typical_evidence_needed)
    return patient_ctx + prot_text, checklist


def _map_weakest_to_agents(weakest: str) -> list[str]:
    text = (weakest or "").lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "medical_necessity": "medical_necessity",
        "medical": "medical_necessity",
        "necessity": "medical_necessity",
        "policy_citation": "policy_citation",
        "policy": "policy_citation",
        "citation": "policy_citation",
        "precedent": "precedent",
    }
    for key, agent in mapping.items():
        if key in text:
            return [agent]
    return list(AGENT_KEYS)


def _agent_user_message(state: HarnessState, extra: str = "") -> str:
    msg = f"""DENIAL LETTER:
{state['denial_summary']}

PATIENT CONTEXT:
{state['patient_context']}

PLAYBOOK GUIDANCE:
{state.get('playbook_context') or 'No specific CARC match.'}"""
    if extra:
        msg += f"\n\n{extra}"
    critiques = state.get("critiques") or []
    if critiques:
        msg += f"\n\nPREVIOUS CRITIQUE TO ADDRESS:\n{json.dumps(critiques[-1], indent=2)}"
    return msg


def _should_run_agent(state: HarnessState, agent_key: str) -> bool:
    if state["round_num"] == 1:
        return True
    targets = state.get("agents_to_revise") or list(AGENT_KEYS)
    return agent_key in targets


def _pass_through_round2(state: HarnessState, agent_key: str) -> dict:
    r1 = (state.get("round_1_outputs") or {}).get(agent_key, {})
    r2 = dict(state.get("round_2_outputs") or {})
    r2[agent_key] = r1
    return {agent_key: r1, "round_2_outputs": r2}


@_apply_weave_op
def medical_necessity_agent(state: HarnessState) -> dict:
    report("medical_necessity", f"Round {state['round_num']}")
    if state["round_num"] == 2 and not _should_run_agent(state, "medical_necessity"):
        return _pass_through_round2(state, "medical_necessity")

    raw = chat_json_with_repair(
        MEDICAL_NECESSITY_SYSTEM,
        _agent_user_message(state),
    )
    try:
        result = validate_agent_output(raw, AgentClaimOutput)
    except Exception:
        result = raw

    updates: dict[str, Any] = {"medical_necessity": result}
    if state["round_num"] == 1:
        r1 = dict(state.get("round_1_outputs") or {})
        r1["medical_necessity"] = result
        updates["round_1_outputs"] = r1
    else:
        r2 = dict(state.get("round_2_outputs") or {})
        r2["medical_necessity"] = result
        updates["round_2_outputs"] = r2
    return updates


@_apply_weave_op
def policy_citation_agent(state: HarnessState) -> dict:
    report("policy_citation", f"Round {state['round_num']}")
    if state["round_num"] == 2 and not _should_run_agent(state, "policy_citation"):
        return _pass_through_round2(state, "policy_citation")

    raw = chat_json_with_repair(
        POLICY_CITATION_SYSTEM,
        _agent_user_message(state),
    )
    try:
        result = validate_agent_output(raw, AgentClaimOutput)
    except Exception:
        result = raw

    updates: dict[str, Any] = {"policy_citation": result}
    if state["round_num"] == 1:
        r1 = dict(state.get("round_1_outputs") or {})
        r1["policy_citation"] = result
        updates["round_1_outputs"] = r1
    else:
        r2 = dict(state.get("round_2_outputs") or {})
        r2["policy_citation"] = result
        updates["round_2_outputs"] = r2
    return updates


@_apply_weave_op
def precedent_agent(state: HarnessState) -> dict:
    report("precedent", f"Round {state['round_num']}")
    if state["round_num"] == 2 and not _should_run_agent(state, "precedent"):
        return _pass_through_round2(state, "precedent")

    extra = state.get("precedent_context") or ""
    raw = chat_json_with_repair(
        PRECEDENT_SYSTEM,
        _agent_user_message(state, extra=f"IMR PRECEDENT DATA:\n{extra}"),
    )
    try:
        result = validate_agent_output(raw, PrecedentOutput)
    except Exception:
        result = raw

    updates: dict[str, Any] = {"precedent": result}
    if state["round_num"] == 1:
        r1 = dict(state.get("round_1_outputs") or {})
        r1["precedent"] = result
        updates["round_1_outputs"] = r1
    else:
        r2 = dict(state.get("round_2_outputs") or {})
        r2["precedent"] = result
        updates["round_2_outputs"] = r2
    return updates


@_apply_weave_op
def insurer_defense_critic(state: HarnessState) -> dict:
    report("critic", f"Round {state['round_num']}")
    pb = _get_playbook()
    playbook_context = ""
    try:
        matches = pb.match_denial(state["denial_letter"])
        if matches:
            playbook_context = pb.build_insurer_defense_context(matches[0])
            if len(matches) > 1:
                playbook_context += f"\n\nAdditional CARC matches: {[m.code for m in matches[1:3]]}"
    except Exception as exc:
        log.warning("Playbook match failed: %s", exc)

    user_msg = f"""DENIAL LETTER:
{state['denial_summary']}

PATIENT'S DRAFT APPEAL (3 agents):

MEDICAL NECESSITY ARGUMENT:
{json.dumps(state.get('medical_necessity', {}), indent=2)}

POLICY CITATION ARGUMENT:
{json.dumps(state.get('policy_citation', {}), indent=2)}

PRECEDENT ARGUMENT:
{json.dumps(state.get('precedent', {}), indent=2)}

{playbook_context}

Your task: identify the SINGLE WEAKEST argument and attack it specifically."""

    raw = chat_json_with_repair(INSURER_DEFENSE_SYSTEM, user_msg)
    try:
        result = validate_agent_output(raw, CritiqueOutput)
    except Exception:
        result = raw
    result["round"] = state["round_num"]

    updates: dict[str, Any] = {"critiques": [result]}
    if state["round_num"] == 1:
        agents = _map_weakest_to_agents(str(result.get("weakest_claim_attacked", "")))
        updates["agents_to_revise"] = agents
        if result.get("revision_needed") is False:
            updates["skip_round_2"] = True
    return updates


@_apply_weave_op
def supervisor_start(state: HarnessState) -> dict:
    report("supervisor", "Starting Round 1")
    return {"round_num": 1}


@_apply_weave_op
def supervisor_round_2(state: HarnessState) -> dict:
    report("supervisor", "Starting Round 2 (targeted revision)")
    r1 = {
        "medical_necessity": state.get("medical_necessity", {}),
        "policy_citation": state.get("policy_citation", {}),
        "precedent": state.get("precedent", {}),
    }
    return {"round_num": 2, "round_1_outputs": r1}


@_apply_weave_op
def supervisor_synthesize(state: HarnessState) -> dict:
    report("supervisor", "Synthesizing final verdict")
    r1 = state.get("round_1_outputs") or {}
    r2 = state.get("round_2_outputs") or r1
    critiques = state.get("critiques") or []

    user_msg = f"""ORIGINAL DENIAL:
{state['denial_letter']}

PATIENT CONTEXT:
{state['patient_context']}

ROUND 1 PATIENT ARGUMENTS:
Medical Necessity: {json.dumps(r1.get('medical_necessity', {}), indent=2)}
Policy Citation: {json.dumps(r1.get('policy_citation', {}), indent=2)}
Precedent: {json.dumps(r1.get('precedent', {}), indent=2)}

ROUND 1 CRITIQUE:
{json.dumps(critiques[0] if len(critiques) > 0 else {}, indent=2)}

ROUND 2 REVISED ARGUMENTS:
Medical Necessity: {json.dumps(r2.get('medical_necessity', {}), indent=2)}
Policy Citation: {json.dumps(r2.get('policy_citation', {}), indent=2)}
Precedent: {json.dumps(r2.get('precedent', {}), indent=2)}

ROUND 2 CRITIQUE:
{json.dumps(critiques[1] if len(critiques) > 1 else {}, indent=2)}

EVIDENCE CHECKLIST (address gaps if any):
{json.dumps(state.get('evidence_checklist', []), indent=2)}

Synthesize the final appeal letter incorporating the strongest arguments and addressing critiques."""

    final_verdict = chat_text(SUPERVISOR_SYNTHESIS_SYSTEM, user_msg, max_tokens=2000)
    return {"final_verdict": final_verdict}


def should_continue_rounds(state: HarnessState) -> str:
    settings = get_settings()
    if state.get("skip_round_2") and state["round_num"] == 1:
        log.info("Skipping Round 2 — critic marked revision_needed=false")
        return "finish"
    if state["round_num"] < settings.max_rounds:
        return "round_2"
    return "finish"


def build_harness():
    workflow = StateGraph(HarnessState)

    workflow.add_node("supervisor_start", supervisor_start)
    workflow.add_node("supervisor_round_2", supervisor_round_2)
    workflow.add_node("supervisor_synthesize", supervisor_synthesize)
    workflow.add_node("medical_necessity", medical_necessity_agent)
    workflow.add_node("policy_citation", policy_citation_agent)
    workflow.add_node("precedent", precedent_agent)
    workflow.add_node("critic", insurer_defense_critic)

    workflow.set_entry_point("supervisor_start")
    workflow.add_edge("supervisor_start", "medical_necessity")
    workflow.add_edge("supervisor_start", "policy_citation")
    workflow.add_edge("supervisor_start", "precedent")
    workflow.add_edge("medical_necessity", "critic")
    workflow.add_edge("policy_citation", "critic")
    workflow.add_edge("precedent", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_continue_rounds,
        {"round_2": "supervisor_round_2", "finish": "supervisor_synthesize"},
    )
    workflow.add_edge("supervisor_round_2", "medical_necessity")
    workflow.add_edge("supervisor_round_2", "policy_citation")
    workflow.add_edge("supervisor_round_2", "precedent")
    workflow.add_edge("supervisor_synthesize", END)

    return workflow.compile()


def get_harness():
    global _harness
    if _harness is None:
        _harness = build_harness()
    return _harness


@_apply_weave_op
def run_harness(denial_letter: str, patient_context: str) -> dict:
    import time

    reset_session_tokens()
    start = time.time()
    report("start", "Initializing harness")

    pb = _get_playbook()
    try:
        matched_entries = pb.match_denial(denial_letter)
        carc_codes_matched = [e.code for e in matched_entries[:3]]
    except Exception:
        carc_codes_matched = []

    try:
        applicable_protections = pb.get_applicable_protections(denial_letter)
    except Exception:
        applicable_protections = []

    playbook_context, evidence_checklist = _build_playbook_context(denial_letter)
    imr_cases = retrieve_similar_cases(denial_letter, patient_context)
    precedent_context = format_precedent_context(imr_cases)

    result = get_harness().invoke(
        {
            "denial_letter": denial_letter,
            "patient_context": patient_context,
            "denial_summary": _summarize_denial(denial_letter),
            "playbook_context": playbook_context,
            "precedent_context": precedent_context,
            "evidence_checklist": evidence_checklist,
            "critiques": [],
            "round_num": 0,
            "medical_necessity": {},
            "policy_citation": {},
            "precedent": {},
            "round_1_outputs": {},
            "round_2_outputs": {},
            "agents_to_revise": list(AGENT_KEYS),
            "skip_round_2": False,
            "final_verdict": "",
        }
    )

    elapsed = time.time() - start
    report("complete", f"Finished in {elapsed:.1f}s")

    r1 = result.get("round_1_outputs") or {}
    r2 = result.get("round_2_outputs") or {}

    result["_metrics"] = {
        "elapsed_seconds": round(elapsed, 1),
        "carc_codes_matched": carc_codes_matched,
        "federal_protections_found": [p["name"] for p in applicable_protections],
        "rounds_completed": result.get("round_num", 0),
        "critiques_generated": len(result.get("critiques", [])),
        "session_tokens": get_session_tokens(),
        "agents_revised_round_2": result.get("agents_to_revise", []),
        "imr_cases_retrieved": len(imr_cases),
    }
    result["_imr_cases"] = imr_cases
    result["_evidence_checklist"] = evidence_checklist
    result["round_1"] = r1
    result["round_2"] = r2
    return result


def main() -> int:
    case_path = Path("data/demo/case_01_oscar_cromolyn.json")
    if not case_path.exists():
        print(f"ERROR: Demo case not found at {case_path}")
        return 1

    case = json.loads(case_path.read_text(encoding="utf-8"))
    print("=" * 80)
    print("TESTING MULTI-AGENT HARNESS")
    print(f"Case: {case['display_name']}")
    print("=" * 80)

    result = run_harness(
        denial_letter=case["denial_letter"]["denial_text"],
        patient_context=json.dumps(case["patient_context"], indent=2),
    )

    print("\nFINAL VERDICT:")
    print("-" * 80)
    print(result.get("final_verdict", "[No verdict generated]"))
    print(f"\nTokens used: {result['_metrics']['session_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
