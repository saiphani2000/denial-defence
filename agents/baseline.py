"""
Baseline single-agent appeal system for comparison.

Architecture: Simple single-call GPT-4o with no revision loop.
"""

from __future__ import annotations

import json
import sys
import io
from pathlib import Path

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

from agents.llm import chat_text, reset_session_tokens
from agents.logging_config import log
from agents.prompts import BASELINE_SYSTEM


def single_agent_appeal(denial_letter: str, patient_context: str) -> str:
    """Single-agent baseline: one GPT-4o call produces the entire appeal."""
    user_message = f"""DENIAL LETTER:
{denial_letter}

PATIENT CONTEXT:
{patient_context}

Write a complete insurance appeal letter addressing this denial."""

    try:
        reset_session_tokens()
        return chat_text(BASELINE_SYSTEM, user_message, max_tokens=2000)
    except Exception as e:
        error_msg = f"Baseline agent failed: {str(e)}"
        log.error(error_msg)
        return f"[Error generating appeal: {error_msg}]"


def main():
    denial_synopsis = """
Type: Medical Necessity
Diagnosis: Morbid Obesity (BMI 42, hypertension, diabetes)
Treatment: Bariatric surgery (gastric sleeve)
Denial Reason: Patient does not meet Class I obesity criteria.
"""
    patient_context = "45-year-old with BMI 42, Type 2 diabetes, failed lifestyle modification."

    print("=" * 80)
    print("TESTING BASELINE SINGLE-AGENT APPEAL")
    print("=" * 80)
    appeal = single_agent_appeal(denial_synopsis, patient_context)
    print(appeal)


if __name__ == "__main__":
    main()
