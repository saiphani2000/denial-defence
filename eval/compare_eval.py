"""
Weave evaluation: compare baseline single-agent vs multi-agent harness.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("WEAVE_PARALLELISM", "20")

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import weave
from openai import OpenAI

from agents.baseline import single_agent_appeal
from agents.config import get_settings
from agents.eval_helpers import imr_row_to_denial_letter, imr_row_to_patient_context
from agents.harness import run_harness

settings = get_settings()
assert settings.openai_api_key, "OPENAI_API_KEY required"
assert settings.wandb_api_key, "WANDB_API_KEY required"

wandb_scorer_client = OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=settings.wandb_api_key,
    default_headers={
        "OpenAI-Project": "sabhisheksagar200-northeastern-university/denial-defense"
    },
)

SCORER_MODEL = settings.scorer_model


def _init_weave():
    weave.init(settings.weave_project)


_init_weave()


@weave.op()
def baseline_system(denial_letter: str, patient_context: str) -> dict:
    start = time.time()
    output = single_agent_appeal(denial_letter, patient_context)
    return {
        "appeal": output,
        "elapsed_seconds": round(time.time() - start, 2),
        "system": "baseline",
    }


@weave.op()
def harness_system(denial_letter: str, patient_context: str) -> dict:
    start = time.time()
    try:
        result = run_harness(denial_letter, patient_context)
        return {
            "appeal": result.get("final_verdict", "[No verdict generated]"),
            "elapsed_seconds": round(time.time() - start, 2),
            "session_tokens": result.get("_metrics", {}).get("session_tokens", 0),
            "system": "harness",
        }
    except Exception as e:
        return {
            "appeal": f"[Harness error: {str(e)}]",
            "elapsed_seconds": round(time.time() - start, 2),
            "system": "harness",
        }


def _unwrap_output(output: str | dict) -> str:
    if isinstance(output, dict):
        return str(output.get("appeal", ""))
    return str(output)


@weave.op()
def survives_attack_scorer(output: str | dict, denial_letter: str) -> dict:
    output = _unwrap_output(output)
    if not output:
        return {"survives": False, "weak_points": ["empty output"], "score": 0.0}

    try:
        response = wandb_scorer_client.chat.completions.create(
            model=SCORER_MODEL,
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.3,
            timeout=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an insurance medical reviewer. Return JSON: "
                        '{"survives": bool, "weak_points": [], "reasoning": ""}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"DENIAL:\n{denial_letter[:1500]}\n\nAPPEAL:\n{output[:3000]}",
                },
            ],
        )
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
        return {
            "survives": result.get("survives", False),
            "weak_points": result.get("weak_points", []),
            "reasoning": result.get("reasoning", ""),
            "score": 1.0 if result.get("survives", False) else 0.0,
        }
    except Exception as e:
        return {"survives": False, "weak_points": [str(e)], "score": 0.0}


@weave.op()
def cites_specific_evidence_scorer(output: str | dict, denial_letter: str) -> dict:
    output = _unwrap_output(output)
    if not output:
        return {"specificity_score": 0, "citations_found": []}

    try:
        resp = wandb_scorer_client.chat.completions.create(
            model=SCORER_MODEL,
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.3,
            timeout=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Return JSON: {"citations_found": [], "specificity_score": int, '
                        '"has_specific_citations": bool}'
                    ),
                },
                {"role": "user", "content": f"APPEAL: {output[:3000]}"},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
        result["specificity_score"] = int(result.get("specificity_score", 0))
        return result
    except Exception as e:
        return {"specificity_score": 0, "citations_found": [], "error": str(e)}


@weave.op()
def invokes_federal_protections_scorer(output: str | dict, denial_letter: str) -> dict:
    output = _unwrap_output(output).lower()
    protections = {
        "MHPAEA": ["mhpaea", "mental health parity", "parity act"],
        "ACA": ["affordable care act", "aca"],
        "ACA_1557": ["section 1557", "1557"],
        "ERISA": ["erisa"],
        "NSA": ["no surprises act", "nsa"],
        "ADA": ["americans with disabilities act"],
    }
    found = [name for name, kws in protections.items() if any(kw in output for kw in kws)]
    return {
        "protections_invoked": found,
        "protection_count": len(found),
        "invokes_any": len(found) > 0,
        "score": float(len(found)),
    }


@weave.op()
def addresses_denial_reason_scorer(output: str | dict, denial_letter: str) -> dict:
    output = _unwrap_output(output)
    if not output:
        return {"directness_score": 0, "addresses_reason": False, "score": 0.0}

    try:
        resp = wandb_scorer_client.chat.completions.create(
            model=SCORER_MODEL,
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0.3,
            timeout=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Score 0-3 directness. Return JSON: {"directness_score": 0-3, '
                        '"addresses_reason": bool, "reasoning": ""}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"DENIAL:\n{denial_letter[:1500]}\n\nAPPEAL:\n{output[:2500]}",
                },
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
        score = int(result.get("directness_score", 0))
        return {
            "directness_score": score,
            "addresses_reason": bool(result.get("addresses_reason", False)),
            "reasoning": result.get("reasoning", ""),
            "score": float(score) / 3.0,
        }
    except Exception as e:
        return {"directness_score": 0, "addresses_reason": False, "error": str(e), "score": 0.0}


@weave.op()
def latency_scorer(output: str | dict, denial_letter: str) -> dict:
    if isinstance(output, dict):
        elapsed = float(output.get("elapsed_seconds", 0))
        return {"elapsed_seconds": elapsed, "score": max(0.0, 1.0 - elapsed / 120.0)}
    return {"elapsed_seconds": 0.0, "score": 0.0}


def build_dataset(n_samples: int = 25) -> weave.Dataset:
    eval_path = settings.eval_parquet_path
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval set not found at {eval_path}. Run: python3 scripts/bootstrap_eval_set.py"
        )

    df = pd.read_parquet(eval_path)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)

    rows = []
    for _, row in df.iterrows():
        denial_letter = imr_row_to_denial_letter(row)
        patient_context = imr_row_to_patient_context(row)
        rows.append(
            {
                "denial_letter": denial_letter,
                "patient_context": patient_context,
                "ground_truth_overturned": bool(row.get("was_overturned", False)),
            }
        )

    return weave.Dataset(name="imr-eval-sample", rows=rows)


async def run_evaluations(n_samples: int = 25):
    print("=" * 80)
    print("DENIAL DEFENSE - COMPARATIVE EVALUATION")
    print(f"Dataset: n={n_samples}, aligned denial-letter format")
    print("=" * 80)

    ds = build_dataset(n_samples=n_samples)
    scorers = [
        survives_attack_scorer,
        cites_specific_evidence_scorer,
        invokes_federal_protections_scorer,
        addresses_denial_reason_scorer,
        latency_scorer,
    ]

    baseline_eval = weave.Evaluation(
        dataset=ds,
        scorers=scorers,
        name=f"baseline_single_agent_n{n_samples}",
    )
    harness_eval = weave.Evaluation(
        dataset=ds,
        scorers=scorers,
        name=f"multi_agent_harness_n{n_samples}",
    )

    baseline_summary = await baseline_eval.evaluate(baseline_system)
    print(f"Baseline summary: {baseline_summary}")

    harness_summary = await harness_eval.evaluate(harness_system)
    print(f"Harness summary: {harness_summary}")

    print(f"\nBoth evals complete — n={n_samples}, {len(scorers)} scorers")


def test_wandb_inference() -> bool:
    try:
        resp = wandb_scorer_client.chat.completions.create(
            model=SCORER_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": 'Return JSON: {"status": "ok"}'}],
            response_format={"type": "json_object"},
            timeout=30,
        )
        print(f"W&B Inference reachable: {resp.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"W&B Inference connection failed: {e}")
        return False


def main() -> int:
    n = int(os.getenv("EVAL_N_SAMPLES", "25"))
    try:
        asyncio.run(run_evaluations(n_samples=n))
        return 0
    except KeyboardInterrupt:
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if os.getenv("SKIP_WANDB_TEST") != "1" and not test_wandb_inference():
        print("ABORT: W&B Inference not reachable. Set SKIP_WANDB_TEST=1 to skip.")
        exit(1)
    exit(main())
