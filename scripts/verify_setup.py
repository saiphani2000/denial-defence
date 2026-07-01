"""
Quick verification script - checks structure without calling APIs.
Run this before adding API keys to verify everything imports correctly.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DISABLE_WEAVE", "true")

print("=" * 80)
print("DENIAL DEFENSE - PRE-FLIGHT CHECK")
print("=" * 80)

errors = []
warnings = []

required_files = [
    "agents/config.py",
    "agents/llm.py",
    "agents/imr_retrieval.py",
    "agents/prompts.py",
    "agents/baseline.py",
    "agents/harness.py",
    "agents/playbook.py",
    "web/app.py",
    "web/templates/index.html",
    "eval/compare_eval.py",
    "data/demo/case_01_oscar_cromolyn.json",
    "data/processed/denial_playbook.json",
    "data/processed/eval_set_imr.parquet",
    "LICENSE",
    ".env.example",
    "requirements.txt",
    "Dockerfile",
]

for f in required_files:
    if not Path(f).exists():
        errors.append(f"Missing file: {f}")
    else:
        print(f"  ✓ {f}")

for pkg in ("openai", "langgraph", "pydantic", "flask", "pandas", "tenacity"):
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError as e:
        errors.append(f"Missing package: {pkg} ({e})")

try:
    from agents import prompts, playbook, schemas, imr_retrieval

    print("  ✓ agent modules import")
except Exception as e:
    errors.append(f"Agent import failed: {e}")

if not Path(".env").exists():
    warnings.append(".env not found — copy .env.example")
if not os.getenv("OPENAI_API_KEY"):
    warnings.append("OPENAI_API_KEY not set")

print()
if errors:
    print("ERRORS:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)

if warnings:
    print("WARNINGS:")
    for w in warnings:
        print(f"  - {w}")

print("\n✅ PRE-FLIGHT CHECK PASSED")
print("Next: cp .env.example .env && python3 web/app.py")
