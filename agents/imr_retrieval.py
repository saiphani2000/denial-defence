"""Retrieve similar overturned CA DMHC IMR cases for the Precedent agent."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from agents.config import get_settings
from agents.logging_config import log

ROOT = Path(__file__).parent.parent
PREVIEW_CSV = ROOT / "data" / "processed" / "eval_set_imr_preview.csv"


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if len(t) > 2}


def _load_imr_frame() -> pd.DataFrame:
    settings = get_settings()
    paths = [
        settings.eval_parquet_path,
        settings.imr_data_path,
        PREVIEW_CSV,
    ]
    for path in paths:
        if not Path(path).exists():
            continue
        try:
            if str(path).endswith(".parquet"):
                df = pd.read_parquet(path)
            elif str(path).endswith(".csv"):
                df = pd.read_csv(path, low_memory=False)
            else:
                continue
            log.info("IMR retrieval loaded %s rows from %s", len(df), path)
            return df
        except Exception as exc:
            log.warning("Failed to load IMR data from %s: %s", path, exc)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _cached_frame() -> pd.DataFrame:
    return _load_imr_frame()


def _score_row(row: pd.Series, query_tokens: set[str]) -> float:
    parts = [
        str(row.get("DiagnosisCategory", "")),
        str(row.get("DiagnosisSubCategory", "")),
        str(row.get("TreatmentCategory", "")),
        str(row.get("TreatmentSubCategory", "")),
        str(row.get("Type", "")),
        str(row.get("Findings", ""))[:500],
    ]
    doc_tokens = _tokenize(" ".join(parts))
    if not doc_tokens or not query_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / max(len(query_tokens), 1)


def retrieve_similar_cases(
    denial_letter: str,
    patient_context: str,
    *,
    top_k: int = 3,
    prefer_overturned: bool = True,
) -> list[dict[str, Any]]:
    """Return top-k IMR cases similar to the denial, preferring overturned outcomes."""
    df = _cached_frame()
    if df.empty:
        return []

    query = f"{denial_letter}\n{patient_context}"
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    work = df.copy()
    if prefer_overturned and "Determination" in work.columns:
        overturn_mask = work["Determination"].astype(str).str.contains("Overturned", case=False, na=False)
        overturned = work[overturn_mask]
        if len(overturned) >= top_k:
            work = overturned

    work = work.copy()
    work["_score"] = work.apply(lambda row: _score_row(row, query_tokens), axis=1)
    work = work.sort_values("_score", ascending=False).head(top_k)

    results: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        if row["_score"] <= 0:
            continue
        ref = str(row.get("ReferenceID", "unknown"))
        determination = str(row.get("Determination", ""))
        findings = str(row.get("Findings", ""))[:800]
        results.append(
            {
                "reference_id": ref,
                "report_year": row.get("ReportYear"),
                "diagnosis": f"{row.get('DiagnosisCategory', '')} — {row.get('DiagnosisSubCategory', '')}",
                "treatment": f"{row.get('TreatmentCategory', '')} — {row.get('TreatmentSubCategory', '')}",
                "determination": determination,
                "type": str(row.get("Type", "")),
                "reviewer_excerpt": findings,
                "similarity_score": round(float(row["_score"]), 3),
            }
        )
    return results


def format_precedent_context(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return (
            "No indexed IMR cases matched locally. Reference general CA DMHC IMR overturn "
            "patterns (~50-60% for well-documented medical necessity denials) without inventing ReferenceIDs."
        )
    lines = ["Retrieved CA DMHC IMR precedents (use these ReferenceIDs when citing):"]
    for case in cases:
        lines.append(
            f"- {case['reference_id']} ({case['report_year']}): {case['diagnosis']} | "
            f"{case['treatment']} → {case['determination']}\n"
            f"  Excerpt: {case['reviewer_excerpt'][:300]}..."
        )
    return "\n".join(lines)
