"""Shared helpers for evaluation dataset formatting."""

from __future__ import annotations

from typing import Any


def imr_row_to_denial_letter(row: Any) -> str:
    """Synthesize an insurer-style denial letter from IMR metadata (not reviewer findings)."""
    diagnosis = f"{row.get('DiagnosisCategory', 'Unknown')} — {row.get('DiagnosisSubCategory', 'N/A')}"
    treatment = f"{row.get('TreatmentCategory', 'Unknown')} — {row.get('TreatmentSubCategory', 'N/A')}"
    imr_type = row.get("Type", "Medical Necessity")

    return f"""NOTICE OF ADVERSE BENEFIT DETERMINATION

We have reviewed your request for coverage of the following service:

Diagnosis: {diagnosis}
Requested treatment/service: {treatment}
Review type: {imr_type}

Based on our medical policy and the clinical information submitted, your request has been DENIED.

Reason for denial: The requested service is not medically necessary for your condition under our coverage criteria. The documentation provided does not demonstrate that you meet the policy requirements for this treatment, including appropriate step therapy, clinical findings, and level-of-care justification.

Reference: Internal medical policy review — CARC 50 (not medically necessary).

You have the right to appeal this decision within the timeframe stated in your plan documents."""


def imr_row_to_patient_context(row: Any) -> str:
    return f"""Report Year: {row.get('ReportYear', 'N/A')}
Diagnosis: {row.get('DiagnosisCategory', '')} — {row.get('DiagnosisSubCategory', 'N/A')}
Treatment requested: {row.get('TreatmentCategory', '')} — {row.get('TreatmentSubCategory', 'N/A')}
Patient age range: {row.get('AgeRange', 'N/A')}
Patient gender: {row.get('PatientGender', 'N/A')}
IMR outcome (ground truth): {row.get('Determination', 'N/A')}"""
