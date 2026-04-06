"""
claim_verifier.py
-----------------
Extracts atomic quantitative claims from LLM-generated EEG reports and
verifies each claim against ground-truth features extracted from the signal.

This is the core of the hallucination detection pipeline in NeuroScribe.
In the final Evidence Verification Agent, these same functions are used to
ground every claim in the report against the extracted feature JSON.

Typical usage:
    from src.llm.claim_verifier import extract_claims, verify_claim

    claims = extract_claims(report_text)
    for claim in claims:
        verdict, gt_value, explanation = verify_claim(claim, feat)
        # verdict: 'VERIFIED' | 'HALLUCINATED' | 'UNVERIFIABLE'
"""

import re
from typing import Optional


def extract_claims(report_text: str) -> list[dict]:
    """
    Parses atomic quantitative claims from a clinical EEG report using regex.

    Detects four claim categories:
        frequency  — e.g. "3 Hz", "4-6 Hz"
        amplitude  — e.g. "150 µV", "200-300 uV"
        duration   — e.g. "40 seconds", "30-sec"
        channel    — e.g. "F7", "T3", "C3-P3"

    Args:
        report_text: Raw LLM report string.

    Returns:
        List of claim dicts with keys:
            category    — one of 'frequency', 'amplitude', 'duration', 'channel'
            claim_text  — the matched substring from the report
            value_lo    — lower bound of the claimed value (None for channels)
            value_hi    — upper bound (same as value_lo for point estimates)
            unit        — 'Hz', 'µV', 'seconds', or 'channel'
    """
    claims = []

        # Number pattern: requires at least one digit (won't match bare ".")
    _NUM = r"\d+(?:\.\d+)?"

    for m in re.finditer(rf"({_NUM})\s*[-\u2013]?\s*({_NUM})?\s*[Hh]z", report_text):
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        claims.append({
            "category":   "frequency",
            "claim_text": m.group(0).strip(),
            "value_lo":   lo,
            "value_hi":   hi,
            "unit":       "Hz",
        })

    # ── Amplitude: "150 µV", "200-300 µV", "150uV" ────────────────────────
    for m in re.finditer(rf"({_NUM})\s*[-\u2013]?\s*({_NUM})?\s*[µu][Vv]", report_text):
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        claims.append({
            "category":   "amplitude",
            "claim_text": m.group(0).strip(),
            "value_lo":   lo,
            "value_hi":   hi,
            "unit":       "µV",
        })

    # ── Duration: "30 seconds", "45-second" ───────────────────────────────
    for m in re.finditer(rf"({_NUM})\s*[-\u2013]?\s*({_NUM})?\s*[Ss]ec", report_text):
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        claims.append({
            "category":   "duration",
            "claim_text": m.group(0).strip(),
            "value_lo":   lo,
            "value_hi":   hi,
            "unit":       "seconds",
        })
        
    # ── Channel names: "F3", "T3-T4", "F7" ───────────────────────────────
    eeg_ch_pattern = r"\b([FfCcPpOoTt][\d]+(?:[-][FfCcPpOoTt][\d]+)?)\b"
    for m in re.finditer(eeg_ch_pattern, report_text):
        claims.append({
            "category":   "channel",
            "claim_text": m.group(0),
            "value_lo":   None,
            "value_hi":   None,
            "unit":       "channel",
        })

    return claims


def verify_claim(
    claim: dict,
    feat: dict,
    tolerances: Optional[dict] = None,
) -> tuple[str, str, str]:
    """
    Checks one extracted claim against ground-truth features.

    Tolerances:
        frequency  — ±1 Hz for point estimates; range must contain GT value
        amplitude  — ±10% of GT RMS (minimum ±20 µV)
        duration   — ±5 seconds
        channel    — must appear in GT top-3 active channels

    Args:
        claim:       Claim dict from extract_claims().
        feat:        Feature dict from src.features.extractor.extract_features().
        tolerances:  Override default tolerance dict.

    Returns:
        (verdict, gt_value_str, explanation_str)
        verdict is one of: 'VERIFIED', 'HALLUCINATED', 'UNVERIFIABLE'
    """
    if tolerances is None:
        tolerances = {"timing": 5.0, "amplitude_pct": 0.10}

    cat = claim["category"]
    lo  = claim["value_lo"]
    hi  = claim["value_hi"]

    if cat == "frequency":
        gt = feat["frequency"]["dominant_hz"]
        if lo != hi:   # range claim
            verdict = "VERIFIED" if lo <= gt <= hi else "HALLUCINATED"
        else:          # point claim
            verdict = "VERIFIED" if abs(lo - gt) <= 1.0 else "HALLUCINATED"
        return verdict, f"{gt} Hz", f"GT dominant freq = {gt} Hz"

    elif cat == "amplitude":
        gt  = feat["amplitude"]["rms_uV"]
        tol = max(gt * tolerances["amplitude_pct"], 20.0)
        if lo is not None:
            verdict = "VERIFIED" if abs(lo - gt) <= tol else "HALLUCINATED"
        else:
            verdict = "UNVERIFIABLE"
        return verdict, f"{gt:.1f} µV (RMS)", f"GT RMS = {gt:.1f} µV  (±10% tol)"

    elif cat == "duration":
        gt  = feat["temporal"]["duration_sec"]
        tol = tolerances["timing"]
        if lo is not None:
            verdict = "VERIFIED" if abs(lo - gt) <= tol else "HALLUCINATED"
        else:
            verdict = "UNVERIFIABLE"
        return verdict, f"{gt} s", f"GT duration = {gt} s  (±{tol}s tol)"

    elif cat == "channel":
        gt_channels = [c.upper().replace(" ", "") for c in feat["spatial"]["top3_channels"]]
        claimed     = claim["claim_text"].upper().replace(" ", "")
        verdict     = "VERIFIED" if claimed in gt_channels else "HALLUCINATED"
        return verdict, str(feat["spatial"]["top3_channels"]), \
               f"GT top-3 = {feat['spatial']['top3_channels']}"

    return "UNVERIFIABLE", "N/A", "Category not supported"


# ── LLM-as-a-Judge: GPT-4o audits and corrects the report ─────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a senior clinical neurologist and EEG expert acting as an "
    "independent auditor. Your job is to verify a clinical EEG report written "
    "by a junior AI assistant and correct any hallucinated or inaccurate values."
)

JUDGE_USER_PROMPT = (
    "Below is an EEG clinical report generated by an AI assistant, followed by "
    "the verified ground-truth signal features extracted directly from the EEG signal.\n\n"
    "=== AI-GENERATED REPORT ===\n"
    "{report}\n\n"
    "=== GROUND-TRUTH SIGNAL FEATURES ===\n"
    "Seizure onset   : {onset:.1f} s\n"
    "Seizure offset  : {offset:.1f} s\n"
    "Duration        : {duration:.1f} s\n"
    "Dominant freq   : {dominant_hz:.1f} Hz\n"
    "RMS amplitude   : {rms_uV:.1f} µV\n"
    "Peak amplitude  : {max_uV:.1f} µV\n"
    "Most active ch  : {most_active}\n"
    "Top-3 channels  : {top3}\n"
    "=====================================\n\n"
    "Your tasks:\n"
    "1. AUDIT: List every quantitative claim in the report (frequency Hz, "
    "amplitude µV, duration seconds, channel names) and mark each as "
    "VERIFIED or HALLUCINATED with the correct ground-truth value.\n\n"
    "2. CORRECT: Rewrite the full report replacing all hallucinated values "
    "with the correct ground-truth values. Keep the clinical language intact.\n\n"
    "Format your response as:\n"
    "## AUDIT\n"
    "<table of claims>\n\n"
    "## CORRECTED REPORT\n"
    "<full corrected report>"
)


def llm_judge_and_correct(client, report: str, feat: dict) -> dict:
    """
    Uses GPT-4o as an independent judge to audit and correct an AI-generated
    EEG report against ground-truth signal features (LLM-as-a-Judge framework).

    Args:
        client: OpenAI client instance (openai.OpenAI).
        report: The AI-generated report text to audit.
        feat:   Feature dict from src.features.extractor.extract_features().

    Returns:
        dict with keys:
            audit_and_correction  — full GPT-4o response (audit table + corrected report)
            corrected_report      — extracted corrected report section
            audit_section         — extracted audit section
    """
    user_msg = JUDGE_USER_PROMPT.format(
        report=report,
        onset=feat["temporal"]["onset_sec"],
        offset=feat["temporal"]["offset_sec"],
        duration=feat["temporal"]["duration_sec"],
        dominant_hz=feat["frequency"]["dominant_hz"],
        rms_uV=feat["amplitude"]["rms_uV"],
        max_uV=feat["amplitude"]["max_uV"],
        most_active=feat["spatial"]["most_active"],
        top3=", ".join(feat["spatial"]["top3_channels"]),
    )
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=1,
        max_completion_tokens=1200,
    )
    full_response = response.choices[0].message.content

    # Split into audit and corrected report sections
    audit_section     = ""
    corrected_report  = ""
    if "## CORRECTED REPORT" in full_response:
        parts = full_response.split("## CORRECTED REPORT", 1)
        audit_section    = parts[0].replace("## AUDIT", "").strip()
        corrected_report = parts[1].strip()
    else:
        audit_section    = full_response
        corrected_report = full_response

    return {
        "audit_and_correction": full_response,
        "audit_section":        audit_section,
        "corrected_report":     corrected_report,
    }
