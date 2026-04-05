"""
report_generator.py
-------------------
Generates LLM-based EEG clinical reports via the OpenAI API.

Baseline 2 (unverified): calls GPT-4o-mini with only the patient ID and
"seizure detected" — no extracted signal features are provided. This is
used to demonstrate that LLMs hallucinate specific clinical details
(frequency, amplitude, channels, duration) when not grounded in evidence.

Typical usage:
    from openai import OpenAI
    from src.llm.report_generator import generate_unverified_report

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    report = generate_unverified_report(client, feat)
"""

UNVERIFIED_PROMPT = (
    "You are a clinical neurologist. Write a formal EEG clinical report "
    "for the following seizure event.\n\n"
    "Patient: {patient}\n"
    "Recording: {file}\n"
    "Seizure detected: Yes\n"
    "Approximate time in recording: {onset:.0f} seconds\n\n"
    "Generate a complete clinical EEG report with the following sections:\n"
    "1. BACKGROUND ACTIVITY\n"
    "2. ICTAL FINDINGS\n"
    "3. IMPRESSION\n"
    "4. CLINICAL CORRELATION\n\n"
    "Include specific details such as frequency (Hz), amplitude (µV), "
    "involved channels, and seizure duration."
)


GROUNDED_PROMPT = (
    "You are a clinical neurologist. Write a formal EEG clinical report "
    "for the following seizure event. You MUST use ONLY the exact numerical "
    "values provided below — do not invent or estimate any values.\n\n"
    "Patient: {patient}\n"
    "Recording: {file}\n\n"
    "=== GROUND-TRUTH SIGNAL FEATURES ===\n"
    "Seizure onset   : {onset:.1f} s\n"
    "Seizure offset  : {offset:.1f} s\n"
    "Duration        : {duration:.1f} s\n"
    "Dominant freq   : {dominant_hz:.1f} Hz\n"
    "RMS amplitude   : {rms_uV:.1f} µV\n"
    "Peak amplitude  : {max_uV:.1f} µV\n"
    "Most active ch  : {most_active}\n"
    "Top-3 channels  : {top3}\n"
    "Delta power     : {delta:.3f}\n"
    "Theta power     : {theta:.3f}\n"
    "Alpha power     : {alpha:.3f}\n"
    "Beta power      : {beta:.3f}\n"
    "Gamma power     : {gamma:.3f}\n"
    "=====================================\n\n"
    "Generate a complete clinical EEG report with:\n"
    "1. BACKGROUND ACTIVITY\n"
    "2. ICTAL FINDINGS\n"
    "3. IMPRESSION\n"
    "4. CLINICAL CORRELATION\n\n"
    "Every quantitative claim (Hz, µV, seconds, channels) MUST match "
    "the ground-truth values above exactly."
)


def generate_grounded_report(client, feat: dict) -> str:
    """
    Generates a clinical EEG report WITH extracted signal features injected
    into the prompt — the LLM is constrained to use only these exact values.

    This is the evidence-grounded approach that prevents hallucination.

    Args:
        client: OpenAI client instance (openai.OpenAI).
        feat:   Feature dict returned by src.features.extractor.extract_features.

    Returns:
        Report text as a string.
    """
    freq = feat["frequency"]
    prompt = GROUNDED_PROMPT.format(
        patient=feat["patient"],
        file=feat["file"],
        onset=feat["temporal"]["onset_sec"],
        offset=feat["temporal"]["offset_sec"],
        duration=feat["temporal"]["duration_sec"],
        dominant_hz=freq["dominant_hz"],
        rms_uV=feat["amplitude"]["rms_uV"],
        max_uV=feat["amplitude"]["max_uV"],
        most_active=feat["spatial"]["most_active"],
        top3=", ".join(feat["spatial"]["top3_channels"]),
        delta=freq["delta_power"],
        theta=freq["theta_power"],
        alpha=freq["alpha_power"],
        beta=freq["beta_power"],
        gamma=freq["gamma_power"],
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=1200,
    )
    return response.choices[0].message.content


def generate_unverified_report(client, feat: dict) -> str:
    """
    Generates a clinical EEG report WITHOUT providing any extracted signal
    features — only patient ID and approximate seizure time are given.

    This is Baseline 2: the LLM must invent specific values, which leads to
    hallucinated frequency, amplitude, channel, and duration claims.

    Args:
        client: OpenAI client instance (openai.OpenAI).
        feat:   Feature dict returned by src.features.extractor.extract_features.

    Returns:
        Report text as a string.
    """
    prompt = UNVERIFIED_PROMPT.format(
        patient=feat["patient"],
        file=feat["file"],
        onset=feat["temporal"]["onset_sec"],
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )
    return response.choices[0].message.content
