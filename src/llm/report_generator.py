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
