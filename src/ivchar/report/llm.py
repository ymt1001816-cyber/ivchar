"""Optional AI-written anomaly summary (Claude via the Anthropic SDK).

Only aggregated statistics, yield and the flagged-outlier table are sent, never
raw curves. The summary is labelled as AI-generated in the report, and the
statistics it describes are computed deterministically by ivchar itself.

Requires ``pip install 'ivchar[llm]'`` and Anthropic credentials
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
"""

from __future__ import annotations

import json

import numpy as np

from ivchar.analysis.batch import Analysis

MODEL = "claude-opus-5-5"

SYSTEM = """You are a device characterization engineer reviewing an automated I-V \
characterization run. Write a short anomaly summary for the report: at most 6 bullet \
points, plain text, each starting with "- ". Cover yield loss and its causes, flagged \
outlier dies with the parameters that flagged them and the most likely physical \
mechanism, and any corner or temperature trend that looks unusual. Cite die IDs, \
corners and numbers from the data. Use only the data given; if something cannot be \
concluded from it, say so. If the data is marked simulated, say so in the first bullet."""


def build_payload(analysis: Analysis, simulated: bool, max_outliers: int = 40) -> dict:
    p = analysis.params
    cols = ["vth_cc_sat", "vth_gm_lin", "ss_sat", "ion", "ioff", "dibl", "ig_max"]
    medians = (p[p.outlier == False].groupby(["corner", "temp_c"])[cols].median()  # noqa: E712
               .reset_index().round(6))
    outliers = p[p.outlier][["corner", "die", "temp_c", "outlier_reason"] + cols].head(max_outliers)
    fails = analysis.spec_rows
    fails = fails[~fails["pass"]][["corner", "die", "fail_reason"]] if len(fails) else fails
    return {
        "data_source": "SIMULATED (compact model, not measured silicon)" if simulated else "measured",
        "units": {"vth": "V", "ss": "mV/dec", "ion/ioff/ig": "A", "dibl": "mV/V"},
        "spec_temp_c": analysis.spec_temp_c,
        "spec_limits": analysis.spec,
        "yield_by_corner": analysis.yields.reset_index().to_dict("records") if len(analysis.yields) else [],
        "spec_failures": fails.to_dict("records") if len(fails) else [],
        "median_by_corner_temp": medians.to_dict("records"),
        "flagged_outliers": outliers.replace({np.nan: None}).round(9).to_dict("records"),
    }


def summarize_anomalies(analysis: Analysis, simulated: bool, model: str = MODEL) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("AI summary needs the Anthropic SDK: pip install 'ivchar[llm]'") from exc

    client = anthropic.Anthropic()
    payload = json.dumps(build_payload(analysis, simulated), default=float)
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "low"},
        # Server-side fallback: if the model declines, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"Characterization results (JSON):\n{payload}"}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to summarize this data")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise RuntimeError(f"empty response (stop_reason={response.stop_reason})")
    return text
