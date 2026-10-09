"""The single prompt template used for SFT, DPO, and every generation run.

Keeping it in one place guarantees training and evaluation see the exact same
format (the manual requires this, and the report must quote it).
"""
from __future__ import annotations

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Wrap a raw user instruction in the template. The response follows directly."""
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())
