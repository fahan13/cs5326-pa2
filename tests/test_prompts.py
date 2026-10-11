"""The generation stop rule: everything from a new '### Instruction:' onward is dropped."""
from alignment.prompts import STOP_MARKER, clean_generation, format_prompt


def test_answer_is_cut_where_the_model_invents_a_new_turn():
    text = "Paris is the capital.\n\n### Instruction:\nWhat about Spain?\n\n### Response:\nMadrid."
    assert clean_generation(text) == ("Paris is the capital.", True)


def test_text_without_marker_is_only_stripped():
    assert clean_generation("  Just an answer.\n") == ("Just an answer.", False)


def test_template_starts_with_the_stop_marker():
    # If these drifted apart, the stop rule would silently stop working.
    assert format_prompt("x").startswith(STOP_MARKER)
