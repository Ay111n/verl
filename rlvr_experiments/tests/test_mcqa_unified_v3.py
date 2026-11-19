"""The live GRPO adapter and FormatEval use identical native choice semantics."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvr.answer_verifier import ANSWER_EXTRACTION_VERSION, verify_answer
from rlvr.reward import score_response
from eval_multilingual_mcqa_vllm import _extract_output
from mcqa_answer_extraction_multilingual import choice_correct


@pytest.mark.parametrize("width", [4, 10])
@pytest.mark.parametrize("finish", ["stop", "length", "abort"])
def test_native_entrypoint_agreement(width, finish):
    labels = "ABCDEFGHIJ"[:width]
    options = {letter: f"text {letter}" for letter in labels}
    gold = labels[-1]
    sample = dict(domain="knowledge", canonical_answer=gold, options_bo=options)
    raw = f"<think>Answer: A</think>ལན་དག་མ་ནི་ **{gold}** ཡིན།\n* A. rejected"
    evaluate = _extract_output(raw, False, width, options)
    direct = verify_answer(sample, prompt="question", completion=raw, finish_reason=finish)
    assert direct["verifier_version"] == evaluate["answer_extraction_version"] == ANSWER_EXTRACTION_VERSION
    assert direct["extracted_answer"] == evaluate["answer_ability"] == gold
    assert direct["choice_method"] == evaluate["answer_ability_extraction_method"]
    assert direct["correctness"] == choice_correct(gold, gold, finish)
    args = dict(domain="knowledge", gold_answer=gold, option_labels=labels, options_bo=options,
                prompt="question", completion=raw, output_tokens=40, finish_reason=finish)
    if finish == "abort":
        with pytest.raises(ValueError):
            score_response(**args)
    else:
        assert score_response(**args)["reward_score"] == direct["correctness"]


def test_unknown_termination_and_historical_input_metadata():
    args = dict(domain="knowledge", gold_answer="B", option_labels=list("AB"), prompt="", completion="Answer: B")
    with pytest.raises(ValueError):
        score_response(**args, finish_reason=None, output_tokens=2)
    for version in ("rlvr-probe-compatible-v1", "rlvr-answer-v2", ANSWER_EXTRACTION_VERSION):
        info = score_response(**args, finish_reason="stop", verifier_version=version)["reward_extra_info"]
        assert info["verifier_version"] == ANSWER_EXTRACTION_VERSION and info["total_reward"] == 1
    assert not score_response(**dict(args,completion="Answer: C"), finish_reason="stop")["reward_score"]


def test_choice_text_from_prepared_legacy_prompt():
    args = dict(domain="knowledge", gold_answer="A", option_labels=list("ABCD"),
                prompt="question\n\nA. red\nB. blue\nC. green\nD. orange\n\nPlease answer.",
                finish_reason="stop", completion="A. blue")
    result = score_response(**args)
    assert result["reward_score"] == 0
    assert result["reward_extra_info"]["choice_method"] == "letter_option_text_conflict"
