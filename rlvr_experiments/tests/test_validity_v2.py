import pytest
from rlvr.reward import score_response

def score(text, **kw):
    return score_response(prompt="<|im_start|>assistant\n<think>\n", completion=text,
                          domain="math", gold_answer="42", profile="R0", **kw)

def test_no_termination_evidence_is_rejected():
    with pytest.raises(ValueError, match="Termination validity"):
        score("work</think>42")

@pytest.mark.parametrize("reason", ["abort", "unknown", "error", ""])
def test_unknown_native_reason_is_rejected(reason):
    with pytest.raises(ValueError, match="Unresolved termination"):
        score("</think>42", finish_reason=reason, output_tokens=8)

@pytest.mark.parametrize("text,n,reason,invalid", [
    ("work</think>42", 16, "stop", False),
    ("work", 4096, None, True),
    ("</think> ", 8, "stop", True),
    ("</think>words", 8, "stop", True),
    ("</think>42", 4096, "stop", False),
    ("</think>42", 4096, "length", True),
    ("work</think><think>unfinished", 20, "stop", True),
])
def test_resolved_fields_are_diagnostics_only(text,n,reason,invalid):
    r=score(text,output_tokens=n,finish_reason=reason);i=r['reward_extra_info']
    for k in ['thinking_closed','thinking_unclosed','final_present','no_reliable_final',
              'answer_extractable','empty_final','extraction_failed','truncated']:
        assert type(i[k]) is bool
    assert i['final_present'] == (not i['no_reliable_final'])
    assert i['validity_invalid'] == invalid
    assert 'component_r1' not in i
    assert i['validity_penalty_raw'] == int(invalid)
    assert r['reward_score'] == i['total_reward'] == float(i['correctness'])

def test_r0_correctness_unchanged_by_validity():
    args=dict(prompt="assistant",completion="<think>x</think>42",domain="math",gold_answer="42",
              output_tokens=4096,finish_reason="length")
    assert score_response(**args,profile="R0")['reward_score']==1
    assert score_response(**args,profile="R0")['reward_score']==1.0

def test_stray_close_cannot_close_later_open():
    i=score_response(prompt='<|im_start|>assistant\n',completion='</think><think>unfinished',
        domain='math',gold_answer='42',profile='R0',output_tokens=12)['reward_extra_info']
    assert i['thinking_unclosed'] is True and i['thinking_closed'] is False
