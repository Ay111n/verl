"""CPU tests for shared answer verification; no model inference."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvr.answer_verifier import verify_answer, VERIFIER_VERSION
from rlvr.reward import score_response
PROMPT = "<|im_start|>assistant\n<think>\n"

def run(text, gold="42", domain="math", finish="stop", policy=None, prompt=PROMPT):
    sample = {"domain": domain, "canonical_answer": gold,
              "option_labels": list("ABCDEFGHIJ"), "numeric_policy": policy or {}}
    return verify_answer(sample, prompt=prompt, completion="Reasoning.</think>" + text,
                         finish_reason=finish, output_tokens=20, max_new_tokens=8192)

@pytest.mark.parametrize("answer,gold", [
    ("42", "42"), ("-42", "-42"), ("+42", "42"),
    ("0.50", "1/2"), ("1/2", "0.5"), ("-2/4", "-0.5"),
    (r"\frac{1}{2}", "0.5"), (r"-\frac{1}{2}", "-0.5"),
    (r"\boxed{\frac{1}{2}}", "0.5"), (r"\boxed{-42}", "-42"),
    ("1e3", "1000"), ("1.25e-2", "1/80"), ("1,000", "1000"),
    ("༤༢", "42"), ("−༢/༤", "-0.5"), ("４２", "42"),
    ("ལན་ནི་ 42 ཡིན།", "42"),
    ("Final answer: 42. Checked in 3 steps.", "42"),
    ("Final answer:\n42", "42"), ("1/2 = 0.5", "0.5"),
    (r"The answer is \(\frac{1}{2}\).", "0.5"),
])
def test_exact_numeric_forms(answer, gold):
    result = run(answer, gold)
    assert result["correctness"] == 1, result
    assert result["pair_eligible"] is True

@pytest.mark.parametrize("answer,gold", [
    ("A","A"),("(B)","B"),("C.","C"),("**D**","D"),("Answer: E","E"),
    ("The correct answer is F.","F"),("ལན་ནི་ G ཡིན།","G"),
    ("Ｈ","H"),(r"\boxed{I}","I"),("(j)","J")])
def test_choice_forms(answer,gold):
    assert run(answer, gold, "knowledge")["correctness"] == 1

@pytest.mark.parametrize("answer", ["A or B", "Answer: A or B", "A/B", "A, B",
    "Final answer: A. Final answer: B.", "A B C D", "Answer: A, B"])
def test_conflicting_choices_never_match_gold(answer):
    for gold in ("A", "B"):
        result = run(answer, gold, "knowledge")
        assert result["correctness"] == 0, result
        assert result["parse_status"] == "ambiguous"
        assert not result["pair_eligible"]

@pytest.mark.parametrize("answer", ["1 or 2", "Final answer: 1 or 2", "1, 2",
    "Answer: 1. Answer: 2.", r"\boxed{1} or \boxed{2}"])
def test_conflicting_numeric_values(answer):
    result = run(answer, "2")
    assert result["parse_status"] == "ambiguous"
    assert result["correctness"] == 0

def test_explicit_final_dominates_discussion_not_last_number():
    assert run("Final answer: 42. Checked in 3 steps.", "3")["correctness"] == 0
    assert run("A is unsuitable. Final answer: B.", "B", "knowledge")["correctness"] == 1
    assert run("Answer: B (vitamin A)", "B", "knowledge")["correctness"] == 1
    assert run("It is a tree.", "A", "knowledge")["correctness"] == 0
    assert run("Answer is a tree.", "A", "knowledge")["correctness"] == 0

@pytest.mark.parametrize("answer", ["1/0", "NaN", "inf", r"\sqrt{2}", "1e999999", "12,34"])
def test_invalid_numeric_forms_are_not_valid_pairs(answer):
    result = run(answer, "1")
    assert result["correctness"] == 0
    assert not result["pair_eligible"]

def test_exact_not_approximate_and_percent_policy():
    assert run("0.3333", "1/3")["correctness"] == 0
    assert run("20%", "20")["parse_status"] == "unsupported_numeric"
    assert run("20%", "20", policy={"percent_mode":"literal"})["correctness"] == 1
    assert run("20%", "0.2", policy={"percent_mode":"ratio"})["correctness"] == 1

def test_final_region_and_termination_are_separate():
    sample={"domain":"math","canonical_answer":"42"}
    no_final=verify_answer(sample,prompt=PROMPT,completion="Reasoning contains 42",
                          finish_reason="length",output_tokens=8192,max_new_tokens=8192)
    assert no_final["parse_status"]=="no_final" and no_final["correctness"]==0
    truncated=run("42",finish="length")
    assert truncated["correctness"]==1 and not truncated["pair_eligible"]
    direct=verify_answer(sample,prompt="assistant",completion="42",finish_reason="stop")
    assert direct["correctness"]==1 and not direct["protocol_valid"] and direct["pair_eligible"]
    at_cap=verify_answer(sample,prompt="assistant",completion="42",finish_reason="stop",
                         output_tokens=8192,max_new_tokens=8192)
    assert not at_cap["truncated"]
    for finish in (None,"abort","unrecognized"):
        result=run("42",finish=finish)
        assert result["correctness"] is None and not result["pair_eligible"]

def test_invalid_gold_is_data_error():
    with pytest.raises(ValueError):
        verify_answer({"domain":"knowledge","canonical_answer":"J","option_labels":["A","B"]},
                      prompt="assistant",completion="J",finish_reason="stop")
    with pytest.raises(ValueError):
        verify_answer({"domain":"math","canonical_answer":"1/0"},
                      prompt="assistant",completion="1",finish_reason="stop")

@pytest.mark.parametrize("domain,text,gold", [
    ("math","42","42"),("math","Final answer: 42. Checked in 3 steps.","3"),
    ("math",r"\frac{1}{2}","0.5"),("knowledge","A or B","B"),
    ("knowledge","Answer: B","B")])
def test_probe_and_training_reward_use_same_verifier(domain,text,gold):
    direct=run(text,gold,domain)
    score=score_response(prompt=PROMPT,completion="Reasoning.</think>"+text,
       domain=domain,gold_answer=gold,option_labels=list("ABCDEFGHIJ"),
       finish_reason="stop",output_tokens=20,max_response_length=8192,
       verifier_version=VERIFIER_VERSION)
    info=score["reward_extra_info"]
    assert score["reward_score"]==direct["correctness"]
    for key in ("verifier_version","parse_status","normalized_answer","pair_eligible"):
        assert info.get(key)==direct.get(key)

def test_v2_reward_refuses_missing_native_termination_and_wrong_version():
    kwargs=dict(prompt=PROMPT,completion="work</think>42",domain="math",
                gold_answer="42",output_tokens=20,verifier_version=VERIFIER_VERSION)
    with pytest.raises(ValueError,match="execution"):
        score_response(**kwargs)
    kwargs["finish_reason"]="stop"
    kwargs["verifier_version"]="unknown"
    with pytest.raises(ValueError,match="verifier"):
        score_response(**kwargs)

def test_tibetan_adjacency_and_symbolic_numeric_rejection():
    assert run("ལན་ནི་༤༢ཡིན།", "42")["correctness"] == 1
    for text in ("2*2", "2π", r"2\\pi", "2^2"):
        result=run(text, "2")
        assert result["correctness"] == 0 and not result["pair_eligible"]

@pytest.mark.parametrize("answer,gold", [
    ("A and C are incorrect. མཐའ་མའི་ལན། H", "H"),
    ("A and B are incorrect. **ལན་དག:**\n**C. Explanation**", "C"),
    ("A is unsuitable. ཡང་དག་པའི་འདེམས་གཞི་ནི་ **F** ཡིན།", "F"),
    ("Option A is unsuitable. Final answer: B.", "B"),
    ("མཐའ་མའི་ལན།\n**H**", "H"),
])
def test_tibetan_answer_heading_and_explicit_conclusion(answer, gold):
    result=run(answer,gold,"knowledge")
    assert result["correctness"]==1, result
    assert result["pair_eligible"]

def test_tibetan_headings_do_not_hide_contradictory_answers():
    result=run("མཐའ་མའི་ལན། A\nམཐའ་མའི་ལན། B","B","knowledge")
    assert result["parse_status"]=="ambiguous"
    assert result["correctness"]==0

def test_tibetan_numeric_heading_with_shad():
    assert run("མཐའ་མའི་ལན། ༤༢","42")["correctness"]==1

@pytest.mark.parametrize("answer,gold", [
    ("A and E are unsuitable.\n\nF", "F"),
    ("A and B are unsuitable.\n\nC. 26-30%", "C"),
    ("The formula contains C and H.\n\nB. Description", "B"),
])
def test_terminal_option_label_after_explanation(answer,gold):
    assert run(answer,gold,"knowledge")["correctness"]==1

def test_terminal_label_cannot_override_explicit_conflicting_answer():
    result=run("Final answer: A.\n\nB","B","knowledge")
    assert result["parse_status"]=="ambiguous" and result["correctness"]==0

@pytest.mark.parametrize("answer,gold", [
    (r"Work: \frac{\sum xy}{\sum x^2}"+"\n\n16", "16"),
    ("Work: 60,000 + 5,000 + 40 + 3\n\n65043", "65043"),
    (r"Work: \frac{\sum xy}{\sum x^2}"+"\n\n"+r"དེས་ན། \(b\) ཡི་རིན་ཐང་ནི་ \(0.28\) ཡིན།", "0.28"),
    (r"Work: \frac{\sum xy}{\sum x^2}"+"\n\n"+r"དེ་ལྟར་\(b = 0.28\)ཡིན།", "0.28"),
])
def test_terminal_numeric_claim_ignores_working_formula(answer,gold):
    result=run(answer,gold)
    assert result["correctness"]==1, result

def test_numeric_terminal_does_not_select_an_incidental_last_number():
    result=run("Working value: 42.\nChecked in 3 steps.","3")
    assert result["correctness"]==0 and result["parse_status"]=="ambiguous"
    result=run("Final answer: 42.\n43","43")
    assert result["correctness"]==0 and result["parse_status"]=="ambiguous"
