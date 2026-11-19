"""Regression checks for four-response outcome-only groups and distributed slots."""
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "knowledge_grpo_intermittent_nll_v3"))
import contract
import sampling
import train


def test_four_answer_credit_and_batch_denominator(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.AdamW([parameter], lr=1e-5)
    def likelihood(model, prompts, responses, *args, **kwargs):
        token = torch.tensor(responses[0], dtype=torch.float)
        return [parameter * token], [torch.zeros_like(token)]
    def update(opt, params, config, policy):
        return dict(grad_norm=abs(float(params[0].grad)))
    monkeypatch.setattr(train, "response_logps_batch", likelihood)
    monkeypatch.setattr(train, "finish_update", update)
    ids = ["q1", "q2"]
    data = {qid:dict(domain="knowledge", prompt="<think>\n", options_bo=dict(zip("ABCD", "abcd")),
                      canonical_answer="A") for qid in ids}
    # First group has one correct, second group has four correct.
    preds = [dict(id=q, rollout_index=i, seed=i, prompt_token_ids=[1],
                  response_token_ids=[n*4+i+1], raw_logprobs=[0.0], finish_reason="stop",
                  completion="reason</think>\n\\boxed{"+("A" if i==3 or n==1 else "B")+"}")
             for n,q in enumerate(ids) for i in range(4)]
    cfg = dict(rollouts_per_question=4, verifier_version="multilingual-final-answer-v3",
               max_new_tokens=8192, head_chunk=512, likelihood_temperature=.6, clip=.2)
    result = train.online(None,opt,[parameter],cfg,dict(policy=0),ids,preds,data,"synthetic",False,{})
    assert result["group_histogram"] == [0,1,0,0,1] and result["correct_responses"] == 5
    assert result["groups"][0]["advantages"] == [-.25,-.25,-.25,.75]
    assert result["groups"][1]["advantages"] == [0,0,0,0]
    assert result["optimizer_updated"] and result["likelihood_responses"] == 4
    assert float(parameter.grad) == -.1875


def test_four_distinct_on_policy_slots_across_three_workers(monkeypatch):
    monkeypatch.setattr(sampling, "sync", lambda *args: 0)
    def lanes(directories, requests, mode):
        assert len(directories) == len(requests) == 3
        assert [len(req["prompts"]) for req in requests] == [22,21,21]
        return [dict(records=req["prompts"], elapsed_seconds=0) for req in requests]
    monkeypatch.setattr(sampling, "all_lanes", lanes)
    ids = [f"q{i}" for i in range(16)]
    data = {q:dict(prompt_token_ids=[i+1]) for i,q in enumerate(ids)}
    cfg = dict(rollouts_per_question=4,sampling_seed=42,max_new_tokens=8192,temperature=.6,top_p=1,top_k=-1)
    rows,_ = sampling.generate(ids,data,cfg,[1,2,3],"tag","adapter","digest",1,"formal")
    assert [(r["id"],r["rollout_index"]) for r in rows] == [(q,i) for q in ids for i in range(4)]
    assert len(set(r["seed"] for r in rows)) == 64
