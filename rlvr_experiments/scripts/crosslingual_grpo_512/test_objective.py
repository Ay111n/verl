"""Regression checks on the prompt contract, gate, next-token alignment and KL gradient."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from paired import training_content
from objective import score_groups, thinking_span
from scripts.luffy_knowledge_388.batched_likelihood import response_logps_batch, response_token_logps
from scripts.opd_math_ab.common import response_hidden, reverse_kl


def test_plain_dev_layout():
    row = {"question":"?", "choices":["a","b","c","d"]}
    for lang, first, last in (("en","Please answer the question.","Answer:"),
                              ("bo","དྲི་བ་ལ་ལན་འདེབས་རོགས།","ལན།")):
        lines = training_content(row,lang).splitlines()
        assert lines[0] == first and lines[-1] == last
        assert lines[4:8] == ["A. a","B. b","C. c","D. d"]
        assert "{{answer:" not in "\n".join(lines)


def test_thinking_mask():
    a,b,eos = 7,8,9
    def span(ids,reason):
        return thinking_span(ids,reason,a,b,{eos})
    assert span([a,10,11,b,12,eos],"stop") == (1,3)
    assert span([a,10,11],"length") == (1,3)
    assert span([a,10,b,12],"length") == (1,2)
    for tokens,reason in [([a,10],"stop"),([a,b,eos],"stop"),([10,b],"stop"),
                          ([a,1,a,2,b],"stop"),([a,1,b,2,b],"stop"),([a,1,eos,2],"length")]:
        assert span(tokens,reason) is None


def test_relative_gate_and_v3():
    cfg = {"rollouts_per_question":4,"verifier_version":"multilingual-final-answer-v3","max_new_tokens":8192}
    english, data, preds = {}, {}, []
    for q,k_en,k_bo in (("a",1,0),("b",2,2),("c",3,2),("d",1,3)):
        view = dict(id=q,domain="knowledge",option_labels=list("ABCD"),options_bo=dict(zip("ABCD","abcd")),
                    canonical_answer="A",prompt="plain",prompt_token_ids=[3])
        data[q]={"bo":view}
        english[q]=[int(i<k_en) for i in range(4)]
        for i in range(4):
            selected = "A" if i<k_bo or i==3 else "B"
            preds.append(dict(id=q,rollout_index=i,seed=i,prompt_token_ids=[3],response_token_ids=[4,5,6],
                              completion="<think>reason</think> "+selected,finish_reason="length" if i==3 else "stop"))
    groups,_ = score_groups(list(data),preds,data,english,cfg)
    assert [(g["tibetan_correct"],g["kd_eligible"]) for g in groups] == [(0,True),(2,False),(2,True),(3,False)]
    assert groups[0]["advantages"] == [0,0,0,0] and groups[1]["advantages"] == [.5,.5,-.5,-.5]


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(23,7)
        self.lm_head = torch.nn.Linear(7,23,bias=False)
        self.lm_head.requires_grad_(False)

    def get_base_model(self):
        return self

    def model(self,input_ids,**unused):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids).cumsum(1))


def test_causal_alignment_and_gradients():
    torch.manual_seed(42)
    model = TinyCausal().double()
    bo,en,reply = [8,1,2],[11,10,3,4,6],[18,5,9,19,12]
    lo,hi = thinking_span(reply,"stop",18,19,{22})
    h = response_hidden(model,bo,reply)
    lp,raw = response_token_logps(model.lm_head,h,torch.tensor(reply),2,.6,True)
    p0,r0 = response_logps_batch(model,[bo],[reply],2,.6,True)
    torch.testing.assert_close(lp,p0[0])
    torch.testing.assert_close(raw,r0[0])
    with torch.no_grad():
        teacher = response_hidden(model,en+reply[:lo],reply[lo:hi])
        t_all = response_hidden(model,en,reply)
        torch.testing.assert_close(teacher,t_all[lo:hi])
    student = h[lo:hi]
    loss = reverse_kl(student,teacher,model.lm_head.weight,1)
    logp = model.lm_head(student).log_softmax(-1)
    logq = model.lm_head(teacher).log_softmax(-1)
    reference = (logp.exp()*(logp-logq)).sum(-1).mean()
    torch.testing.assert_close(loss,reference)
    actual = torch.autograd.grad(.1*loss/64,model.embed.weight,retain_graph=True)[0]
    expected = torch.autograd.grad(.1*reference/64,model.embed.weight)[0]
    torch.testing.assert_close(actual,expected)
    assert actual.norm() > 0 and teacher.grad is None and model.lm_head.weight.grad is None


def test_single_optimizer_update_matches_two_dense_group_means(monkeypatch):
    from contextlib import nullcontext
    import objective

    torch.manual_seed(31)
    model = TinyCausal().double()
    model.disable_adapter = nullcontext
    params = list(model.embed.parameters())
    opt = torch.optim.SGD(params, lr=.001)
    cfg = dict(rollouts_per_question=4, verifier_version="multilingual-final-answer-v3",
        max_new_tokens=8192, kd_weight=.1, kd_temperature=1., kd_chunk=2, head_chunk=2,
        likelihood_temperature=.6, clip=.2)
    audit = dict(think_open=18,think_close=19)
    data, predictions, reference = {}, [], 0.
    english = {"zero":[1,0,0,0],"mixed":[1,1,0,0]}
    for uid in english:
        bo,en = [3,6],[10,2,7]
        view = dict(id=uid, domain="knowledge", option_labels=list("ABCD"), options_bo=dict(zip("ABCD","abcd")),
                    canonical_answer="A",prompt="plain",prompt_token_ids=bo)
        data[uid] = {"bo":view,"en":{"prompt_token_ids":en}}
        for rep in range(4):
            ok = uid=="mixed" and rep<2
            tokens = [18,5,rep+2,19,14+int(ok)]
            student = response_hidden(model,bo,tokens)
            logits = model.lm_head(student).float()
            target = torch.tensor(tokens)
            raw = -F.cross_entropy(logits.detach(),target,reduction="none")
            predictions.append(dict(id=uid,rollout_index=rep,seed=rep,prompt_token_ids=bo,response_token_ids=tokens,
                raw_logprobs=raw.tolist(),finish_reason="stop",completion="<think>r</think> "+("A" if ok else "B")))
            if uid=="mixed":
                lp = -F.cross_entropy(logits/.6,target,reduction="none")
                reference += -((lp-lp.detach()).exp()*(int(ok)-.5)).mean()/8
            if uid=="zero":
                teacher = response_hidden(model,en,tokens).detach()
                pl = model.lm_head(student[1:3]).log_softmax(-1)
                ql = model.lm_head(teacher[1:3]).log_softmax(-1)
                reference += .1*(pl.exp()*(pl-ql)).sum(-1).mean()/8
    expected = torch.autograd.grad(reference,params)[0]
    snapshots = []
    def finish(optimizer, parameters, config, version):
        snapshots.append(parameters[0].grad.clone())
        return dict(grad_norm=snapshots[-1].norm().item(),grad_clipped=False)
    monkeypatch.setattr(objective,"finish_update",finish)
    event = objective.update(model,opt,params,cfg,{"cursor":1,"policy":1},list(english),predictions,
                             data,english,audit,{22},"test",{})
    assert len(snapshots)==1 and event["optimizer_updated"] and event["kd_tokens"]==8
    assert event["kd_eligible"]==1 and event["group_histogram"]==[1,0,1,0,0]
    torch.testing.assert_close(snapshots[0],expected,rtol=2e-6,atol=1e-8)
