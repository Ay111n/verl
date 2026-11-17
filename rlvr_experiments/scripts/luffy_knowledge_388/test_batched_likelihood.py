"""Numerical checks of response masks, sequence weighting, and independent attention."""
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from common import response_logps,online_terms,demonstration_loss
from batched_likelihood import response_logps_batch


def model_and_data():
    torch.manual_seed(13)
    config=Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=128,
        attention_dropout=0.,pad_token_id=0,eos_token_id=2,tie_word_embeddings=False)
    config._attn_implementation="eager"
    model=Qwen3ForCausalLM(config).eval()
    return model,[[1,0,5],[6,7],[8,9,10,11,12]],[[4,0,9,2],[3,2],[13,14,15,16,17,2]]


def test_causal_eos_and_raw_likelihood():
    model,prompts,responses=model_and_data()
    for temperature in (1.,.6):
        got,raw=response_logps_batch(model,prompts,responses,chunk_size=3,temperature=temperature,return_raw=True)
        for i,(p,r) in enumerate(zip(prompts,responses)):
            expected=response_logps(model,p,r,chunk_size=2,temperature=temperature)
            original=response_logps(model,p,r,chunk_size=2)
            torch.testing.assert_close(got[i],expected,rtol=1e-5,atol=1e-5)
            torch.testing.assert_close(raw[i],original,rtol=1e-5,atol=1e-5)
            assert got[i].numel()==len(r) and not raw[i].requires_grad


def test_equal_sequence_weighted_gradient_for_three_losses():
    model,prompts,responses=model_and_data();params=tuple(model.parameters())
    for arm in ("A","B","C"):
        sequential=[response_logps(model,p,r,chunk_size=2,temperature=.6) for p,r in zip(prompts,responses)]
        batched=response_logps_batch(model,prompts,responses,chunk_size=5,temperature=.6)
        def objective(values):
            loss=0
            for i,lp in enumerate(values):
                advantage=[-.4,.6,-.2][i]
                on,kl=online_terms(lp,lp.detach(),lp.detach()+.03,advantage,.2)
                loss=loss+(on+.001*kl)/3
                if arm=="B":loss=loss-.0125*lp.mean()/3
                if arm=="C":loss=loss+.25*demonstration_loss(lp,.8,.1)/3
            return loss
        left=objective(sequential);right=objective(batched)
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
        a=torch.autograd.grad(left,params);b=torch.autograd.grad(right,params)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-6)


def test_no_cross_sequence_attention():
    model,prompts,responses=model_and_data()
    with torch.no_grad():
        first=response_logps_batch(model,prompts,responses)
        changed=[p.copy() for p in prompts];changed[0]=[20,21,22]
        second=response_logps_batch(model,changed,responses)
    assert not torch.allclose(first[0],second[0])
    torch.testing.assert_close(first[1],second[1],rtol=0,atol=0)
    torch.testing.assert_close(first[2],second[2],rtol=0,atol=0)


def test_full_update_preserves_question_and_response_weights(monkeypatch):
    import copy
    from contextlib import nullcontext
    import train
    import batched_update
    template,prompts,responses=model_and_data()
    rows=[];outputs=[]
    for i,prompt in enumerate(prompts[:2]):
        rows.append({"id":str(i),"prompt_token_ids":prompt,"old_bo_rewards":[0,1,0,1],
                     "probe_selection":{"stratum":"synthetic"},"demo":{"response_token_ids":responses[i]}})
        for j in range(4):
            ids=responses[(i+j)%3]
            outputs.append({"id":str(i),"rollout_index":j,"prompt_token_ids":prompt,
                            "response_token_ids":ids,"raw_logprobs":[-1.]*len(ids),"reward":int(j<1+i)})
    def verifier(row,output,budget):
        return {"correctness":output["reward"],"parse_status":"ok","termination_status":"stop",
                "protocol_valid":True,"truncated":False,"answer_extractable":True,
                "final_present":True,"thinking_closed":True}
    monkeypatch.setattr(train,"verify",verifier);monkeypatch.setattr(batched_update,"verify",verifier)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:0)
    monkeypatch.setattr(torch.cuda,"max_memory_reserved",lambda:0)
    class NoUpdate:
        def __init__(self,model):self.model=model;self.param_groups=[{"lr":0.}]
        def zero_grad(self,set_to_none=True):self.model.zero_grad(set_to_none=set_to_none)
        def step(self):pass
    cfg={"max_new_tokens":8192,"max_model_len":32768,"chunk_size":2,"likelihood_temperature":.6,
         "clip":.2,"kl_coef":.001,"nll_alpha":.0125,"demo_scale":.25,"gamma":.1,
         "max_grad_norm":100.,"learning_rate":1e-6,"warmup_updates":5}
    generated={"records":outputs,"elapsed_seconds":1.,"adapter_sha256":"synthetic"}
    for arm in ("A","B","C"):
        left=copy.deepcopy(template);right=copy.deepcopy(template)
        left.disable_adapter=lambda:nullcontext();right.disable_adapter=lambda:nullcontext()
        lp=list(left.parameters());rp=list(right.parameters())
        a=train.update(left,NoUpdate(left),lp,rows,generated,arm,cfg,0,1,1)
        b=batched_update.update(right,NoUpdate(right),rp,rows,generated,arm,cfg,0,1,1,micro_batch=3,head_chunk=5)
        for x,y in zip(lp,rp):torch.testing.assert_close(x.grad,y.grad,rtol=3e-4,atol=3e-6)
        streamed=copy.deepcopy(template);streamed.disable_adapter=lambda:nullcontext()
        sp=list(streamed.parameters());accumulated=[torch.zeros_like(x) for x in sp]
        for index in (1,0):
            portion={**generated,"records":outputs[index*4:index*4+4]}
            batched_update.update(streamed,NoUpdate(streamed),sp,[rows[index]],portion,arm,
                {**cfg,"max_grad_norm":float("inf")},0,1,1,micro_batch=3,head_chunk=5)
            for total,parameter in zip(accumulated,sp):total.add_(parameter.grad,alpha=1/len(rows))
        for expected,actual in zip(lp,accumulated):torch.testing.assert_close(expected.grad,actual,rtol=3e-4,atol=3e-6)
        for field in ("rewards","correct","zero_correct_groups","all_correct_groups","mixed_groups","rollout_count"):
            assert a[field]==b[field]
        av={(x["id"],x["rollout_index"]):x for x in a["online"]}
        bv={(x["id"],x["rollout_index"]):x for x in b["online"]}
        for key in av:
            for field in ("advantage","policy_loss","sampled_kl","mean_logprob"):
                assert abs(av[key][field]-bv[key][field])<1e-5


def test_sdpa_causal_padding_and_gradients():
    # Exercise the same HF attention-dispatch path as the GPU model, in FP32.
    model,prompts,responses=model_and_data();model.set_attn_implementation("sdpa")
    params=tuple(model.parameters())
    single=[response_logps(model,p,r,chunk_size=2,temperature=.6) for p,r in zip(prompts,responses)]
    batch=response_logps_batch(model,prompts,responses,chunk_size=5,temperature=.6)
    for left,right in zip(single,batch):torch.testing.assert_close(left,right,rtol=2e-5,atol=2e-5)
    a=torch.autograd.grad(sum(x.mean() for x in single),params)
    b=torch.autograd.grad(sum(x.mean() for x in batch),params)
    for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=3e-4,atol=3e-6)
