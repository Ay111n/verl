"""CPU acceptance of objectives, causal labels, and actual exposure accounting."""
from collections import Counter
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from common import advantages, online_terms, demonstration_loss, plan_indices, lr_for_update, response_logps

def test_advantages():
    for correct in range(5):
        r=torch.tensor([1.]*correct+[0.]*(4-correct),dtype=torch.float64)
        a,d=advantages(r,"C")
        torch.testing.assert_close(a,r-(correct+1)/5)
        assert abs(float(d)-(4-correct)/5)<1e-12
        b,_=advantages(r,"A")
        assert abs(float(b.sum()))<1e-12
    assert advantages([0,0,0,0],"A")[0].abs().sum()==0
    assert advantages([1,1,1,1],"C")[0].abs().sum()==0

def test_shaping_gradient():
    lp=torch.tensor([-15.,-4.,-2.,-.2],dtype=torch.float64,requires_grad=True)
    loss=demonstration_loss(lp,.8,.1)
    gradient,=torch.autograd.grad(loss,lp)
    p=lp.detach().exp()
    torch.testing.assert_close(gradient,-.8*.1*p/(p+.1).square()/4)
    assert torch.autograd.gradcheck(lambda x: demonstration_loss(x,.8,.1),(lp,))

def test_clipping_and_stopped_old():
    lp=torch.tensor([-.3,-1.,-3.],dtype=torch.float64,requires_grad=True)
    old=lp.detach().clone().requires_grad_()
    ref=lp.detach().clone().requires_grad_()
    on,kl=online_terms(lp,old,ref,.4,.2)
    g,=torch.autograd.grad(on+.001*kl,lp)
    torch.testing.assert_close(g,torch.full_like(lp,-.4/3))
    assert float(kl.detach())==0 and old.grad is None and ref.grad is None
    lp2=torch.tensor([math.log(1.5)],dtype=torch.float64,requires_grad=True)
    on,_=online_terms(lp2,torch.zeros_like(lp2),torch.zeros_like(lp2),1.,.2)
    assert float(torch.autograd.grad(on,lp2)[0])==0

def test_causal_response_and_eos():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(13,5);self.lm_head=torch.nn.Linear(5,13,bias=False)
        def model(self,input_ids,use_cache=False):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    torch.manual_seed(42);model=Toy()
    prompt=[2,3,4];response=[7,8,12]
    got=response_logps(model,prompt,response,chunk_size=2)
    full=model.lm_head(model.emb(torch.tensor([prompt+response[:-1]]))).float()
    labels=torch.tensor([[-100,-100,7,8,12]])
    expected=-F.cross_entropy(full[0],labels[0],reduction="none")[2:]
    torch.testing.assert_close(got,expected)
    assert got.numel()==3

def test_budget_and_last_batch():
    cfg={"epochs":3,"batch_size":16,"order_seed":42,"learning_rate":1e-6,"warmup_updates":5}
    plan=plan_indices(388,cfg)
    assert len(plan)==75 and sum(len(b["indices"])*4 for b in plan)==4656
    assert Counter(i for b in plan for i in b["indices"])==Counter({i:3 for i in range(388)})
    assert [len(plan[i]["indices"]) for i in (24,49,74)]==[4,4,4]
    # Data-parallel summation with the actual global question count.
    for n in (16,4):
        vals=torch.arange(n*4,dtype=torch.float64).reshape(n,4)
        got=sum(vals[i::2].sum()/(n*4) for i in (0,1))
        torch.testing.assert_close(got,vals.mean())
    assert lr_for_update(1,cfg)==2e-7 and lr_for_update(75,cfg)==1e-6


def test_optimizer_checkpoint_roundtrip():
    import io
    first=torch.nn.Parameter(torch.tensor([.5,-.3]))
    original=torch.optim.AdamW([first],lr=1e-3,weight_decay=0.)
    for _ in range(2):
        original.zero_grad();first.square().sum().backward();original.step()
    stream=io.BytesIO()
    torch.save({"step":2,"optimizer":original.state_dict(),"parameter":first.detach().clone()},stream)
    stream.seek(0);state=torch.load(stream,weights_only=False)
    assert {int(v["step"]) for v in state["optimizer"]["state"].values()}=={2}
    second=torch.nn.Parameter(state["parameter"].clone())
    resumed=torch.optim.AdamW([second],lr=9.)
    resumed.load_state_dict(state["optimizer"])
    for parameter,optimizer in [(first,original),(second,resumed)]:
        optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
    torch.testing.assert_close(first,second,rtol=0,atol=0)
