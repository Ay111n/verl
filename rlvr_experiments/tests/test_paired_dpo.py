import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Qwen3Config,Qwen3ForCausalLM
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/same_source_dpo_316"))
from paired_dpo import response_logps,terms

def test_native_response_shift_and_gradient():
    torch.manual_seed(42)
    c=Qwen3Config(vocab_size=83,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0)
    m=Qwen3ForCausalLM(c).double().eval()
    prompt=[2,5,8];response=[11,17,23,1]
    actual=response_logps(m,prompt,response,chunk_size=2)
    ids=torch.tensor([prompt+response])
    logits=m(ids).logits[0,len(prompt)-1:-1]
    expected=F.log_softmax(logits.float(),-1).gather(1,torch.tensor(response)[:,None]).flatten()
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    g1=torch.autograd.grad(actual.sum(),m.model.embed_tokens.weight,retain_graph=True)[0]
    g2=torch.autograd.grad(expected.sum(),m.model.embed_tokens.weight)[0]
    torch.testing.assert_close(g1,g2,atol=1e-5,rtol=1e-5)
    assert len(actual)==4

def test_sum_dpo_mean_auxiliary_and_gate():
    c=torch.tensor(-10.,requires_grad=True);r=torch.tensor(-20.,requires_grad=True)
    b,a,d=terms(c,r,-10.,-20.,10,10,2.,1.)
    torch.testing.assert_close(d["dpo"],torch.tensor(2.).log())
    torch.testing.assert_close(d["margin"],torch.tensor(1.))
    torch.testing.assert_close(a,torch.tensor(1.))
    dc,dr=torch.autograd.grad(a,(c,r))
    assert dc<0 and dr>0
    _,zero,_=terms(c,r,-10.,-20.,10,10,2.,0.)
    assert zero==0
    _,satisfied,_=terms(c,r,-10.,-20.,10,10,.5,1.)
    assert satisfied==0
