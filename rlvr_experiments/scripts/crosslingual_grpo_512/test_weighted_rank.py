import torch
from types import SimpleNamespace

from weighted_rank import choice_loss, choice_scores, option_pairs


def test_minimum_of_two_margins_and_frequency():
    pairs = option_pairs("B",list("BBBBBCCC"),list("BBBBDDDD"))
    assert [p["weight"] for p in pairs] == [1/16]
    assert option_pairs("B",list("BBBBBDDD"),["D"]*8)[0]["weight"] == 2/8
    assert option_pairs("B",list("BBBBBBBD"),["D"]*8)[0]["weight"] == 6/8
    assert option_pairs("B",["B"]*8,["D"]*4+["C"]*2+[None]*2)[0]["weight"] == 2/8


def test_ties_invalids_and_disagreeing_teacher():
    assert option_pairs("B",["B"]*4+["D"]*4,["D"]*8)[0]["weight"] == 0
    assert option_pairs("B",["B"]*3+["A"]*5,["B"]*4+["D"]*4)[0]["weight"] == 0
    assert option_pairs("B",[None]*8,["D"]*8)[0]["weight"] == 0
    assert option_pairs("B",["B"]*8,[None]*8) == []


def test_auxiliary_gradient_never_sees_target_letter():
    embedding = torch.nn.Embedding(19,5)
    head = torch.nn.Linear(5,19,bias=False)
    head.weight.requires_grad_(False)
    class Decoder(torch.nn.Module):
        def forward(self,input_ids,use_cache):
            assert input_ids.tolist() == [[10,11,12]]
            return SimpleNamespace(last_hidden_state=embedding(input_ids).cumsum(1))
    base = SimpleNamespace(lm_head=head,model=Decoder())
    model = SimpleNamespace(get_base_model=lambda:base)
    label_ids = torch.tensor([1,3,4,6])
    scores = choice_scores(model,[10,11,12],label_ids)
    expected = head(embedding(torch.tensor([10,11,12])).sum(0))[label_ids]
    torch.testing.assert_close(scores,expected)
    pairs = option_pairs("B",list("BBBBBBBD"),["D"]*8)
    loss = choice_loss(scores,"ABCD","B",pairs)
    (g,) = torch.autograd.grad(loss,scores,retain_graph=True)
    assert g[1]<0 and g[3]>0 and g[0]==g[2]==0
    loss.backward()
    assert embedding.weight.grad[10:13].abs().sum()>0
