"""The revisit may change the policy, never the ordinary pool, cursor or trigger."""
from copy import deepcopy
from revisit import record_event

CFG = dict(check_every_online=3, nll_batch=16, nll_limit=8)


def test_first_revisit_can_update_or_be_skipped_after_restore():
    qids=[str(i) for i in range(16)]
    initial=dict(tx=7,policy=7,cursor=6,online=6,nll=1,pending=["waiting"],due=[],uses=dict.fromkeys(qids,1),
                 revisit_due=qids,revisit_batches=0,revisit_updates=0)
    for active in (True,False):
        state=deepcopy(initial)
        event=dict(kind="revisit",groups=[dict(id=q) for q in qids],optimizer_updated=active)
        record_event(state,event,{},CFG)
        assert (state["policy"],state["tx"],state["revisit_batches"],state["revisit_updates"])==(7+int(active),8,1,int(active))
        assert (state["online"],state["cursor"],state["pending"],state["due"])==(6,6,["waiting"],[])
        assert state["uses"]==initial["uses"] and not state["revisit_due"] and "scheduled_nll_ids" not in event


def test_regular_check_schedules_exact_fifo_then_one_immediate_revisit():
    before=[str(i) for i in range(17)]
    state=dict(tx=10,policy=10,cursor=8,online=8,nll=1,revisit_updates=1,revisit_batches=1,pending=before[:],
               due=[],revisit_due=[],uses={})
    event=dict(kind="online",groups=[dict(id="new",rewards=[1,0,0])],optimizer_updated=True)
    record_event(state,event,{"new":{"english_references":[1]}},CFG)
    assert state["due"]==before[:16] and event["scheduled_nll_ids"]==before[:16]
    record_event(state,dict(kind="nll",references=[dict(id=q) for q in before[:16]]),{},CFG)
    assert state["revisit_due"]==before[:16] and state["pending"]==before[16:] and state["due"]==[]
    assert state["uses"]==dict.fromkeys(before[:16],1) and state["online"]==state["cursor"]==9
