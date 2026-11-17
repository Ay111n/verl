"""Check finished-only callbacks and original-order reconstruction without GPU work."""
from types import SimpleNamespace as Obj
from stream_worker import stream_generate


def test_finish_order_does_not_change_return_order():
    class Fake:
        def __init__(self):self.llm_engine=self;self.round=0
        def enqueue(self,prompts,**kwargs):return ["7","8","9"]
        def has_unfinished_requests(self):return self.round<2
        def step(self):
            self.round+=1
            return [Obj(request_id="8",finished=True),Obj(request_id="7",finished=False)] if self.round==1 else [Obj(request_id="9",finished=True),Obj(request_id="7",finished=True)]
    observed=[]
    result=stream_generate(Fake(),[1,2,3],[1,2,3],None,lambda index,out:observed.append((index,out.request_id)))
    assert observed==[(1,"8"),(2,"9"),(0,"7")]
    assert [x.request_id for x in result]==["7","8","9"]
