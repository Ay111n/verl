"""Check finished-only callbacks and original-order reconstruction without GPU work."""
from types import SimpleNamespace as Obj
from stream_worker import stream_generate


def test_finish_order_does_not_change_return_order():
    class Fake:
        def __init__(self):
            self.llm_engine=self;self.round=0
            self.output_processor=Obj(request_states={x:Obj(external_req_id=x) for x in ("7","8","9")})
        def enqueue(self,prompts,**kwargs):return ["7","8","9"]
        def has_unfinished_requests(self):return self.round<2
        def step(self):
            self.round+=1
            return [Obj(request_id="8",finished=True),Obj(request_id="7",finished=False)] if self.round==1 else [Obj(request_id="9",finished=True),Obj(request_id="7",finished=True)]
    observed=[]
    result=stream_generate(Fake(),[1,2,3],[1,2,3],None,lambda index,out:observed.append((index,out.request_id)))
    assert observed==[(1,"8"),(2,"9"),(0,"7")]
    assert [x.request_id for x in result]==["7","8","9"]


def test_installed_vllm_internal_ids_map_to_external_outputs(monkeypatch):
    from vllm.v1.engine.input_processor import InputProcessor
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION","0")
    requests=[Obj(request_id=str(i),external_req_id=None) for i in (71,72,73)]
    for request in requests:InputProcessor.assign_request_id(request)
    assert all(x.request_id!=x.external_req_id for x in requests)
    class Fake:
        def __init__(self):
            self.llm_engine=self;self.done=False
            self.output_processor=Obj(request_states={x.request_id:x for x in requests})
        def enqueue(self,prompts,**kwargs):return [x.request_id for x in requests]
        def has_unfinished_requests(self):return not self.done
        def step(self):
            self.done=True
            return [Obj(request_id=str(i),finished=True) for i in (73,71,72)]
    observed=[]
    result=stream_generate(Fake(),[1,2,3],[1,2,3],None,lambda index,out:observed.append((index,out.request_id)))
    assert observed==[(2,"73"),(0,"71"),(1,"72")]
    assert [x.request_id for x in result]==["71","72","73"]
