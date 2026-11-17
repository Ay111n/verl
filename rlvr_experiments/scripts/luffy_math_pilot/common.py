"""Small Math mixed-policy pilot; shared numerical and verifier implementations."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.same_source_dpo_316.paired_dpo import load_model,read_rows,save,sha,adapter_hash,runtime
from scripts.luffy_knowledge_388.batched_likelihood import response_logps_batch
from rlvr.verifier import Candidate,make_prompt
from rlvr.answer_verifier import verify_answer
ROOT=PROJECT.parents[1]
def render(tok,text):
    return tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
def prompt_text(row):
    return make_prompt(Candidate(row["domain"],row,b""))
def verify(row,pred,budget):
    return verify_answer(row,prompt=prompt_text(row),completion=pred["completion"],
        finish_reason=pred["finish_reason"],output_tokens=len(pred["response_token_ids"]),max_new_tokens=budget)
def advantages(rewards):
    r=torch.as_tensor(rewards,dtype=torch.float32)
    assert r.shape[-1]==3 and bool(((r==0)|(r==1)).all())
    mean=(r.sum(-1,keepdim=True)+1)/4
    return (r-mean).detach(),(1-mean.squeeze(-1)).detach()
def online_loss(logp,old,adv,clip=.2):
    ratio=(logp-old.detach()).exp()
    a=torch.as_tensor(adv,device=logp.device).detach()
    return -torch.minimum(ratio*a,ratio.clamp(1-clip,1+clip)*a).mean()
def demo_loss(logp,adv,gamma=.1):
    a=torch.as_tensor(adv,device=logp.device).detach()
    return -a*torch.sigmoid(logp-math.log(gamma)).mean()
def rpc(root,request,timeout=3600):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    name=request["request_id"];assert "/" not in name
    req=root/(name+".request.json");res=root/(name+".response.json")
    if req.exists():assert json.loads(req.read_text())==request
    else:save(req,request)
    end=time.monotonic()+timeout
    while not res.exists():
        if time.monotonic()>end:raise TimeoutError(name)
        time.sleep(.5)
    value=json.loads(res.read_text())
    if value.get("error"):raise RuntimeError(value["error"])
    assert value["request_id"]==name
    return value
def norm(grads):
    return float(torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt())

def rpc_parallel(root, request, workers, timeout=3600):
    """Run disjoint rollouts on identical adapters; restore original request order."""
    from concurrent.futures import ThreadPoolExecutor
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    keys=[(r["id"],r["rollout_index"]) for r in request["prompts"]]
    assert len(keys)==len(set(keys)) and workers
    save(root/(request["request_id"]+".request.json"),request)
    start=time.monotonic();assignments=[]
    for rank,worker in enumerate(workers):
        subset=request["prompts"][rank::len(workers)]
        if subset:assignments.append((Path(worker),{**request,"request_id":request["request_id"]+f"-w{rank}","prompts":subset}))
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures=[executor.submit(rpc,path,part,timeout) for path,part in assignments]
        results=[future.result() for future in futures]
    records={};timings=[]
    for (path,part),result in zip(assignments,results,strict=True):
        assert result["adapter_sha256"]==request["adapter_sha256"]
        expected={(r["id"],r["rollout_index"]):r for r in part["prompts"]}
        assert len(result["records"])==len(expected)
        for row in result["records"]:
            key=(row["id"],row["rollout_index"])
            assert key in expected and key not in records
            assert row["prompt_token_ids"]==expected[key]["prompt_token_ids"]
            assert row["seed"]==expected[key]["seed"]
            records[key]=row
        timings.append({"rpc":str(path),"request_id":part["request_id"],"responses":len(result["records"]),"seconds":result["elapsed_seconds"]})
    assert set(records)==set(keys)
    value={"request_id":request["request_id"],"adapter_sha256":request["adapter_sha256"],
           "records":[records[key] for key in keys],"elapsed_seconds":time.monotonic()-start,"parallel_workers":timings}
    save(root/(request["request_id"]+".response.json"),value)
    return value
