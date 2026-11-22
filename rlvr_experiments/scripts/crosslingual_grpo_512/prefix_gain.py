"""Continue visit two, teaching an English opening only after a paired improvement."""
import gzip
import json
import os
from pathlib import Path
import sys
import time

from restate import ROOT, RUN as PARENT, OUT as PARENT_WEIGHTS, load_inputs, read, save, sha, update
from restate import AutoTokenizer, torch, load_model, restore, commit, generate, sync, judge, supervise
from candidate import save_responses
from sampling import all_lanes

NAME = "formateval_priority512_prefix_gain_20261122"
PROJECT = PARENT.parent.parent
RUN = PARENT.parent / NAME
OUT = PARENT_WEIGHTS.parent / NAME
CONFIG = PROJECT / "configs" / (NAME + ".json")
SOURCE = PARENT_WEIGHTS / "checkpoint-0016"
PREFIX_RUN = PARENT.parent / "formateval_priority512_prefix_rescue_20261121"


def freeze():
    cfg, audit, data = load_inputs()
    tok = AutoTokenizer.from_pretrained(ROOT/cfg["model_path"],local_files_only=True)
    assert sha(SOURCE/"adapter_model.safetensors") == read(PARENT/"steps/checkpoint-0016.json")["output_adapter_sha256"]
    path = PREFIX_RUN/"prefixes.json"
    assert sha(path) == read(PREFIX_RUN/"preflight.json")["prefixes_sha256"]
    prefixes = read(path)
    weak = [g["id"] for i in range(1,17) for g in read(PARENT/"steps"/f"checkpoint-{i:04d}.json")["groups"] if sum(g["rewards"])<=1]
    assert len(set(weak))==len(weak)==255
    frozen = {}
    for q in weak:
        if q not in prefixes:
            continue
        view = data[q]["bo"]
        forced = prefixes[q]["forced"]
        assert forced.startswith("<think>\n") and "</think>" not in forced
        encode = lambda s: tok.encode(s,add_special_tokens=False)
        context = encode(view["prompt"]+"<think>\n")
        full = encode(view["prompt"]+forced)
        assert full[:len(context)]==context and full[:len(view["prompt_token_ids"])]==view["prompt_token_ids"]
        budget = cfg["max_new_tokens"]-(len(full)-len(view["prompt_token_ids"]))
        assert 0 < budget < cfg["max_new_tokens"] and len(full)>len(context)
        frozen[q]=dict(text=forced,context=context,target=full[len(context):],prompt_ids=full,budget=budget)
    assert len(frozen)==182
    save(RUN/"prefix_targets.json",frozen)
    save(RUN/"preflight.json",dict(parent=str(SOURCE),adapter_sha256=sha(SOURCE/"adapter_model.safetensors"),
        candidates=182,prefix_sha256=sha(RUN/"prefix_targets.json"),parent_prompt_sha256=audit["prompt_sha256"],
        recovery_sha256=sha(SOURCE/"recovery.pt"),config_sha256=sha(CONFIG)))


def assisted(batch, native, frozen, bo, cfg, workers, tag, cp, digest):
    requests=[]
    for r in native:
        q=r["id"]
        if q in frozen:
            t=frozen[q]
            requests.append(dict(id=q,rollout_index=r["rollout_index"],seed=r["seed"],
                prompt_token_ids=t["prompt_ids"],max_new_tokens=t["budget"]))
    if not requests:
        return {},[]
    jobs=[dict(request_id=f"{tag}-hint-w{i}",mode="generate",adapter=str(cp),adapter_sha256=digest,
               preload_adapter=True,prompts=requests[i::2],return_logprobs=True,
               **{k:cfg[k] for k in ("max_new_tokens","temperature","top_p","top_k")}) for i in range(2)]
    results=all_lanes(workers,jobs,"generate")
    correct={}
    records=[]
    for result in results:
        for r in result["records"]:
            q=r["id"];t=frozen[q]
            assert len(r["response_token_ids"])<=t["budget"]
            full=t["text"]+r["completion"]
            verdict=judge(bo[q],full,r["finish_reason"],len(r["response_token_ids"])+cfg["max_new_tokens"]-t["budget"],cfg)
            correct[q]=correct.get(q,0)+verdict["reward"]
            records.append(dict(id=q,rep=r["rollout_index"],seed=r["seed"],completion=r["completion"],
                response_token_ids=r["response_token_ids"],judgment=verdict,adapter_sha256=digest))
    assert len(records)==len(requests)==4*len(correct)
    return correct,records


def train():
    cfg=read(CONFIG);_,_,data=load_inputs()
    assert os.environ["CUDA_VISIBLE_DEVICES"]=="3"
    free,total=torch.cuda.mem_get_info()
    assert free>=cfg["actor_min_free_mib"]*1024**2
    torch.cuda.set_per_process_memory_fraction((free-cfg["actor_reserve_mib"]*1024**2)/total)
    audit=read(RUN/"preflight.json")
    assert sha(CONFIG)==audit["config_sha256"] and sha(RUN/"prefix_targets.json")==audit["prefix_sha256"]
    frozen=read(RUN/"prefix_targets.json")
    targets={q:(v["context"],v["target"]) for q,v in frozen.items()}
    bo={q:d["bo"] for q,d in data.items()}
    batches=read(PARENT/"training_plan.json")["batches"][16:]
    assert len(batches)==16 and {q for b in batches for q in b}==set(bo)
    workers=[RUN/"rpc_workers"/f"worker{i}" for i in range(2)]
    base=ROOT/cfg["model_path"]
    OUT.mkdir(parents=True,exist_ok=True)
    last=max(OUT.glob("checkpoint-[0-9][0-9][0-9][0-9]"),default=SOURCE)
    model=load_model(str(base),cfg["initial_seed"],adapter=str(last),trainable_adapter=True)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    state,digest=restore(last,model,opt,read(last/"training_config.json") if last==SOURCE else cfg)
    if last==SOURCE:
        assert state["cursor"]==state["policy"]==16 and digest==audit["adapter_sha256"]
    for step in range(state["cursor"]+1,33):
        start=time.monotonic();batch=batches[step-17];tag=f"b{step:03d}-{digest[:16]}"
        native,times=generate(batch,bo,cfg,workers,tag,last,digest,step,"native-visit-2")
        hinted,records=assisted(batch,native,frozen,bo,cfg,workers,tag,last,digest)
        counts={q:sum(judge(bo[q],r["completion"],r["finish_reason"],len(r["response_token_ids"]),cfg)["reward"]
                      for r in native if r["id"]==q) for q in hinted}
        weights={q:max(0,hinted[q]-counts[q])/4 for q in hinted}
        evt=update(model,opt,params,cfg,state,batch,native,bo,targets,2,prefix_weights=weights)
        evt.update(kind="online",step=step,input_adapter_sha256=digest,hinted=hinted,native_candidate=counts,
                   weights=weights,hint_responses=len(records),**times)
        evt["responses_sha256"]=save_responses(step,state["policy"],digest,native,evt["groups"],data,run=RUN)
        path=RUN/"responses"/f"hint-{step:04d}.jsonl.gz"
        with gzip.open(path.with_suffix(".tmp"),"wt",encoding="utf8") as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        path.with_suffix(".tmp").replace(path);evt["hint_sha256"]=sha(path)
        state["cursor"]=state["tx"]=step
        state["policy"]+=int(evt["optimizer_updated"]);state["online"]=state["policy"]
        last,digest,secs=commit(OUT,RUN,model,opt,state,evt,cfg,base)
        print(step,evt["correct_responses"],sum(w>0 for w in weights.values()),time.monotonic()-start,flush=True)
    sync(workers,"final-"+digest[:16],last,digest)
    ev=[read(OUT/f"checkpoint-{i:04d}"/"event.json") for i in range(17,33)]
    assert sum(e["hint_responses"] for e in ev)==728
    final=dict(state,epochs=1,question_visits=512,online_responses=2048,hint_responses=728,
               supervised=sum(sum(w>0 for w in e["weights"].values()) for e in ev),
               adapter=str(last),adapter_sha256=digest)
    save(OUT/"final.json",final);save(RUN/"complete.json",final)


if __name__=="__main__":
    if sys.argv[1]=="train":train()
    elif sys.argv[1]=="freeze":freeze()
    elif sys.argv[1]=="run":supervise(CONFIG,RUN,online_actor=Path(__file__).resolve())
