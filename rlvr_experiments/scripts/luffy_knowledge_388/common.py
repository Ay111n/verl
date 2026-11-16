"""Synchronous Tibetan rollout plus fixed English demonstration primitives."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE.parent / "same_source_dpo_316"))
from paired_dpo import load_model, response_logps, adapter_hash, read_rows, save, sha, runtime
sys.path.remove(str(HERE.parent / "same_source_dpo_316"))
from rlvr.paths import load_paths
from rlvr.verifier import Candidate, make_prompt, render_prompt
from rlvr.answer_verifier import verify_answer, VERIFIER_VERSION

def paths():
    return load_paths("luffy_knowledge_388_l20")

def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def prompts(tok, row):
    candidate = Candidate("knowledge", row, b"")
    text, _ = render_prompt(tok, candidate)
    assert hashlib.sha256(text.encode()).hexdigest() == row["old_bo_prompt_sha256"]
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == row["old_bo_prompt_tokens"]
    return text, ids

def verify(row, output, budget):
    result = verify_answer(row, prompt=make_prompt(Candidate("knowledge", row, b"")),
                          completion=output["completion"], finish_reason=output["finish_reason"],
                          output_tokens=len(output["response_token_ids"]), max_new_tokens=budget)
    assert result["execution_valid"] and result["correctness"] in (0, 1)
    return result

def plan_indices(count, cfg):
    batches = []
    for epoch in range(cfg["epochs"]):
        order = list(range(count))
        random.Random(cfg["order_seed"] + epoch).shuffle(order)
        for start in range(0, count, cfg["batch_size"]):
            batches.append({"epoch":epoch+1,"indices":order[start:start+cfg["batch_size"]]})
    return batches

def rollout_seed(cfg, epoch, qid, index):
    value = f'luffy-v1|{cfg["sampling_seed"]}|{epoch}|{qid}|{index}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") % (2**31-1)

def advantages(rewards, arm):
    values = torch.as_tensor(rewards)
    if not values.is_floating_point(): values = values.float()
    assert values.shape[-1] == 4 and bool(((values==0)|(values==1)).all())
    mean = (values.sum(-1, keepdim=True) + 1) / 5 if arm == "C" else values.mean(-1, keepdim=True)
    return (values - mean).detach(), (1 - mean.squeeze(-1)).detach()

def online_terms(logp, old_logp, ref_logp, advantage, clip):
    assert logp.ndim==1 and logp.numel()>0 and logp.shape==old_logp.shape==ref_logp.shape
    ratio = (logp-old_logp.detach()).exp()
    adv = torch.as_tensor(advantage, device=logp.device).detach()
    policy = -torch.minimum(ratio*adv, ratio.clamp(1-clip,1+clip)*adv).mean()
    delta = ref_logp.detach()-logp
    kl = (delta.exp()-delta-1).mean()
    return policy, kl

def demonstration_loss(logp, advantage, gamma):
    adv = torch.as_tensor(advantage, device=logp.device).detach()
    # Stable algebraic form of exp(logp)/(exp(logp)+gamma).
    return -adv * torch.sigmoid(logp - __import__("math").log(gamma)).mean()

def lr_for_update(update, cfg):
    return cfg["learning_rate"] * min(1., update/cfg["warmup_updates"])

def setup_dist():
    rank,world=int(os.environ.get("RANK",0)),int(os.environ.get("WORLD_SIZE",1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK",0)))
    if world>1: dist.init_process_group("nccl")
    return rank,world

def barrier():
    if dist.is_initialized(): dist.barrier()

def reduce_gradients(params):
    for p in params:
        if p.grad is None: p.grad=torch.zeros_like(p)
        if dist.is_initialized(): dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

def grad_copy(params):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

def grad_norm(grads):
    return float(torch.stack([g.double().square().sum() for g in grads]).sum().sqrt())

def grad_summary(left,right):
    nl,nr=grad_norm(left),grad_norm(right)
    dot=float(torch.stack([(x.double()*y.double()).sum() for x,y in zip(left,right)]).sum())
    return {"online_plus_kl_grad_norm":nl,"demo_grad_norm":nr,
            "demo_online_cosine":dot/(nl*nr) if nl*nr else None}

def gather(value, world):
    if world==1: return [value]
    values=[None]*world
    dist.all_gather_object(values,value)
    return values

def rpc(root, request, timeout=7200):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    name=request["request_id"]
    assert "/" not in name
    req=root/(name+".request.json");res=root/(name+".response.json")
    if req.exists(): assert json.loads(req.read_text())==request, "Changed replay request"
    else: save(req,request)
    deadline=time.monotonic()+timeout
    while not res.exists():
        if time.monotonic()>deadline: raise TimeoutError(name)
        time.sleep(1)
    value=json.loads(res.read_text())
    if value.get("error"): raise RuntimeError(value["error"])
    assert value["request_id"]==name
    if request.get("mode")!="shutdown":
        assert value["adapter_sha256"]==request.get("adapter_sha256")
    return value

def generate(rows, adapter, name, epoch, cfg, rpc_root):
    requests=[]
    for row in rows:
        for index in range(4):
            requests.append({"id":row["id"],"rollout_index":index,
                "prompt_token_ids":row["prompt_token_ids"],
                "seed":rollout_seed(cfg,epoch,row["id"],index)})
    return rpc(rpc_root,{"request_id":name,"mode":"generate",
        "adapter":str(adapter),"adapter_sha256":sha(Path(adapter)/"adapter_model.safetensors"),
        "prompts":requests,"max_new_tokens":cfg["max_new_tokens"],
        "temperature":cfg["temperature"],"top_p":cfg["top_p"],"top_k":cfg["top_k"]})

def frozen_manifest(model):
    selected={}
    for name,p in model.named_parameters():
        if not p.requires_grad and ("embed_tokens" in name or "lm_head" in name or
            "layers.0.self_attn.q_proj.base_layer.weight" in name):
            selected[name]=hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert selected and all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
    return selected
