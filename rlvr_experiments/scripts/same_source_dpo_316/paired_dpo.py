"""Offline DPO pilot primitives; frozen reference is the complete merged SFT."""
import hashlib
import math
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def load_model(path, seed=42, adapter=None, trainable_adapter=False, *, lora_rank=16, lora_alpha=32):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa",disable_mmap=True,device_map={"":f"cuda:{torch.cuda.current_device()}"})
    model.config.use_cache=False
    if adapter:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=trainable_adapter)
    else:
        model=get_peft_model(model,LoraConfig(r=lora_rank,lora_alpha=lora_alpha,lora_dropout=0.,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none",task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    model.train()
    return model

def response_logps(model,prompt_ids,response_ids,chunk_size=128,temperature=1.0,return_raw=False):
    """Causal response likelihood; optional temperature and detached raw diagnostics."""
    assert prompt_ids and response_ids and math.isfinite(float(temperature)) and temperature>0
    base=model.get_base_model() if hasattr(model,"get_base_model") else model
    device=next(base.parameters()).device
    tokens=torch.tensor([prompt_ids+response_ids[:-1]],dtype=torch.long,device=device)
    target=torch.tensor(response_ids,dtype=torch.long,device=device)
    hidden=base.model(input_ids=tokens,use_cache=False).last_hidden_state[0,len(prompt_ids)-1:]
    assert len(hidden)==len(target)
    values=[]
    def token_ce(h,t):
        logits=base.lm_head(h).float()
        policy=-F.cross_entropy(logits if temperature==1.0 else logits/temperature,t,reduction="none")
        if not return_raw:return policy
        if temperature==1.0:raw=policy.detach()
        else:
            with torch.no_grad():raw=-F.cross_entropy(logits.detach(),t,reduction="none")
        return torch.stack((policy,raw))
    for start in range(0,len(target),chunk_size):
        h=hidden[start:start+chunk_size];t=target[start:start+chunk_size]
        values.append(checkpoint(token_ce,h,t,use_reentrant=False) if torch.is_grad_enabled() else token_ce(h,t))
    if return_raw:
        combined=torch.cat(values,dim=1)
        return combined[0],combined[1].detach()
    return torch.cat(values)

def terms(chosen,rejected,ref_chosen,ref_rejected,nc,nr,target,gate,beta=.1,alpha=1.):
    dpo=-F.logsigmoid(beta*((chosen-rejected)-(ref_chosen-ref_rejected)))
    nll=-chosen/nc
    margin=chosen/nc-rejected/nr
    auxiliary=gate*torch.relu(torch.as_tensor(target,device=margin.device)-margin).square()
    return dpo+alpha*nll,auxiliary,{"dpo":dpo,"nll":nll,"margin":margin,
        "active":(gate>0 and bool((margin.detach()<target).item()))}

def pair_scores(model,row,language="bo"):
    p=row["prompt_"+language+"_token_ids"]
    return tuple(response_logps(model,p,row[s]["response_token_ids"]).sum() for s in ["chosen","rejected"])

def adapter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def runtime():
    import importlib.metadata as md,sys
    return {"python":sys.executable,"packages":{s:md.version(s) for s in ["torch","transformers","peft","numpy","safetensors"]},
        "device":torch.cuda.get_device_name(),"dtype":"bfloat16 base; float32 LoRA and reductions",
        "attention":"torch SDPA","chunk_size":128,"padding":False,"response_eos_included":True}
