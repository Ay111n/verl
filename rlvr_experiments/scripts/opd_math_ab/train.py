"""Two-condition online reverse-KL pilot. Each batch uses the current student."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from transformers import AutoTokenizer
from common import (
    paths, read_rows, save, sha, render, prompt_text, teacher_text, response_hidden,
    reverse_kl, load_model, adapter_hash, tensor_manifest, current_identity,
    rpc, score_verification,
)


def adapter_save(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    return sha(path / "adapter_model.safetensors")


def generate(rows, tok, adapter, request_id, cfg, rpc_dir):
    prompts = [{"id": r["id"], "prompt_token_ids":
                tok.encode(render(tok, prompt_text(r)), add_special_tokens=False)} for r in rows]
    return rpc(rpc_dir, {
        "request_id": request_id, "mode": "generate", "adapter": str(adapter) if adapter else None,
        "adapter_sha256": sha(Path(adapter)/"adapter_model.safetensors") if adapter else None,
        "seed": cfg["seed"], "prompts": prompts, "max_new_tokens": cfg["max_new_tokens"],
        "temperature": cfg["temperature"], "top_p": cfg["top_p"], "top_k": cfg["top_k"],
    })


def kd_backward(model, tok, row, output, arm, divisor, chunk):
    response = output["response_token_ids"]
    student_prompt = output["prompt_token_ids"]
    expected = tok.encode(render(tok, prompt_text(row)), add_special_tokens=False)
    assert student_prompt == expected
    tp = tok.encode(render(tok, teacher_text(row, arm)), add_special_tokens=False)
    assert len(tp) + len(response) <= 32768
    assert len(response) > 0
    if output["finish_reason"] == "stop":
        assert response[-1] == tok.eos_token_id, ("Unexpected native stop", response[-1])
    torch.cuda.synchronize()
    start = time.monotonic()
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        teacher = response_hidden(model, tp, response).detach()
    torch.cuda.synchronize()
    teacher_seconds = time.monotonic() - start
    model.train()
    start = time.monotonic()
    student = response_hidden(model, student_prompt, response)
    loss = reverse_kl(student, teacher, model.get_base_model().lm_head.weight, chunk)
    assert torch.isfinite(loss), (row["id"], loss.detach())
    (loss / divisor).backward()
    torch.cuda.synchronize()
    backward_seconds = time.monotonic() - start
    return {
        "id": row["id"], "loss": float(loss.detach()),
        "student_prompt_tokens": len(student_prompt), "teacher_prompt_tokens": len(tp),
        "response_tokens": len(response), "teacher_seconds": teacher_seconds,
        "student_forward_backward_seconds": backward_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
    }


def checkpoint(model, optimizer, path, step, cfg, initial_hash, frozen_hash):
    path.mkdir(parents=True, exist_ok=True)
    adapter_dir = path / f"checkpoint-{step:04d}"
    digest = adapter_save(model, adapter_dir)
    state = {"step": step, "optimizer": optimizer.state_dict(), "config": cfg,
             "adapter": str(adapter_dir), "adapter_sha256": digest,
             "initial_adapter_hash": initial_hash, "frozen_hash": frozen_hash,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
             "python_rng": random.getstate()}
    tmp = path/"recovery.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(path/"recovery.pt")
    save(path/"latest.json", {k:state[k] for k in
         ["step","adapter","adapter_sha256","initial_adapter_hash","frozen_hash"]})
    return adapter_dir


def fixed_probe(model, prompt, response):
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        h = response_hidden(model, prompt, response[:16])
    return h.detach().clone()


def validate_adapter(model, tok, row, output, adapter, rpc_dir, label):
    ids = output["response_token_ids"][:32]
    p = output["prompt_token_ids"]
    request = {
        "request_id": label, "mode": "score", "seed": 42,
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": sha(Path(adapter)/"adapter_model.safetensors") if adapter else None,
        "prompts": [{"id": row["id"], "prompt_token_ids": p+ids, "score_start": len(p)}],
    }
    result = rpc(rpc_dir, request)
    model.eval()
    with torch.no_grad():
        h = response_hidden(model, p, ids)
        logits = model.get_base_model().lm_head(h).float()
        hf = torch.log_softmax(logits, -1).gather(
            1, torch.tensor(ids, device=logits.device)[:, None]).squeeze(1).cpu()
    vl = torch.tensor(result["records"][0]["logprobs"])
    error = float((hf-vl).abs().mean())
    assert error < .08, ("HF/vLLM fixed-token mismatch", error)
    return {"mean_abs_token_logprob_difference": error, "hf":hf.tolist(), "vllm":vl.tolist()}


def smoke(p, cfg, tok, rows):
    run = p["run_root"]/"smoke"
    run.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((p["run_root"]/"preflight.json").read_text())
    examples = [rows[i] for i in preflight["smoke_indices"]]
    model = load_model(str(p["model_path"]), cfg["seed"])
    optimizer = torch.optim.AdamW(
        [x for x in model.parameters() if x.requires_grad], lr=cfg["learning_rate"],
        betas=(.9,.999), eps=1e-8, weight_decay=0.)
    frozen = tensor_manifest(model)
    initial = adapter_hash(model)
    model_dir = p["model_output"]/"smoke"
    zero = model_dir/"initial"
    adapter_save(model, zero)
    outputs = generate(examples, tok, zero, "smoke-initial-16", cfg, p["run_root"]/"rpc")
    save(run/"rollouts.json", outputs)
    fixed = outputs["records"][0]
    probe_prompt = tok.encode(render(tok, teacher_text(examples[0], "solution")), add_special_tokens=False)
    before = fixed_probe(model, probe_prompt, fixed["response_token_ids"])
    validation0 = validate_adapter(model,tok,examples[0],fixed,zero,p["run_root"]/"rpc","smoke-score-zero")
    # Worst observed combined teacher+response context and longest response.
    lengths = [(len(tok.encode(render(tok,teacher_text(r,"solution")),add_special_tokens=False))
                +len(o["response_token_ids"])) for r,o in zip(examples,outputs["records"])]
    indices = list(dict.fromkeys([
        max(range(16),key=lambda i:lengths[i]),
        max(range(16),key=lambda i:len(outputs["records"][i]["response_token_ids"])),
        0,
    ]))
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for arm in ("answer","solution"):
        optimizer.zero_grad(set_to_none=True)
        for i in indices:
            result = kd_backward(model,tok,examples[i],outputs["records"][i],arm,len(indices),cfg["kl_chunk"])
            samples.append({"arm":arm,**result})
            print(json.dumps({"event":"smoke_backward",**samples[-1]}),flush=True)
        grad = torch.nn.utils.clip_grad_norm_(
            [v for v in model.parameters() if v.requires_grad],cfg["max_grad_norm"])
        assert torch.isfinite(grad) and grad > 0
        optimizer.step()
    changed = adapter_hash(model)
    assert initial != changed
    assert frozen == tensor_manifest(model)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    after = fixed_probe(model,probe_prompt,fixed["response_token_ids"])
    teacher_delta = float((before-after).abs().max())
    assert teacher_delta == 0., teacher_delta
    trained = model_dir/"updated"
    adapter_save(model,trained)
    validation1 = validate_adapter(model,tok,examples[0],fixed,trained,p["run_root"]/"rpc","smoke-score-updated")
    adapter_effect = max(abs(a-b) for a,b in zip(validation0["vllm"],validation1["vllm"]))
    assert adapter_effect > 1e-7
    report = {
        "passed":True, "runtime":current_identity(cfg["kl_chunk"]), "config":cfg,
        "rollout_count":16,"backpropagation_examples":len(samples),
        "initial_adapter_hash":initial,"updated_adapter_hash":changed,
        "teacher_fixed_prefix_hidden_max_difference":teacher_delta,
        "frozen_parameter_checksums_unchanged":True,
        "max_observed_teacher_plus_response_tokens":max(lengths),
        "truncated_rollouts":sum(x["finish_reason"]=="length" for x in outputs["records"]),
        "rollout_response_lengths":[len(x["response_token_ids"]) for x in outputs["records"]],
        "samples":samples,"zero_adapter_backend_check":validation0,
        "updated_adapter_backend_check":validation1,
        "loaded_adapter_effect_max_token_logprob":adapter_effect,
        "scope":"Train-side resource and engineering check; updates discarded before formal A/B."
    }
    save(run/"summary.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["samples","runtime","zero_adapter_backend_check","updated_adapter_backend_check"]}),flush=True)


def train(p, cfg, tok, rows, arm):
    assert json.loads((p["run_root"]/"smoke/summary.json").read_text())["passed"]
    out = p["model_output"]/arm
    run = p["run_root"]/arm
    out.mkdir(parents=True,exist_ok=True)
    run.mkdir(parents=True,exist_ok=True)
    recovery = out/"recovery.pt"
    state = torch.load(recovery,map_location="cpu",weights_only=False) if recovery.exists() else None
    if state:
        assert state["config"] == cfg
        assert sha(Path(state["adapter"])/"adapter_model.safetensors")==state["adapter_sha256"]
    model = load_model(str(p["model_path"]),cfg["seed"],
                       adapter=state["adapter"] if state else None,
                       trainable_adapter=bool(state))
    trainable = [v for v in model.parameters() if v.requires_grad]
    assert all("lora_" in n for n,v in model.named_parameters() if v.requires_grad)
    optimizer = torch.optim.AdamW(trainable,lr=cfg["learning_rate"],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        step = state["step"]
        initial = state["initial_adapter_hash"]
        frozen = state["frozen_hash"]
        adapter = Path(state["adapter"])
    else:
        step=0
        initial=adapter_hash(model)
        frozen=tensor_manifest(model)
        adapter=checkpoint(model,optimizer,out,step,cfg,initial,frozen)
    own_start = {"arm":arm,"initial_adapter_hash":initial,"train_sha256":sha(p["train_data"]),
                 "config":cfg,"runtime":current_identity(cfg["kl_chunk"]),"trainable_parameters":sum(v.numel() for v in trainable),
                 "trainable_names":[n for n,v in model.named_parameters() if v.requires_grad]}
    save(run/"initialization.json",own_start)
    other = p["run_root"]/("solution" if arm=="answer" else "answer")/"initialization.json"
    if other.exists():
        assert json.loads(other.read_text())["initial_adapter_hash"]==initial
    batches = [rows[i:i+cfg["batch_size"]] for i in range(0,len(rows),cfg["batch_size"])]
    assert len(batches)<=64
    torch.cuda.reset_peak_memory_stats()
    for number,batch in enumerate(batches):
        if number < step:continue
        start=time.monotonic()
        generated=generate(batch,tok,adapter,f"{arm}-step-{number:04d}",cfg,p["run_root"]/"rpc")
        optimizer.zero_grad(set_to_none=True)
        stats=[]
        for row,output in zip(batch,generated["records"],strict=True):
            assert row["id"]==output["id"]
            v=score_verification(row,output["completion"],output["finish_reason"],
                                 len(output["response_token_ids"]),cfg["max_new_tokens"])
            stats.append({**kd_backward(model,tok,row,output,arm,len(batch),cfg["kl_chunk"]),
                          "verification":v,"finish_reason":output["finish_reason"]})
            print(json.dumps({"event":"sample","arm":arm,"step":number+1,
                              **{k:v for k,v in stats[-1].items() if k!="verification"}}),flush=True)
        assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
        grad=torch.nn.utils.clip_grad_norm_(trainable,cfg["max_grad_norm"])
        assert torch.isfinite(grad)
        optimizer.step()
        adapter=checkpoint(model,optimizer,out,number+1,cfg,initial,frozen)
        record={"arm":arm,"step":number+1,"samples":stats,"mean_sequence_kl":sum(s["loss"] for s in stats)/len(stats),
                "grad_norm":float(grad),"rollout_seconds":generated["elapsed_seconds"],
                "elapsed_seconds":time.monotonic()-start,"adapter":str(adapter),
                "adapter_sha256":sha(adapter/"adapter_model.safetensors")}
        save(run/f"step-{number+1:04d}.json",record)
        print(json.dumps({k:v for k,v in record.items() if k!="samples"}),flush=True)
    assert frozen==tensor_manifest(model)
    save(run/"complete.json",{"arm":arm,"steps":len(batches),"examples":len(rows),
        "adapter":str(adapter),"adapter_sha256":sha(adapter/"adapter_model.safetensors"),
        "initial_adapter_hash":initial,"final_adapter_hash":adapter_hash(model),
        "frozen_parameter_checksums_unchanged":True,"config":cfg})
    print(json.dumps({"event":"arm_complete","arm":arm,"steps":len(batches)}),flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--mode",choices=["smoke","train"],required=True)
    a.add_argument("--arm",choices=["answer","solution"],default="answer")
    a.add_argument("--config",type=Path,required=True)
    args=a.parse_args()
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))==1
    p=paths()
    cfg=json.loads(args.config.read_text())
    tok=AutoTokenizer.from_pretrained(str(p["model_path"]),local_files_only=True)
    rows=read_rows(p["train_data"])
    assert sha(p["train_data"])==cfg["train_sha256"]
    if args.mode=="smoke":smoke(p,cfg,tok,rows)
    else:train(p,cfg,tok,rows,args.arm)
if __name__=="__main__":
    main()
