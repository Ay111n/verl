"""Verify completed native-EOS response backpropagation on a cached smoke sample."""
import json
import torch
from transformers import AutoTokenizer
from common import paths, read_rows, load_model, save
from train import kd_backward

p=paths()
cfg=json.loads((p["run_root"]/"smoke/summary.json").read_text())["config"]
outputs=json.loads((p["run_root"]/"smoke/rollouts.json").read_text())["records"]
complete=[r for r in outputs if r["finish_reason"]=="stop"]
tok=AutoTokenizer.from_pretrained(str(p["model_path"]),local_files_only=True)
assert all(r["response_token_ids"][-1]==tok.eos_token_id for r in complete)
output=min(complete,key=lambda r:len(r["response_token_ids"]))
row=next(r for r in read_rows(p["train_data"]) if r["id"]==output["id"])
model=load_model(str(p["model_path"]),cfg["seed"])
records=[]
for arm in ["answer","solution"]:
    model.zero_grad(set_to_none=True)
    records.append({"arm":arm,**kd_backward(model,tok,row,output,arm,1,cfg["kl_chunk"])})
    grad=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
    assert torch.isfinite(grad) and grad>0
    assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
save(p["run_root"]/"smoke/completed_response_check.json",
     {"passed":True,"completed_native_eos_responses":len(complete),"eos_token_id":tok.eos_token_id,
      "backpropagation":records,"optimizer_updates":0})
print(json.dumps({"passed":True,"completed_native_eos_responses":len(complete),
                  "checked_response_tokens":len(output["response_token_ids"])}),flush=True)
