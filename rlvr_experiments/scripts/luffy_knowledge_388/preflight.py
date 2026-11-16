"""CPU data/verifier/template acceptance and exact run contract."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from common import paths, read_rows, save, sha, prompts, verify, plan_indices, object_sha, VERIFIER_VERSION

def main():
    p=paths();cfg=json.loads((p["project_dir"]/"configs/luffy_knowledge_388_20261116.json").read_text())
    rows=read_rows(p["train_data"])
    assert len(rows)==len({r["id"] for r in rows})==388 and sha(p["train_data"])==cfg["train_sha256"]
    tok=AutoTokenizer.from_pretrained(str(p["model_path"]),local_files_only=True)
    lengths=[];rewards=[]
    for r in rows:
        assert r["domain"]=="knowledge"
        text,ids=prompts(tok,r);d=r["demo"]
        assert "|en|" in d["source_key"] and len(d["response_token_ids"])<=8192
        assert d["response_token_ids"][-1]==tok.eos_token_id
        assert hashlib_token(d["response_token_ids"])==d["response_token_sha256"]
        assert tok.decode(d["response_token_ids"],skip_special_tokens=False)==d["completion"] or tok.decode(d["response_token_ids"],skip_special_tokens=False).removesuffix(tok.eos_token)==d["completion"]
        v=verify(r,d,8192)
        assert v["correctness"]==1 and v["pair_eligible"] and v["protocol_valid"]
        lengths.append(len(ids)+len(d["response_token_ids"]));rewards.append(sum(r["old_bo_rewards"]))
    assert max(lengths)==8879 and max(lengths)<32768
    plan=plan_indices(388,cfg)
    assert len(plan)==75
    smoke=[max(range(388),key=lambda i:lengths[i]),
           next(i for i,x in enumerate(rewards) if x==0),
           next(i for i,x in enumerate(rewards) if x==2),
           next(i for i,x in enumerate(rewards) if x==4)]
    assert len(set(smoke))==4
    report={"passed":True,"rows":388,"maximum_prompt_plus_demo":max(lengths),
        "smoke_indices":smoke,"plan_sha256":object_sha(plan),"config":cfg,
        "train_sha256":sha(p["train_data"]),"verifier_version":VERIFIER_VERSION,
        "verifier_sha256":sha(p["project_dir"]/"rlvr/answer_verifier.py"),
        "updates":75,"rollouts_per_arm":4656,"no_model_loaded":True}
    save(p["run_root"]/"preflight.json",report)
    save(p["run_root"]/"training_plan.json",plan)
    print(json.dumps(report,ensure_ascii=False))

def hashlib_token(ids):
    import hashlib
    return hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()
if __name__=="__main__":main()
