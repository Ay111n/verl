"""CPU prompt alignment and context limits; no generation or holdout scoring."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from common import paths, read_rows, save, sha, render, prompt_text, teacher_text

def main():
    p = paths()
    tok = AutoTokenizer.from_pretrained(str(p["model_path"]), local_files_only=True)
    train = read_rows(p["train_data"])
    dev = read_rows(p["eval_data"])
    records = []
    for row in train:
        student = render(tok, prompt_text(row))
        prompts = [student] + [render(tok, teacher_text(row, arm)) for arm in ("answer", "solution")]
        ids = [tok.encode(text, add_special_tokens=False) for text in prompts]
        assert all(len(x)>0 for x in ids)
        assert row["reference_reasoning"] not in student
        assert all(len(x)+16384 <=32768 for x in ids), (row["id"], list(map(len, ids)))
        records.append({"id": row["id"], "prompt_lengths": list(map(len, ids))})
    dev_lengths = [len(tok.encode(render(tok, prompt_text(r)), add_special_tokens=False)) for r in dev]
    assert max(dev_lengths)+16384<=32768
    assert len(train)==342 and len(dev)==512
    assert not set(r["id"] for r in train)&set(r["id"] for r in dev)
    longest = sorted(range(len(records)), key=lambda i: records[i]["prompt_lengths"][2], reverse=True)[:8]
    first = [i for i in range(len(records)) if i not in longest][:8]
    result = {"train_rows":len(train), "eval_rows":len(dev), "train_sha256":sha(p["train_data"]),
              "eval_sha256":sha(p["eval_data"]), "max_context":32768, "max_new_tokens":16384,
              "max_training_prompt_lengths":[max(r["prompt_lengths"][i] for r in records) for i in range(3)],
              "max_eval_prompt_length":max(dev_lengths), "native_eos_id":tok.eos_token_id,
              "smoke_indices":longest+first, "records":records}
    save(p["run_root"]/"preflight.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)
if __name__=="__main__":
    main()
