"""Freeze content-bounded English hints for native Tibetan online rollouts."""
import json
from pathlib import Path
import re
import sys
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/"scripts/knowledge_grpo_intermittent_nll_v3"), str(ROOT/"finetuning/rl/scripts")]
import probe_formateval_prefix94 as probe
from probe_formateval_prefix94 import sha, read, save, digest

NAME = "formateval_priority512_prefix_rescue_20261121"
CONFIG = PROJECT/"configs"/(NAME+".json")
RUN = PROJECT/"runs"/NAME
OUT = ROOT/"finetuning/model/rl"/NAME


def prepare():
    cfg = read(CONFIG)
    plain = PROJECT/"runs"/cfg["baseline"]
    en = PROJECT/"runs"/cfg["teacher_source"]
    pre, audit, teacher = read(plain/"preflight.json"), read(en/"preflight.json"), read(en/"teacher_complete.json")
    assert sha(en/"teacher_complete.json") == cfg["teacher_sha256"]
    assert sha(en/"training_prompts.json") == teacher["prompt_sha256"] == audit["prompt_sha256"]
    assert sha(plain/"training_prompts.json") == pre["prompt_file_sha256"] == cfg["plain_prompt_sha256"]
    ds=ROOT/cfg["training_data"]
    assert sha(ds)==cfg["train_sha256"]==pre["train_sha256"]==audit["train_sha256"]
    assert sha(ds.parent/"manifest.json")==cfg["dataset_manifest_sha256"]
    assert read(ds.parent/"manifest.json")["verification"]["protected_matches"]==0
    assert pre["model_hashes"]==audit["model_hashes"] and pre["model_path"]==str(ROOT/cfg["model_path"])==audit["model_path"]
    assert sha(ROOT/"benchmark/eval/scripts/mcqa_answer_extraction_multilingual.py")==cfg["choice_verifier_sha256"]
    assert sha(Path(probe.__file__))==cfg["prefix_rule_script_sha256"]
    base=ROOT/cfg["model_path"]
    for filename,h in pre["model_hashes"].items():
        assert sha(base/filename)==h,filename
    tok=Tokenizer.from_file(str(base/"tokenizer.json"))
    bo, paired=read(plain/"training_prompts.json"),read(en/"training_prompts.json")
    assert len(bo)==512 and list(bo)==list(paired)==list(teacher["rewards"])
    assert digest("\n".join(bo)+"\n")==cfg["id_order_sha256"]
    refs={}
    for uid, q in bo.items():
        eng=paired[uid]["en"]
        assert eng["canonical_answer"]==q["canonical_answer"] and q["options_bo"]==paired[uid]["bo"]["options_bo"]
        options=list(eng["options_en"].values())+list(q["options_bo"].values())
        patterns=[probe.LABEL]+[re.compile(r"(?<!\w)"+r"\s+".join(re.escape(w) for w in opt.split())+r"(?!\w)",re.I) for opt in options if opt.strip()]
        for rep,ok in enumerate(teacher["rewards"][uid]):
            if not ok:
                continue
            fn=en/"teacher_responses"/uid/f"{rep}.json.gz"
            r=read(fn)
            assert (r["id"],r["rep"],r["prompt_sha256"],r["finish_reason"])==(uid,rep,eng["prompt_sha256"],"stop")
            assert probe.judge(eng,r["completion"],"stop",r["output_tokens"])["reward"]==1
            ids=r["response_token_ids"]
            if ids.count(tok.token_to_id("</think>"))!=1 or ids[0]!=tok.token_to_id("<think>"):
                continue
            thought=tok.decode(ids[1:ids.index(tok.token_to_id("</think>"))],skip_special_tokens=False)
            hits=[m.start() for rx in patterns if (m:=rx.search(thought))]
            if not hits:
                continue
            ends=list(probe.END.finditer(thought[:min(hits)]))
            if not ends:
                continue
            prefix=thought[:ends[-1].end()].strip()
            if not re.search(r"[A-Za-z]{2,}",prefix):
                continue
            forced="<think>\n"+prefix+"\n"
            tokens=tok.encode(q["prompt"]+forced,add_special_tokens=False).ids
            assert tokens[:len(q["prompt_token_ids"])]==q["prompt_token_ids"]
            if len(tokens)+cfg["max_new_tokens"]>cfg["max_model_len"]:
                continue
            refs[uid]=dict(source_rep=rep,source_sha256=sha(fn),forced=forced,prompt_token_ids=tokens,prompt_sha256=digest(q["prompt"]+forced))
            break
    RUN.mkdir(parents=True,exist_ok=True)
    save(RUN/"prefixes.json",refs)
    save(RUN/"training_prompts.json",bo)
    save(RUN/"preflight.json",dict(questions=len(bo),eligible=len(refs),config_sha256=sha(CONFIG),prompt_file_sha256=sha(RUN/"training_prompts.json"),
         prefixes_sha256=sha(RUN/"prefixes.json"),teacher_sha256=cfg["teacher_sha256"],model_hashes=pre["model_hashes"],train_sha256=sha(ds)))
    print(f"Frozen {len(refs)} eligible references on {len(bo)} questions.",flush=True)
