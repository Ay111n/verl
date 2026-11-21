"""Three frozen epochs of offline DPO plus chosen-response token-mean NLL."""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from peft import get_peft_model_state_dict, set_peft_model_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "same_source_dpo_316"))
from paired_dpo import adapter_hash, load_model, pair_scores, read_rows, runtime, save, sha, terms

ROOT = Path("/home/nlp/data/lyl")
NAME = "native_dpo399_20261122"
DATA = ROOT / "finetuning/rl/final_data" / NAME
RUN = ROOT / "verl/rlvr_experiments/runs" / NAME
OUT = ROOT / "finetuning/model/rl" / NAME
MODEL = ROOT / "finetuning/model/sft/en_cot_49800/merged/checkpoint-9000"


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    files = [DATA / f"epoch{i}.jsonl" for i in range(1, 4)]
    epochs = [read_rows(f) for f in files]
    ids = {r["id"] for r in epochs[0]}
    assert len(ids) == 399
    assert all(len(rows) == 399 and {r["id"] for r in rows} == ids for rows in epochs)
    hashes = [sha(f) for f in files]
    model = load_model(str(MODEL))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0)
    statefile = RUN / "resume.pt"
    completed = 0
    if statefile.exists():
        state = torch.load(statefile, map_location="cpu", weights_only=False)
        assert state["hashes"] == hashes
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        completed = state["step"]
        del state
    if rank == 0:
        save(RUN / "config.json", dict(runtime=runtime(), epochs=3, effective_batch=8,
             beta=.1, alpha=1, lr=1e-5, warmup=4, base=str(MODEL), epoch_sha256=hashes))
    started = time.monotonic()
    for epoch, rows in enumerate(epochs):
        for start in range(0, 399, 8):
            step = epoch * 50 + start // 8 + 1
            if step <= completed:
                continue
            batch = rows[start:start+8]
            lr = 1e-5 * (step / 4 if step <= 4 else (151-step) / 146)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(3, device="cuda", dtype=torch.float64)
            for row in batch[rank::world]:
                with model.disable_adapter(), torch.no_grad():
                    rc, rj = pair_scores(model, row)
                c, j = pair_scores(model, row)
                nc, nr = (len(row[s]["response_token_ids"]) for s in ("chosen", "rejected"))
                loss, _, detail = terms(c, j, rc, rj, nc, nr, 0., 0.)
                assert torch.isfinite(loss)
                (loss / len(batch)).backward()
                metrics += torch.stack([loss.detach(), detail["dpo"].detach(), detail["nll"].detach()])
                del loss, detail, c, j, rc, rj
            for p in params:
                dist.all_reduce(p.grad)
            dist.all_reduce(metrics)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if rank == 0:
                record = dict(step=step, epoch=epoch+1, ids=[r["id"] for r in batch],
                              lr=lr, grad_norm=float(norm), elapsed=time.monotonic()-started,
                              losses=(metrics/len(batch)).tolist())
                tmp = statefile.with_suffix(".tmp")
                torch.save(dict(step=step, hashes=hashes, adapter=get_peft_model_state_dict(model),
                                optimizer=optimizer.state_dict()), tmp)
                tmp.replace(statefile)
                save(RUN / "steps" / f"{step:04}.json", record)
                print(json.dumps(record), flush=True)
            dist.barrier()
    digest = adapter_hash(model)
    hashes_by_rank = [None] * world
    dist.all_gather_object(hashes_by_rank, digest)
    assert len(set(hashes_by_rank)) == 1
    if rank == 0:
        path = OUT / "checkpoint-0150"
        model.save_pretrained(path)
        AutoTokenizer.from_pretrained(MODEL, local_files_only=True).save_pretrained(path)
        save(RUN / "complete.json", dict(step=150, visits=1197, path=str(path), digest=digest,
             adapter_file_sha256=sha(path/"adapter_model.safetensors")))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

