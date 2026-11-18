"""A checkpoint directory is the transaction for a policy update and its queue effects."""
from pathlib import Path
import json
import os
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from contract import read
from scripts.luffy_math_pilot.common import save, sha


def snapshots(out):
    result = sorted(out.glob("checkpoint-[0-9][0-9][0-9][0-9]"))
    assert [p.name for p in result] == [f"checkpoint-{i:04d}" for i in range(len(result))]
    return result


def flush_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def commit(out, run, model, opt, state, event, cfg, model_path):
    before = time.monotonic()
    dest = out / f'checkpoint-{state["tx"]:04d}'
    if dest.exists():
        raise FileExistsError(dest)
    staging = Path(tempfile.mkdtemp(dir=out, prefix=".pending-"))
    model.save_pretrained(staging, safe_serialization=True)
    for item in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
                 "added_tokens.json", "vocab.json", "merges.txt"):
        src = model_path / item
        if src.is_file():
            try:
                os.link(src, staging / item)
            except OSError:
                shutil.copyfile(src, staging / item)
    digest = sha(staging / "adapter_model.safetensors")
    event = dict(event, policy_version=state["policy"], tx=state["tx"], output_adapter_sha256=digest)
    save(staging / "state.json", state)
    save(staging / "event.json", event)
    save(staging / "training_config.json", cfg)
    torch.save(dict(optimizer=opt.state_dict(), python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()), staging / "recovery.pt")
    save(staging / "manifest.json", dict(adapter_sha256=digest, tx=state["tx"], policy_version=state["policy"],
                                        base_model=str(model_path), config_sha256=sha(staging/"training_config.json")))
    for p in staging.iterdir():
        if p.is_file():
            with p.open("rb") as f:
                os.fsync(f.fileno())
    flush_dir(staging)
    staging.rename(dest)
    flush_dir(out)
    save(run / "steps" / (dest.name + ".json"), event)
    save(run / "latest.json", dict(checkpoint=str(dest), tx=state["tx"], policy_version=state["policy"],
                                  adapter_sha256=digest, cursor=state["cursor"], online=state["online"],
                                  nll=state["nll"], queue_size=len(state["pending"]), time_unix=time.time()))
    return dest, digest, time.monotonic() - before


def restore(checkpoint, model, optimizer, cfg):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    assert read(checkpoint / "training_config.json") == cfg
    meta = read(checkpoint / "manifest.json")
    digest = sha(checkpoint / "adapter_model.safetensors")
    assert digest == meta["adapter_sha256"]
    disk = load_file(str(checkpoint / "adapter_model.safetensors"))
    live = get_peft_model_state_dict(model)
    assert set(disk) == set(live) and all(torch.equal(v, live[k].detach().cpu()) for k,v in disk.items())
    saved = torch.load(checkpoint / "recovery.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    state = read(checkpoint / "state.json")
    assert all(int(v["step"]) == state["policy"] for v in optimizer.state.values())
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    torch.cuda.set_rng_state(saved["cuda"])
    return state, digest
