"""Resume the frozen three-arm pilot for two additional complete LR cycles."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from paired_dpo import adapter_hash, load_model, object_sha, read_rows, runtime, save, sha
from train import load_inputs, losses
from continuation_diagnostics import (
    accumulate_components, cycle_lr, gradient_summary, optimizer_digest,
    score_epoch, vector_norm,
)


def check_resume_scores(values, prior, targets):
    cached = {}
    for path in sorted(prior.glob("reference.shard*.jsonl")):
        for row in read_rows(path):
            assert row["key"] not in cached
            cached[row["key"]] = row
    assert len(cached) == 632
    checks = []
    for row, target in zip(values, targets):
        for side in ("chosen", "rejected"):
            ref = cached[row["pair_id"] + "|bo|" + side]
            difference = abs(row[side + "_sum"] - ref["sum_logprob"])
            length = target["nc" if side == "chosen" else "nr"]
            checks.append((difference, difference / length))
    maximum = max(v[0] for v in checks)
    per_token = max(v[1] for v in checks)
    assert maximum <= .05 and per_token <= 1e-5, (maximum, per_token)
    return {"compared_sequences": len(checks), "max_sum_logp_difference": maximum,
            "max_mean_logp_difference": per_token,
            "prior_sha256": {p.name: sha(p) for p in sorted(prior.glob("*.jsonl"))}}


def save_checkpoint(model, opt, params, exposure, config, step, output, rank, world):
    digest = adapter_hash(model)
    hashes = [None] * world
    dist.all_gather_object(hashes, digest)
    assert len(set(hashes)) == 1
    steps = sorted({int(v["step"]) for v in opt.state.values()})
    assert steps == [step], steps
    state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
    rngs = [None] * world
    dist.all_gather_object(rngs, state)
    if rank == 0:
        dest = output / f"checkpoint-{step}"
        assert not dest.exists(), dest
        dest.mkdir(parents=True)
        model.save_pretrained(dest)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(config["model"], local_files_only=True).save_pretrained(dest)
        torch.save({
            "optimizer": opt.state_dict(), "torch_rng_state": state["cpu"],
            "cuda_rng_state": state["cuda"], "rng_by_rank": rngs,
            "step": step, "exposure": exposure, "config": config,
            "parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
        }, dest / "training_state.pt")
        save(dest / "checkpoint_manifest.json", {
            "step": step, "pairs_seen": len(exposure), "adapter_trainable_hash": digest,
            "optimizer_steps": steps, "optimizer_entries": len(opt.state),
            "exposure_sha256": object_sha(exposure),
            "files": {p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        })
    dist.barrier()
    return digest


def run(a):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 2
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    records, targets, plan = load_inputs(a.data)
    continuation = json.loads((a.run / "continuation_plan.json").read_text())
    preflight = json.loads((a.run / "source_preflight.json").read_text())["arms"][a.arm]
    assert continuation["source_plan_sha256"] == sha(a.data / "training_plan.json")
    assert continuation["data_sha256"] == sha(a.data / "pairs.jsonl")
    assert continuation["targets_sha256"] == sha(a.data / "targets.jsonl")
    source = Path(preflight["source"])
    assert sha(source / "training_state.pt") == preflight["training_state_sha256"]
    assert sha(source / "adapter_model.safetensors") == preflight["adapter_file_sha256"]
    state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
    original = state["config"]
    assert state["step"] == 40 and len(state["exposure"]) == 316
    assert object_sha(state["exposure"]) == plan["order_sha256"]
    for key in plan:
        assert original[key] == plan[key], key
    weight = original["lambda"]
    target_key = original["target_key"]
    assert weight == (0. if a.arm == "base" else continuation["loss_settings_unchanged"]["lambda_true_and_shuffled"])
    assert target_key == ("shuffled_target" if a.arm == "shuffled" else "target")
    model = load_model(a.model, adapter=str(source), trainable_adapter=True)
    initial = adapter_hash(model)
    assert initial == preflight["adapter_trainable_hash"], (initial, preflight)
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 504 and all(p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW(params, lr=plan["learning_rate"], betas=tuple(plan["betas"]),
                            eps=plan["eps"], weight_decay=plan["weight_decay"])
    before_digest = optimizer_digest(state["optimizer"])
    opt.load_state_dict(state["optimizer"])
    assert optimizer_digest(opt.state_dict()) == before_digest
    assert len(opt.state) == 504 and {int(v["step"]) for v in opt.state.values()} == {40}
    for p in params:
        for key in ("exp_avg", "exp_avg_sq"):
            assert opt.state[p][key].shape == p.shape and opt.state[p][key].dtype == p.dtype
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    exposure = list(state["exposure"])
    del state
    out = a.run / a.arm
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "completion.json").exists()
    assert not (out / "steps.rank0.jsonl").exists()
    config = {**original, "source_step": 40, "final_step": 120, "additional_updates": 80,
              "model": a.model, "source_adapter": str(source), "source_preflight": preflight,
              "continuation_plan_sha256": sha(a.run / "continuation_plan.json"),
              "source_adapter_trainable_hash": initial, "runtime": runtime(),
              "component_probe": "separate autograd.grad observations; actual combined loss.backward unchanged",
              "rng_restore": "source rank0 CPU and two visible CUDA states; dropout=0 and frozen explicit data orders",
              "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}}
    if rank == 0:
        save(out / "config.json", config)
    started = time.time()
    scores = score_epoch(model, records, targets, plan, target_key, 40, out, rank, world)
    check = check_resume_scores(scores, a.run / "prior_scores" / a.arm, targets)
    assert optimizer_digest(opt.state_dict()) == before_digest
    if rank == 0:
        save(out / "resume_verified.json", {
            "step": 40, "adapter_trainable_hash": initial, "optimizer_tensor_digest": before_digest,
            "optimizer_entries": 504, "optimizer_steps": [40], **check,
            "elapsed_seconds": time.time() - started,
        })
    # No arm begins updating until all three source probability checks pass.
    deadline = time.time() + 1200
    while not all((a.run / arm / "resume_verified.json").exists() for arm in ("base", "true", "shuffled")):
        assert time.time() < deadline, "Waiting for all three resume checks timed out"
        time.sleep(2)
    dist.barrier()
    probe_steps = {41, *range(50, 121, 10)}
    largest = max(range(len(targets)), key=lambda i: targets[i]["gate"] *
                  max(0., targets[i][target_key] - targets[i]["initial_bo_margin"]) ** 2)
    for epoch in continuation["additional_epochs"]:
        assert sorted(epoch["indices"]) == list(range(316))
        assert object_sha([records[i]["pair_id"] for i in epoch["indices"]]) == epoch["pair_order_sha256"]
        probe_steps.add((epoch["epoch"] - 1) * 40 + epoch["indices"].index(largest) // 8 + 1)
    if rank == 0:
        save(out / "gradient_probe_plan.json", {"steps": sorted(probe_steps),
             "largest_initial_own_aux_pair_id": records[largest]["pair_id"]})
    new_tokens = 0
    step = 40
    with (out / f"steps.rank{rank}.jsonl").open("w", encoding="utf-8") as stream, (
            out / f"gradients.rank{rank}.jsonl").open("w", encoding="utf-8") as probes:
        for epoch in continuation["additional_epochs"]:
            epoch_seen = []
            for start in range(0, 316, plan["effective_batch"]):
                step += 1
                indices = epoch["indices"][start:start + plan["effective_batch"]]
                lr = cycle_lr(step, plan)
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.zero_grad(set_to_none=True)
                diagnostic = step in probe_steps
                buffers = {k: [torch.zeros_like(p) for p in params] for k in ("dpo", "nll", "aux")} if diagnostic else None
                values = []
                for i in indices[rank::world]:
                    r, t = records[i], targets[i]
                    base, aux, d, c, j = losses(model, r, t, target_key, plan)
                    loss = base + weight * aux
                    assert torch.isfinite(loss)
                    if diagnostic:
                        accumulate_components(
                            {"dpo": d["dpo"], "nll": plan["alpha"] * d["nll"], "aux": aux},
                            buffers, params, len(indices),
                        )
                    # Preserve the original numerical update path.
                    (loss / len(indices)).backward()
                    values.append({"base": float(base.detach()), "aux": float(aux.detach()),
                        "dpo": float(d["dpo"].detach()), "nll": float(d["nll"].detach()),
                        "margin": float(d["margin"].detach()), "active": float(d["active"]),
                        "chosen_delta": float(c.detach()) - t["bo_chosen"],
                        "rejected_delta": float(j.detach()) - t["bo_rejected"]})
                    del base, aux, d, c, j, loss
                for param in params:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                packed = torch.tensor([sum(v[k] for v in values) for k in values[0]],
                                      device="cuda", dtype=torch.float64)
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                averages = {k: float(v) / len(indices) for k, v in zip(values[0], packed)}
                if diagnostic:
                    for group in buffers.values():
                        for gradient in group:
                            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                    gd = gradient_summary(buffers, [p.grad for p in params], weight)
                    before_update = [p.detach().clone() for p in params]
                    param_norm = vector_norm(before_update)
                    del buffers
                norm = torch.nn.utils.clip_grad_norm_(params, plan["max_grad_norm"], error_if_nonfinite=True)
                opt.step()
                if diagnostic:
                    update_norm = vector_norm([p.detach() - b for p, b in zip(params, before_update)])
                    gd.update(step=step, epoch=epoch["epoch"], batch_pair_ids=[records[i]["pair_id"] for i in indices],
                              lr=lr, lambda_value=weight, parameter_norm=param_norm,
                              update_norm=update_norm, relative_update=update_norm / param_norm,
                              clip_coefficient=min(1., plan["max_grad_norm"] / (float(norm) + 1e-6)))
                    probes.write(json.dumps(gd) + "\n"); probes.flush()
                    del before_update, gd
                for i in indices:
                    new_tokens += len(records[i]["prompt_bo_token_ids"]) * 2 + targets[i]["nc"] + targets[i]["nr"]
                    exposure.append(records[i]["pair_id"]); epoch_seen.append(records[i]["pair_id"])
                record = {"step": step, "epoch": epoch["epoch"], "pairs": len(exposure),
                          "new_pairs": len(exposure) - 316, "batch_pairs": len(indices),
                          "lr": lr, "gradient_norm": float(norm), "diagnostic": diagnostic,
                          "elapsed_seconds": time.time() - started, "new_input_tokens": new_tokens,
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, **averages}
                stream.write(json.dumps(record) + "\n"); stream.flush()
                if rank == 0:
                    save(out / "status.json", record)
                    print(json.dumps(record), flush=True)
            assert object_sha(epoch_seen) == epoch["pair_order_sha256"]
            assert step == epoch["epoch"] * 40
            digest = save_checkpoint(model, opt, params, exposure, config, step, a.output, rank, world)
            score_epoch(model, records, targets, plan, target_key, step, out, rank, world)
            assert adapter_hash(model) == digest
            assert {int(v["step"]) for v in opt.state.values()} == {step}
    assert step == 120 and len(exposure) == 948
    if rank == 0:
        final = a.output / "checkpoint-120"
        save(out / "completion.json", {
            "status": "completed", "arm": a.arm, "source_step": 40, "optimizer_steps": 120,
            "additional_updates": 80, "pairs_seen": 948, "new_pairs_seen": 632,
            "unique_pairs": 316, "exposure_sha256": object_sha(exposure), "new_input_tokens": new_tokens,
            "adapter": str(final), "adapter_final_sha256": adapter_hash(model),
            "adapter_files": {p.name: sha(p) for p in final.iterdir() if p.is_file()},
            "elapsed_seconds": time.time() - started, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "config_sha256": sha(out / "config.json"), "data_parallel_world_size": world,
        })
    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=["base", "true", "shuffled"], required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
