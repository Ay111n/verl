# Same-source English–Tibetan margin DPO pilot

Explicitly authorized on 2026-11-16, isolated from historical outcome-only R0.
The user-requested pilot supersedes the local R0-only/no-automatic-training rule
for this task. Historical contracts and results are preserved.

## Data and objective

The sibling finetuning exporter retains 316 preselected BO-origin chosen/rejected
pairs (178 knowledge, 138 mathematics), with structural protocol validity and
complete native token sequences of at most 8192 tokens. Selection does not depend
on English margins. Both completions are scored under paired English/Tibetan
questions. Outputs retain English reasoning followed by Tibetan answers.

Native response tokens include EOS and exclude prompt tokens. S is summed log
probability and s=S/T:
- DPO = -logsigmoid(beta * [(S_bo+ - S_bo-) - (S_ref_bo+ - S_ref_bo-)]).
- NLL = -S_bo+/T+.
- m_bo = S_bo+/T+ - S_bo-/T-.
- t = s_ref_en+ - s_ref_en-, g = 1[t>0], both frozen.
- aux = g * relu(t-m_bo)^2.
- total = DPO + alpha*NLL + lambda*aux.

Base has lambda=0. True uses same-question English targets. Shuffled keeps the
same gate and permutes positive targets without fixed points within domain and
response-length quartiles. This tests target-value correspondence conditional
on the English-positive gate, not removal of all English information.
Initial hinge activation and gradient magnitudes may differ and are reported.

## Frozen budget

One epoch, 316 pairs, global batch 8 (last batch 4), 40 updates; seed42; AdamW
lr=1e-5, four warmup updates then linear decay, weight decay0, gradient clip1;
beta=.1, alpha=1; LoRA rank16, alpha32, dropout0 on q/k/v/o and gate/up/down.
All arms start with the same complete merged SFT and identical zero-B LoRA
initialization and global order. Two replicas per arm split each global batch;
loss/global-batch-size gradients are summed before clipping/updating.
Single-device execution is also supported. Parallelism adds no exposure.

Lambda is fixed from 16 domain/length-stratified training pairs: 0.1 times the
aggregate base-gradient norm divided by the aggregate true-auxiliary gradient
norm. True/shuffled share lambda. Calibration makes no optimizer update.
DEV does not select lambda, data, or checkpoint.

The HF reference cache shares the training model class, precision, native token
mask, shifted positions and chunked output projection. vLLM scores are diagnostic
only. Calibration checks zero-LoRA versus complete-SFT equivalence and reference
versus gradient-enabled scores before updating. Output logits use 128-token
chunks with backward recomputation; backbone uses gradient checkpointing and
torch SDPA. No TRL or separate flash-attn package is required.

## Entrypoints

Use explicit machine-configured paths and the inference interpreter.

1. Sibling finetuning export_same_source_dpo_316.py --root THESIS_ROOT.
2. cache_reference.py --data PAIRS --model SFT --output CACHE --shard I --shards 4.
3. freeze_reference.py --data PAIRS --reference CACHE.
4. train.py calibrate --data DATA_DIR --model SFT --run RUN_DIR.
5. Each of base/true/shuffled: python -m torch.distributed.run --standalone
   --nproc_per_node=2 -- train.py train --data DATA_DIR --model SFT --run RUN_DIR
   --arm ARM --output ADAPTER_DIR.
6. Sibling finetuning merge_same_source_dpo_adapter.py, then benchmark
   evaluate_same_source_dpo_316.py for all arms and a fresh SFT baseline.

Logs/configuration: ignored rlvr_experiments/runs. Data/reference targets:
finetuning. Models: finetuning/model/rl. Independent DEV outputs: benchmark.
Do not commit runtime products or machine configuration to public Git.

## Evaluation and interpretation

This English-CoT route uses Tibetan questions/instructions, thinking-on, no
English reference, context32768/output32000, temperature.6, top_p.95, top_k20,
min_p0. Native benchmark scoring is reused. Two independent TP1 replicas per model handle alternating canonical indices and
reassemble exactly 666 unique outputs. Fixed per-question seed
42+canonical-index is shared across four runs. Historical ON/OFF scores are
contextual; the fresh SFT run supplies the paired baseline.

Compare true vs base and true vs shuffled on all 666 questions with paired
transitions, bootstrap accuracy-difference intervals and exact McNemar. Report
extraction, truncation, format and output-length behavior. Training margins
describe fitting, not generalization. One seed/short epoch is a feasibility
pilot. No TEST inference or separate RLVR validation split.
