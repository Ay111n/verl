LUFFY-style English-source demonstration transfer for Tibetan Knowledge

Authorization: 2026-11-16 user explicitly requested implementation, CPU/GPU
acceptance, then matched A/B/C training for 75 updates and final 666DEV.
This authorization covers the otherwise non-automatic training/data/model rule.

Data and scientific comparison
388 unchanged reviewed Knowledge question IDs; one fixed, answer-verified,
complete English-origin trajectory each. Native output is English reasoning
plus Tibetan final. Only Tibetan questions enter online generation and
demonstration teacher forcing. English input is provenance, not a second loss
branch. No sentence/step alignment, external judging, or live teacher generation.
Training data/manifest live in finetuning/rl/final_data/luffy_knowledge_388_20261116.
Previous BO rewards only calibrate the initial NLL gradient scale and define
diagnostic difficulty strata; online updates use each arm's fresh policy.

A: online centered-advantage policy loss + KL.
B: A + alpha * per-response demonstration NLL.
C: mixed 4-online+1-correct-demo mean baseline, online loss +
   0.25 * (-A_demo * mean(sigmoid(logp-log(0.1)))) + KL.
All response means include native EOS and exclude prompts. Mean across four
online responses and then across the actual batch questions; no group std.
KL is sampled low-variance exp(ref_lp-lp)-(ref_lp-lp)-1, coefficient .001,
on online response tokens only, frozen original SFT through disabled LoRA.
Reward is unified rlvr-answer-v2 correctness 0/1. Truncation/format is diagnostic.
A correct reliable answer before truncation retains reward1; failed executions
are errors. Demonstration completeness filtering does not alter online reward.
No entropy or length reward. Online clip .2, one update per fresh rollout batch.

The old-policy HF denominator is the detached logp from the unchanged model
before its single optimizer update. Dropout=0, no PPO data reuse, and exactly
one update per batch make this equal to a separate old HF policy forward.
Ratio is numerically1 before updating; its numerator retains the policy gradient.
All optimized policy, frozen-reference and demonstration likelihoods use
log_softmax(logits / 0.6), matching the configured rollout temperature.
The full vocabulary softmax is used in training; rollout top-k/top-p filtering
remains a conventional sampling approximation and is not importance-corrected.
vLLM raw (temperature1) probabilities and fixed32 raw NLL are diagnostics only.
They are never substituted for the optimized policy or an English proposal.
Off-policy p_phi=1 and shaping follow the official practical LUFFY implementation;
this is not unbiased correction for the filtered English proposal.
LUFFY: https://arxiv.org/html/2504.14945v3
Implementation commit: Simplified-Reasoning/LUFFY
01f4ae066249e6965ce0d5512ccdf99a12895c74
Our 4+1, LoRA, clipping/KL and response-mean aggregation are explicit adaptations.

Fixed configuration
Same complete SFT en_cot_49800 merged checkpoint-9000. Same step0 LoRA seed42,
rank16/alpha32/dropout0, q/k/v/o/gate/up/down, BF16 base and FP32 LoRA.
AdamW lr1e-6, first5-update linear warmup, betas.9/.999, eps1e-8, weight_decay0,
grad norm clip1. Group batch16, last batch4, order seeds42/43/44, 3epochs,
75updates,1164 question visits and4656 newly sampled answers per arm.
Context32768/output8192, temperature.6/top_p.95/top_k20.
Demo max prompt+response8879; no 8192-total-length truncation.
Two independent actor replicas manually sum globally normalized gradients.
Actor GPUs3/4; dedicated TP1 vLLM GPU5; no other task devices are modified.
Sequential arms. Policy is synchronized only after committed optimizer update;
unique LoRA ID reload for every request and prefix caching disabled.
Every step adapter is immutable for policy provenance. Rolling recovery includes
optimizer/RNG;0/25/50/75 additionally contain full archived training state.
B alpha is frozen from two16-question initial gradient batches; no DEV tuning.

Acceptance and outputs
test_core.py plus existing test_answer_verifier_v2.py: analytic gradients,
causal labels/EOS, clip/stopped denominators, unequal final batch and exposures.
preflight.py: all388 IDs, fixed source hashes, prompt/token identity, verifier,
EOS, full demonstration lengths, exact75-step exposure plan.
train.py --mode calibrate: no optimizer updates; same initial LoRA.
train.py --mode smoke --smoke-id <unique-attempt>: four training questions including longest demo; two
fresh C updates (32rollouts), disabled-base invariance, HF/vLLM scoring,
LoRA reload effect, distributed equality, optimizer state validation.
Smoke updates are discarded; formal arms restart from identical step0.
A repeated acceptance attempt must use a new smoke-id and fresh policy-specific
rollouts; prior adapters and diagnostics are retained. Exact bitwise replay of
a BF16 GPU update is not assumed. The canonical smoke summary points to the
accepted attempt. Only a fully passed attempt unlocks formal training.

Launch from rlvr_experiments in verified inference environment:
python scripts/luffy_knowledge_388/preflight.py
CUDA_VISIBLE_DEVICES=5 python scripts/luffy_knowledge_388/worker.py --rpc runs/luffy_knowledge_388_20261117/rpc
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --standalone --nproc_per_node=2 -- scripts/luffy_knowledge_388/train.py --mode calibrate
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --standalone --nproc_per_node=2 -- scripts/luffy_knowledge_388/train.py --mode smoke --smoke-id smoke-temperature-1
CUDA_VISIBLE_DEVICES='' python scripts/luffy_knowledge_388/run_pipeline.py

Evaluation reuses benchmark/eval/scripts/evaluate_same_source_dpo_316.py
with this run name: exact previous SFT/DPO ON protocol, BO questions/instructions,
4choices, no English reference, 32768/32000, temperature.6/top_p.95/top_k20,
per-question seed42+index, twoTP1 replicas. Only step75, no intermediate DEV orTEST.
This RL reasoning pilot retains the current conversation's matched ON protocol;
the separate cross-reference SFT/distillation OFF experiment is not modified.
Primary metric666DEV answer ability accuracy, primary comparisonC-B with paired
bootstrap/McNemar; SFT baseline reused only after exact protocol/content checks.
Train46-question recovery is diagnostic, not held-out generalization.
Online/demo gradient norms and cosine are recorded each update; the online
gradient includes its common KL. Sampled objective value alone is not success.

Final CPU comparison stage
finish_analysis.py validates the hash-pinned analysis and SFT baseline artifacts,
waits for the three final DEV evaluations, and runs the benchmark-owned analyzer.
--check validates its five inputs without starting inference or waiting.
The stage uses no GPU and writes analysis_complete.json only after all raw online
rewards, policy hash chains, budgets and matched DEV protocols pass the audit.
Training/evaluation failures stop this stage with a separate failure record.
The returned SFT predictions are reused unchanged, not sampled again.

Temperature correction, 2026-11-17
The original luffy_knowledge_388_20261116 run was stopped after A step24.
It sampled at T0.6 but optimized raw T1 likelihoods; no B/C or DEV was run.
Its outputs/checkpoints are retained as superseded debugging evidence only.
Corrected run luffy_knowledge_388_20261117 reuses the identical frozen388 data,
recalibrates alpha at T0.6, repeats CPU/GPU acceptance, and initializes all arms
from the same original SFT. The old A weights/optimizer are not resumed.
CPU regression checks scaled normalization and analytic gradients; GPU smoke
compares the optimized likelihood to dense logits/T and raw scores to vLLM.
Official temperature implementation: Simplified-Reasoning/LUFFY commit
01f4ae066249e6965ce0d5512ccdf99a12895c74, luffy/verl/verl/workers/actor/dp_actor.py.

Read-only runtime audit
audit_runtime.py --run-root <run> --max-step <completed-A-step> --out <directory>
aggregates immutable completed updates, real generated token counts, stage times
and logged completion tails. It loads no model and performs no new sampling or
optimizer updates. Larger-data costs and hypothetical speedups are explicitly
projections, not performance benchmark results. It saves source hashes and a
worker-log prefix snapshot so later appends do not invalidate the evidence.

Authorized hold and bounded throughput acceptance, 2026-11-17
User chose: finish A at75, hold B/C, optimize first. hold_after_a.py takes over
only the identified pipeline/analysis controllers, lets the active A process
finish unchanged, then shuts down its dedicated sampler. Models and optimizer
checkpoints are retained. Do not restart run_pipeline.py while this hold applies.

The candidate implementation is separate from the running formal train.py:
- batched_likelihood.py: right-padded independent sequences and response-only
  head computation, with the same temperature, causal/EOS masks and sequence
  means. batched_update.py length-buckets microbatches without changing the
  actual question/response weights, KL, demonstration terms or gradients.
- stream_worker.py publishes only complete four-response question groups using
  the installed vLLM enqueue/step interface. A full-batch optimizer update still
  waits for every question; this is not a stale-policy asynchronous algorithm.
- benchmark_actor.py compares the original routine to microbatches1/2/4 and
  larger head chunks on the immutable existing step51 data and checkpoint50.
  The optimizer is a no-op: no parameter update is performed.
- benchmark_streaming.py tests two samplers plus one actor on16 questions, then
  replays exactly those outputs to compare the accumulated full-batch gradient.
  The component-sum timing is a counterfactual, not another timed full run.
- run_performance_acceptance.py waits for hold_complete.json and release of
  this task's GPUs3/4/5. It runs one64-output single-sampler batch, the same
  input/seed budget across two samplers, and one64-output overlapped batch.
  Total192 new diagnostic answers plus two short scoring warmups. No B/C
  training, DEV, or model updates are started. B/C remain held afterward.

CPU acceptance:106 tests, including seven new mask/gradient/order tests and the
existing99 objective/verifier checks. These are not GPU throughput results.
GPU acceptance requires gradient cosine>=.999, relative norm change<=.02,
relative gradient difference<=.05, and actor logprob MAE<=.03 on real outputs.
Real Adam moments are excluded from the gradient benchmark and disclosed in
its memory report. Every optimization result must state its device count,
actual token count, sampled length/termination outcomes and measured scope.
Same seeds do not guarantee bitwise identical GPU outputs across batch shapes.
The streaming benchmark currently verifies the combined gradient; production
integration must also retain the formal per-step online/demo gradient diagnostics.
These candidates have not replaced the formal trainer or justified a speedup
claim until performance_acceptance.json and its component results are checked.

Right-padding attention detail
Only real response positions contribute to the likelihood. Because all padding
is strictly to the right, causal attention already prevents real positions from
seeing it. The batched helper therefore omits an explicit padding mask, retaining
torch SDPA's causal Flash/GQA path. The CPU tests compare real response values
and A/B/C gradients against unpadded sequences, including token0 inside actual
content and unequal prompt/response lengths. This is not left-padding inference.

Comparable original resource allocation
benchmark_legacy_dp.py replays the already-produced single-sampler64 outputs
through the unmodified train.update C loss on two actors. This adds no sampled
answers or model updates. It reports actual two-actor time instead of dividing
a one-actor measurement by two. The original serial-stage baseline is the sum
of separately measured generation and training times; the report explicitly
identifies it as such. All streamed timings still use two samplers and one actor.

Supplementary scheduling probe
While the actor benchmarks ran, an additional64-answer two-sampler probe used
historical step51 response-token totals to balance16 questions into two groups
of8 (largest estimated costs first, then the lower-cost shard with capacity).
Every question retained its four original seeds and generation parameters.
The exact prior hash, placement and requests are retained in the run folder.
This adds64 diagnostic answers to the main192-answer suite. It does not change
formal training or the even/odd placement in the streaming benchmark. The latter
waits for empty worker queues before starting its timed interval.

The added SDPA CPU check covers causal right-padding and gradients in FP32.
CPU equivalence alone does not approve a BF16 GPU microbatch configuration;
the recorded per-candidate GPU thresholds must pass, without relaxed tolerances.

Installed vLLM request-ID compatibility
In vLLM0.19.1 enqueue() returns randomized internal request IDs, while the
finished RequestOutput uses the original external ID. stream_worker.py reads
that mapping from output_processor.request_states before processing results.
The regression test exercises the installed InputProcessor.assign_request_id
implementation. The first streaming attempt failed this mapping and its logs
remain intact; it performed no optimizer update.
retry_streaming_acceptance.py reuses all completed measurements and retries
only streaming plus the original two-actor replay in a fresh namespace. Each
sampler must first pass a four-response, eight-token streaming interface probe.
Those eight short responses are interface checks, not training answers. Failed
full requests are counted separately from completed diagnostic answers.
Untimed legacy-model loading can overlap the offline numerical replay, but a
gate prevents its measured update from running until the first actor exits.
The accepted explicit microbatch size is1. Larger sizes remain experimental
and must not be enabled unless both numerical and throughput checks justify it.
