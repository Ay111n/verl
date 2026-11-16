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
vLLM raw generation probabilities are recorded as a backend diagnostic, never
used as an English-source importance denominator.
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
CUDA_VISIBLE_DEVICES=5 python scripts/luffy_knowledge_388/worker.py --rpc runs/luffy_knowledge_388_20261116/rpc
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --standalone --nproc_per_node=2 -- scripts/luffy_knowledge_388/train.py --mode calibrate
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --standalone --nproc_per_node=2 -- scripts/luffy_knowledge_388/train.py --mode smoke --smoke-id smoke-acceptance-3
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
