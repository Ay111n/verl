Small Math English-reference mixed-policy pilot (2026-11-17)

User explicitly authorized one small training run, reference correction where needed,
and final evaluation; no ablations. Later steering replaced the old Math128 evaluation
with full 666DEV. This overrides legacy R0-only/no-automatic-training rules for this run.
No other experiment or running process is modified.

Start: 2026-11-17 14:55:20 Asia/Shanghai. Four-hour end-to-end deadline 18:55:20.
At most two GPUs: actor0 + sampler5, then two TP1 evaluation replicas0/5.
Training allocation ends at17:20:20, with a measured next-batch guard.
No repeated DEV checkpoint selection; evaluate only the last completed fixed-plan adapter.
A budget-limited prefix is disclosed, never called a full pass.

Data: first128 Math questions in the existing frozen500 candidate order, independent
of English or Tibetan outcomes.98 original complete English references pass the added
structure/script checks.30 require full English re-solving with the gold numerical
value as a consistency hint. Exactly one repair attempt, output8192; never overwrite
an incorrect completion's final answer. Accept only complete, verifier-correct,
nonempty English-dominant think + Tibetan final with native EOS. Final-answer and
Unicode tests do not certify every reasoning step or semantic translation.
The finalized pilot has101 training questions:98 cached references and3 accepted
full re-solves;27 failed repairs are excluded. This gives7 updates and303 fresh
online responses in the fixed plan. Original order, hashes and reasons are retained.
Data construction belongs to finetuning/rl/scripts/prepare_luffy_math_pilot.py.
Reference source is recorded as cached/no-gold or gold-conditioned full re-solve.

Student: original en_cot_49800 merged checkpoint9000, LoRA16/alpha32/dropout0,
all seven attention/MLP projection classes; BF16 base/FP32 adapters.
Only Tibetan question enters the student prompt. The English-origin response is a
teacher-forced target including its own Tibetan answer; no splicing of two completions.

One epoch, batches16 questions (last smaller), fresh3 on-policy responses per question
plus one fixed successful reference. One optimizer update per rollout batch, no PPO reuse.
Rewards are unchanged numerical-verifier outcomes,0/1. Mixed mean=(sum3rewards+1)/4;
online advantage=reward-mean and demo advantage=1-mean, with no std normalization.
Online clipped surrogate eps.2; denominator is detached current HF logp before its
single update. Reference loss=-A_demo*mean(sigmoid(logp-log(.1))).
Equal four response means per question, then actual-batch question mean.
No KL/entropy/format/length rewards. Full-softmax sampling and optimized likelihood
both useT.6, top_p1/top_k-1; output cap8192. Frozen-old probabilities retain no gradients.
LoRA AdamW5e-6,2-update warmup,gradient clip1,weight decay0.
The practical demo denominator1 is not an unbiased correction of the filtered English
proposal. LUFFY-style adaptation: https://arxiv.org/html/2504.14945v5
Response-mean aggregation, LoRA,3+1 and short budget are explicit adaptations.

Imports the unique existing model loader and batched causal response-likelihood helper.
Numerical helper uses microbatch1, response-only head chunks512, native EOS and no prompt loss.
CPU tests cover advantages, detached denominator, clipping and shaping gradients.
First real rollout batch checks raw HF/vLLM logprob MAE<.05 for every trajectory before
any optimizer step. All frozen parameter gradients must remain absent; LoRA changes
and every adapter reload SHA are checked. Per-step online/demo gradient norms retained.

Evaluation: unchanged benchmark/eval/scripts/evaluate_same_source_dpo_316.py,
full Tibetan4-choice666DEV, ON, context32768/output32000,T.6/top_p.95/top_k20,
seed42+canonical index, two TP1 replicas, same input/template/extractor as frozen
SFT355/666 baseline. The earlier Math12816K protocol is not run.
Math training to multi-domain666DEV is a transfer check. One arm versus S0 gives
only an overall trend, not attribution to English reference or superiority over
plain GRPO/SFT. No TEST and no ablation.
Prediction scoring/analysis remains in benchmark. Source hashes, configs, logs and
ordinary outputs are returned and verified before eligible remote-output cleanup;
formal data, model and recovery states remain.

Entrypoints:
python scripts/luffy_math_pilot/test_core.py
CUDA_VISIBLE_DEVICES=5 python scripts/luffy_math_pilot/worker.py --model <S0> --rpc <run>/rpc
CUDA_VISIBLE_DEVICES=0 python scripts/luffy_math_pilot/train.py --config configs/luffy_math_pilot_20261117.json
CUDA_VISIBLE_DEVICES='' python scripts/luffy_math_pilot/run_pipeline.py --config configs/luffy_math_pilot_20261117.json
