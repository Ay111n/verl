English-solution feedback for Tibetan Math: minimal A/B online-distillation pilot

Authorization: 2026-11-16 user requested the next experiment, at most two GPUs.
This is a separate named experiment; it does not change R0 defaults.

Scientific contrast
  Student always receives the Tibetan question and existing Tibetan suffix.
  Native SFT output protocol remains English reasoning plus Tibetan final.
  Both teachers use the same frozen SFT checkpoint-9000, Tibetan and English
  questions, and the same gold. Only solution adds the cached verified English
  reasoning. The initial SFT base serves as teacher with LoRA disabled.
  No additional teacher training, GRPO, NLL, reward reweighting, or token labels.

Data
  500 previously fixed Math candidates were reserved on the training side before
  selecting references; 342 have a completed correct nonempty English reference.
  The first qualifying original rollout index is selected; Tibetan outcomes are
  not used for inclusion. Source is the 2026-11-15 cached 1000-question probe.
  512 independent Math development questions are not filtered by English success.
  ID, exact/lexical question overlap and observed SFT containment are screened.
  This is not semantic decontamination, final TEST, Knowledge, or 666DEV.
  Manifests and formal catalog live in finetuning/rl/final_data/opd_math_ab_20261116
  and benchmark/datasets/rlvr_math_holdout_20261116.

Loss and implementation
  For every actual student prefix, full-vocabulary KL(p_student || q_teacher).
  Mean across response positions (including native EOS), then mean across samples.
  Prompt and padding are excluded by causal hidden-position slicing.
  Teacher is stop-gradient and eval mode. Output head remains frozen.
  custom ReverseKL recomputes position chunks for exact loss/hidden gradients;
  no top-K or vocabulary truncation. CPU double/float and CUDA BF16 are checked
  against native autograd, including numerical gradcheck and identical p=q.
  This is exact relative to the specified BF16-head/FP32-softmax computation,
  not a claim of real-number numerical exactness.
  One response per prompt; one optimizer update per fresh rollout batch.
  A and B use their own changing policy. Old BO trajectories are not training data.
  Each policy checkpoint is hashed; vLLM removes the old adapter and loads a new
  LoRA ID. Prefix caching is disabled. GPU0 shares teacher/student base; GPU2
  generates. The smoke compares fixed-token HF/vLLM scores before/after updates
  and verifies unchanged disabled-adapter teacher outputs.

Locked first-pilot budget
  342 samples once, effective batch 8, microbatch 1, 43 optimizer updates.
  Last batch has 6 samples and uses its actual sample count for normalization.
  LoRA r16/alpha32/dropout0; seven attention/MLP linear projections, no lm_head.
  AdamW lr5e-6, betas .9/.999, eps1e-8, weight_decay0, grad_clip1, no scheduler.
  Seed20261116. Sampling temperature.6/top_p.95/top_k20; KL temperature1.
  Context32768, completion16384; native EOS. Same lengths/settings for A and B.
  All checkpoints are preserved on L20; recovery.pt points to the last committed
  optimizer state and adapter. No holdout-based checkpoint selection.
  The primary paper reference is COPSD arXiv2605.09548v1 section5.3/AppendixD:
  https://arxiv.org/html/2605.09548v1
  Its reported rank64/alpha128, batch32, temperature1.1 and completion2048 differ.
  This pilot transfers fixed-teacher full-vocabulary reverse-KL, not that full
  setup or a validated Tibetan result. Lower rank/batch, longer output budget
  and existing sampling protocol are explicit task/hardware choices.

Evaluation
  benchmark/eval/scripts/opd_math_ab/evaluate.py
  Baseline S0, final answer arm, final solution arm, same 512 Tibetan-only inputs,
  one response and fixed per-question seeds, no English/gold/reference.
  Unified rlvr-answer-v2 exact numeric scoring. Structural protocol and truncation
  are diagnostic metrics. Valid extracted final on truncated output follows the
  existing verifier; completed-correct is reported separately.
  benchmark/eval/analysis/opd_math_ab_20261116/compare.py reports solution-answer
  primary difference, solution-S0, paired gains/harms, question bootstrap 95% CI
  and exact McNemar. One training seed is a pilot; question bootstrap cannot
  cover training randomness. Matched updates are not equal compute.

Run (L20 inference environment; machine paths in configs/paths.json)
  python scripts/opd_math_ab/test_loss.py
  python scripts/opd_math_ab/preflight.py
  CUDA_VISIBLE_DEVICES=2 python scripts/opd_math_ab/worker.py --rpc runs/opd_math_ab_20261116/rpc
  CUDA_VISIBLE_DEVICES=0 python scripts/opd_math_ab/train.py --mode smoke --config configs/opd_math_ab_20261116.json
  CUDA_VISIBLE_DEVICES='' python scripts/opd_math_ab/run_pipeline.py

The controller requires a passing smoke. While training, eval clients have no CUDA
visibility and reuse GPU2. Only after training exits may a second worker use GPU0.
All model generations run TP1; the task never requests any third GPU.

Evaluation scheduling revision (2026-11-17 Beijing time)
  All final S0/A/B evals submit 64 questions per RPC, with at most 16 active
  sequences. Final results use eval/results/opd_math_ab_20261116/batch64.
  Earlier partial batch16 S0 outputs remain separate and are not compared.
  Training batch8 and all scientific settings remain unchanged. A verified
  active answer trainer can be adopted with --adopt-answer-pid; it is not
  restarted. The original formal protocol is preserved; execution_revision.json
  records the scheduling change, source hashes and adopted process identity.
