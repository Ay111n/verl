# Fixed mixed200 English-reference GRPO pilot

Explicit user authorization: fixed200 (140Knowledge/60Math), reuse correct English traces,
fully re-solve failed/missing traces, at most four GPUs, entire run within four hours,
full666DEV versus the frozen S0 baseline. This overrides legacy R0-only/no-auto-train
rules only for this run. No TEST, no ablations, no changes to the frozen200 IDs.

Start 2026-11-17 18:31:45 Asia/Shanghai; deadline22:31:45. Reference allocation55min;
training stops admitting batches by21:01:45; final666DEV retains32768/32000 ON,
T0.6/top_p0.95/top_k20, seed42+canonical index and the same two-replica jobs as S0.
Four reference workers use0/2/3/5, then actor0+samplers2/3/5, then eval0/2.
All phases together never exceed four GPUs. Only owned process groups are stopped.

The original luffy_math_pilot loss is unchanged: 3on-policy Tibetan responses plus
one verified English-origin complete response, reward-minus-group-mean advantages,
online clipped surrogate and demo sigmoid shaping gamma0.1. LoRA16/alpha32,
AdamW5e-6, one epoch,13updates, train cap8192. No reward reweighting, KL or entropy term.
The shared prompt renderer now selects the row's actual domain so Knowledge choices
are present. Three disjoint sampling lanes use identical adapter versions and preserve
all per-row seeds, identities and response order before the single optimizer update.
CPU objective tests plus parallel-identity/domain tests run before training; the first
actual batch must also pass HF/vLLM raw-logprob MAE<0.05 per response.

Reference preparation belongs to finetuning/rl/scripts/prepare_mixed200_references.py.
The5 valid cached references retain native EOS. Historical text omits the displayed EOS,
so text/token alignment ignores only that trailing display marker, never target tokens.
Missing references first use S0 English-only solving; failed attempts use full re-solves
with the gold as a consistency hint, separately labeled. Require correct outcome,
closed English-dominant reasoning, Tibetan final, valid native EOS and context length.
Never splice the final from a different response, patch a wrong final, silently drop,
or replace fixed questions. If all200 cannot pass within the allocation, stop before
training and retain partial references for an explicit follow-up. Outcome checks do
not prove every reasoning step; gold-conditioned traces can rationalize the answer.

Reusable entrypoints remain in scripts/luffy_math_pilot; this folder contains only
the bounded mixed200 lifecycle. Data and model artifacts are excluded from Git.
