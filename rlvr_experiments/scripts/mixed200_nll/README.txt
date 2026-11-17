User-authorized fresh S0 run on the exact existing200 references. Only reference objective changes to -0.1*A_ref*mean(logp), at unchanged temperature0.6 and normalization1/(4B). All seeds, 13updates,3online samples,8192train and32000 full666DEV remain fixed. Actor0+samplers2/3/5, evaluation0/2; max4GPUs. No full-pool sampler. Original R0-only/no-auto-training restriction overridden for this expressly requested run. Compare frozen S0 and previous mixed200. No claim of improvement before evaluation.

Recovery after step 1:
- Two samplers rejected checkpoint-0001 on the first hash read; old logs did not
  record actual digest. Nine subsequent concurrent reads matched the saved hash.
  A transient read/publication issue is a hypothesis, not a proven root cause.
- Publish future adapter directories via fsync plus atomic rename. Retry reads
  for at most 10 seconds, logging actual/expected digests; persistent mismatch fails.
- Explicit --resume restores adapter, Adam state and Python/Torch/CUDA RNG, checks
  exact config, tensor equality and question order, and skips completed updates.
- Keep original failure responses in recovery_incident_0001; retry only failed
  subrequests with unchanged seeds. Successful subrequests and step1 are retained.
- CPU: 16 core/parallel/recovery tests pass. Runtime recovery audit is emitted
  as resume-0001.json after weights and optimizer are verified.
