# 硬件配置

1. 当前保留的双卡参数：configs/hardware/dual_l20_tp1_dp2.yaml。完整算法和 sampling 继承 configs/base.yaml，使用同步 trainer；两张卡共置 actor/rollout。
2. configs/hardware/dual_l20_tp1_dp2.project_overlay.resolved.json 是当前自定义配置层的完整解析结果，不含上游 Hydra 默认值，也不是实测配置。环境和路径占位符按 profile 解析；local 模型未配置时为 null。
3. contracts/hardware/historical_dual_l20.resolved.json 是原始完整 Hydra 解析快照，原样保留，实际 TP2/DP1。它不能证明 TP1/DP2 通过新的硬件门槛。当前 TP1/DP2 的全量 Hydra 合同及实测验收未包含在本地副本中，execution/reference.json 保持缺失，不根据早期快照拼接伪造。
4. three_gpu_plan_20261022/ 是原三卡规划，launch_ready=false。原文包含迁移前本地路径、原服务器状态和哈希，作为历史设计保持原文；当前代码位置为此 RLVR 工程。本轮不实现三卡、不运行 GPU。
