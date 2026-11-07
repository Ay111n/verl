# R0 RLVR 工程

R0 = verifier correctness，取值 0 或 1；validity 仅记录，不参与奖励。

LoRA rank=32、alpha=64；TP1/DP2；32 prompts × 8 responses；response 4096；50 updates。完整配置见 configs/base.yaml 和 configs/hardware。

入口 scripts/run_experiment.py --experiment R0，默认仅预览。contracts/ 管理独立合同；runs/ 保存未来输出。数据使用当前可用文件，翻译脚本不变。

execution/reference.json 缺失，新版 verl 兼容性尚未验收，启动器保留旧版本限制；缺项必须报错，不能伪造通过。
