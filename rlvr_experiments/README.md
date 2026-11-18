# R0 RLVR 工程

R0 = verifier correctness，取值 0 或 1；validity 仅记录，不参与奖励。

LoRA rank=32、alpha=64；TP1/DP2；32 prompts × 8 responses；response 4096；50 updates。完整配置见 configs/base.yaml 和 configs/hardware。

入口 scripts/run_experiment.py --experiment R0，默认仅预览。contracts/ 管理独立合同；runs/ 保存未来输出。数据使用当前可用文件，翻译脚本不变。

旧 B0–B4 分桶及其 R0 训练、验证和准备数据已于 2026-11-18 从本地和 L20 永久删除。原冻结合同及路径仅供查阅。重新评测前需要准备新的输入和对应合同。

冻结 manifest 保留原始路径，核验时按配置的训练和评测根目录读取对应文件，并检查原有哈希。当前原 checkpoint-6000 合并模型缺失，tokenizer 参考包仅用于协议核验，模型缺项仍由 readiness 报告。

execution/reference.json 缺失，新版 verl 兼容性尚未验收，启动器保留旧版本限制；缺项必须报错，不能伪造通过。
