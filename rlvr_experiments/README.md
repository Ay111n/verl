# R0 RLVR 工程

R0 = verifier correctness，取值 0 或 1；validity 仅记录，不参与奖励。

LoRA rank=32、alpha=64；TP1/DP2；32 prompts × 8 responses；response 4096；50 updates。完整配置见 configs/base.yaml 和 configs/hardware。

入口 scripts/run_experiment.py --experiment R0，默认仅预览。contracts/ 管理独立合同；runs/ 保存未来输出。数据使用当前可用文件，翻译脚本不变。

历史数据的只读 CPU 核验使用 `python scripts/run_experiment.py --experiment R0 --profile server --run-dir /tmp/rlvr-path-check --validate-data`。L20 上使用根目录规范中的 inference 解释器，Windows 本机使用 `--profile local`。旧 400 条内部评测和 2,000 条验证集保持 `evaluation_status=retired`，来源和评测 Parquet 位于 benchmark 的退役归档。活动评测与数据准备入口会明确报出退役限制。

冻结 manifest 保留原始路径，核验时按配置的训练和评测根目录读取对应文件，并检查原有哈希。当前原 checkpoint-6000 合并模型缺失，tokenizer 参考包仅用于协议核验，模型缺项仍由 readiness 报告。

execution/reference.json 缺失，新版 verl 兼容性尚未验收，启动器保留旧版本限制；缺项必须报错，不能伪造通过。
