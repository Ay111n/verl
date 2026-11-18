# 200 道 Knowledge 题的在线 GRPO 与间歇 NLL

训练入口使用同名配置 `configs/knowledge_xl_grpo_nll_200_20261119.json`。在 L20 的 `inference` 环境运行 `contract.py` 检查模型、固定样本、评测隔离和参考原文，再运行 `run.py`。控制器占用一张更新卡和至多三张采样卡，按实际空闲卡选择设备。完整 DEV 调用已有双副本 benchmark 入口。

16 题测速使用单独的模型目录、随机请求键与优化器。控制器将单遍、双遍和 DEV 的预计分钟数写入运行目录的 `forecast.json`。向实验负责人报告后，将该文件的 SHA-256 以字段 `forecast_sha256` 写入同目录的 `forecast_reported.json`，控制器即按估算选定的遍数从 S0 开始正式训练。

每个 `checkpoint-NNNN` 原子保存 LoRA、tokenizer、优化器、RNG、批次游标、待执行的 NLL 及 FIFO 顺序。恢复使用已经发布的最后一个完整目录。GPU 进程退出后重新运行同一控制器，可以重用同策略版本已经完成的分片响应。运行目录的 `steps` 保留每次更新的奖励、监督来源及队列记录。轮末模型和最终模型的位置登记在模型目录的 `epoch-N.json` 和 `final.json`。
