# V4 初始发布状态

证据截止：2026-10-17 18:09（北京时间）。本页记录准备状态，后续运行以新的终态验收记录为准。

当前已登记 **24 个全新正式训练位置，已核实完成 0 个**。两种底座各自比较 M0 与 M3，两个固定划分、三个随机种子。真实 TRAIN 资源和模型资格测试尚未完成，正式共同训练预算仍未冻结，因此当前没有新架构提升结论。旧 V3 的失败与完成结果均保留，不填入本轮位置。

监控修复已覆盖 `/proc` 进程消失竞态，并加入独立工作进程超时保护。实际计算环境的 CPU 回归为 **148/148 通过**（测试25.688秒，包装流程35.079秒）；最初一次134通过、14个归档路径错误也在摘要中保留。这些测试使用小型随机模型和控制流程夹具，不是正式科学结果。

两个固定版本的模型文件已准备，Qwen3的12个文件共8,060,915,998字节已在主存储和持久存储逐文件核对SHA。仓库公开版本清单和校验摘要，预训练权重从官方固定版本下载。node1资源门控在CPU繁忙时等待，候选GPU空闲不会单独触发训练。

`configs/development_protocol.json`、`configs/canary_m2_a2_g8.json` 与 `docs/evaluation_protocol.json` 当前均为开发／TRAIN资格配置，不是正式冻结的运行许可。模型维度不同导致跨底座接口参数也不同：1.5B为1,618,568，4B为1,884,808；每个底座内M0/M3容量相同，通信均为58×128×float32＝29,696逻辑字节。跨底座比较会分别报告底座主效应与连接器改进的交互差异。

## 可移植 CPU 检查

在本目录使用Linux或WSL，设置 `MM_STANDARD_ROOT` 为本目录的 `standard` 绝对路径，设置 `CUDA_VISIBLE_DEVICES` 为空，运行：

```bash
export MM_STANDARD_ROOT="$PWD/standard"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -B -m unittest discover -s tests -p 'test*.py' -v
```

完整测试需要Torch、Transformers及数值依赖；实际已验证版本见 `resources/cpu_verification_summary.json`。Windows会明确跳过Linux进程组相关测试。`tests/test_archive.py` 仅将默认规范目录改为仓库内 `standard`，环境变量覆盖和其余AST语句不变；原始与公开SHA同时记录。

模型无关检查也可执行：

```bash
python -S -m unittest discover -s tests -p test_evaluate.py -v
python -S -m unittest discover -s tests -p test_queue.py -v
python -S -m src.mm_evaluate validate-protocol --protocol docs/evaluation_protocol.json --development
```

冻结划分的原始数据继续位于仓库 `data/a1`；本轮没有重划分或修改旧结果。当前未公开测试预测值，因为还没有通过正式验收的新运行。

## 本地工程资格尝试：资源停止

2026-10-17的一个Qwen3-4B本地工程尝试已实际启动，并因进程组主机内存超过预先约定的上限而停止：监督器记录峰值4,918,353,920字节，上限4,831,838,208字节（4.5GiB），终止原因为 `host_rss_limit`，退出码−15；退出后的进程组成员为空。该尝试另设11.5GiB GPU分配器预算，运行环境与node1不同，仅用于资源资格检查。

原始日志显示3/3权重分片加载完成，尚无已记录的优化进度、历史、检查点、预测或评分；已记录优化更新数为0，不据此推断未观察到的内部执行。监督器原始时间为10:44:49.116414至10:46:25.233715 UTC，原收据的单调计时耗时90.49482445299873秒原值保留，不用墙钟差覆盖。

本次属于资源资格失败，不能作为M3科学负分或架构不可实现的证据。当前接受的TRAIN canary仍为0，正式完成仍为0/24。逐字段来源SHA、原始安全日志和资源限制见 `engineering_results/qualification_attempts.json`；规范node1源码和评分没有被本地工程副本替换。后续内存优化仍在独立工程副本开发，尚无通过资格测试的结论。
