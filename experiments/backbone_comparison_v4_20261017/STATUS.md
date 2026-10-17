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
