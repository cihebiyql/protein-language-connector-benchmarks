# 仓库与方法文献复现评估

核查日期：2026-10-15。依据已取回的完整 5 轮对话，执行范围采用最后一轮确定的 T1/T2/T3：ESM2-150M + Qwen2.5-1.5B-Instruct，先比较普通预测头、输入连接和内部连接。历史轮次中的大型蛋白质聊天模型与通用多 Agent 系统作为后续对照。

**结论：首轮三测试具备可实现的模型接口基础；现有公开仓库没有直接提供 Benz2022/Kliche2024 上的同一套基准。** 需要由本项目建立数据适配器、固定划分与评测器。OPUS 与 ProtT3 可以作为第二阶段独立复现对象。代码下载完成与模型实验跑通分别记录。

本次已浅克隆 12 个仓库；保留各自 `.git`，固定当前 commit。已下载 10 份有效 PDF（含 OPUS 补充表），均核验文件头并用 pypdf 解析页数；另外保存 ProtNote 完整 XML、Pep-Instructions 出版社完整 HTML、BioReason-Pro 官方元数据。清单为 `sources/repository_manifest.json` 与 `papers/methods/manifest.json`。`sources/method_scope_inventory.json` 覆盖五轮中明确命名的 64 项方法、论文、基础模型和主要证据资源，逐项标注已取得、其他分支负责、延后及获取受阻。所有仓库仅进行静态审阅，未执行其训练、安装或下载脚本。

## 1. 仓库清单和第一轮作用

| 仓库 | 固定 commit 前 12 位 | 已取回代码的作用 | 当前复现判断 |
|---|---|---|---|
| [OPUS-PLLM](https://github.com/Fanchuana/OPUS-PLLM) | `a9597043fb0c` | 真实公开蛋白质→LLM输入软 token 基线 | 组件入口已核验；老环境及量化栈需要独立适配 |
| [ProtT3](https://github.com/acharkq/ProtT3) | `f427daca2cf9` | Q-Former 输入连接参考 | 有训练/评测代码及 OSF 资源；README 命令存在问题 |
| [MiniMind](https://github.com/jingyaogong/minimind) | `f659b55761b7` | 透明的 Transformer 中间状态实验台 | 可参考逐层循环；第一轮无需重新训练该模型 |
| [MiniMind-V](https://github.com/jingyaogong/minimind-v) | `1862b633fc08` | 现成连续向量→输入位置投影结构 | 可借鉴接口；原仓库输入模态是图像 |
| [C2C](https://github.com/thu-nics/C2C) | `bdb932aa6363` | 跨 LLM KV 投影与融合 | 有公开 Fuser；模型与本轮蛋白质接口不同 |
| [Draft-KV](https://github.com/Svardfox/Draft-KV) | `e369f1b26298` | T2 正确/错配/接收端独立对照的直接依据 | 有代码；已训练桥权重仍标为 coming soon |
| [LatentMAS](https://github.com/Gen-Verse/LatentMAS) | `9a9e4d331eb1` | 潜在状态传递与记忆参考 | 通用 LLM 多 Agent 评测，第一轮无需运行 |
| [StateBridge](https://github.com/YanwenPneg/StateBridge) | `3f6bf5442c6e` | 隐状态尺度与几何对齐参考 | 适用于其假设下的 LLM 通信；不能直接套用为蛋白质连接 |

## 2. OPUS-PLLM：原始论文与可获取组件

原始论文：[Advancing generative large language models toward discriminative performance in protein function prediction](https://link.springer.com/article/10.1186/s13059-026-04109-8)，Genome Biology，2026-05-21，DOI `10.1186/s13059-026-04109-8`。本地正文 28 页，附加文件 1 为 28 页补充表。

源码中的真实连接流程为 ESM2-650M → 池化/模态对齐 → 8 个投影 token → Galactica/Llama 等语言模型。`multi_modality_model/cstp_v3/modelling.py:21` 明确使用 `esm2_t33_650M_UR50D`；论文与源码的 1280 维编码器不同于本轮拟用的 ESM2-150M/640 维，已发布权重不能直接替换编码器后继续使用。

具体入口：

- `repos/OPUS-PLLM/multi_modality_model/multi_modality_v1/eval/run_opus_ddp.py`：批量推理，读取 `input/instruction/output` JSON 列表。
- `.../eval/run_opus_online.py`：单条交互推理。
- `.../model/builder.py:71`：根据路径中的 `opt` 或 `galactica` 选择 `OpusOPTForCausalLM`，因此小版本有实际加载分支。
- `.../model/protein_mlp/builder.py:11`：默认 `n_tokens=8`。
- `.../eval/metrics_computing_opi.py`：按数据文件名选择既有指标；本轮 Recall@3/MRR 与配对方向指标需另接评测器。

模型组件入口已经用 Hugging Face API 实际核验：旧作者链接 `YifanXu24/OPUS-PLLM-Galactica-1.3B-Base` 已跳转为 [OPUS-PLLM-Authors/OPUS-PLLM-Galactica-1.3B-Base](https://huggingface.co/OPUS-PLLM-Authors/OPUS-PLLM-Galactica-1.3B-Base)，公开且无需门禁，revision `78c3a936a755fbfa9cc03c5ab80c3782cce9552c`。文件包括 `modality_encoding_adapter.ckpt`、`modality_refinement_projection.bin`、`lora_adapter/adapter_model.safetensors`。基础模型 [facebook/galactica-1.3b](https://huggingface.co/facebook/galactica-1.3b) 也公开可取，revision `f711c69357d598defb703ddce93c5d7f7bc6e6da`。本分支只核验入口，没有重复下载模型权重。

官方原始环境：Python 3.10、Torch 2.4.0/CUDA 11.8、xformers 0.0.27.post2；requirements 还包括 Transformers 4.46.3、bitsandbytes 0.42、PEFT 0.11.1、DeepSpeed 0.15.1、fair-esm 2.0。论文训练使用 A100-80GB，一阶段单卡、后两阶段四卡。这些是原文配置，不是本机显存测量。面向较新 GPU 的执行应先验证当前 Torch/CUDA，再逐个处理自定义语言模型、量化和注意力兼容性。

静态复现障碍：批量脚本硬编码 `batch_size=8`，直接调用 `.cuda()`，默认温度 0.1；`--load-4bit` 用 `type=bool` 且默认 True，命令行传字符串 `False` 仍会被 Python 当作真值。已有 `.sh` 包含作者绝对路径。正式基准应明确关闭采样、从单样本开始，并修正参数解析和路径。上述问题可在独立适配分支解决，当前原始仓库保持不变。

许可证：代码 GPL-3.0；原文说明数据 CC BY 4.0；Galactica 模型卡为 CC BY-NC 4.0。源码许可、适配器许可、基础模型许可需要分别保留。官方训练数据入口为 OPUS-InstructionCorpus、OPUS-InstructionCorpus-Evol，第一轮对话已明确不以这些语料预热。

## 3. ProtT3：更贴近 ESM2-150M 的输入连接对照

[ACL 正式论文页](https://aclanthology.org/2024.acl-long.324/)，DOI `10.18653/v1/2024.acl-long.324`，本地 18 页。`model/blip2_opt.py:75` 默认蛋白编码器为 `facebook/esm2_t30_150M_UR50D`，语言模型默认 `facebook/galactica-1.3b`。Q-Former 读取残基表示，将可学习查询映射后拼到 LLM 输入，适合作为后续更复杂输入连接的真实方法对照。

训练入口 `stage1.py`、`stage2.py`；结果评测文件实际为 `read_results.py`。官方 [OSF 数据及 checkpoint 页](https://osf.io/23azs/?view_only=PRIVATE_ACCESS_REMOVED 已通过 API 确认可列出资源，包括 `SwissProtV3.zip`（148,410,120 bytes）、`proteinkg25.zip`（138,380,545 bytes）以及 checkpoint 文件夹，哈希及下载 URL 保存在仓库清单。数据下载与本轮 Benz/Kliche 数据分开安排。

原环境为 Python 3.8/CUDA 11.7，依赖 flash-attn、OpenDelta、Salesforce LAVIS、DeepSpeed、PyTorch Lightning，多个包要求从最新 Git 安装，版本未完整锁定。第一轮项目不应为跑 B1 或小投影器引入这套依赖。

README 中的 `python read_results` 缺少 `.py`；“加载 checkpoint 复现”下的 stage2 命令仍为 `--mode train`，而源码 `stage2.py:90-94` 明确 `train` 会调用 `trainer.fit`，`eval` 才调用 `validate`。需要根据源码改为评测模式后使用。根目录没有 LICENSE；部分模型文件含 Salesforce BSD-3-Clause 标识，不能据此给整个仓库统一指定许可。

## 4. MiniMind 与 MiniMind-V：接口位置及冻结策略

`repos/minimind/model/model_minimind.py:12` 当前默认 8 层、768 维。`FeedForward` 在 136 行，`MiniMindBlock` 在 183 行，`MiniMindModel` 在 201 行，逐层循环在 226 行附近。最终返回的 `hidden_states` 为最终状态张量，采集中间层需要显式 hook 或逐层存储。

`repos/minimind-v/model/model_vlm.py:23` 的 `MMVisionProjector` 为 LayerNorm → Linear → GELU → Linear；在输入位置写入连续向量后进入逐层 Transformer。该实现可帮助核对本轮输入端连接与内部层连接的区别。

冻结策略需要区分脚本：`train_pretrain_vlm.py:102` 默认 `freeze_llm=2`，只训练投影；`train_sft_vlm.py:102` 默认 `freeze_llm=1`，会解冻首尾层。实际执行位于 `trainer/trainer_utils.py:82-95`。聊天中笼统写“默认1”，应按这两个阶段细分。首轮继续采用已训练 Qwen 的方案；若以后用 MiniMind 做机制验证，应显式检查所有 `requires_grad` 及优化器参数。

## 5. 隐状态通信方法与 T2 的对应关系

| 方法 | 论文与本地 PDF | 实际代码入口 | 可迁移到本轮的内容 |
|---|---|---|---|
| C2C | [2510.03215](https://arxiv.org/abs/2510.03215)，29页；PDF有ICLR2026页眉 | `script/evaluation/unified_evaluator.py`、`rosetta/model/wrapper.py` | 层位映射、KV投影、接收端融合的实现参照 |
| Draft-KV | [2609.34754v2](https://arxiv.org/abs/2609.34754v2)，41页 | `script/draft_kv/run_stage3_eval_all.sh` | 正确消息、错配消息、接收端独立三条件；区分系统收益与内容收益 |
| LatentMAS | [2511.20639](https://arxiv.org/abs/2511.20639)，35页 | `run.py`、`models.py`、`methods/latent_mas.py` | 连续潜在状态及记忆传递；通用LLM侧参考 |
| StateBridge | [2608.13317](https://arxiv.org/abs/2608.13317)，18页 | `methods/state_bridge.py`、`models.py` | 范数校准和几何兼容性；从语言词表得到的锚点不能直接代替蛋白质配对监督 |
| PLM层分析 | [2602.20449](https://arxiv.org/abs/2602.20449)，16页 | 本轮只取论文 | 为比较中间层与最后层提供研究依据；具体任务仍需实测 |

Draft-KV v2 的 arXiv 说明明确修正了模板残留的会议发表页眉，按预印本引用。其仓库 README 声明原实验为单张 96GB GPU、Torch 2.6/CUDA12.4/Transformers4.52.4，桥 checkpoint 仍为 coming soon。论文数字只能作为作者报告引用，本地未复现。

C2C 的 [Fuser 权重页](https://huggingface.co/nics-efc/C2C_Fuser) 已核验公开，revision `f01fc3258b305e280e04c7238f4f2cf31b7dc70d`；这些权重对应特定 LLM 对，不能作为 ESM→Qwen 的已训练接口。另有许可元数据不一致：根 LICENSE 为 Apache-2.0，`pyproject.toml` 写 MIT，应保留原文件并在后续复用时确认。

## 6. 最小基准的工程接口建议

最新对话固定的小接口可直接由标准 PyTorch/Transformers 实现：LayerNorm(640) → Linear(640,256) → GELU → Linear(256,1536) → LayerNorm(1536)，加门控标量；共 563,201 可训练参数。它是本项目实验配置，不应命名为已复现 OPUS 或 ProtT3。

首次运行必须证明四项：ESM 返回第15/30层真实残基表示；Qwen 输入端及第7/14层后写入位置正确；冻结 Qwen 后梯度仍能回到投影器；关门控能恢复无分子消息条件。位置命名建议按人类层数记录，在代码中对应 block 索引6/13，避免“第14层后”与索引14混用。

T1 使用独立已知阳性召回与 MRR，T2 复用同一测试样本做正确/无/错配/恒定消息比较，T3 使用同靶点配对方向与 A/B 交换一致性。模型配置、JSON结构、token数、样本顺序、随机种子和供体列表应固定。相同供体扰动与问法改写属于同一生物学样本的重复测量。

本报告的证据层级为：仓库和PDF已实际取得；加载与评测入口已静态核实；是否在本机完成模型前向、反向和真实数据评测由项目运行报告给出。下一步成本最低的路径是先验证 B1 + L30-J0/L15-J14，随后扩至六组层位与 T2，再进入配对任务。

## 7. 补齐直接领域基线与全部方法范围

| 方法 | 本地资料 | 代码入口和资源条件 | 扩展成本 |
|---|---|---|---|
| [ProtNote](https://github.com/microsoft/protnote) | MIT仓库；[Europe PMC完整XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12054973/fullTextXML) | `bin/main.py`、`bin/test_models.py`；[Zenodo](https://zenodo.org/records/13897920)原始数据包约17.6GB | 适合开放文字功能打分；不是生成式解释的同构基线。OUP PDF403，PMC PDF返回HTML，XML已完整保存 |
| [Prot2Text-V2](https://github.com/ColinFX/Prot2Text-V2) | MIT仓库；NeurIPS2025正式PDF28页 | `scripts/generate_instruct.py`、`scripts/benchmark.py`；默认ESM2-3B + Llama3.1-8B-Instruct | 原README使用Python3.8、Torch2.3/cu121/Transformers4.40.2，并列图处理依赖；需要独立环境和统一输入条件 |
| [BioReason-Pro](https://github.com/bowang-lab/BioReason-Pro) | MIT仓库；bioRxiv API确认v1为2026-03-20、v2为2026-07-19；PDF多入口429/403 | `predict.py`、`eval.py`、`train_protein_llm.py`；ESM3模型需要接受许可门禁；默认完整流水线含InterPro和GO-GPT | 官方推荐A100-80GB推理，Python>=3.11、Torch2.6–2.7、vLLM/Unsloth/ESM/FlashAttention，依赖间存在已注明冲突 |
| [Pep-Instructions](https://github.com/kjY7836/pepinstruction) | 仓库及[出版社全文HTML](https://oup.silverchair-cdn.com/article-minimal/8762935) | `code/llama2_newsplit_train.py`、各任务`test.py`；[Llama2适配权重入口](https://huggingface.co/Codelife176/Pep-instruction) | 根目录无LICENSE；requirements包含作者本地`file:///workspace/bitsandbytes`及大量整机环境包，不能直接原样安装 |

Pep-Instructions 的浅克隆触发 Git LFS 自动取回仓库所附 JSON/CSV 语料约168MiB；没有另外下载其模型权重。训练脚本预期的 `enhanced_train.json` 与仓库实际 `train.json` 名称不同，评测README要求设置本地路径和外部API参数。运行前应统一改成本地、固定模型、可审计评测，避免把外部服务结果混入本轮同底座比较。

范围清单还列出 ProtST、ProteinCLIP、Prot2Text、ProtLLM、ProTrek、Prot2Chat、ProCyon、InterPLM、MapDiff、CaLM、Flamingo、DeepStack、Coconut、FActScore、DataSAIL 等历史参考，以及交由数据分支获取的原始实验文献。对话只给出缩写的 IAA 和泛称 Model Stitching 的研究线保持待唯一识别状态；清单没有猜测一个具体 DOI 后冒充完成获取。

## 8. 本次新增的直接相关研究：T2 已有生物领域先例

本次额外发现、原对话没有引用的论文：[When Do Biological Reasoning Models Use Their Biological Inputs?](https://arxiv.org/abs/2610.00898)，2026-10-01预印本，本地29页PDF；[官方代码入口](https://github.com/mims-harvard/bio-mirage)列入范围清单，当前未额外克隆。

该文直接研究 BioReason-Pro、Prot2Text-V2 等六种生物推理模型，通过固定问题并扰动单一生物输入、构造证据冲突、探针和SFT/RL比较，区分“信息存在于表示”与“下游模型使用信息”。其正文第6页/Fig.3b报告：在将不同蛋白的 GO-GPT 预测文字、InterPro 注释和 ESM3 表示组合成冲突输入的构造条件中，BioReason-Pro 的回答在99.7%的冲突中与 GO-GPT 来源一致。这个数值针对该受控冲突设置，不能理解为正常生物任务中99.7%的输出完全不使用蛋白输入。文中另设106个有实验 GO NOT 注释的冲突样本，98.1%跟随GO-GPT/InterPro来源，两个分母和条件应分别报告。

因此，本项目 T2 属于必需的有效性检验，单独加入错配/关闭消息消融已不足以构成方法创新。研究增量需要来自公平控制下的内部读写层规律、真实分子差异任务和相对强流水线的额外收益。该文也报告 Prot2Text-V2 在其蛋白名称冲突设置中97.0%跟随序列来源，进一步支持将它纳入后续强对照，而不能预设所有现有蛋白质语言模型都忽视分子输入。
