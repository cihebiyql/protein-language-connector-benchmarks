# 五个原作者系统的复现准入盘点

2026-10-16，本轮只读核对本项目源码、README、配置、下载回执及三个既有解释器的包元数据。没有导入作者仓库、反序列化检查点、运行 GPU、安装依赖或启动原系统。五项目前都未达到原系统执行准入；它们缺少的环节各不相同。

| 系统 | 已有数据与模型 | 当前执行阻塞 | 本轮基础测试与原系统的差别 |
|---|---|---|---|
| OPUS-PLLM | 18 项官方基准文件，合计 20,973,373 字节；两套完整指令语料分别 1,534,241,673、3,327,988,260 字节；固定源码 commit `a9597043…`。 | 项目模型目录没有作者的基础 Llama3／Galactica，以及模态编码适配器、投影器、LoRA 权重；实际编码器代码请求 ESM2-t33-650M，项目已有的是 t30-150M。作者依赖栈也未准备。 | Qwen＋小型连接器的 A1 结果，不能替代 OPUS 的 18 项原任务生成及原评价管线。 |
| ProtT3 | SwissProtV3 的 train／validation／test 为 430,595／10,000／10,000 条配对；OntoProteinDatasetV2 为 422,315／10,000／10,000 行；两包已有官方 SHA 与 CRC 回执。固定源码 `f427daca…`。 | 作者 stage1、SwissProt 微调、PDBQA 检查点尚未落地；PubMedBERT 和默认 Galactica-1.3B 未在项目模型目录找到；PDBDataset QA 数据未落地；LAVIS、OpenDelta、DeepSpeed 等栈缺失。 | 当前 `qformer_style` 是受控小型查询交叉注意力基线，没有执行作者的两阶段对齐、LoRA 及原 captioning／QA 任务。 |
| ProtNote | 固定源码 `b29e17fa…`；完整 data.zip 为 17,594,525,298 字节，官方 MD5 通过。全 71 个文件已安全解压并独立重读 29,364,788,715 字节；包括 seed 12／22／32／42／52 五份已落盘的作者检查点，每份 3,412,866,491 字节，以及注释、标签嵌入、词表和数据文件。 | 缺专用运行环境、二进制成员内部结构与安全加载核验；主资产下载、解压与文件完整性验收已完成。额外 outputs.zip／ablation_models.zip 仅针对整篇结果与消融回放，不是主模型推理一概必须的材料。 | B1 受控对齐器及零样本评价原理测试，没有加载这五份已取得的作者权重，也没有执行原始集合上的作者 ProtNote。 |
| DeepLC-v3.1.13 | 源码包内有 3 个真实 `.keras` 容器，分别 17,006,761、7,753,787、10,838,168 字节；已新计算 SHA，并只读核对容器目录与 JSON 元数据；另有 10 份示例 CSV 和既有 RT 数据。 | 缺 TensorFlow／Keras、psm_utils、deeplcretrainer。此版 pyproject 要求 Python≥3.11，PLM 环境 3.10 不满足；现有 3.11 环境虽满足 Python 条件，仍缺运行栈。RT 科学来源门槛另未解决。 | 本轮 ridge／kNN 的 iRT 诊断未执行 DeepLC。113,215 行标签已经确认是 iRT；98,192 行化学身份仍不确定，2026 精确源版本亦待核，不能据此声称作者 2026 原评价复现。 |
| PepInter | 固定源码 `7a93c909…`；分类检查点 6,900,954,326 字节、回归检查点 7,011,734,946 字节，均已有官方 MD5 通过回执；亲和力 train／test CSV 已在本地。 | 缺专用运行环境及原检查点配置／状态覆盖核验；没有执行检查点反序列化。作者 CLI 固定 GPU、batch 128、8 workers、max_len 1024，尚无本机适配实测。原分类完整数据及原训练框架仍未形成完整本地训练合同。 | A1 的本项目预测器和 BASE03 数字呈现管线都没有调用 PepInter 原权重，因此不能算 PepInter 推理或训练复现。 |

## 各系统的具体依据与下一道准入门

**OPUS-PLLM。** [README](../docs/PUBLICATION_SCOPE.md) 明确要求基础模型与三类 OPUS 权重，并提供 `run_opus_ddp.py`、`eval_run_multichoice.py` 等入口。[编码器实现](../docs/PUBLICATION_SCOPE.md) 使用 `esm2_t33_650M_UR50D()`。下一道门是固定具体 Model Zoo 变体及全部权重，匹配其环境，随后为某一原始基准冻结推理与评价合同；不应把本地 Qwen 当成作者基础模型直接替换。

**ProtT3。** [README](../docs/PUBLICATION_SCOPE.md) 区分 stage1、stage2 与加载作者检查点；[stage1 配置](../docs/PUBLICATION_SCOPE.md) 默认 PubMedBERT，[stage2 配置](../docs/PUBLICATION_SCOPE.md) 默认 Galactica-1.3B。QA 还依赖 PDBDataset。[数据回执](../docs/PUBLICATION_SCOPE.md)仅覆盖已落地的两套配对语料。作者 README 的 stage2“加载检查点”示例仍写 `--mode train`，正式推理合同必须核对实际入口行为，不能照抄后当成仅推理。

**ProtNote。** [README](../docs/PUBLICATION_SCOPE.md) 分别说明 `bin/main.py` 主模型推理，以及依赖 `data`、`outputs`、消融权重的整篇 `Results.ipynb` 回放。[最终下载验证](../docs/PUBLICATION_SCOPE.md)现已确认原始三项大文件 3/3 完成，其中 ProtNote 包官方 MD5 为 `703141e8eb7a89c83190122f0adfde35`，SHA256 为 `a5efe8fef2411efc96bbaccdc872ed1d58b62d727b2a3d830a117d7db64e8f50`。因此主模型权重不能再列为“未取得”。[安全容器盘点](../docs/PUBLICATION_SCOPE.md)确认 71 个文件、11 个目录，五份作者检查点均真实存在于包内；25 份 FASTA 与 7 份 JSON 已完整读取并通过成员 CRC。检查点、标签嵌入张量及注释 pickle 仍未反序列化；容器存在不证明内部模型结构或成功加载。

先前“正在解压”的观察已成为历史：2026-10-16 **10:13:53（北京时间）**，[独立验收](../docs/PUBLICATION_SCOPE.md)确认全部 71 个文件已落入 [unpacked_20261016](../docs/PUBLICATION_SCOPE.md)，逐文件重读合计 29,364,788,715 字节，大小、源成员 CRC 和 SHA 全部一致；32 份 FASTA／JSON 的 SHA 与旧 ZIP 扫描 **32/32 相同**。五份主权重的实际文件路径和 SHA 已录入机器清单，并可在该目录下的 `data/models/ProtNote/` 找到。[进程终态回执](../docs/PUBLICATION_SCOPE.md)确认解压及独立验收均以 0 退出且相关进程已消失。完整 [解压清单](../docs/PUBLICATION_SCOPE.md)保留每个文件的实际路径、大小、CRC 与 SHA。本次事实刷新不重新读取模型张量，不改变训练准入或原系统未执行状态。

GO 原随机划分的 train／dev／test 为 **418,015／52,841／51,751** 条 FASTA 记录；EC 为 **208,823／26,724／25,892** 条。它们不是可相加的独立蛋白总数，包内还保留其他 `fake_*`、版本和零样本集合。**FASTA header 的首字段是记录 ID，其余字段含 GO／EC 答案；以后模型可见序列输入必须只取正文序列，把 header 标签解析到独立金标准字段，不能整段 header 送入模型。**[base_config.yaml](../docs/PUBLICATION_SCOPE.md) 的标签编码器为 E5-large-instruct；ZIP 中已经有缓存标签嵌入，是否可直接用于具体主模型推理须核验其内部结构、索引和配置，而非直接断言一定还要下载整个 E5。outputs.zip（18.7 GB）与 ablation_models.zip（41.4 GB）属于整篇结果、输出与消融回放所需的额外材料；主模型推理的下一道门是已取得资产的安全加载、匹配环境及冻结任务合同。

**DeepLC。** [pyproject.toml](../docs/PUBLICATION_SCOPE.md) 的版本和 Python 要求比 README 老的 GUI 安装段更明确，应采用 Python≥3.11、TensorFlow≥2.15且<3 的实际包合同。三份 Keras 文件包含 JSON 配置及权重成员，绝非 LFS 文本指针，但这尚未证明加载、校准或预测成功。可以先为包内示例制定独立的受限推理合同；论文 RT 主表评价还须另过[来源与化学身份门槛](bench_v02_panel_results.md)。

**PepInter。** [README](../docs/PUBLICATION_SCOPE.md) 声明作者在 A800、Python 3.12、torch 2.7.1+cu126 测试；[pyproject.toml](../docs/PUBLICATION_SCOPE.md) 自身允许 Python≥3.10、torch≥2.5，故版本差异本身不等于已证明不兼容。[原 CLI](../docs/PUBLICATION_SCOPE.md) 使用 `weights_only=False`、`strict=False` 加载；下载哈希通过不能代替反序列化安全性、缺失键与预期模型覆盖核验。源码包含 [astra 兼容层](../docs/PUBLICATION_SCOPE.md)，CLI 也设置 `load_from_esmc=False`，因此没有把“未下载外部 ESMC”武断写成一切推理必然缺失的条件。原训练完整性与已发布检查点推理需分别登记。

## 环境与硬件证据边界

本轮仅使用 `importlib.util.find_spec` 与 `importlib.metadata.version`，没有导入陌生包。检查了 Windows 当前解释器和 WSL 两个现有环境；完整包存在性及版本保存在机器回执。

WSL PLM 环境为 Python 3.10.12，另一 web311 环境为 3.11.15；二者均有 torch 2.12.1+cu130、Transformers 4.57.6，均未发现 TensorFlow、Keras、pytorch-lightning、DeepSpeed、flash-attn、fair-esm、LAVIS、OpenDelta、PEFT、xformers、bitsandbytes、DeepLC、psm_utils、deeplcretrainer、PepInter、trifast、cuequivariance_torch、torcheval 和 torchdata。包元数据存在也不证明二进制兼容或完整模型能运行；检查范围之外的远端环境保持未知。

作者资源说明与本机实测分开：OPUS 文档推荐 CUDA 11.8；ProtT3 提供四张 GPU 的示例，未据此断言必须四卡；ProtNote 配置注释涉及 V100／A100；DeepLC 本轮检查的文档未明确强制 GPU；PepInter 声明 A800 测试。**本轮没有测量这五个原系统在本机的最小显存、内存、耗时或当前 GPU 可用容量**，故不提供猜测数字，也没有依据过期资源目录宣称“可启动”。

机器可读文件：[original_system_readiness.json](../docs/PUBLICATION_SCOPE.md)。其顶层 `systems` 分别列出五项状态、资产、具体阻塞、证据路径、作者资源说明与本地测量边界；22 份原静态证据和新增的 5 份最终资产／解压证据保存 SHA256。资产事实刷新时间另行记录，原环境／硬件观察时间保持不变，未重复探测。原作者系统的执行身份继续为未声称完成，与当前基础架构测试结果分开。
