# Molecular Memory V3 数据与缓存合同

此模块实现新 scope 的数据准备、同次双层 ESM 抽取命令及只读缓存加载。旧样本、划分、底座、缓存和结果保持原字节。2026-10-17 的正式抽取和训练由独立执行协议调度；本实现交付时仅运行 CPU 小夹具与只读来源校验，未启动真实底座或 GPU。

## 冻结数据与进程隔离

显式 `project_root` 指向旧项目根；数据相对位置固定为 `data/processed/bench_v02/a1_locked_v2/<scenario>/`。`src/mm_data.py:FROZEN_SHAS` 锁两个场景各自的 samples、test_gold、benchmark_lock、manifest 共 8 个 SHA。`load_frozen_scenario(project_root, scenario)` 先核 8 资产中本场景的四个文件，再逐项核该 lock.pins 全闭包；不允许绝对引用、`..`、反斜线引用、盘符或解析后逃出注册根的符号链接。读取 JSON/JSONL 拒重复键、非有限/溢出数字、坏行和空行，不丢行。

本次只读实核：known 场景 2062/147/152，double-new 场景 2063/431/442；两 lock 的并集为 **48 文件、78,547,471 B**。逐场景核唯一 ID、test gold 与 samples.test 的 ID/顺序/原始精度 pKD 完全一致。两场景人群重叠，各自 TRAIN/VAL/TEST 语义保留，不把它们合并训练。

准备进程命令（在新 scope 根执行；输出目录须不存在）：

```sh
python -m src.mm_data --project-root /registered/old_project --output-root /new_scope/data_views
```

`prepare_views(project_root, output_root)` 是独立审计/准备阶段，可读含 test gold 的原始 samples；训练进程只用 `read_input_view(view_root, scenario, split)` 打开已准备的单个 split，不再读取原 samples 或独立 gold。输出为：

- `<scenario>/train.jsonl`、`validation.jsonl`：原顺序，保留原始精度 `pkd` 标签。
- `<scenario>/test_inputs.jsonl`：原顺序，无 `pkd` 或其它标签、DOI、来源分组、raw_metadata；仅身份信封和允许输入。
- `manifest.json`：8 资产来源 SHA、完整 lock.pins 收据、每视图路径/行数/SHA、10 槽顺序。执行器须把该 manifest SHA 固定在 run pins 中。

`sample_id` 和 `split` 是输入视图的追踪信封；`collate` 不把它们交给模型。序列只交给 ESM/cache 查找；提示词只有匿名任务描述与显式允许的上下文。labelled view 禁止 test。独立 test_gold 仅在选点记录写定后的评分阶段由评价器读取。本模块不替代执行器的进程访问门禁。

父执行 lane 已实际生成本 scope 的 `data_views/`；本 lane 随后只读复核该 manifest SHA 为 **b505c29ea8b78cd4d6d9fb6f0bca52f5b09cd3a25f43ea8d727d1ca5066e91f0**，所有引用均为 portable relative path，无 Windows 绝对路径。六个 split view 的 SHA/行数读取均通过，TRAIN/VAL 有 `pkd`，152/442 条 test input 全无 `pkd`。这是已生成数据视图的收据，不能解释为 ESM 抽取完成。

## 10 槽条件接口

模型输入固定为 `condition_ids: int64[B,10]`、`observed: bool[B,10]`。每 observed 槽的 ID 为：

`int(SHA256(UTF8(slot_name + NUL + canonical_JSON(value))), big_endian) % 4096`

canonical JSON 使用键排序、紧凑分隔符、UTF-8 原字符、不允许非有限数。算法不拟合 TRAIN/VAL/TEST 类别字典；4096 桶存在可预期哈希碰撞，不解释为可逆物理量编码。missing 槽 ID 的占位值为 0，**由 observed=false 选择独立 missing embedding，不能解释为数值 0**。真实 false 和真实数值 0 都可为 observed。

| 槽 | 固定字段/组合 | observed 规则 |
|---|---|---|
| 0 | chemical_state.c_terminal | 非 null/空串 |
| 1 | chemical_state.chemical_modification | 同上 |
| 2 | chemical_state.cyclization_method | 同上 |
| 3 | chemical_state.explicitly_representable | 同上，false 仍观测 |
| 4 | chemical_state.n_terminal | 非 null/空串 |
| 5 | chemical_state.residue_modification | 同上 |
| 6 | chemical_state.topology | 同上 |
| 7 | assay | 同上 |
| 8 | measurement_context = 精确 `{temperature_kelvin, ph, buffer}` | 任一真实字段非缺失；组合中其它字段仍为 null |
| 9 | target_context = 精确 `{target_sequence_role, actual_construct_verified, actual_construct_sequence_missing}` | role 存在，或 verified=true，或 construct 确有观测 |

只从上表及固定 missing/verified 标记构造条件，绝不把来源 dict 直通。类别仅允许 string/bool/null；数值温度/pH 仅允许有限 real，bool 不作为数字。实际 construct 序列文字永远不输入；这里只保留其 missing 状态，未实现 construct resolver。当前 A1 所有温度/pH/buffer/construct 缺失、verified=false，assay 与 reference-target role 存在。slot8 全 missing；slot9 的 observed 来自明确的参考靶标角色，并不表示构建体已核实。两个复合槽是为固定 10 槽而明确登记的有损类别组合，没有插补实验条件。

提示词用原匿名 system/user 文本、原 `pKD=-log10(KD[M])` 语义与严格上下文白名单。化学仅七键，其它嵌套键被忽略；允许键内若为对象/数组则拒绝，不能借嵌套注入标签。可读 prompt 仍分别显示温度、pH、buffer、missing 和 role，不以条件哈希替代可读事实。

## 因果 token 与模型 batch

`collate(rows, tokenizer, store, device='cpu', supervised=True)` 返回单 dict 模型 kwargs：

```text
input_ids            int64 [B,T], left padded
attention_mask       int64 [B,T], pads=0
labels               int64 [B,T], 仅 supervised 存在，prompt/pad=-100
prompt_end_index     int64 [B], 物理 padded 坐标中最后真实 prompt token
memory_inputs:
  p15,p30,t15,t30     float32 [B,8,640], frozen raw pooled source
  condition_ids      int64 [B,10]
  observed           bool [B,10]
```

四个 raw feature tensor 可 detach；可训练投影/条件/关系模块由模型 forward 内创建，不在 collation 中预生成或 detach。`store.pooled(sequence)` 返回 `{'H15':tensor[8,640], 'H30':tensor[8,640]}`。

训练答案仍为 `{"pKD":<两位小数>}` + EOS；评分使用原精度 gold。label 与 input 对齐，由模型损失做 causal shift。EOS 监督，prompt/pad 不监督。`prompt_end_index = T-(prompt_len+answer_len)+prompt_len-1`；prompt-only 为最后 active 位置。numeric head 应读取该位置的 post-Qwen-finalnorm 因果状态。输入/标签/ID 信封分开，runner 从原 rows 的 `pkd` 构造独立 numeric target。`supervised=False` 不读取 gold、也不返回 labels，适用于 test_inputs 生成。测试已经覆盖 gold 改变时 prompt token 前缀、raw memory 和条件不变；模型 lane 另核因果数值 head 不变。

## ESM 层语义：已核源码，未运行底座

固定 ESM2 revision `a695f6045e2e32885fa60af20c13cb35398ce30c`，30 block、640 维。新提取器显式读取 block15 hook 和 `last_hidden_state`，不保存全部 31 份 hidden states：

- **H15**：`encoder.layer[14]` 的 tuple 第 0 项，完成第 15 块，最终 encoder LN 之前。
- **H30**：`EsmModel.last_hidden_state`，完成第 30 块及 `emb_layer_norm_after` 之后。

2026-10-17 经已核 SSH/<GPU_HOST> 身份，只读单文件 `<PYTHON_ENV>/lib/python3.11/site-packages/transformers/models/esm/modeling_esm.py`，SHA256 **6f867571aeee0fb9f444cb23ac5e8ca189aa4b05424aad986c9c74b6ce412cb8**。未 import torch、构造模型或初始化 CUDA。TF4.48.1 的 `EsmEncoder.forward` 568–654 行按块赋值 `hidden_states=layer_outputs[0]`，循环结束后执行 `emb_layer_norm_after`，再返回 `last_hidden_state=hidden_states`；`EsmModel.forward` 813–932 行取 `encoder_outputs[0]` 并返回它为 `last_hidden_state`。因此本方案不依赖不同版本 hidden_states 汇总 API 的隐式归一化行为。

`python -m src.mm_extract inspect --output /new_scope/esm_semantics.json` 仅读取安装源码并生成版本/源码 SHA/两个 forward 原文的收据。生产 extract 入口同时锁 TF4.48.1 和上面实际源码 SHA；变更环境需新资格/协议，不能只改版本文字。当前 CPU 测试在已装 TF4.57.6 上构造随机 tiny ESM（30 块、8 维），验证 hook H15 与 index15、H30 与 last_hidden_state 相等；这不是对真实预训练模型的执行证据。

## 完整残基覆盖与 8 槽池化

每个序列按不重叠连续半开区间 `[start,stop)` 切为最多 1022 残基；每 chunk 一次 forward 同抓两层，去 BOS/EOS，按原顺序拼接完整 `[L,640]`。tokenizer 必须输出恰好 `Lchunk+2` tokens，不 truncation。所有 chunk、两层 raw 及保存 tensor 均检查 finite；hooks 在异常路径 finally 清理。

完整拼接后以 float32、无参数 `adaptive_avg_pool1d(...,8)` 得 `[8,640]`，才按登记 storage dtype 保存。没有先逐 chunk 池化或截断后取均值。每槽覆盖 `[floor(jL/8), ceil((j+1)L/8))`，短序列的区间可重叠。manifest 保存确切区间、chunk coverage、逐层 raw mean-square 与重叠平均 slot 广播回原位置的 reconstruction MSE。该失真只作缓存诊断，不作训练损失或选点指标。

`raw_full_coverage=true` 的含义是池化前完整残基都参与；它**不表示保存了全残基 tensor，也不表示池化无损**。保存 `raw_residues_saved=false`、`pooling_is_lossy=true`；多 chunk 时 `whole_sequence_joint_attention=false`，没有跨 chunk 注意力。不能把槽位置当结构坐标。

缓存路径必须是新 scope 下尚不存在的独立目录，如 `cache/esm2_150m_layers15_30_<extractor_sha>/`。入口不会覆盖旧缓存或补造 L15。每个唯一序列文件 `sequences/<sequence_sha>.pt` 仅有 H15/H30 两个 tensor；manifest 收集完整 chunk、coverage、lossy pooling、模型六文件 pins、安装源码、提取器、view manifest、dtype/device/torch 版本、payload SHA/bytes、无标签声明。torch payload 用 `weights_only=True` CPU 读取；所有 `SequenceStore.pooled` 调用先核 payload SHA，再查 coverage/shape/finite。

生产命令仅实现，尚未运行：

```sh
python -m src.mm_extract extract \
  --view-root /new_scope/data_views \
  --model-root /registered/models/facebook--esm2_t30_150M_UR50D \
  --output-root /new_scope/cache/esm2_150m_layers15_30_<extractor_sha> \
  --device cuda:0 --compute-dtype bfloat16 --storage-dtype float32 --max-residues 1022
```

CPU 命令把 `--device` 改为 `cpu`，`--compute-dtype float32`。这里不承诺 CPU/GPU 浮点输出逐位相等；正式执行前应固定单一 dtype/backend/批形与源 SHA。模型六文件 SHA 直接来自旧已核 `sources/model_manifest.json`；加载前全核，`local_files_only=True` 禁止网络下载。可选 `--model-pins` 文件必须逐项等于本模块六文件常量，不放宽 frozen revision。

`SequenceStore(cache_root).manifest_sha256` 为 run pin 入口；`store.provenance()` 返回该值和实际加载 payload 的 SHA 收据；`store.verify()` 可在训练前无 tensor 反序列化地核全部 payload。未访问缓存不算已核文件。实现只提供命令，不接管 GPU 分配、时间预算或真实执行授权。

完整抽取默认 `--phase full`。开始、每 25 个序列及结束输出 UTC、elapsed wall seconds、完成/预期序列数和 raw 残基覆盖。相同节点的新输出根内以临时文件原子更新 `manifest.partial.json`，保留已经产生的文件；异常写 `cache_status=failed` 并抛错，不写 `manifest.json`。只有已完成的 sequence SHA 集与预期全部一致时才写 final manifest。进程被外部杀死时，最多最后 24 条的 payload 可能未进入最后 partial checkpoint；这些孤立文件不构成成功或隐式 resume。输出根存在即拒绝，不自动复用未知 partial cache。

真实设备 canary 的独立命令入口（仅实现，不由本 lane 执行）：

```sh
python -m src.mm_extract extract --phase train-canary --max-sequences 4 \
  --scenario new_peptide_known_target --view-root /new_scope/data_views \
  --model-root /registered/models/facebook--esm2_t30_150M_UR50D \
  --output-root /new_scope/cache/train_canary_unique_revision \
  --device cuda:0 --compute-dtype bfloat16 --storage-dtype float32
```

canary 仅读所选场景 TRAIN view；按 `(length,sequence_sha)` 排序取 2–4 个等距位置，固定包含最短和最长，不读标签、不按表现选例。记录 `phase=train-canary`、TRAIN scenario、预期 SHA/数量，独立目录。`SequenceStore` 拒绝 canary 和 partial cache 作为正式训练输入；正式 full 不允许 `--max-sequences`。4 条真实 canary 只用于后续冻结的设备/数值/吞吐资格，不能补充测试表现。

## 交付验证与范围

`CUDA_VISIBLE_DEVICES=''`、CPU threads2，WSL 现有 Python3.10/Torch2.12.1/TF4.57.6 下 **16 项 unittest 通过**：严格 JSON/坏行；extra metadata 不入 prompt/condition；允许键内嵌套拒绝；10 槽 missing/false/zero；construct 文字不入输入；leftpad/答案/EOS/prompt_end；gold 改变输入不变；root escape/SHA 篡改；锁 pin 篡改；1022 分块；同次两层/有损池化；异常 hook 清理与冻结；缓存覆盖/哈希；随机 tiny ESM 层语义；failed partial 无 final；canary 仅 TRAIN/固定长度选择/不被正式 store 接纳。

Windows 标准库只读重新核两场景真实源数据及 48 项闭包，通过；父生成的 `data_views` 也已重新核验。预训练 ESM 缓存、模型训练和正式评价由父执行 lane 落盘，不在本交付中宣称完成。没有安装依赖、运行论文代码、修改旧资产或改变评价标准。
