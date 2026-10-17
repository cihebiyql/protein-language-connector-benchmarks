# Molecular Memory v3 单run训练与独立评价

`src/mm_train.py`实现一个scenario、一个variant、一个seed的完整作业。模型构建、条件与缓存、独立评分分别由`mm_model.py`、`mm_data.py/mm_extract.py`、`mm_evaluate.py`提供。新任务固定截止2026-10-17 10:29:40+08:00；父controller负责启动准入、进程树超时与最终验收，worker的`complete`只表示本阶段正常完成。

## 入口与阶段

从本scope根目录使用`python -m src.mm_train`；直接执行脚本也支持。所有数据、缓存、模型和输出路径显式传入，输出目录必须尚不存在。每个phase使用独立新目录，fit与evaluate均绑定同一冻结配置、scenario、variant和seed。

```text
python -m src.mm_train --config configs/development.json --project-root PROJECT \
  --cache-root PAIRED_CACHE --view-root VIEWS --output NEW_CANARY \
  --scenario new_peptide_known_target --variant M0 --seed 17 --device cuda:0 \
  --phase train-canary --max-updates 4

python -m src.mm_train --config configs/FROZEN.json --project-root PROJECT \
  --cache-root PAIRED_CACHE --view-root VIEWS --output NEW_FIT \
  --scenario new_peptide_known_target --variant M0 --seed 17 --device cuda:0 --phase fit

python -m src.mm_train --config configs/FROZEN.json --project-root PROJECT \
  --cache-root PAIRED_CACHE --view-root VIEWS --output NEW_EVALUATION \
  --scenario new_peptide_known_target --variant M0 --seed 17 --device cuda:0 \
  --phase evaluate --checkpoint NEW_FIT/selected.pt
```

`train-canary`只读取TRAIN视图，拟合TRAIN数值标准化并执行恰好指定的optimizer更新；小TRAIN集合可按固定seed顺序跨epoch循环。history区分完整epoch与被4update截断的片段，`epochs_completed`只计完整epoch。结束后固定TRAIN文件前`max(2,generation_batch_size)`行执行真实greedy24、strict parser、numeric评分、同θ on/off C1和JSONL保存，使batch8 canary实际覆盖8行而非只测2行；先在同一prefix用保持shape／mask／prompt anchor的未来答案token扰动验证numeric_z精确不变，失败即保留资格记录并停。无validation/test读取或选点。它用于时间、显存、端到端支持性资格，分数不用于选择预算或架构。完整冻结的无标签paired cache由`SequenceStore`核验；extract自身的少量序列canary缓存不能冒充完整训练缓存。

`fit`只读取TRAIN与全validation，固定公共epoch数，不因某seed早期收益为负而提前停止。每epoch以答案token的CE总和／token总数选择，严格`<`更新最优值，因此精确平局保留最早epoch。它不打开test inputs或test gold，也不生成test预测。`evaluate`要求fit正常完成、selected SHA和身份匹配、同一代码／缓存闭包，加载所选完整trainable state，完成全validation与test的两类输出，并评价test C1。

正式fit/evaluate要求`status="frozen"`且`budget_signature`等于`training`字典的排序、compact JSON的SHA256。draft canary结束后由父任务根据实际资源统一冻结预算；未测时间或显存不会被伪造成资格。矩阵及36位置顺序由父controller冻结，worker不按test分数改顺序、删seed、重选预算或跨scenario延续训练状态。

## 固定训练与输出合同

- `CE = answer_loss_sum / answer_token_count`，只监督原2位小数JSON答案＋EOS，prompt及padding为`-100`。
- `numeric MSE = mean((head_z - (pkd-μTRAIN)/σTRAIN)^2)`；μ与总体σ只用该scenario的TRAIN原精度pKD，σ下限`1e-8`。`numeric_loss_weight`默认1，与CE相加；推理以`head_z*σTRAIN+μTRAIN`恢复原pKD。
- 初始公共配置为3epoch、AdamW `lr=3e-4`、`weight_decay=.01`、batch4、accumulation1、全体活跃参数梯度norm clip1。实际正式值必须由独立冻结配置记录，所有匹配臂共同使用。
- LM是本地精确权重的Qwen2.5-1.5B-Instruct，CUDA为bf16，CPU小模型为fp32；冻结参数、保持eval。训练期间保留注入点之后的autograd。可训练参数、保存的浮点模块state均为fp32；不对整模型转换fp32而破坏base dtype。
- Python／NumPy／Torch设置同seed，Torch CPU线程2，最高fp32 matmul精度、TF32关闭、cudnn benchmark关闭、确定性算法严格开启，autocast关闭。`CUDA_LAUNCH_BLOCKING`默认0，可由controller显式设定；实际值写入receipt。预算canary与正式必须使用相同执行flag。

生成固定greedy、最大24新tokens、`use_cache=False`，`generation_batch_size`由公共配置指定。数据左padding，初始`prompt_end_index`在追加tokens时保持固定。每行输出到首EOS（含EOS）或24token；批内已结束行后续写masked padding直到整批结束。没有截断原序列或宽松解析。ledger分开记录prompt／padding、各forward的实际query槽与有效token、未结束行、首EOS长度及24上限；整批的真实计算包含已结束行，不能用单行有效输出长度冒充实际成本。batch改变属于新任务的预冻结吞吐选择，必须用TRAIN支持性canary并统一应用于全部臂。

LM JSON生成与numeric head独立保存、独立评分。`*_lm_predictions.jsonl`保留raw文本、raw token IDs、首EOS长度和strict parse；`*_numeric_predictions.jsonl`保留反标准化pKD或显式失败。numeric输出不替换语言生成结果，失败不丢行、不裁剪或手工修正。

## C1与检查点

`c1_predictions.jsonl`在同一个selected θ上记录每行开启／关闭整个分支的答案NLL sum/count，以及两个prompt-only numeric head输出。正的`off NLL-on NLL`或`off numeric MAE-on numeric MAE`表示该固定模型在此干预下改善对应输出。off仍运行冻结LM及numeric head，且不等于传全零memory。test teacherforcing仅经`mm_data_evaluation.collate_evaluation(...,phase='evaluate')`显式能力；训练路径保留原`mm_data.collate`对test监督的拒绝。

TRAIN均值memory、uniform memory、严格生物donor P/H本轮为未执行或无金标准；C1仅支持同θ的计算依赖／分布敏感性解释。分别训练两系统的端到端差异与同θ干预分开分析。

每epoch保存全部非LM模块参数及buffers、optimizer、TRAIN标准化、身份、历史、选点及gate／routes诊断。每epoch首次batch，以及第2、4次optimizer更新前，记录CE与numeric MSE各自的模块梯度norm与连接情况，覆盖gate初态和打开后的路径，均在weight decay／clipping前；零gate初态下下游task梯度为零可能是预期行为，不能把AdamW decay当task学习。最优epoch的序列化字节原样复制为`selected.pt`，evaluate再原样复制到自身目录；不重新训练或重选。

## 来源与失败

run manifest固定记录四个原A1资产SHA、prepared view manifest及来源lock闭包、当前源码SHA、模型manifest与实际本地权重字节核验、paired cache manifest及每payload SHA。fit只打开并核验TRAIN／validation的已分离视图，原混合samples及test gold的一次准备审计由数据模块记录。来源检查不允许旧H30投影缓存替代新H15/H30。

可评分`artifacts`固定六项：history、selection、checkpoint、LM、numeric、C1。其他统计、ledger、canary评分引用位于`provenance.audit_artifacts`。fit目录本身尚无test预测；独立评分器应接收evaluate阶段目录。异常写`failed.json`，manifest状态改为training/evaluation_failed；保留已写部分输出，父controller决定invalid/partial及完整3seed覆盖，worker不造成功记录。既有输出目录直接拒绝复用。

## 已验证范围

`tests/test_train.py`在CPU、隐藏CUDA、无预训练权重／下载条件下验证TRAIN标准化、masked CE、冻结梯度、确定更新次数、完整epoch、最早平局、检查点字节及optimizer、scenario身份、首EOS／固定anchor／no_grad恢复、C1、strict序列化，以及独立fit→evaluate的test读取顺序。另以真实`MolecularMemoryLM`＋随机小LM检查训练计算图与完整模块保存。真实Qwen／GPU吞吐、OOM边界和正式预算需由实际canary验证。
