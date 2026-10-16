# 连接器基准的参数、通信与归因审计

本审计只读现有源码、冻结数据和八个真实运行归档；没有训练、GPU 调用、大语言模型加载、评价规则修改。唯一写入为本报告和同目录 `budget_audit.json`。后者保存源码 SHA、八项运行配置、逐张量参数计数、逐步日志核验、长度分布及拟议矩阵。现有八个连接器检查点均在 CPU 读取，所有张量的实际 numel 与 capacity 清单一致，全部为 float32；八份 run.jsonl 均独立数出 1,548 个 optimizer_step 事件。

## 1. 当前基准实际上比较了什么

比较的是**四种不同容量、压缩方式、读取次数、接收位置和梯度路径的系统**。它们共享冻结 ESM2 残基缓存、冻结 Qwen2.5-1.5B、允许的实验上下文、训练划分和生成规则，但不是“只改变插入位置”的试验。

统一记号：肽长度 P、全长参考靶蛋白长度 T、有效源残基 N=P+T、某次前向的有效文本长度 L、memory_dim d=128、LM hidden H=1536、配置槽数 K=16。实际张量还包含批内 padding，物理计算通常随 B×Nmax、B×Lmax 增长。Qwen 为 28 层、12 个 query heads、2 个 KV heads、head_dim=128；最大位置数 32,768。

源码依据：[connectors.py:30](../../historical/scripts/foundation_20261016/connectors.py)、[Qwen config](../../docs/PUBLICATION_SCOPE.md)。

| 方法 | 可训练参数 | float32 权重净字节 | 实际源读取/压缩 | 进入 LM 的位置/状态 |
|---|---:|---:|---|---|
| constant | 24,576 | 98,304 | 不使用残基内容；16×1536 直接学习 | 16 个输入前缀位置 |
| input_mlp | 282,752 | 1,131,008 | 拼接所有残基，经 type+LN 后按顺序整体均分为16槽，逐槽 MLP | 16 个输入前缀位置 |
| qformer_style | 350,848 | 1,403,392 | 16 个学习 query 对全部 N 个残基做一次4头跨注意力 | 16 个输入前缀位置 |
| internal_cross_attention | 548,608 | 2,194,432 | 每个文本位置对全部 N 个残基做一次4头跨注意力 | 第14个 decoder block **之后**，向每个有效文本状态加 residual；没有额外 LM 位置 |

**internal 的配置 `num_memory_tokens=16` 没有参与其键值压缩或读取。它实际读 N 个键，并非16个。** `insertion_layer=14` 是 one-based，对应 `model.layers[13]` 的 forward hook，随后还有 block15–28 共14个 block。

依据：[模块构造:47](../../historical/scripts/foundation_20261016/connectors.py)、[池化和前缀:135](../../historical/scripts/foundation_20261016/connectors.py)、[内部读取:154](../../historical/scripts/foundation_20261016/connectors.py)。

## 2. 精确参数构成

所有 Linear 均含 bias；MHA 的 q/k/v 内部投影也存在，不能把 `source_projection` 视为唯一 K/V 投影。

| 模块 | 公式 | 参数 |
|---|---|---:|
| 分子 type embedding | 2×640 | 1,280 |
| source LayerNorm | 2×640 | 1,280 |
| Linear 640→128 | 640×128+128 | 82,048 |
| Linear 128→1536 | 128×1536+1536 | 198,144 |
| MHA128 in-projection | 3×128²+3×128 | 49,536 |
| MHA128 out-projection | 128²+128 | 16,512 |
| MHA128 合计 | 4×128²+4×128 | 66,048 |
| Qformer 学习 query | 16×128 | 2,048 |
| internal query LayerNorm | 2×1536 | 3,072 |
| internal query Linear1536→128 | 1536×128+128 | 196,736 |

因此 MLP=2,560+82,048+198,144=282,752；Qformer=2,560+82,048+66,048+198,144+2,048=350,848；internal=2,560+82,048+66,048+198,144+3,072+196,736=548,608。constant=16×1536=24,576。三个含蛋白内容的方法与 constant 均不等参数，internal 与 Qformer 也相差197,760个参数。冻结 backbone 各为1,543,714,304参数。

现有 MLP 使用 GELU，Qformer/internal 使用 MHA；internal 有 query norm、动态 query、无显式 gate，直接加非零初始化 residual；prefix 不存在同型 residual gate。现有 source/query Linear 使用 PyTorch 默认初始化，学习 queries/constant 初始化为 Normal(0,0.02²)。相同 seed 不能保证不同模块创建顺序下的对应张量逐位相同。

## 3. 通信预算必须分三段记录

本报告的“字节”是张量的逻辑表示大小，不是 GPU 显存峰值、总线实测流量、Shannon 信息量或已经证明有效的信息位数。把这几个量混作“16-token通信预算”会得出错误结论。

**源→连接器。** 三个含内容的方法均先把完整缓存搬入：cache residues30 为 float16，collate 强制 float32，[缓存序列化:74](../../historical/scripts/bench_v02/cache_panel_features.py)、[collate:153](../../historical/scripts/foundation_20261016/run_connectors.py)。每条有效源为 N×640×4=2,560N 字节，另有 bool mask N 字节和 int64 type IDs 8N 字节；真实分配受 padding 影响。constant 为统一 runner 仍搬运并验证源张量，但其预测不依赖内容。

**连接器瓶颈。** MLP 池化后为16×640 float32=40,960字节，再输出16×1536。Qformer 有16×128 float32 query/message=8,192字节；它访问的 projected source 为 N×128 float32=512N字节。internal 也有512N字节 projected source，但其 message 为 L×128 float32=512L字节。MHA 内部有效 K、V 为两份128维投影，逻辑总计1,024N字节。Qformer 的跨注意力边数为4×16×N，internal 为4×L×N；它们不相等。

**连接器→接收端。** 三种 prefix 的16×1536 embedding 最终转换为 bfloat16，即49,152字节，插入序列开头；文本的有效 position_ids 全部后移16。它们经28层 LM，所有文本 query 在每层可以读取 prefix。若按 Qwen GQA 的额外逻辑 K/V 计数，每层16×2(K,V)×2(KVheads)×128×2字节=16,384字节，28层为458,752字节；这是逻辑计算规模，**不是已分配 KV cache**，因为本实现 use_cache=False。

这个位移不改变文本内部任意两个 token 的相对 RoPE 距离；不能仅凭 position_ids 后移就认定它损坏文本表示。增加前缀本身、文本对前缀的相对位置与额外注意力路径才是实际结构变化。prefix codec 的消息不随当前问题/已生成文本变化，internal query则随其改变。

internal 在一个位置向全部 L 个文本状态写入 L×1536 bfloat16 residual，即3,072L字节；cast 前的 output projection 结果是 float32，即6,144L字节。该读写只发生在一个 hook，query 由当前文本状态决定，随后通过剩余14层传播。它既没有16-slot写入限制，也没有 prefix 的28层逐层额外 K/V。

**逐自回归步骤。** greedy_generate 不使用 KV cache，每产生一个 token 都重新运行整个 prompt、source codec 和 reader。生成 g 个 token 时，单条活跃样本的动态 query 总数为 gLprompt+g(g−1)/2，非一次 Lprompt。prefix 在 g 次 forward 中也重复构造并重复穿过 LM。批内 padding、已完成行和24-token上限会进一步改变物理工作量。应另外报告实际 g、各批 Lmax/Nmax、reader调用数、wall time、GPU峰值；旧归档没有完整 per-layer 实测字节/FLOPs tracing，不能追认为已有测量。

依据：[前缀与位置:174](../../historical/scripts/foundation_20261016/connectors.py)、[no-cache生成:215](../../historical/scripts/foundation_20261016/connectors.py)。

## 4. 两个分子的份额并不均衡

memory 总是**先肽、后全长参考靶蛋白**。MLP 在这个拼接张量上整体 adaptive_avg_pool1d，而不是每个分子各分配8槽。其第 k 个池化区间为 `[floor(kN/16),ceil((k+1)N/16))`，边界可重叠；该槽的肽几何系数为区间与 `[0,P)` 交集长度/区间长度。这里统计的是 source LN 后各残基对槽均值的几何系数，不能解释为经 MLP 后的实际因果贡献。

| 划分 | train 肽占 N 中位数 | test 肽占 N 中位数 | test MLP有效肽几何系数中位数 | test含肽槽中位数 | test纯肽槽中位数 | test N中位数/最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 已知靶 | 3.071% | 1.854% | 1.852% | 1 | 0 | 719 / 2,559 |
| 双新 | 3.067% | 2.331% | 2.315% | 1 | 0 | 551 / 3,068 |

两场景训练最大 N 均为3,429。全部分布和验证集统计见 JSON。本审计仅统计序列长度，未据此改 test、指标、样本权重或阈值。

Qformer/internal 对拼接后的全部 N 个键做 softmax。P/N 只是**均匀注意力下的参照份额**；它们可能学会强烈关注肽。旧运行没有保存完整 attention weights，不能宣称实际肽权重就是1.85%或2.33%。全长蛋白较大、肽在整体池化中被稀释是待检验机制，不是已经证明的失败原因。ESM 长序列缓存还采用不重叠≤1022残基分块，跨块不做 ESM 上下文注意力；此限制所有方法相同。[缓存分块规则:74](../../historical/scripts/bench_v02/cache_panel_features.py)

## 5. 优化、选择与梯度路径

四种连接器、两个场景共八项：seed17；已知靶 train/val/test=2062/147/152，双新2063/431/442；batch4；3轮；每轮 ceil(train/4)=516步，总计1,548步；AdamW lr3e−4、weight_decay0.01、clip_grad_norm1，无 scheduler、无梯度累积；每轮次序由 Python Random(seed+epoch)生成。精选仅依据完整验证集 answer-token NLL、相等取最早轮。

| 方法 | 已知靶最佳轮/NLL | 双新最佳轮/NLL |
|---|---|---|
| constant | 3 / 0.658262 | 3 / 0.669903 |
| input_mlp | 3 / 0.662964 | 3 / 0.671714 |
| qformer_style | 3 / 0.654320 | 2 / 0.670584 |
| internal | 3 / 0.659349 | 2 / 0.669394 |

训练 label 为两位小数 `{"pKD":数字}` 加 EOS。prompt label 全为−100，loss 按所有有效答案 token 求均值，包含 JSON语法、字段、数字和 EOS；不是每个样本等权的 pKD MAE，也不是只对数字部分计算。最终 test gold 保留原始精度、严格 JSON、无裁剪、最多24新token。格式 token 和常见数值分布的学习都能降低 NLL；更低 NLL 不保证更低生成 MAE。constant 与含蛋白方法的 correct/off NLL 改善接近，支持进一步分离格式/分布适配和蛋白内容作用，但不单独证明模型完全忽视蛋白。

LM 参数冻结不等于整个 LM 处于 no_grad。prefix 的梯度需要穿过28个冻结 decoder block 回到连接器；internal 的注入前14层没有上游可训练源，通常只需 forward，梯度从后14层传回 reader/source。internal 的动态 query 来自冻结前半段文本状态；上半段有真实梯度 Jacobian，backbone 权重不更新。teacher-forcing 时 query 可见前面的正确答案 token，但仍遵守 causal mask，无未来答案通过 source 泄漏。off 会完全移除干预，恢复普通冻结 LM 的位置和 logits；它不是同参数的内容空白干预，也不保留 prefix 的额外位置。

依据：[标签:88](../../historical/scripts/foundation_20261016/run_connectors.py)、[loss:190](../../historical/scripts/foundation_20261016/run_connectors.py)、[训练与选择:276](../../historical/scripts/foundation_20261016/run_connectors.py)。

旧数值预测器另有 BASE01 374,081参数、MEANPOOL 324,673参数、seeds17/42/2026、30轮、effective batch64、990步。它们拟合 train 标准化 pKD 的 MSE，按 val MAE 选模型，训练曝光约为新连接器的10倍；尽管 optimizer steps 较少，不能称计算预算较少。BASE03 从两种旧预测器按 val MAE 选一个，再用冻结 LM 生成，数值预测性能和文本输出失败须分别呈现。旧预测器是必要能力参照，**不是同损失、同训练曝光、同容量的插入位置对照**。[旧训练协议:249](../../historical/scripts/bench_v02/train_panel_baselines.py)

## 6. 为什么不能把当前差异归因插入位置

混杂包括：282,752/350,848/548,608容量不同；MLP/学习query/文本动态query不同；16个前缀槽与 N个全残基键不同；整体池化与学习注意力不同；16次固定query与 L次动态query不同；28层与14层剩余接收传播不同；额外位置和文本 RoPE位置不同；query norm/投影和 residual 形式不同；无gate的随机初始化扰动不同；source内容和语法适配未用同容量null分离；同seed不代表同对应权重；单seed不估计优化随机性。双新曾有 CUDA失败和同步标志恢复记录，环境模式也须在新矩阵中统一并保留旧失败归档。

三轮仍属固定预算基础实验，不能把“没有稳定优势”直接解释为内部机制无效；同样不能据此更换测试样本、最佳seed、评价指标、错误处理或用gold修正输出。

## 7. 可实现的最小匹配矩阵（建议，尚未执行）

### 7.1 先定义真正能隔离的主问题

主比较采用**同一 reader 在 embedding之后与 decoder block14之后的 residual 注入**。两者均不新增 LM token，均每个有效文本token读取一次同一16×128消息，均写入1536维 residual。它检验的是**读取/注入深度**，不应把早期 reader 命名为原先 input_mlp 或经典 prefix。

共享codec：8个肽query只读肽、8个靶query只读靶，共16×128，source LN/type/640→128和MHA权重共用，16个学习query按分子固定分配。这样分子配额是预先定义的结构因素，所有接收位置相同。codec 参数=1,280+1,280+82,048+66,048+2,048=**152,704**。K/V读入可仍为全部残基，源编码侧注意力边数固定为4×(8P+8T)，发送给 reader 的消息严格16×128 float32=8,192字节。

统一 reader：query LN1536(3,072)+Linear1536→128(196,736)+MHA128(66,048)+Linear128→1536(198,144)+标量gate(1)=**464,001参数**。两个位置使用完全相同的norm、projection、gate函数和初值，保存共享初始化tensor SHA，不靠“同seed”猜测相同。reader无dropout；不添加未参与forward的dummy参数凑数。gate初值/初始化分布须在新的 prereg 中固定，不能按test选择。

严格深度诊断应**冻结同一个codec state**，每个样本产生逐位相同消息，两位置各训练464,001参数。codec可采用预先固定的无标签初始化/固定编码，并明确这测的是固定消息的可用性；如采用训练过codec，拟合只能用TRAIN且事先定好统一选择方式。端到端版可以从同一初始化各自学习codec，总计**616,705可训练参数**；此时只是同容量/同初值，最终消息不再逐位相同，应单列结论。

| 层级 | 注入条件 | 内容条件 | 推荐seed | 每场景运行数 |
|---|---|---|---|---:|
| 核心固定消息 | embedding后 / block14后 | 真实 / 内容中性 | 17、42、2026 | 12 |
| 信号负控扩展 | 同上 | 训练时独立新噪声；heldout噪声预先冻结 | 同三seed | 6 |
| 联合优化扩展 | 同上 | 真实 / 内容中性，同codec架构共同训练 | 同三seed | 12 |

先做核心12项×2场景；另外两层须按资源排程，但冻结测试集及旧结果原样保留。三seed用于初步稳定性估计，不能把3个seed当足以支持很强的总体推断。

### 7.2 真正 prefix 应作为次级系统对照

可用**同一 reader**，以固定的16×1536 query buffer（非额外可训练参数，各槽不同且初始化预先冻结）读同一codec，生成16前缀；与内部 reader拥有完全同样的464,001参数，codec联训则均616,705。全部参数必须实际参与forward并审计grad，避免参数“挂名匹配”。

即便如此，prefix为16个固定query，而内部为 L个文本动态query；prefix多16位置、文本位移16、通过28层，内部不占位置、通过14层；一次49,152字节prefix写入与 L次3,072字节状态写入并不相等。这些是架构本身的区别，**不能同时宣称同 query次数、同接收端字节、同剩余层数和同prefix结构**。应分别报告“共享源消息槽/精度匹配”和“实际接收计算/流量未匹配”，必要时另做固定compute/latency约束的系统比较，不能靠空算补FLOPs得出机制公平。

### 7.3 同参数 null、随机内容与消融

内容中性对照保留完整codec+reader、同训练预算、同初始化与参数，给两种分子预先固定的规范占位源，固定槽和mask；确保没有sample ID、长度、来源或label从占位构造泄漏。codec之前的constant源应经验证在所有样本输出相同；只把输入置零但保留变长mask/类型比例，可能仍泄漏长度。

若需要源编码侧也保留计算形状，可在肽/靶各自的原始长度下重复该分子的固定值，分子独立归一化读取后理论输出与长度无关，并验证实际输出一致性；同时记录浮点误差。固定消息会使部分内容依赖参数的梯度为零，这是null机制的预期，不应另挂无效参数伪装有效容量。报告应区分“模块/名义参数完全相同”与“在此输入条件下可辨识的有效自由度相同”。

随机内容对照应独立于label：训练每次产生独立噪声、heldout噪声在运行前固定seed和分布（尺度只可由TRAIN估计），不按sample ID生成可记忆且跨split关联的伪特征。其作用是检验“有一个可训练干预”的正则化/格式适配，不能替代真实生物学错配金标准。已有严格 C1 donor为0时，P/H保持未定义；跨样本随意置换不能冒充符合原准入规则的C1错配。

建议真实/中性/噪声各自都训练同型系统；在已训练真实系统上另做off、零消息、固定消息的推理干预，分别标记分布外干预，不与重新训练的null混合。三种内容条件和fixed/joint codec分层预先冻结，避免在test上挑有利的对照。

## 8. 新实验应冻结的分析合同

沿用旧样本、物理group、test gold、严格parse、24token、全样本分母规则、无裁剪、原精度计分和val NLL选择。全方法相同batch、epochs、optimizer、gradient clip、序列化、backbone revision、dtype和CUDA执行模式；新记录实际曝光、answer tokens、optimizer steps、16-slot消息dtype/字节、每次reader query/key数和调用数、通信写入维度、wall time/显存。shared codec / reader 初始张量须逐位核验。

先看每seed原始结果，再给同seed配对差值、seed均值与标准差、最小/最大值；同一个seed中的test-group bootstrap与跨seed波动分开，不把重复seed预测当独立新增样本。现有test已经多次查看，本轮属于在同一冻结benchmark上预先登记的受约束跟进，不能包装为全新未见的最终确认测试。模型/训练改进先通过TRAIN/VALIDATION定型，最终需要外部独立资料作确认，但不得替换或重划现有test。

准确率/回归质量、parse失败率、消息利用和系统代价分别报告。若任一生成失败，保留全部n与失败原因，full_sample_mae继续未定义，valid-only只按其原分母解释；不新增有利的后处理或罚分口径。来源/构建体不确定性、严格donor缺失及尚未完成的其他任务金标准，仍保持原状态。
