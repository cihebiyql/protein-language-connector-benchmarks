# Molecular Memory模型实现合同

实现位置：`src/mm_model.py`，CPU测试：`tests/test_model.py`。本实现不加载模型、下载资源或启动训练。真实Qwen的接入与完整任务验收由新scope执行器负责。

## API

`MolecularMemoryLM(base_lm, variant="M3", hidden_dim=1536, esm_dim=640, memory_dim=128, num_heads=4, hook_layer=13)`。`hook_layer`为零基索引，默认在第14个decoder block之后注入。要求底座提供`get_input_embeddings()`、`model.layers`及`model.norm`，禁止gradient checkpointing。

`forward(input_ids, attention_mask, prompt_end_index, memory_inputs=None, labels=None, off=False)`：

| 输入 | 合同 |
|---|---|
| input_ids | int64 `[B,T]` |
| attention_mask | binary `[B,T]`，与输入同设备，支持左padding |
| prompt_end_index | 必传int64 `[B]`，指向每行最后一个答案前prompt token，必须active；不会从labels或最后答案位置猜测 |
| memory_inputs | 精确六键：p15、p30、t15、t30、condition_ids、observed；任何额外metadata键拒绝 |
| p15/p30/t15/t30 | frozen finite float `[B,8,640]`；每个bank须由数据层覆盖完整序列后pool；模型不截取序列 |
| condition_ids | int64 `[B,10]`、范围0—4095；固定哈希由数据层完成，不做TRAIN类别拟合 |
| observed | bool `[B,10]`；false使用对应槽独立missing embedding，忽略该槽hash ID的值 |
| labels | 可选int64 `[B,T]`、`-100`忽略；只允许在prompt末端之后的active位置监督 |

返回dict：`logits[B,T,V]`、`numeric[B]`、同一tensor别名`numeric_z[B]`、`routes={view[B,2], bank[B,4]}`、`gate=tanh(alpha)`，以及`loss`。`labels=None`时loss为None；否则loss仅为shifted answer-token CE的sum/count。训练器负责TRAIN-only数值标准化、joint MSE权重和选点；模型不读取gold数值，不裁剪预测，也不把numeric回填语言生成。

生成由训练/评价层用固定prompt_end_index逐次追加token并调用forward，模型始终传`use_cache=False`。不在本模型内部更换greedy、EOS或strict parser策略。

`trainable_state_dict()`返回全部非底座参数及固定buffer的CPU tensor副本；`load_trainable_state_dict(state)`核对精确键、shape、dtype、finite后恢复。53个参数tensor＋2个buffer，共55个tensor；不含底座权重。变体标识及数据/协议版本由外层checkpoint metadata绑定，不能跨变体误当同一科学结果。

`capacity_report()`返回逐参数数量；`communication_report(B,T,hidden_element_bytes)`返回单次forward逻辑预算；`diagnostics(reset=False)`返回累计实际forward次数、padding/active位置、调用/通信账本、末次路由及gate。路由在forward时仅detach保留，调用diagnostics时才转换CPU列表。

## 六种活动对照

| 配置 | 第一个/第二个逻辑视图 | router anchor | relation K/V |
|---|---|---|---|
| M0 | H30/H30 | 固定q0 | peptide self |
| M1 | H15/H30 | 固定q0 | peptide self |
| M2 | H15/H30 | prompt末端注入前状态 | peptide self |
| M3 | H15/H30 | prompt末端注入前状态 | target cross |
| A_L | H30/H30 | prompt末端注入前状态 | target cross |
| A_Q | H15/H30 | 固定q0 | target cross |

每个视图先对raw pooled 640维向量作无仿射LayerNorm，再通过各自独立`Linear640→128`；同视图p/t共享该投影。加入4类type embedding中的p/t类型、2类**逻辑view-slot** source embedding和固定sin/cos槽位置。source表示逻辑视图身份：H30/H30对照仍有两个不同view标记，不能将其说成实际H15来源。两个视图各自有独立relation MHA和MLP；relation=`MLP(concat(p, attention(p, partner), p*attention(p, partner)))`，MLP为384→128→128、GELU。self和cross均使用8个query/8个K/V，同参数、同调用形状；target始终另经target reader参与，两臂都不制造负例。

条件采用共享`Embedding4096×128`，observed=false取`Embedding10×128`对应missing槽；再加condition type和10槽sin位置。missing是有效状态，不是padding；全部10槽参与attention。具体十槽字段与受限composite由`mm_data`的固定合同定义。

文本共享`LayerNorm1536 + Linear1536→128`产生每token query。p/t/c/r各一个独立MHA128、4 heads；p/t/r的reader跨两个视图共享，condition只读一次。view router的2个softmax权重共享应用于p/t/r的两个读出；bank router的4个softmax权重再混合p/t/c/r。两个router均Linear128→{2,4}、权重与bias全零初始化，因此初始view=1/2、bank=1/4。

问题anchor固定取注入前query的prompt_end_index；静态q0为1536维linspace(-1,1)，经过同query LN/projection再入router，非恒定以避免LayerNorm消掉全常量。静态router仍参与前向/训练，但输入恒定导致可辨识自由度更少；“同名义参数”不等于有效统计容量相同。每token的bank reader query始终来自该token状态；去问题条件化只控制router anchor，不宣称完全删除文本依赖。

混合128维输出经共享Linear128→1536写回，外乘标量tanh(alpha)，alpha初值0。输出投影和其余分支正常非零初始化。零gate第一步分支任务梯度为0是预期，gate应先获得信号；非零gate测试另外核实所有活动模块均有任务梯度。

## 冻结、因果性与异常清理

底座所有参数requires_grad=false；`train()`也始终保持底座eval。forward不在整个LM外套no_grad，注入之后的冻结decoder运算仍保留连接器反向梯度。全部连接器参数FP32，连接器计算关闭autocast，残差仅在写入时转为底座hidden dtype。冻结原始ESM输入禁止requires_grad。

numeric head是Linear1536→1，读取**最终model.norm输出**的显式prompt_end_index。路由anchor与numeric均不读取后续teacher suffix；CPU因果LM测试对任意替换并延长suffix逐配置检查numeric/routes完全相等。真实底座还须在工程资格中复核，不用tiny测试冒充实际模型验收。

注入hook与最终norm捕获hook都仅在一次forward作用，成功/异常均在finally中移除；禁止并发或重入forward。off严格绕过全部memory/condition/relation/reader注入，仅运行同position_ids、mask、use_cache=False的底座；off logits与直接底座逐bit相等。numeric head仍可读取off底座的prompt state，routes为None；不把off叫作另一次同监督训练。

## 精确默认名义参数

以下由CPU shape-only fake底座实例化后逐参数numel求和；不加载或运行真实底座。

| 模块 | 参数 |
|---|---:|
| 两个分子投影 | 164096 |
| type(4) / source(2) | 512 / 256 |
| 条件共享词表 / missing(10) | 524288 / 1280 |
| query LN / projection | 3072 / 196736 |
| 两个relation MHA＋MLP | 263680 |
| p/t/c/r四个reader | 264192 |
| view / bank router | 258 / 516 |
| 输出投影 | 198144 |
| 外gate | 1 |
| numeric head | 1537 |
| **每配置总计** | **1618568** |

两个无训练buffer：slot_positions[10,128]、static_anchor[1536]。所有配置注册相同参数及shape；H30/H30仍使用两个独立投影，self relation仍调用两个完整relation模块，静态路由并非关掉router。词表embedding行是否在某批被索引和gate0造成的首步零梯度不等同于未注册/禁用模块，也不能据此声称每一参数每一步都有非零任务梯度。MLP GELU、无仿射raw LN、sin位置和softmax无额外参数。

## 逻辑通信与实际调用

全部bank槽有效。每行两视图各p8+t8+r8，加共享c10，共58个128维FP32槽：**29696 bytes/forward**；不能沿用旧8192 bytes声明。raw选择输入为两视图×p/t×8×640，经FP32转换的逻辑投影输入81920 bytes/row；H30/H30可能复用相同原tensor，logical预算不是独立信息量，也不是实际传输次数。

每次实际调用：reader p2/t2/c1/r2共**7**，relation attention共**2**；每个reader MHA有4 heads。reader K/V逻辑边界59392 bytes/row；relation K/V16384 bytes/row；reader query为3584×T bytes/row，relation query8192 bytes/row。attention edges为reader 4×T×58及relation 4×2×8×8；残差写回为B×T×1536×底座hidden element bytes。固定58槽里已包含由分子生成的relation，不能把它当作独立外部证据。

这些是module-boundary逻辑张量账本，不是PCIe/网络流量、Shannon信息或FLOPs；同形状也不保证GPU kernel时间完全相等。全部padding query仍实际计算，账本另列active位置。无新LM token、无KV cache；逐token生成每次重跑完整累计文本并重建可训练banks，须累加实际EOS前的forward账本。额外condition ID/missing输入、离线ESM抽取、tokenizer、LM算力、optimizer状态与训练标签均不冒充这个单一payload预算。

## CPU验证证据与边界

WSL Ubuntu-22.04既有Torch环境，CUDA_VISIBLE_DEVICES为空，OMP/MKL threads=2。11个独立CPU测试全部通过：六配置shape/同参数图/初始uniform路由，严格off，所有变体suffix隔离，零gate先有信号及非零gate全模块任务梯度，冻结底座，无底座tensor的weights_only序列化含buffer，成功与异常hook清理，输入合同，missing不受hash占位改变，实际7＋2模块调用及通信账本，H15选择/self-cross/static-query敏感性。测试使用随机自有tiny因果LM，不是科学效果结果，也没有真实Qwen/ESM forward、GPU或fit。
