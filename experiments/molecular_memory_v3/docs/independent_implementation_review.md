# 独立实现审查

结论：**APPROVE_WITH_LIMITS；Architecture CLEAR。未发现需请求修改的实质bug。** 此结论针对实现与CPU合同，不构成预训练模型、GPU或科学预测有效性验收。

审查范围：`src/mm_model.py`、`src/mm_data.py`、`src/mm_extract.py`及对应测试；补读`mm_train.py`的checkpoint保存、身份验证和恢复调用。只写本报告与JSON，不改实现、测试、旧数据、底座或依赖。

## 实证结果

在Ubuntu-22.04既有`<PYTHON_ENV>/bin/python`执行，CUDA_VISIBLE_DEVICES为空、OMP/MKL threads=2，PYTHONDONTWRITEBYTECODE=1。当前unit suite **44/44通过，2.682秒**，包括并行开发期间新增的缓存partial/canary门禁测试。首轮28/28通过，后续改动后已重跑。

额外执行12组独立对抗检查：6配置×FP32/BF16，构造随机极小Qwen2（3层、12维，eager attention），没有调用from_pretrained、下载权重或加载真实Qwen。任意替换并延长teacher suffix后numeric和两组routes最大绝对差均为**0**；左侧再补两个padding并相应平移prompt_end_index，numeric最大绝对差也为**0**。打开gate后两路分子投影梯度均非零、底座所有parameter.grad仍为None。另核collator会随答案长度变动正确平移batch中其它行的prompt_end_index。共13条额外检查逐项保存于JSON。CUDA未初始化。

命令：

```text
wsl -d Ubuntu-22.04 --cd <PROJECT>/experiments/molecular_memory_v3_20261017 -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 <PYTHON_ENV>/bin/python -m unittest discover -s tests -p test_*.py -v
```

## 核查矩阵

|核查项|证据与判断|
|---|---|
|numeric是否读gold suffix|`mm_data.py:280–285`显式计算最后prompt token位置，labels仅在之后；`mm_model.py:303`读取finalnorm的该位置。标签替换/加长与真实tiny Qwen双dtype检查均保持数值不变。通过。|
|router anchor是否标签泄漏|`mm_model.py:238–243`从单个注入hook之前的prompt_end query取anchor；没有均值池化完整teacher序列。suffix不变性实测通过。|
|冻结LM后的hook梯度|`mm_model.py:297`未用整体no_grad包裹LM；注入后冻结层仍保留对可训练残差的反向图。模块全梯度测试与随机Qwen梯度检查通过。|
|zero gate后sender梯度|`mm_model.py:259`零gate首步分支梯度为0符合公式，alpha仍有信号；非零gate时投影、关系、reader、router等模块任务梯度通过。未把零初始化声称为首步全部参数非零梯度。|
|条件allowlist|`mm_data.py:111–152`逐项取七化学字段、受限测量/靶标上下文，prompt同用context_view；source ID/raw metadata/gold变化不改变prompt或条件。允许键内对象/数组拒绝。|
|ESM15/30语义|`mm_extract.py:26–27,86–98`直接hook第15块、取last_hidden_state归一化最终层；random tiny30层ESM测试通过。生产版本锁TF4.48.1及已登记源码SHA，当前TF4.57.6微型测试不冒充生产资格。|
|六臂容量和通信活动|六臂注册相同图和形状；所有臂都执行p/t/r双view、c单view读出（7次reader）与2次relation attention。H30/H30保留两个独立投影；self/cross同形，静态anchor仍有可训练router。对应测试通过。|
|全残基coverage再pool|`mm_extract.py:109–112`先拼接所有有序chunk raw，核长度，再adaptive平均池化8槽；记录有损、无跨chunk attention、未保存raw。测试覆盖分块、短序列、同次双层和异常hook清理。|
|缓存完成与失败状态|审查期间owner补齐partial进度、失败状态、TRAIN-only canary、complete/full门禁和来源版本锁，新增测试通过。未完成/仅canary缓存不供正式SequenceStore。|
|checkpoint是否遗漏|`mm_model.py:114–129`保存全部非base参数与两个buffer、恢复检查键/shape/dtype/finite；CPU weights_only往返预测完全一致。`mm_train.py:251–264,440–445,571–583`额外绑定identity、选择epoch和checkpoint SHA。未见当前调用遗漏。|

## 解释范围与剩余验证

A_Q禁用的是router的样本条件anchor，reader仍使用逐token隐藏状态query。该范围已在model_implementation.md明确，不能把此臂解释成“完全无文本依赖”。静态输入router的有效可辨识容量与动态router不同，当前合同只主张名义参数/活动形状匹配。

允许字段的名称、类型被约束，允许string的语义内容并非通用泄漏检测器；本固定数据实验仍依赖冻结且审计过的原始来源。没有发现gold字段直通路径。

SequenceStore.verify只核文件指纹；pooled另核覆盖/shape/finite。当前用途把manifest SHA固定于run pin；不能将任意自写manifest的verified=true当来源证据。

预训练Qwen/ESM的实际dtype、数值稳定性、显存、端到端耗时及科学有效性仍由独立资格/正式运行确认。当前审查未运行这些负载。源文件与测试的精确SHA快照在JSON中；并行开发后续若改动核心文件，应以新SHA补审。
