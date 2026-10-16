# NormFormer / ReZero 官方源码静态补核

两篇论文的官方代码关联明确，已新增独立、精确commit的源码归档。只做下载、字节核验、安全提取与静态阅读；未导入或运行下载的代码，未安装依赖、加载模型、调用GPU或改动当前36项。已有残差文献报告及审查文件原字节保持，保护指纹保存在同名JSON。

## 来源、版本与许可证

先检索本项目既有repos文件名，未发现可复用的NormFormer/ReZero/fairseq源码。两篇本地原PDF物理第1页的实际超链接分别指向`pytorch/fairseq/tree/main/examples/normformer`与`majumderb/rezero`。GitHub公共API将前者规范到`facebookresearch/fairseq`；该仓库NormFormer README再次给出同名论文、arXiv ID和作者联系人。ReZero README、包作者信息与MIT版权信息均对应论文作者。

| 代码 | 固定commit | commit日期 | 顶层许可 | 原ZIP字节 |
|---|---|---|---|---:|
| facebookresearch/fairseq | `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99` | 2025-09-30 | MIT | 14,039,779 |
| majumderb/rezero | `e2c94a825c5564217e8cf4d75a28d59cab1d7029` | 2021-06-02 | MIT | 6,849 |

这是当前官方仓库的固定源码快照，**不是已经核实的论文原实验commit**。两篇PDF仍保持各自arXiv v2/preprint身份；没有用较晚代码日期改写论文版本。许可证原件已保留，证据文件SHA包含在JSON中。

原ZIP SHA256：

- fairseq：`fe8bc0f9aae8bf027752e700840a5402a9af3dba71cd5daade7e3300a645a991`。
- ReZero：`f63f24876d32e539a47245f17256a00c4097e35d63c0863a3555ecff1bf39584`。

[隔离仓库manifest](../../docs/PUBLICATION_SCOPE.md)保存URL、commit API元数据、完整原archive及逐文件指纹。

## 安全提取与完整性边界

所有成员先校验绝对路径、父目录逃逸、Windows驱动/反斜线、Unix文件类型及解析后目录包含关系，写入采用exclusive模式，不使用通用extractall。

fairseq首次整体预检发现symlink，**在任何fairseq source写入前拒绝**。保留了[首次拒绝记录](../../docs/PUBLICATION_SCOPE.md)和完整原ZIP；随后仅建立**普通文件的partial view**，不称完整checkout或完整解包。以下4个link成员始终拒绝，不创建、不跟随、不解析到目标：

- `examples/adaptive_span/truncated_bptt_lm_task.py`
- `examples/mr_hubert/simple_kmeans`
- `examples/wav2vec/unsupervised/kaldi_self_train/st/steps`
- `examples/wav2vec/unsupervised/kaldi_self_train/st/utils`

官方ZIP另捆绑3个HuBERT `.npy`测试fixture，共7,324,224字节；未加载这些数组，也未另外获取数据。它们保留在原ZIP，已从source-only partial view排除并记录omission。最终fairseq提取1629个普通源码/文档文件，ReZero提取全部8个普通文件；没有创建symlink。

所有提取文件均独立核对原ZIP内容、磁盘字节与manifest的SHA一致；其中**13份静态证据文件**还在报告JSON逐项登记。原ZIP合计14,046,628字节，源码视图合计13,747,376字节，总计27,794,004字节；每仓库archive+view均低于50MiB，总计低于100MiB。未另下模型或外部数据集。

## NormFormer：分支LN位置、缩放与训练recipe

官方[NormFormer README](../../docs/PUBLICATION_SCOPE.md)要求`--scale-attn --scale-fc --scale-heads`；较小模型才建议额外`--scale-resids`，其1.3B例没有该选项。

- [transformer_layer.py:277](../../docs/PUBLICATION_SCOPE.md)：`attn_ln`由scale_attn开启；`c_attn`是nheads个可训练ones；`ffn_layernorm`由scale_fc开启；`w_resid`是hidden维可训练ones，由scale_resids单独开启。
- [forward:451](../../docs/PUBLICATION_SCOPE.md)：self-attention返回后依次为c_attn缩放、attn_ln、dropout、residual相加。额外LN在**新增attention分支内**，不是对整个相加结果归一化。
- [FFN:506](../../docs/PUBLICATION_SCOPE.md)：activation(fc1)和activation dropout后做ffn_layernorm，再fc2；可选w_resid乘的是**skip residual**，然后与FFN输出相加。

一个值得保留的实现细节：此快照的c_attn在MHA返回后，把输出分成head-sized slices再缩放；[multihead_attention.py:773](../../docs/PUBLICATION_SCOPE.md)显示返回前已经过out_proj，快速路径也传入out_proj权重。因此不能把这段代码逐字描述成“在WO前缩放未投影head”；原论文物理第4页讨论过把缩放移出attention的效率实现。机制核查需同时保留论文公式和此commit实际顺序。

[train_lm.sh](../../docs/PUBLICATION_SCOPE.md)公开recipe使用FSDP、8GPU、memory-efficient-fp16、activation checkpoint、2048-token序列、polynomial decay、750 warmup及seed1。1.3B函数指定24层、hidden2048、32heads、max-update上限286102；README给LR6e−4。CLI虽名为`--optimizer adam`，但[实现文档:45](../../docs/PUBLICATION_SCOPE.md)说明是AdamW式decoupled weight decay；recipe betas为0.9/0.98。

README第12/27行明确原Table2使用更大的私有数据；公开训练命令的步数上限也不是论文等GPU时间实际完成步数收据。源码存在并不等于可以用公开数据精确复现论文成绩。

## ReZero：零标量直接乘法，无tanh、无分支LN

[rztx.py:47](../../docs/PUBLICATION_SCOPE.md)显式建立可训练`resweight=Tensor([0])`。encoder的同一个scalar被self-attention和FFN共用（74、80行）；decoder同一个scalar被self-attention、cross-attention及FFN共用（115、139、142、148行）。它是直接乘法，**没有tanh**。

这两个layer实现没有LayerNorm；[README:54](../../docs/PUBLICATION_SCOPE.md)要求外层Transformer的norm保持None。包构造器默认dropout0.1、ReLU，README演示8heads；这些默认值不是原论文AppendixB中GELU、dropout0.2、2heads的实验配置。仓库提供layer/package和作者教程链接，未提供论文全部数据、模型与实际训练收据；未下载或执行教程。

## 与当前迁移的区别及可支持的结论

当前系统保留预训练Qwen的既有归一化，冻结base/eval，只训练一个读取16×128固定蛋白消息的Reader；gate为`tanh(alpha)`、alpha初值0，每个Reader只在一个注入位置使用。它既不是上述完整NormFormer预训练，也不是移除LayerNorm并从头训练的ReZero堆叠。

当前已存在query LN，但没有新增分支输出LN。未来“δ输出内归一化/固定TRAIN尺度校准+零gate”仍只是独立、未执行的迁移假说，不能将NormFormer原成绩或ReZero的全网isometry结论直接套用。零gate下F的任务梯度为0也不表示所有F参数首步必定不变，当前AdamW可对non-None零grad做weight decay。

本补核能够支持“官方源码确有这些初始化、分支顺序及可学习缩放”，不能支持“它们已改善本课题”。当前36项、冻结数据/测试标准、旧结果及已有文献审查报告均未改变。
