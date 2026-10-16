# 残差尺度与零初始化：两篇原论文及后续检验方案

**现有四项统计提示应检查“相对残差尺度”，尚不能解释内部连接器的因果效果。** 下面两篇原论文分别支持关注分支归一化与零初始化，但都没有验证冻结Qwen上的蛋白连接器。本次只收集原文、CPU提取与文献证据，未训练、未校准任何新系数、未修改当前36项矩阵或已有源码/协议/结果。

## 当前观察的准确范围

四项均为known-target、seed17、由原完整val NLL选中的epoch1。统计来自该轮验证前向的全部有效prompt及teacher-forced答案位置，共34,968个位置；不是仅答案token、最后token或坐标均值，也不是下面拟议的TRAIN prompt-only校准统计。

| 配置 | hidden L2/token RMS | gated δ/h L2比 | alpha |
|---|---:|---:|---:|
| early-real | 1.005459 | 0.242108 | 0.006421 |
| early-neutral | 1.005459 | 0.264415 | 0.005827 |
| mid-real | 719.011396 | 0.011299 | −0.020794 |
| mid-neutral | 719.011396 | 0.012773 | −0.020559 |

逐项history路径、原始精度和SHA见同名JSON。这个比例不等于语义信息量、梯度范数、Jacobian敏感度或预测贡献；方向、后续层、优化与常量内容均可能参与。不能由四项关联推出“mid被淹没”，也不能用V2/V3批量资格失败证明mid架构较差。

## 原论文一：NormFormer

Sam Shleifer、Jason Weston、Myle Ott，**arXiv:2110.09456v2，2021-11-01，14页，稿面标注ICLR2022审稿稿**。已先查本地项目文件及工作台题名/ID，未找到可直接复用的原文，随后取得官方版本。

原文§2.1，物理第2页的head scaling系数初始化为1；物理第3页的**未编号公式**在attention输出分支上增加LN：

`NormScaledMHA(x) = x + LN(HeadScaleMHA(LN(x),LN(x),LN(x)))`。

另在FFN首层GELU后增加LN。§4.1（物理第6–7页）分析的是层间梯度大小失配。作者1.3B CLM配置为LR6e−4，Table2（物理第5页）等GPU时间比较baseline286K步与NormFormer275K步；额外算子有真实开销。§4.2（物理第8页）还报告可学习skip缩放在1.3B上无益，不能把“加缩放”当作普遍改进。[本地PDF，第3页](../../docs/PUBLICATION_SCOPE.md) · [官方PDF](https://arxiv.org/pdf/2110.09456v2)

迁移时应区分：论文的可选ResScale缩放skip的x，本报告候选缩放新增δ；本地记录的是activation/residual范数，原论文分析的是梯度。两者都不能称为原论文在本任务的复现。

## 原论文二：ReZero

Thomas Bachlechner、Bodhisattwa Prasad Majumder、Huanru Henry Mao、Garrison W. Cottrell、Julian McAuley，**arXiv:2003.04887v2，2020-06-25，14页，preprint/under review身份**。

物理第1页Eq.(1)：`x_(i+1)=x_i+alpha_i F(x_i)`；物理第2页明确alpha初值为0，F的初始**任务梯度**为0。物理第8页Table3在enwiki8到1.2BPB时列出8800步与warmup基线13690步，速度比1.56倍，不能写成“少56%步数”。AppendixB（物理第12页）：12层、hidden/embed512、2heads、context512、batch1080、attention/FFN dropout0.2、GELU、LAMB固定LR0.016。[本地PDF，第1页](../../docs/PUBLICATION_SCOPE.md) · [官方PDF](https://arxiv.org/pdf/2003.04887v2)

本实现使用tanh(alpha)，仅在零点保留相同的零输出和一阶门控性质，离开零点后不等同原论文的无界alpha。零gate保留的是**原冻结LM行为**，不意味着整个冻结LM成为identity或已有全网dynamical-isometry证明。此外，当前AdamW wd=0.01可对non-None零grad参数做weight decay；任务梯度为0不等于F参数首步完全不更新。

## 可检验的迁移假说，尚未执行

当前Reader已有query LN，但它不归一化最终写入的δ，也不使δ与接收状态h的相对尺度一致。后续可另立研究协议检验两个方案，当前36项及评分保持原样。

**方案S1：固定、无标签的TRAIN尺度校准，仍用零gate。** 对每个placement及预定seed，用冻结base和未训练Reader，仅读取固定TRAIN prompt，不拼接gold答案。令u为未门控Reader输出，`r(a)=sqrt(mean_valid_TRAIN_prompt_tokens ||a||²)`，固定` s=r(h)/r(u)`后使用：

`h_new = h + tanh(alpha) * s * u，alpha_init=0`。

这把未门控分支的初始潜在相对RMS统一为1，是预先定义的单位，并非论文提供或由val/test挑出的最优目标。校准样本ID、padding规则、初始化SHA和标量需冻结；分母无效则拒绝校准，不能用val/test补算。s同时改变gate梯度与有效优化参数化，故后续差异也不能仅归因通信内容。

**方案S2：在δ分支内增加输出LN，再零gate。** 在1536维output projection后、gate前使用无affine的`v=LN(u)`，保留`h_new=h+tanh(alpha)*c*v`。如要隔离“归一化形状”与“相对幅度”，c也应按同一TRAIN规则取`r(h)/r(v)`，与S1配对比较。只加LN而不校准相对尺度，不能保证early/mid的相对注入一致。

不能改成`LN(h+delta)`：即使alpha=0，它仍把h改成LN(h)，破坏原本的冻结base保留性质。此方案只是借鉴NormFormer的**分支内输出归一化**与ReZero的零初始化，不是完整复刻两篇网络。

未来检验应预先固定placement×方案、real/neutral及多seed，报告训练梯度、gate轨迹、raw/gated δ/h、val选择和实际代价，并保留原始配置。比例趋于接近本身也不是任务有效性证明；模型结论仍需原定任务指标和固定样本验证。

## 参数、通信与资产记录

无affine LN增加0个可训练参数，Reader仍464,001；若采用affine LN，则两位置均增加3072参数，改为467,073，必须单独登记。16×128 float32消息仍为每样本每forward 8192逻辑字节；额外LN和缩放有O(BLH)运算、激活/反向开销，TRAIN校准前向也要计账。参数或消息尺寸相同不能宣称同实测计算预算。

两份PDF保存于独立新目录，原始字节不修复、不覆盖：

- NormFormer：2,201,887字节；SHA256 `c3bd7416cf2e37290a3a0ac807f193c344a905605afb86aad9bef3fd14bfc502`。
- ReZero：1,314,235字节；SHA256 `54eea1741fad2e7a20e52f55c64b34977f1206c5e3c8e14804ea5385c299c070`。PyMuPDF提示一处colorspace解析警告，14页文字均非空，警告保存在provenance中；公式以上述官方/本地物理页为准。

下载、逐页CPU文本、OpenAlex精确身份、工作台新增paper ID及检索状态保存在[资产manifest](../../docs/PUBLICATION_SCOPE.md)与[入库收据](../../docs/PUBLICATION_SCOPE.md)。未下载模型/数据集，未执行论文代码。

两篇均已作为**新paper**入库：NormFormer `22de1a65-affd-5b7f-a816-57fe0c0fca3b`，ReZero `4917e4f7-6691-54c2-a869-452a0c28a9cb`。本地PyMuPDF正文分块及关键词检索完成；语义索引使用现有远程SiliconFlow BAAI/bge-m3 API，分别35/35、45/45目标完成，两次实际semantic查询均命中对应新paper。没有占用本机GPU或提交全库任务；未改旧campaign/status。
