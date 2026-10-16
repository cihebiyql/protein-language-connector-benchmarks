# 输出分布与残差源码补证的独立审查

**结论：通过；未发现新的评价标准漂移、无依据因果／显著性／正式校准主张。P0—P3发现均为0。** 本审查针对固定17:26首八格与已下载的证据快照，没有读取活跃案例、运行模型或第三方源码，没有GPU、新fit、下载或旧文件修改。

科学冻结SHA仍为 `9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208`。本次112项只读核对全部通过：重算2,376条首八格保存预测的MAE、全局平均并列秩Spearman、ddof=0分布及精确值频数；核对594条旧TRAIN mean保存预测；复核2份源码ZIP、13份静态证据文件的磁盘／ZIP／登记SHA及18份旧保护报告指纹。完整值、来源和逐项结果见[JSON记录](additional_diagnostics_review.json)。

## 输出诊断没有替换主评价

[输出分布报告](output_distribution_diagnostic_8cases.md)明确保留原完整总体MAE，原全局Spearman只是已有登记指标，未升格为新的主要端点。它正确指出`metrics.target_macro.spearman`的容器名字不代表逐靶点相关系数宏平均。mean／总体SD／min／max／精确不同值数及频数另列为**观察test后的描述性统计**，没有新增阈值、优胜规则、重分箱、异常值剔除或拟合。

八格确有不同程度的输出收缩和离散集中；151/152输出6及423/442输出5是字面频数事实，不依赖新定义的collapse阈值。报告同时保留其他3—5取值系统和小幅正全局rho，没有称八格完全常数或所有排序信息为零。正rho仅说明这些保存样本中的单调关联；未将其写成统计显著性、生物因果或残基消息来源的证明。

mid-real均值更接近gold均值且MAE下降被表述为与整体位置解释相容；报告明确均值接近不决定MAE，没有估计误差中“多少由校准造成”，也没有把它称为已验证的条件／概率校准。双新mid-real的MAE改善而rho下降这一反向证据保留，避免把回归误差和排序效果互换。

TRAIN mean继续是已保存的独立数值通道，未重拟合；其常数秩方差为零，Spearman保持null而非0。报告未在test寻找最优常数、平移／缩放拟合或挑选新检查点。neutral仍保留允许prompt上下文，三seed汇总保持null，当前36项及MAE／覆盖／失败处理不变。

## 官方源码与论文身份、迁移边界一致

[源码补证报告](residual_code_evidence.md)将fairseq 2025-09-30及ReZero 2021-06-02的固定commit明确标为较晚官方源码快照，没有冒称对应论文原实验commit，也没有改写两份arXiv v2／preprint身份。fairseq是保留原ZIP的普通文件partial view，symlink与捆绑fixture的omission、首次拒绝记录均披露，没有声称完整功能checkout或已复现论文。

本次静态阅读实际下载文件确认：

- NormFormer的`attn_ln`位于新增attention分支内，在dropout和residual相加之前；FFN额外LN在activation／activation dropout之后、fc2之前。`w_resid`乘skip residual，不是新增消息分支。
- 此commit的head-sized slice缩放位于MHA返回之后，而MHA返回前已有`out_proj`；报告没有把它逐字等同于WO前未投影head缩放。
- ReZero实际layer建立零初始化可训练标量，encoder／decoder分别在各自多个子层共用该标量，直接相乘，没有tanh、没有layer内LN。README的外层`norm=None`与包默认值、原AppendixB配方的区别保留。
- NormFormer公开README披露原Table2私有大数据；公开训练命令的最大更新上限未被当成论文等GPU时间的实际完成收据。静态代码存在没有被当成原成绩可完整复现。

当前冻结Qwen的单个16×128消息Reader、query LN及tanh gate与两套原网络的差别已明确。输出分支LN／TRAIN尺度校准仍是独立、**未执行**的未来假说，不把原论文成绩或ReZero全网性质直接套给当前任务。零任务梯度与AdamW仍可能weight decay的界限也正确保留。

本次没有发现需要修改既有报告、源码、矩阵或评价的事项。只读重算中的1e−12是浮点一致性核查容差，不是新基准接受阈值。本审查没有重新下载资料、复读全部PDF或执行源码；论文关联依据已存provenance与静态文件，不是新的全套论文复现验收。
