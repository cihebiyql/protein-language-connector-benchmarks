# 蛋白质语言连接器：基准、文献与可分析结果

Protein-language connector benchmarks, literature, frozen data, and auditable results.

**2026-10-17更新：[分子记忆V3框架与已验收基线](experiments/molecular_memory_v3/README.md)已公开。** 包含新模型、自动训练／评测／消融代码、两场景均值／中位数／Ridge／kNN的完整预测和拟合状态，以及真实TRAIN试跑证据。36组新消融队列的远端最终结果尚未取回，不计为已完成，也不宣称新连接器取得稳定优势。

**2026-10-16历史实验状态：36项预登记实验中，30项验收通过、1项池化尝试验收拒绝、5项未启动。三种子结果尚未确立稳定的中层连接器优势。** 未验收尝试的原始数值保留在隔离状态，不进入正式统计；该历史36项与V3的新36项分别登记。

| 真实消息，pKD MAE（越低越好） | early，三seed均值±样本SD | mid，三seed均值±样本SD |
|---|---:|---:|
| 新肽／已见靶点 | 1.961566 ± 0.259145 | 2.070668 ± 0.209683 |
| 新肽／新靶点 | 1.568852 ± 0.141596 | 1.606038 ± 0.243510 |

两场景的深度效应均为两个种子不利、一个有利。门控初始化对照也随种子翻转；池化两个场景正式覆盖均为0／3，mean和SD保持null。

## 开始分析

Python 3.10或以上，默认分析只使用标准库，不需要GPU、模型、账号或下载。

```bash
git clone https://github.com/cihebiyql/protein-language-connector-benchmarks.git
cd protein-language-connector-benchmarks
python -m unittest discover -s tests_public -v
python -m analysis.analyze_results validate --repo-root .
python -m analysis.analyze_results analyze --repo-root . --output analysis_output
```

具体命令及输出字段以[分析指南](docs/ANALYSIS_GUIDE.md)为准。完整运行状态见[36项机器结果](results/matched_budget/final_summary.json)，方法与证据见[研究报告](reports/benchmark_clarity_20261016/FINAL_REPORT.md)。

## 内容索引

| 目录 | 内容 |
|---|---|
| `analysis/`、`tests_public/` | 无模型结果复算、严格解析、完整分母和seed覆盖验证 |
| `results/matched_budget/` | 全部36项状态、31个实际尝试的预测／C1／history／选点及原验收结果 |
| `results/legacy/` | 早期探索、既有数值／神经基线及其他面板的历史结果；预算与本轮不同 |
| `data/a1/` | 两场景冻结samples、gold和lock；PPIKB派生数据，CC BY 4.0 |
| `data/rt_target_diagnostic/` | CC0目标域RT诊断数据；原跨条件A2仍HOLD |
| `checkpoints/` | 自有Reader、固定投影与未训练初态的纯tensor导出，详见[说明](docs/CHECKPOINTS.md) |
| `historical/` | 历史训练源码、测试、配置与观测环境；不作为当前可直接启动的训练入口 |
| `reports/` | 基准定义、参数与通信账本、机制分析、文献证据、审查与图表 |
| `third_party/papers/` | 经逐文件许可核验的原字节论文；保留各自原许可 |
| `catalogs/` | 全部文献版本、作者仓库、数据／模型官方下载地址、commit、大小和校验值 |
| `publication/` | source→public双SHA、元数据规范化及未打包项目记录 |

## 比较的是什么

冻结ESM2-150M第30层残基表示和Qwen2.5-1.5B-Instruct，只训练同构Reader。early在首decoder块前、mid在第14块后动态读取；两者都是464,001个名义训练参数，每次读取16×128 float32，即8,192逻辑字节，不增加LM位置。

每配置固定3轮、1,548次更新，按原完整验证答案NLL最早最小值选点，greedy最多24个新token，严格单字段有限数值JSON。任一解析失败则完整MAE为null；不足3／3有效种子不报告三seed均值或SD。实际FLOPs、累计EOS读取和传播路径另外记账。

数据、主终点、分组、失败分母及原选点规则均保留。两个场景独立训练但来源总体重叠，不能当成两次独立外部复验。数据缺少充分核实的实际构建物、温度、pH等信息，当前是数据库pKD预测基准。

## 如何解释结果

位置／内容比较是分别训练和验证选点系统的总效应。C1的branch on/off是对答案概率的计算干预，包含格式适配；它不直接等于分子信息量。合法独立供体仍为0，P/H保持undefined，新固定消息系统的C2尚未执行。

门控六对中near-unit的验证NLL全部更高，但测试MAE有四对更低；因此不拿NLL或范数替代任务端点，也不根据test重新选择checkpoint。尺度与优化轨迹提供机制线索，尚未唯一定位不稳定的原因。基于Flamingo、BLIP-2、ProtT3、NormFormer、ReZero等原文提出的方案及新增预算见[改进方案](reports/benchmark_clarity_20261016/evidence_backed_improvement_plan.md)。

## 公开副本与历史原件

公开前对本机路径、主机身份和行政元数据作规范化，并逐文件记录原SHA与公开SHA；原科研文件不修改。冻结数据、原始预测、C1和history等允许且无隐私的文件保持原字节；所选模型只作CPU纯tensor封装转换，不做新训练或推理。

历史训练合同的2026-10-16窗口已结束；历史runner保留SHA门槛和原时限。公共化路径或元数据副本不冒充旧原件，也不自动重新开启旧运行。先用上面的无模型命令复核已有结果，详见[历史代码与复现范围](docs/HISTORICAL_CODE.md)。

## 论文与许可

全部本地论文／附件版本均登记于[文献目录](catalogs/papers.md)。许可明确的全文原样附带；没有公开重分发许可的版本提供题名、官方链接、版本和SHA，不镜像全文。论文、数据、作者代码与模型权重分别适用各自条款，参见[许可范围](docs/PUBLICATION_SCOPE.md)及[第三方论文声明](third_party/papers/NOTICE.md)。

本仓库没有擅自为原项目代码增加新的通用开源许可。公开可见、GitHub可fork与获得第三方内容的任意再许可是不同事项。

大型预训练模型、原始数据包与上游仓库树不重复上传，保留[官方资产索引](catalogs/official_downloads.json)与[58个上游仓库索引](catalogs/upstream_repositories.json)。已下载不等于许可允许重新分发；数据与权重的存在也不等于执行了对应作者基准。

