# 同肽多问题：共享4B分子接口的最终验证

完整实验约定见 [PROTOCOL.md](PROTOCOL.md)，机器可执行标准在 [config.json](config.json) 的 `decision` 中。

一批相同的肽、两个实测性质（RT和CCS）、一个组合问题。三个方案各用同一个共享生成连接器回答全部问题，固定三个随机种子，共九组。主成绩来自Qwen自由生成的yes/no，不使用任务专属预测头。TEXT对照得到仅用TRAIN拟合的专用预测属性；训练行使用嵌套分组OOF输出。

本轮只检验既定分析性质范围内，分子表示是否比预测属性文字带来可靠增益。按预设收益、跨种子及分子替换标准自动决定继续或暂停。旧实验全部保留，不按成绩修改测试集或评价标准。

## 执行顺序

1. `src/qa_data.py` 固定公开配对数据、隔离旧序列和8mer组、生成问题阈值与置换。
2. `src/qa_features.py` 提取ESM第15/30层，生成TRAIN内嵌套交叉拟合属性对照。
3. `src/qa_train.py --phase smoke` 只用TRAIN做技术资格测试。
4. 冻结全部代码、数据、特征、属性和配置的SHA，先同步预注册协议到GitHub。
5. `resources/node_pipeline.py --stage launch` 在node1独立训练九组，逐组验收，调用分析及发布程序；无需本机在线。
6. `src/qa_analyze.py` 重算实际生成答案并执行停止规则，`resources/publish_v6.py` 上传允许公开的代码、划分、检查点、回答、报告及SHA回执。

实际运行记录在 `STATUS.json`、`runs/` 和 `analysis/`；公开完成状态在 `COMPLETION_STATUS.json`。预注册阶段没有正式结果，不能将技术试跑当成有效性证据。

## 复现与公开范围

公开的数据来自 [Meier等人2021论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7896072/) 的 [固定作者处理输出](https://raw.githubusercontent.com/MannLabs/DeepCollisionalCrossSection/5187be8348ecea98565ef56c7be51a8dc49d8524/output/evidence_aligned_PT.csv)，引用与处理范围见协议。保留固定版本、源文件指纹、实际处理代码和新划分。

为避免复制冻结底座和大特征文件，公开包保留其SHA及提取程序；本机安装路径转成 `${QWEN3_4B_MODEL_PATH}`、`${ESM150M_MODEL_PATH}` 和 `${EXPERIMENT_SCOPE}`。`PUBLICATION_MANIFEST.json` 同时记录原始与公开文件指纹及变换。原始节点冻结清单的配置SHA对应节点原文件；在其他机器配置模型路径后应另生成复现清单。

TEXT额外上游预测器及文本token成本独立记录；三者连接器参数量相同，不等于整个系统成本相同。训练种子是同一测试面板上的三次拟合，不是三个独立生物学数据集。此实验也不是原论文外部测试划分复现。
